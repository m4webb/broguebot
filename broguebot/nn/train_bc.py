"""Behavioral cloning: imitate logged trajectories.

Samples fixed-length windows from episodes, teacher-forces the recurrent
unroll from a zero hidden state (windows are long enough that the memory
warms up), cross-entropy on actions.

  .venv/bin/python -m broguebot.nn.train_bc --data data/traj --out runs/bc
"""

import argparse
import contextlib
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .featurize import batch as batch_feats, featurize
from .model import BroguePolicy, Config, count_params
from .trajlog import episode_files, load_episode


class WindowSampler:
    def __init__(self, files: list[str], window: int, cache: int = 64,
                 gamma: float = 0.999):
        self.files = files
        self.window = window
        self.cache = {}
        self.cache_max = cache
        self.rng = random.Random(0)
        self.gamma = gamma

    def _episode(self, path: str) -> dict:
        if path not in self.cache:
            if len(self.cache) >= self.cache_max:
                self.cache.pop(next(iter(self.cache)))
            ep = load_episode(path)
            # discounted return-to-go per step (for the value-head warmup);
            # computed once over the full episode, then sliced per window
            r = ep["rewards"].astype(np.float32)
            ret = np.zeros_like(r)
            run = 0.0
            for i in range(len(r) - 1, -1, -1):
                run = r[i] + self.gamma * run
                ret[i] = run
            ep["returns"] = ret
            self.cache[path] = ep
        return self.cache[path]

    def sample(self, batch_size: int) -> dict:
        # Every window must be exactly self.window long so the batch stacks
        # into a rectangular tensor; episodes shorter than the window are
        # skipped (resampled) rather than yielding a ragged batch.
        obs, acts, rets = [], [], []
        attempts = 0
        while len(obs) < batch_size:
            attempts += 1
            if attempts > 1000 * batch_size:
                raise RuntimeError(
                    f"too few episodes with length >= window={self.window}; "
                    "lower --window or collect longer episodes")
            ep = self._episode(self.rng.choice(self.files))
            T = len(ep["actions"])
            if T < self.window:
                continue
            start = self.rng.randrange(0, T - self.window + 1)
            end = start + self.window
            feats = [featurize(ep["frames"][t].tobytes())
                     for t in range(start, end)]
            obs.append(batch_feats(feats))
            acts.append(ep["actions"][start:end].astype(np.int64))
            rets.append(ep["returns"][start:end])
        out = {k: torch.as_tensor(np.stack([o[k] for o in obs]))
               for k in obs[0]}
        return (out, torch.as_tensor(np.stack(acts)),
                torch.as_tensor(np.stack(rets)))


class StreamSampler:
    """Streaming (stateful) sampler for truncated-BPTT BC.

    Keeps `batch` parallel cursors, each walking one episode front-to-back in
    fixed `window`-sized steps. Across training steps the caller carries the
    GRU hidden forward per row (detached = truncated BPTT) and zeroes it only
    where `reset` is True (a row that just (re)started an episode). So the
    hidden state accumulates over the WHOLE episode prefix while backprop spans
    only `window` steps — memory-use without the long-fixed-window overfit.
    Windows never straddle an episode boundary, so no in-window done masking.
    """

    def __init__(self, files: list[str], window: int, batch: int,
                 cache: int = 64, gamma: float = 0.999):
        self.files = files
        self.window = window
        self.batch = batch
        self.gamma = gamma
        self.cache = {}
        self.cache_max = max(cache, batch * 2)
        self.rng = random.Random(0)
        self.eps = [None] * batch       # current episode per stream
        self.cursor = [0] * batch
        self.fresh = [True] * batch     # just (re)started -> zero hidden

    _episode = WindowSampler._episode    # same caching + returns computation

    def _assign(self, i: int):
        while True:
            ep = self._episode(self.rng.choice(self.files))
            if len(ep["actions"]) >= self.window:
                self.eps[i], self.cursor[i], self.fresh[i] = ep, 0, True
                return

    def sample(self):
        """Returns (obs(B,W,...), acts(B,W), rets(B,W), reset(B,) bool)."""
        obs, acts, rets, reset = [], [], [], []
        for i in range(self.batch):
            if (self.eps[i] is None or
                    self.cursor[i] + self.window > len(self.eps[i]["actions"])):
                self._assign(i)
            ep = self.eps[i]
            s = self.cursor[i]
            e = s + self.window
            feats = [featurize(ep["frames"][t].tobytes()) for t in range(s, e)]
            obs.append(batch_feats(feats))
            acts.append(ep["actions"][s:e].astype(np.int64))
            rets.append(ep["returns"][s:e])
            reset.append(self.fresh[i])
            self.cursor[i] = e
            self.fresh[i] = False
        out = {k: torch.as_tensor(np.stack([o[k] for o in obs]))
               for k in obs[0]}
        return (out, torch.as_tensor(np.stack(acts)),
                torch.as_tensor(np.stack(rets)),
                torch.tensor(reset, dtype=torch.bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/bc")
    ap.add_argument("--config", default="small", choices=["small", "base"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction,
                    default=True, help="bf16 autocast (CUDA only)")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                    default=True, help="recompute encoder in backward")
    ap.add_argument("--chunk", type=int, default=128,
                    help="frames per encoder mini-batch in the unroll (0=all); "
                    "sets backprop peak memory — keep ~128 to stay in 12GB")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder (~1.2x; adds startup cost)")
    ap.add_argument("--skip-truncated", action=argparse.BooleanOptionalAction,
                    default=True, help="drop episodes that hit the step cap "
                    "(stuck loops) rather than ending naturally")
    ap.add_argument("--value-coef", type=float, default=0.0,
                    help="if >0, also train the value head to predict returns "
                    "(MSE) — calibrates it so PPO --init doesn't start with a "
                    "random value fn (fixes PPO-from-confident-base collapse)")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--stateful", action="store_true",
                    help="streaming truncated-BPTT BC: carry GRU hidden across "
                    "consecutive windows of each episode (memory-use without "
                    "long-fixed-window overfit). Use a short --window (16-32).")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of episodes held out to measure the "
                    "train/val accuracy gap (overfitting); 0 disables")
    ap.add_argument("--disable-actions", default="",
                    help="comma-separated action names to forbid (mask logits "
                    "to -inf; their targets are ignored in the CE loss), e.g. "
                    "'explore,descend,ascend' for manual tile-by-tile play")
    args = ap.parse_args()
    dev = args.device

    def amp_ctx():
        if args.amp and dev == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    files = []
    for d in args.data.split(","):
        files += episode_files(d.strip(), skip_truncated=args.skip_truncated)
    if not files:
        raise SystemExit(f"no episodes in {args.data}")
    # held-out split (by episode) to measure the overfitting gap
    random.Random(1234).shuffle(files)
    n_val = int(len(files) * args.val_frac)
    val_files = files[:n_val]
    train_files = files[n_val:] or files
    print(f"{len(files)} episodes ({len(train_files)} train / {len(val_files)} "
          f"val), device={args.device}, "
          f"mode={'stateful' if args.stateful else 'window'}")

    def make_sampler(fs):
        if args.stateful:
            return StreamSampler(fs, args.window, args.batch, gamma=args.gamma)
        return WindowSampler(fs, args.window, gamma=args.gamma)

    sampler = make_sampler(train_files)
    val_sampler = make_sampler(val_files) if val_files else None

    @torch.no_grad()
    def val_acc(n=20):
        if val_sampler is None:
            return float("nan")
        model.eval()
        h = (model.initial_state(args.batch, dev) if args.stateful else None)
        accs = []
        for _ in range(n):
            if args.stateful:
                vo, va, _vr, vreset = val_sampler.sample()
                hid = h.detach() * (~vreset).to(dev).float().unsqueeze(-1)
            else:
                vo, va, _vr = val_sampler.sample(args.batch)
                hid = model.initial_state(args.batch, dev)
            vo = {k: v.to(dev) for k, v in vo.items()}
            va = va.to(dev)
            with amp_ctx():
                vl, _vv, h = model.unroll(vo, hid)
            accs.append((vl.float().argmax(-1) == va).float().mean().item())
        model.train()
        return sum(accs) / len(accs)

    model = BroguePolicy(getattr(Config, args.config)()).to(args.device)
    model.grad_checkpoint = args.grad_checkpoint
    model.encode_chunk = args.chunk
    if args.compile:
        model.encoder = torch.compile(model.encoder)
    disabled = [s for s in args.disable_actions.split(",") if s] \
        if args.disable_actions else []
    disabled_idx = None
    if disabled:
        from ..env import ACTION_INDEX
        model.set_disabled_actions(disabled)
        disabled_idx = torch.tensor([ACTION_INDEX[n] for n in disabled],
                                    device=dev)
        print(f"manual play: disabled actions {disabled} "
              "(masked logits + ignored in CE)")
    print(f"params: {count_params(model)/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    # stateful mode carries one persistent hidden across training steps
    # (truncated BPTT); window mode re-zeros it every step.
    carry = (model.initial_state(args.batch, args.device)
             if args.stateful else None)
    best_val = -1.0
    for step in range(1, args.steps + 1):
        if args.stateful:
            obs, acts, rets, reset = sampler.sample()
        else:
            obs, acts, rets = sampler.sample(args.batch)
        obs = {k: v.to(args.device) for k, v in obs.items()}
        acts = acts.to(args.device)
        rets = rets.to(args.device)
        if args.stateful:
            # detach (cut BPTT at the window boundary) and zero rows that
            # just (re)started an episode
            keep = (~reset).to(args.device).float().unsqueeze(-1)
            hidden = carry.detach() * keep
        else:
            hidden = model.initial_state(acts.shape[0], args.device)
        with amp_ctx():
            logits, values, new_hidden = model.unroll(obs, hidden)
        if args.stateful:
            carry = new_hidden
        logits, values = logits.float(), values.float()
        flat_acts = acts.reshape(-1)
        if disabled_idx is not None:
            # a disabled action as the target would give CE on a -inf logit
            # (= inf loss); ignore those steps (rare in human manual play)
            flat_acts = flat_acts.clone()
            flat_acts[torch.isin(flat_acts, disabled_idx)] = -100
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             flat_acts, ignore_index=-100)
        v_loss = F.mse_loss(values, rets) if args.value_coef > 0 \
            else torch.zeros((), device=dev)
        loss = ce + args.value_coef * v_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            acc = (logits.argmax(-1) == acts).float().mean().item()
            va = val_acc() if (step % 250 == 0 or step == 1) else None
            vstr = f" val_acc {va:.3f}" if va is not None else ""
            print(f"step {step}: loss {ce.item():.3f} acc {acc:.3f}{vstr} "
                  f"vloss {v_loss.item():.3f} "
                  f"({step/(time.time()-t0):.1f} it/s)", flush=True)
            # keep the highest-val-acc checkpoint (pre-overfit), separate from
            # the final one — the overfit base capped the last attempt
            if va is not None and va > best_val:
                best_val = va
                torch.save({"model": model.state_dict(),
                            "config": args.config, "val_acc": va, "step": step,
                            "disabled": disabled},
                           os.path.join(args.out, "bc_best.pt"))
        if step % 500 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "config": args.config, "disabled": disabled},
                       os.path.join(args.out, "bc.pt"))
    print(f"saved {os.path.join(args.out, 'bc.pt')}; best val_acc {best_val:.3f} "
          f"-> {os.path.join(args.out, 'bc_best.pt')}")


if __name__ == "__main__":
    main()
