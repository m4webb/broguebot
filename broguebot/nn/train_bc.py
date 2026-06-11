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
    def __init__(self, files: list[str], window: int, cache: int = 64):
        self.files = files
        self.window = window
        self.cache = {}
        self.cache_max = cache
        self.rng = random.Random(0)

    def _episode(self, path: str) -> dict:
        if path not in self.cache:
            if len(self.cache) >= self.cache_max:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = load_episode(path)
        return self.cache[path]

    def sample(self, batch_size: int) -> dict:
        obs, acts = [], []
        for _ in range(batch_size):
            ep = self._episode(self.rng.choice(self.files))
            T = len(ep["actions"])
            w = min(self.window, T)
            start = self.rng.randrange(0, T - w + 1)
            feats = [featurize(ep["frames"][t].tobytes())
                     for t in range(start, start + w)]
            obs.append(batch_feats(feats))
            acts.append(ep["actions"][start:start + w].astype(np.int64))
        out = {k: torch.as_tensor(np.stack([o[k] for o in obs]))
               for k in obs[0]}
        return out, torch.as_tensor(np.stack(acts))


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
    args = ap.parse_args()
    dev = args.device

    def amp_ctx():
        if args.amp and dev == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    files = episode_files(args.data)
    if not files:
        raise SystemExit(f"no episodes in {args.data}")
    print(f"{len(files)} episodes, device={args.device}")
    sampler = WindowSampler(files, args.window)

    model = BroguePolicy(getattr(Config, args.config)()).to(args.device)
    model.grad_checkpoint = args.grad_checkpoint
    model.encode_chunk = args.chunk
    print(f"params: {count_params(model)/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        obs, acts = sampler.sample(args.batch)
        obs = {k: v.to(args.device) for k, v in obs.items()}
        acts = acts.to(args.device)
        hidden = model.initial_state(acts.shape[0], args.device)
        with amp_ctx():
            logits, _values, _ = model.unroll(obs, hidden)
        logits = logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               acts.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            acc = (logits.argmax(-1) == acts).float().mean().item()
            print(f"step {step}: loss {loss.item():.3f} acc {acc:.3f} "
                  f"({step/(time.time()-t0):.1f} it/s)", flush=True)
        if step % 500 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "config": args.config},
                       os.path.join(args.out, "bc.pt"))
    print("saved", os.path.join(args.out, "bc.pt"))


if __name__ == "__main__":
    main()
