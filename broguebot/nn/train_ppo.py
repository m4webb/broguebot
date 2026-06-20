"""Recurrent PPO over vectorized BrogueEnvs.

Single process: N threaded envs collect a T-step segment, then PPO
updates with GAE. Hidden states are carried across segments and reset at
episode boundaries (also masked during the training-time re-unroll).
Optionally initialize from a BC checkpoint (--init): initialization only,
no KL tether — the optimizer is free to leave the prior behind.

  .venv/bin/python -m broguebot.nn.train_ppo --out runs/ppo --envs 8
"""

import argparse
import contextlib
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..env import VectorEnv, wipe_gamedata
from .rewards import REWARDS
from .rnd import RND, CountNovelty
from .featurize import batch as batch_feats, featurize
from .model import BroguePolicy, Config, count_params


def amp_ctx(enabled: bool, dev: str):
    """bf16 autocast on CUDA when enabled; a no-op otherwise."""
    if enabled and dev == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def masked_unroll(model, obs_seq, hidden, dones):
    """Like model.unroll but resets hidden where dones[t-1] is set."""
    B, T = obs_seq["glyphs"].shape[:2]
    flat = {k: v.reshape(B * T, *v.shape[2:]) for k, v in obs_seq.items()}
    z = model.encode_frames(flat).reshape(B, T, -1)
    logits, values = [], []
    for t in range(T):
        if t > 0:
            keep = (~dones[:, t - 1]).float().unsqueeze(-1)
            hidden = hidden * keep
        hidden = model.memory(z[:, t], hidden)
        h = model.post(torch.cat([hidden, z[:, t]], dim=-1))
        logits.append(model.policy_head(h) + model.action_mask)
        values.append(model.value_head(h).squeeze(-1))
    return torch.stack(logits, 1), torch.stack(values, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--config", default="small", choices=["small", "base", "fine"])
    ap.add_argument("--init", help="BC checkpoint to start from")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--segment", type=int, default=128)
    ap.add_argument("--updates", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--entropy-final", type=float, default=None,
                    help="if set, entropy coef anneals linearly --entropy -> this "
                    "over the run (a constant bonus over-explores late; anneal to "
                    "e.g. 0.001 lets the policy sharpen as it converges)")
    ap.add_argument("--entropy-anneal-updates", type=int, default=None,
                    help="anneal --entropy -> --entropy-final over the FIRST this-many "
                    "updates, then hold at the floor (default: over the whole run). "
                    "A short horizon (e.g. 400) lets the policy commit early instead "
                    "of churning at high entropy — day3 cold-start wasted ~600 updates "
                    "on the plateau because the full-run anneal kept ent_coef ~0.0055 "
                    "until upd ~700.")
    ap.add_argument("--gamedata", default="gamedata/ppo")
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
    ap.add_argument("--reward", default="default", choices=list(REWARDS),
                    help="reward shaping: default|explore|survival|dense")
    ap.add_argument("--rnd", action="store_true",
                    help="IGNORE the extrinsic reward and train on pure RND "
                    "intrinsic 'interestingness' (novelty of the on-screen "
                    "glyph+RGB state). Single self-supervised signal; descent/"
                    "exploration/combat emerge from novelty-seeking.")
    ap.add_argument("--rnd-lr", type=float, default=1e-4,
                    help="learning rate for the RND predictor network")
    ap.add_argument("--rnd-mode", default="rnd", choices=["rnd", "count"],
                    help="intrinsic novelty type: rnd (prediction-error; can "
                    "collapse) or count (SimHash pseudo-counts; collapse-proof)")
    ap.add_argument("--rnd-bits", type=int, default=22,
                    help="count mode: SimHash bits. Fewer = coarser = per-level "
                    "novelty saturates sooner (more descent pressure); too few "
                    "= states collide into noise.")
    ap.add_argument("--rnd-depth-key", action="store_true",
                    help="count mode: stratify novelty buckets by dungeon depth, "
                    "so a new level is maximally novel. Diagnostic: does a "
                    "depth-salient novelty metric induce descent?")
    ap.add_argument("--disable-actions", default="",
                    help="comma-separated action names to forbid (mask logits "
                    "to -inf), e.g. 'explore,descend,ascend' to force manual "
                    "tile-by-tile navigation. Recorded in the checkpoint.")
    ap.add_argument("--max-steps", type=int, default=20000,
                    help="per-episode step cap. Lower (~2500) for macro-free "
                    "manual play: long wandering episodes give huge-variance "
                    "returns that destabilize PPO; capping forces efficient "
                    "descent and steadies training.")
    args = ap.parse_args()
    dev = args.device

    model = BroguePolicy(getattr(Config, args.config)()).to(dev)
    model.grad_checkpoint = args.grad_checkpoint
    model.encode_chunk = args.chunk
    if args.compile:
        model.encoder = torch.compile(model.encoder)
    if args.init:
        ckpt = torch.load(args.init, map_location=dev)
        model.load_state_dict(ckpt["model"])
        print("initialized from", args.init)
    disabled = [s for s in args.disable_actions.split(",") if s] \
        if args.disable_actions else []
    if disabled:
        model.set_disabled_actions(disabled)
        print(f"manual play: disabled actions {disabled}")
    print(f"params: {count_params(model)/1e6:.2f}M device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    nov_cls = {"rnd": RND, "count": CountNovelty}[args.rnd_mode]
    rnd = nov_cls(dev, lr=args.rnd_lr, bits=args.rnd_bits,
                  depth_key=args.rnd_depth_key) if args.rnd else None
    if rnd is not None:
        print(f"intrinsic novelty ON ({args.rnd_mode}) — extrinsic reward IGNORED")
    os.makedirs(args.out, exist_ok=True)

    wipe_gamedata(args.gamedata)
    vec = VectorEnv(args.envs, args.gamedata, reward_fn=REWARDS[args.reward],
                    max_steps=args.max_steps)
    frames = vec.reset()
    hidden = model.initial_state(args.envs, dev)
    ep_returns, ep_depths = [], []
    best_depth = 0.0   # best smoothed eval depth -> ppo_best.pt (the trace
                       # oscillates, so the last %50 checkpoint is often a trough)
    N, T = args.envs, args.segment
    # entropy coefficient anneals linearly --entropy -> --entropy-final over the
    # run. A constant bonus over-explores late (day3: best plateaued while entropy
    # drifted up); annealing lets the policy sharpen as it converges.
    ent_final = args.entropy if args.entropy_final is None else args.entropy_final
    anneal_n = args.entropy_anneal_updates or args.updates

    for update in range(1, args.updates + 1):
        frac = min(1.0, (update - 1) / max(1, anneal_n - 1))
        ent_coef = args.entropy + frac * (ent_final - args.entropy)
        t0 = time.time()
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done = \
            [], [], [], [], [], []
        seg_hidden = hidden.detach()
        with torch.no_grad():
            for _ in range(T):
                feats = batch_feats([featurize(f.raw) for f in frames])
                obs = {k: torch.as_tensor(v, device=dev)
                       for k, v in feats.items()}
                with amp_ctx(args.amp, dev):
                    logits, value, _aux, hidden = model(obs, hidden)
                # Carry hidden/value in fp32: recurrent state and GAE math
                # are precision-sensitive; only the encoder ran in bf16.
                logits, value = logits.float(), value.float()
                hidden = hidden.float()
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                results = vec.step(act.tolist())
                frames = [r[0] for r in results]
                rew = torch.tensor([r[1] for r in results], device=dev)
                done = torch.tensor([r[2] for r in results], device=dev)
                hidden = hidden * (~done).float().unsqueeze(-1)
                for r in results:
                    if r[2]:
                        ep_returns.append(r[3].get("episode_return", 0.0))
                        ep_depths.append(r[3].get("depth", 1))
                buf_obs.append(feats)
                buf_act.append(act)
                buf_logp.append(dist.log_prob(act))
                buf_val.append(value)
                buf_rew.append(rew)
                buf_done.append(done)
            feats = batch_feats([featurize(f.raw) for f in frames])
            obs = {k: torch.as_tensor(v, device=dev)
                   for k, v in feats.items()}
            with amp_ctx(args.amp, dev):
                _, last_value, _, _ = model(obs, hidden)
            last_value = last_value.float()

        acts = torch.stack(buf_act, 1)              # N,T
        logps = torch.stack(buf_logp, 1)
        vals = torch.stack(buf_val, 1)
        rews = torch.stack(buf_rew, 1)
        dones = torch.stack(buf_done, 1)
        obs_seq = {k: torch.as_tensor(
            np.stack([b[k] for b in buf_obs], axis=1), device=dev)
            for k in buf_obs[0]}

        rnd_loss = 0.0
        if rnd is not None:
            # intrinsic 'interestingness' = novelty of each observation; replaces
            # the extrinsic reward entirely. Train the predictor on the same obs.
            flat = {k: v.reshape(N * T, *v.shape[2:]) for k, v in obs_seq.items()}
            raw = rnd.reward_raw(flat).reshape(N, T).float()
            rews = rnd.normalize(raw, args.gamma)
            rnd_loss = rnd.distill(flat)

        adv = torch.zeros_like(rews)
        last_gae = torch.zeros(N, device=dev)
        next_val = last_value
        for t in reversed(range(T)):
            mask = (~dones[:, t]).float()
            delta = rews[:, t] + args.gamma * next_val * mask - vals[:, t]
            last_gae = delta + args.gamma * args.lam * mask * last_gae
            adv[:, t] = last_gae
            next_val = vals[:, t]
        ret = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(args.epochs):
            with amp_ctx(args.amp, dev):
                logits, values = masked_unroll(model, obs_seq, seg_hidden,
                                               dones)
            logits, values = logits.float(), values.float()
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(acts)
            ratio = (new_logp - logps).exp()
            pg = -torch.min(
                ratio * adv,
                ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            v_loss = F.mse_loss(values, ret)
            ent = dist.entropy().mean()
            loss = pg + 0.5 * v_loss - ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        if update % 5 == 0 or update == 1:
            sps = N * T / (time.time() - t0)
            rs, ds = ep_returns[-50:] or [0], ep_depths[-50:] or [1]
            ret_disp = (f"intr {rews.mean().item():.3f} rndL {rnd_loss:.3f}"
                        if rnd is not None else f"ret {sum(rs)/len(rs):.2f}")
            print(f"upd {update}: loss {loss.item():.3f} ent {ent.item():.2f} "
                  f"entc {ent_coef:.4f} "
                  f"{ret_disp} depth {sum(ds)/len(ds):.2f} "
                  f"eps {len(ep_returns)} ({sps:.0f} sps)", flush=True)
        if update % 50 == 0:
            torch.save({"model": model.state_dict(),
                        "config": args.config, "disabled": disabled},
                       os.path.join(args.out, "ppo.pt"))
            # also keep the highest smoothed-depth checkpoint (>=100 eps so the
            # average is meaningful), since ppo.pt may land on an oscillation trough
            smooth = sum(ep_depths[-100:]) / len(ep_depths[-100:]) \
                if ep_depths else 0.0
            if len(ep_depths) >= 100 and smooth > best_depth:
                best_depth = smooth
                torch.save({"model": model.state_dict(), "config": args.config,
                            "smooth_depth": smooth, "update": update,
                            "disabled": disabled},
                           os.path.join(args.out, "ppo_best.pt"))
                print(f"  [best] depth {smooth:.2f} @ upd {update} -> ppo_best.pt",
                      flush=True)
    vec.close()


if __name__ == "__main__":
    main()
