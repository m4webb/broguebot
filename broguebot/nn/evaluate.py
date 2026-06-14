"""Evaluate a checkpoint on a fixed seed suite for clean A/B comparisons.

  .venv/bin/python -m broguebot.nn.evaluate --ckpt runs/ppo/ppo.pt --games 50
"""

import argparse
import collections

import torch

from ..env import BrogueEnv, wipe_gamedata
from .featurize import batch as batch_feats, featurize
from .model import BroguePolicy, Config


EVAL_SEEDS = [1000003 + 7919 * i for i in range(200)]  # fixed suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = BroguePolicy(getattr(Config, ckpt["config"])()).to(args.device)
    model.load_state_dict(ckpt["model"])
    # re-apply the manual-play action mask the checkpoint was trained with
    # (non-persistent buffer, so it isn't in state_dict)
    if ckpt.get("disabled"):
        model.set_disabled_actions(ckpt["disabled"])
        print(f"disabled actions: {ckpt['disabled']}")
    model.eval()

    wipe_gamedata("gamedata/eval")
    env = BrogueEnv("gamedata/eval", max_steps=args.max_steps)
    depths, causes = [], collections.Counter()
    for i in range(args.games):
        frame = env.reset(EVAL_SEEDS[i % len(EVAL_SEEDS)])
        hidden = model.initial_state(1, args.device)
        done = False
        while not done:
            obs = {k: torch.as_tensor(v, device=args.device)
                   for k, v in batch_feats([featurize(frame.raw)]).items()}
            with torch.no_grad():
                logits, _v, _a, hidden = model(obs, hidden)
            act = torch.distributions.Categorical(
                logits=logits / args.temperature).sample().item()
            frame, _r, done, info = env.step(act)
        depths.append(info.get("depth", 1))
        causes[info.get("killer") or info.get("cause") or "?"] += 1
        print(f"game {i+1}: depth {depths[-1]} "
              f"({info.get('killer', '?')}, {info['steps']} steps)",
              flush=True)
    env.close()
    print(f"\navg depth {sum(depths)/len(depths):.2f}  "
          f"max {max(depths)}  dist {collections.Counter(depths)}")
    for cause, n in causes.most_common(10):
        print(f"  {n:3d}  {cause}")


if __name__ == "__main__":
    main()
