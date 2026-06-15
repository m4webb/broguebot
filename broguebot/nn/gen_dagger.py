"""DAgger data collection: roll out the BC policy, label its states with the oracle.

The manual-teacher BC clones well (val 0.805) but rollouts cap at depth ~1.35 from
compounding errors (BC distribution shift): the policy drifts to states the teacher
never visited and has no correct label there. DAgger fixes this — run the LEARNER,
and at every state it visits record the EXPERT's move (Brogue's autoexplore oracle,
BB_ORACLE_EXPORT). Aggregating these corrections with the original data and
retraining teaches the policy to recover from its own mistakes.

  .venv/bin/python -m broguebot.nn.gen_dagger --ckpt runs/manual_bc2/bc_best.pt \
      --out data/dagger --episodes 200 --seed 10000
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ..env import ACTION_INDEX, BrogueEnv, wipe_gamedata
from ..ipc import FRAME_SIZE
from .featurize import batch as batch_feats, featurize
from .model import BroguePolicy, Config

DIR2ACTION = np.array(
    [ACTION_INDEX[n] for n in ("move_n", "move_s", "move_w", "move_e",
                               "move_nw", "move_sw", "move_ne", "move_se")],
    dtype=np.int16)
_REC = FRAME_SIZE + 1


def convert(binpath: str):
    """Oracle export -> (frames, oracle-move actions, zero rewards). The oracle
    move is the LABEL regardless of what the policy did, so no move-filtering."""
    with open(binpath, "rb") as fh:
        data = fh.read()
    n = len(data) // _REC
    if n == 0:
        return None
    frames = np.stack([np.frombuffer(data, np.uint8, FRAME_SIZE, i * _REC)
                       for i in range(n)])
    dirs = np.array([data[i * _REC + FRAME_SIZE] for i in range(n)], np.uint8)
    return frames, DIR2ACTION[dirs], np.zeros(n, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="data/dagger")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10000)  # disjoint from train+eval
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--gamedata", default="gamedata/dagger")
    ap.add_argument("--tmp", default="/tmp/bbdagger")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = BroguePolicy(getattr(Config, ckpt["config"])()).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    if ckpt.get("disabled"):
        model.set_disabled_actions(ckpt["disabled"])

    wipe_gamedata(args.gamedata)
    env = BrogueEnv(args.gamedata, max_steps=args.max_steps)
    t0 = time.time()
    tot = 0
    depths = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        binp = os.path.join(args.tmp, f"ep_{seed}.bin")
        try:
            os.remove(binp)
        except FileNotFoundError:
            pass
        os.environ["BB_ORACLE_EXPORT"] = binp
        frame = env.reset(seed)
        hidden = model.initial_state(1, args.device)
        done = False
        info = {}
        with torch.no_grad():
            while not done:
                obs = {k: torch.as_tensor(v, device=args.device)
                       for k, v in batch_feats([featurize(frame.raw)]).items()}
                logits, _v, _a, hidden = model(obs, hidden)
                act = torch.distributions.Categorical(
                    logits=logits / args.temperature).sample().item()
                frame, _r, done, info = env.step(act)
        conv = convert(binp) if os.path.exists(binp) else None
        moves = 0
        if conv is not None and len(conv[1]) >= 20:
            frames, actions, rewards = conv
            moves = len(actions)
            meta = json.dumps({"seed": seed, "depth": info.get("depth", 1),
                               "truncated": bool(info.get("truncated"))})
            np.savez_compressed(
                os.path.join(args.out, f"ep_{seed}.npz"),
                frames=frames, actions=actions, rewards=rewards,
                meta=np.frombuffer(meta.encode(), np.uint8))
            tot += moves
            depths.append(info.get("depth", 1))
        try:
            os.remove(binp)
        except FileNotFoundError:
            pass
        if ep == 0 or (ep + 1) % 10 == 0:
            md = np.mean(depths) if depths else 0
            print(f"ep {ep + 1}/{args.episodes} seed={seed} oracle_labels={moves} "
                  f"policy_depth={info.get('depth')} | kept {len(depths)} "
                  f"policy_mean_depth {md:.2f} ({tot / (time.time() - t0):.0f}/s)",
                  flush=True)
    env.close()
    md = np.mean(depths) if depths else 0
    print(f"\ndone: {len(depths)} eps, {tot} oracle-labeled states, "
          f"policy mean depth {md:.2f}")


if __name__ == "__main__":
    main()
