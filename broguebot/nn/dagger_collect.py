"""DAgger data collection: roll out a policy in the IPC env and label each
state IT visits with the scripted teacher's action. Writes a trajlog dataset
of (frame, teacher_action) — states from the policy's distribution, actions
from the expert. This corrects the policy's own mistakes (e.g. not resting at
low HP) the way plain BC on teacher trajectories can't.

  python -m broguebot.nn.dagger_collect --ckpt runs/ppo_warm/ppo.pt \
      --out data/dagger1 --episodes 300 --seed 9000
"""

import argparse
import time

import torch

from ..brain import Brain, BotConfig, EpisodeOver
from ..env import ACTIONS, BrogueEnv, wipe_gamedata
from .featurize import batch as batch_feats, featurize
from .model import BroguePolicy, Config
from .scripted_actor import (_StubPane, frame_to_snapshot, inventory_from_frame,
                             key_to_action)
from .trajlog import TrajectoryWriter


def teacher_label(brain, frame, policy_keys):
    """The expert action for the policy's current state (DAgger query)."""
    snap = frame_to_snapshot(frame)
    brain.pane._frame = frame
    brain.inv = inventory_from_frame(frame)
    brain.inv_dirty = False
    if snap.mode != "game":
        return None                       # prompt frame: policy handles it
    brain.prev_keys = policy_keys         # reflect that the policy just acted
    try:
        brain.observe(snap)
        act = brain.decide(snap)
    except EpisodeOver:
        return None
    for k in act.keys:
        idx = key_to_action(k)
        if idx is not None:
            return idx
    return None


def collect(ckpt_path, out_dir, episodes, seed_start, device, gamedata,
            max_steps, temperature):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BroguePolicy(getattr(Config, ckpt["config"])()).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path} ({ckpt['config']})")

    wipe_gamedata(gamedata)
    env = BrogueEnv(gamedata, max_steps=max_steps)
    writer = TrajectoryWriter(out_dir)
    t0 = time.time()
    tot_steps = tot_labeled = 0
    depths = []
    for ep in range(episodes):
        seed = seed_start + ep
        frame = env.reset(seed)
        brain = Brain(_StubPane(), BotConfig(headless=False))
        brain.pane._frame = frame
        hidden = model.initial_state(1, device)
        writer.start_episode(env.seed)
        done = False
        last_keys = []
        steps = labeled = 0
        while not done:
            obs = {k: torch.as_tensor(v, device=device)
                   for k, v in batch_feats([featurize(frame.raw)]).items()}
            with torch.no_grad():
                logits, _v, _a, hidden = model(obs, hidden)
            pol = torch.distributions.Categorical(
                logits=logits / temperature).sample().item()
            label = teacher_label(brain, frame, last_keys)
            if label is not None:
                writer.record(frame.raw, label, 0.0)
                labeled += 1
            key = ACTIONS[pol][1]
            last_keys = [key] if isinstance(key, str) else []
            frame, _r, done, info = env.step(pol)
            steps += 1
        writer.end_episode(info)
        tot_steps += steps
        tot_labeled += labeled
        depths.append(info.get("depth", 1))
        if (ep + 1) % 10 == 0 or ep == 0:
            sps = tot_steps / (time.time() - t0)
            print(f"ep {ep+1}/{episodes} seed={seed} depth={info.get('depth')} "
                  f"labeled {labeled}/{steps} | {sps:.0f} sps "
                  f"meanDepth={sum(depths)/len(depths):.2f}", flush=True)
    env.close()
    print(f"\ndone: {episodes} eps, {tot_labeled} labeled states "
          f"({tot_labeled/max(tot_steps,1):.0%} of {tot_steps} steps), "
          f"mean depth {sum(depths)/len(depths):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="policy to roll out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gamedata", default="gamedata/dagger")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    collect(args.ckpt, args.out, args.episodes, args.seed, args.device,
            args.gamedata, args.max_steps, args.temperature)


if __name__ == "__main__":
    main()
