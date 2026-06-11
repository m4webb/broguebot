"""Generate behavioral-cloning data by driving the scripted Brain over the
IPC env and recording every (frame, action, reward) step.

  .venv/bin/python -m broguebot.nn.gen_scripted --out data/scripted \
      --episodes 200 --seed 1

The recorded actions are env action indices (0..NUM_ACTIONS-1), exactly the
targets train_bc.py expects. Episodes are independent; on a finished game the
env emits a DONE frame and the actor is reset for the next seed.
"""

import argparse
import time

from ..env import BrogueEnv, wipe_gamedata
from .scripted_actor import ScriptedActor
from .trajlog import TrajectoryWriter


def generate(out, episodes, gamedata, seed_start, flee_hp, max_steps):
    wipe_gamedata(gamedata)
    env = BrogueEnv(gamedata, max_steps=max_steps)
    writer = TrajectoryWriter(out)
    actor = ScriptedActor(flee_hp=flee_hp)
    t0 = time.time()
    total_steps = 0
    depths = []
    for ep in range(episodes):
        seed = seed_start + ep if seed_start is not None else None
        frame = env.reset(seed)
        actor.reset()
        writer.start_episode(env.seed)
        done = False
        steps = 0
        while not done:
            action = actor(frame)
            nxt, reward, done, info = env.step(action)
            writer.record(frame.raw, action, reward)
            frame = nxt
            steps += 1
        path = writer.end_episode(info)
        total_steps += steps
        depths.append(info.get("depth", 1))
        sps = total_steps / (time.time() - t0)
        print(f"ep {ep + 1}/{episodes} seed={env.seed} steps={steps} "
              f"depth={info.get('depth')} result={info.get('history_result','?')}"
              f" | {sps:.0f} steps/s  -> {path and path.split('/')[-1]}",
              flush=True)
    env.close()
    n = len(depths)
    print(f"\ndone: {n} episodes, {total_steps} steps, "
          f"mean depth {sum(depths)/n:.2f}, max depth {max(depths)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/scripted")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1,
                    help="first seed (episodes use seed, seed+1, ...); "
                    "pass -1 for random seeds")
    ap.add_argument("--flee-hp", type=int, default=35)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--gamedata", default="gamedata/scripted")
    args = ap.parse_args()
    seed_start = None if args.seed < 0 else args.seed
    generate(args.out, args.episodes, args.gamedata, seed_start,
             args.flee_hp, args.max_steps)


if __name__ == "__main__":
    main()
