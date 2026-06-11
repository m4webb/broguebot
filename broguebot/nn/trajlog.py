"""Episode trajectory recording: raw frames + actions + rewards.

Raw frame bytes are stored (not featurized) so the featurizer can evolve
without invalidating data. npz-compressed; dungeon frames are highly
redundant so deflate gets a useful ratio.
"""

import json
import os
import time

import numpy as np

from ..ipc import FRAME_SIZE


class TrajectoryWriter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._reset()

    def _reset(self):
        self.frames, self.actions, self.rewards = [], [], []
        self.seed = None

    def start_episode(self, seed: int):
        self._reset()
        self.seed = seed

    def record(self, raw: bytes, action: int, reward: float):
        self.frames.append(np.frombuffer(raw, dtype=np.uint8))
        self.actions.append(action)
        self.rewards.append(reward)

    def end_episode(self, info: dict) -> str | None:
        if not self.frames:
            return None
        path = os.path.join(
            self.out_dir, f"ep_{int(time.time()*1000)}_{self.seed}.npz")
        np.savez_compressed(
            path,
            frames=np.stack(self.frames),
            actions=np.asarray(self.actions, dtype=np.int16),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            meta=np.frombuffer(
                json.dumps({"seed": self.seed, **{
                    k: v for k, v in info.items()
                    if isinstance(v, (int, float, str, bool))}}).encode(),
                dtype=np.uint8))
        self._reset()
        return path


def load_episode(path: str) -> dict:
    z = np.load(path)
    ep = {"frames": z["frames"], "actions": z["actions"],
          "rewards": z["rewards"],
          "meta": json.loads(bytes(z["meta"]).decode())}
    assert ep["frames"].shape[1] == FRAME_SIZE
    return ep


def episode_files(data_dir: str) -> list[str]:
    return sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.endswith(".npz"))


def collect(env, actor, writer: TrajectoryWriter, episodes: int,
            seed_fn=None) -> list[dict]:
    """Drive `env` with `actor(frame) -> action`, recording every step."""
    results = []
    for _ in range(episodes):
        seed = seed_fn() if seed_fn else None
        frame = env.reset(seed)
        writer.start_episode(env.seed)
        done = False
        while not done:
            action = actor(frame)
            next_frame, reward, done, info = env.step(action)
            writer.record(frame.raw, action, reward)
            frame = next_frame
        writer.end_episode(info)
        results.append(info)
    return results
