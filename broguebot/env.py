"""Gym-style environment over the brogue IPC platform.

The action space is the game's own keyboard, one keycode per action — no
macros, no hidden conveniences. Item use, throwing (with full manual
targeting: direction keys aim, Tab cycles targets, Enter fires), menus,
confirmations and quantity prompts are all operated by the policy through
the same primitives a human uses. Context decides meaning exactly as it
does on a keyboard: 'a' is "apply" at the dungeon prompt and "slot a"
inside an item menu.

Episode flow: reset() spawns a fresh sandboxed game; step() returns
(frame, reward, done, info). At game over the patched platform emits a
DONE frame and exits, so a death costs one action; Brogue's own run
history (killer, depth, turns) is merged into info.
"""

import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from .game import last_history_record, prune_recordings
from .ipc import BrogueIPC, Frame

# brogue keycodes (Rogue.h): RETURN_KEY is \n (10), not \r
TAB, ENTER, SHIFT_TAB, ESCAPE, SPACE, BACKSPACE = 9, 10, 25, 27, 32, 127

ACTIONS = [
    # movement (also: aim in targeting mode, navigate in cursor mode)
    ("move_nw", "y"), ("move_n", "k"), ("move_ne", "u"),
    ("move_w", "h"), ("move_e", "l"),
    ("move_sw", "b"), ("move_s", "j"), ("move_se", "n"),
    # run: shift+direction, travels until something interesting happens
    ("run_nw", "Y"), ("run_n", "K"), ("run_ne", "U"),
    ("run_w", "H"), ("run_e", "L"),
    ("run_sw", "B"), ("run_s", "J"), ("run_se", "N"),
    # turn-passing and automation
    ("rest", "z"), ("long_rest", "Z"), ("search", "s"), ("explore", "x"),
    ("descend", ">"), ("ascend", "<"),
    # item verbs (each opens the game's own selection prompt)
    ("apply", "a"), ("equip", "e"), ("remove", "r"), ("drop", "d"),
    ("throw", "t"), ("rethrow", "T"), ("swap", "w"), ("call", "c"),
    ("relabel", "R"), ("inventory", "i"),
    # remaining letters: inventory slots in menus, no-ops at the prompt
    ("letter_f", "f"), ("letter_g", "g"), ("letter_m", "m"),
    ("letter_o", "o"), ("letter_p", "p"), ("letter_q", "q"),
    ("letter_v", "v"),
    # digits: numpad movement/rest at the prompt, digits in number prompts
    ("digit_0", "0"), ("digit_1", "1"), ("digit_2", "2"),
    ("digit_3", "3"), ("digit_4", "4"), ("digit_5", "5"),
    ("digit_6", "6"), ("digit_7", "7"), ("digit_8", "8"),
    ("digit_9", "9"),
    # UI keys: acknowledge, confirm/travel-to-cursor, cancel, target cycle
    ("ack", SPACE), ("confirm", ENTER), ("cancel", ESCAPE),
    ("tab", TAB), ("shift_tab", SHIFT_TAB), ("backspace", BACKSPACE),
]
ACTION_INDEX = {name: i for i, (name, _) in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)
KEYCODES = [k if isinstance(k, int) else ord(k) for _, k in ACTIONS]


def default_reward(prev: Frame, cur: Frame, info: dict) -> float:
    """Depth progress is the spine; tiny step cost discourages dithering."""
    r = -0.0005
    if prev is not None:
        r += 1.0 * max(0, cur.stats.deepest - prev.stats.deepest)
        r += 0.0001 * max(0, cur.stats.gold - prev.stats.gold)
    if info.get("won"):
        r += 10.0
    return r


class BrogueEnv:
    """One sandboxed brogue game driven through the IPC platform."""

    def __init__(self, gamedata: str, reward_fn=default_reward,
                 max_steps: int = 20000, wizard: bool = False):
        self.gamedata = gamedata
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.wizard = wizard
        self.game: BrogueIPC | None = None
        self.frame: Frame | None = None
        self.steps = 0
        self.seed = None
        self.episode_return = 0.0
        self._last_was_macro = False
        self._rng = random.SystemRandom()

    # ------------------------------------------------------------ lifecycle

    def reset(self, seed: int | None = None) -> Frame:
        self.close()
        self.seed = seed if seed is not None else self._rng.randrange(1, 2**31)
        self.started = time.time()
        self.game = BrogueIPC(self.gamedata, seed=self.seed,
                              wizard=self.wizard)
        self.frame = self.game.read_frame()
        if self.frame is None:
            raise RuntimeError("brogue exited before first frame")
        self.steps = 0
        self.episode_return = 0.0
        return self.frame

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None

    # ------------------------------------------------------------ stepping

    def step(self, action: int):
        if self.frame is None or self.game is None:
            raise RuntimeError("call reset() first")
        name, key = ACTIONS[action]
        prev = self.frame
        frame = self.game.step(key)
        self.steps += 1

        info = {"action": name, "seed": self.seed, "steps": self.steps}
        done = False
        if frame is None or frame.done or frame.stats.game_has_ended:
            done = True
            frame = self._finish_episode(frame, info)
        elif self.steps >= self.max_steps:
            done = True
            info["truncated"] = True
        self.frame = frame
        reward = self.reward_fn(prev, frame, info)
        self.episode_return += reward
        if done:
            info["episode_return"] = self.episode_return
            info["depth"] = frame.stats.deepest
        return frame, reward, done, info

    def _finish_episode(self, frame: Frame | None, info: dict) -> Frame:
        """Skip post-game screens (acknowledgments, recording prompt)."""
        last = frame or self.frame
        for i in range(60):
            if frame is None or frame.done:
                break
            key = ENTER if i % 2 else SPACE
            frame = self.game.step(key)
            if frame is not None:
                last = frame
        hist = last_history_record(self.gamedata, self.started)
        if hist:
            info.update(hist)
            info["won"] = "victor" in (hist.get("history_result") or "").lower()
        try:
            prune_recordings(self.gamedata)
        except OSError:
            pass
        self.close()
        return last


class VectorEnv:
    """N BrogueEnvs stepped concurrently.

    os.read/write release the GIL, so plain threads overlap the games'
    turn computation; no extra processes needed.
    """

    def __init__(self, n: int, gamedata_root: str, **env_kwargs):
        self.envs = [BrogueEnv(os.path.join(gamedata_root, f"env{i}"),
                               **env_kwargs) for i in range(n)]
        self.pool = ThreadPoolExecutor(max_workers=n)

    def reset(self, seeds: list | None = None) -> list:
        seeds = seeds or [None] * len(self.envs)
        return list(self.pool.map(lambda ev: ev[0].reset(ev[1]),
                                  zip(self.envs, seeds)))

    def step(self, actions: list, auto_reset: bool = True) -> list:
        """Returns [(frame, reward, done, info)]; resets finished envs."""
        def one(pair):
            env, action = pair
            frame, reward, done, info = env.step(action)
            if done and auto_reset:
                frame = env.reset()
            return frame, reward, done, info
        return list(self.pool.map(one, zip(self.envs, actions)))

    def close(self):
        for env in self.envs:
            env.close()
        self.pool.shutdown(wait=False)


def wipe_gamedata(root: str):
    """Remove env sandbox dirs (recordings, history) for a fresh run."""
    if os.path.isdir(root):
        shutil.rmtree(root)
