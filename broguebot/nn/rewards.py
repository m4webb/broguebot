"""Pluggable reward functions for PPO reward-shaping experiments.

env.BrogueEnv takes a `reward_fn(prev_frame, cur_frame, info) -> float`. The
default (env.default_reward) is depth-progress + a little gold, minus a tiny
step cost. These variants add fairness-safe shaping signals computed purely
from what the frame already exposes (the post-FOV display buffer), so they
never leak omniscient state.

Pick one with train_ppo's --reward flag (wired separately). REWARDS maps a
name to a function. Keep the default available as a control.
"""

from ..env import default_reward
from ..ipc import COLS, ROWS, MAP_OFF

# dungeon map window inside the 100x34 display buffer (Rogue.h: STAT_BAR_WIDTH
# +1 cols on the left, MESSAGE_LINES on top, 2 rows at the bottom)
_MX, _MY = 21, 3
_MW, _MH = COLS - 21, ROWS - 5


def explored_count(frame) -> int:
    """Number of known (non-blank) cells in the dungeon map window — a proxy
    for how much of the level the player has revealed. Reads glyphs directly
    off the raw frame to stay cheap. Glyph 32 (space) == unknown/unseen."""
    raw = frame.raw
    n = 0
    for x in range(_MX, _MX + _MW):
        base = MAP_OFF + x * ROWS * 8
        for y in range(_MY, _MY + _MH):
            off = base + y * 8
            g = raw[off] | (raw[off + 1] << 8)
            if g != 32:
                n += 1
    return n


def explore_reward(prev, cur, info, w_explore: float = 0.0005) -> float:
    """Default reward + a bonus for revealing new map cells on the same level.

    Encourages mapping the floor (which surfaces stairs, items, and routes)
    instead of dithering. Suppressed across a depth change, where the cell
    count legitimately drops as a fresh level starts mostly unknown."""
    r = default_reward(prev, cur, info)
    if prev is not None and cur.stats.depth == prev.stats.depth:
        revealed = explored_count(cur) - explored_count(prev)
        if revealed > 0:
            r += w_explore * revealed
    return r


def survival_reward(prev, cur, info, w_alive: float = 0.001) -> float:
    """Default reward + a small per-step bonus for being alive, to offset the
    step cost so the agent isn't nudged toward suicidal dives. The terminal
    DONE frame is the death step; we credit non-terminal steps only."""
    r = default_reward(prev, cur, info)
    if not (cur is None or cur.done or cur.stats.game_has_ended):
        r += w_alive
    return r


def dense_reward(prev, cur, info) -> float:
    """Exploration + survival shaping together over the depth-progress spine."""
    r = explore_reward(prev, cur, info)
    if not (cur is None or cur.done or cur.stats.game_has_ended):
        r += 0.001
    return r


REWARDS = {
    "default": default_reward,
    "explore": explore_reward,
    "survival": survival_reward,
    "dense": dense_reward,
}
