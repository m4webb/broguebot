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


def hp_reward(prev, cur, info, w_dmg: float = 0.005,
             w_heal: float = 0.002) -> float:
    """Default reward MINUS a penalty for HP lost (and a smaller reward for HP
    regained). Targets the warm-start failure mode (dies fighting mid-tier
    monsters): pressure to take less damage without farming risk — the best
    case is taking zero damage (reward 0), so unlike a per-step alive bonus it
    can't be gamed by stalling, and w_heal<w_dmg makes damage->heal cycles net
    negative. HP is the stats block's quantized bar; only credited on
    non-terminal steps (the DONE frame's HP is unreliable)."""
    r = default_reward(prev, cur, info)
    if prev is not None and not (cur is None or cur.done
                                 or cur.stats.game_has_ended):
        dhp = cur.stats.hp_pct - prev.stats.hp_pct
        if dhp < 0:
            r += w_dmg * dhp          # dhp<0 -> penalty
        elif dhp > 0:
            r += w_heal * dhp
    return r


def eel_reward(prev, cur, info, w_eel: float = 0.01, flat: float = 0.15) -> float:
    """Default reward MINUS a penalty for taking EEL damage — the #1 depth-2
    killer (day3 death analysis: eels kill 18-26 games at depth 2, more than any
    other cause). Eels ambush from deep water and are submerged/invisible until
    they strike, so the reliable signal is the attack MESSAGE, not a visible
    monster: we fire only on a freshly-appeared message mentioning "eel" (the
    hit event), and scale by HP lost (flat fallback when the 20-cell HP bar
    doesn't register the small drop). Event-based (one hit = one penalty) and
    non-farmable (best case = take no eel damage = 0 penalty), so it teaches
    water-edge avoidance without the over-caution of a blanket HP penalty.
    Fairness-safe: messages are exactly what the player reads on screen."""
    r = default_reward(prev, cur, info)
    if prev is not None and not (cur is None or cur.done
                                 or cur.stats.game_has_ended):
        new_msgs = [m for m in cur.messages if m not in prev.messages]
        if any("eel" in m.lower() for m in new_msgs):
            dhp = cur.stats.hp_pct - prev.stats.hp_pct
            r -= w_eel * (-dhp if dhp < 0 else flat)
    return r


def dense_reward(prev, cur, info) -> float:
    """Exploration + survival shaping together over the depth-progress spine."""
    r = explore_reward(prev, cur, info)
    if not (cur is None or cur.done or cur.stats.game_has_ended):
        r += 0.001
    return r


def deep_reward(prev, cur, info, k: float = 1.0) -> float:
    """Like default, but the depth bonus scales WITH depth: reaching a new
    deepest level L pays k*L (so 4->5 pays 5) instead of a flat 1. The hard
    frontier is worth far more than shallow progress, amplifying PPO's gradient
    from the rare successful deep descents to try to push past the ~3.6 plateau.
    Event-based (each new deepest pays once) so not farmable; and since reaching
    deep requires surviving, it rewards successful descent, not suicide."""
    r = -0.0005
    if prev is not None:
        for L in range(prev.stats.deepest + 1, cur.stats.deepest + 1):
            r += k * L
        r += 0.0001 * max(0, cur.stats.gold - prev.stats.gold)
    if info.get("won"):
        r += 10.0
    return r


REWARDS = {
    "default": default_reward,
    "explore": explore_reward,
    "survival": survival_reward,
    "dense": dense_reward,
    "hp": hp_reward,
    "deep": deep_reward,
    "eel": eel_reward,
}
