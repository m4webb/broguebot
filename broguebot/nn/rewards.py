"""Pluggable reward functions for PPO reward-shaping experiments.

env.BrogueEnv takes a `reward_fn(prev_frame, cur_frame, info) -> float`. The
default (env.default_reward) is depth-progress + a little gold, minus a tiny
step cost. These variants add fairness-safe shaping signals computed purely
from what the frame already exposes (the post-FOV display buffer), so they
never leak omniscient state.

Pick one with train_ppo's --reward flag (wired separately). REWARDS maps a
name to a function. Keep the default available as a control.
"""

import numpy as np

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


def frontier_reward(prev, cur, info, w: float = 0.0002) -> float:
    """Default reward + a BOUNDED, non-farmable exploration nudge: pay only when
    the level's explored-cell count reaches a NEW maximum (frontier pushed). The
    per-step cell-reveal bonus (explore_reward) is farmable in macro-free play —
    the agent wanders/re-reveals to milk it and never descends (observed: explore
    return ballooned 3.5->12 while depth stalled at ~1.4). Rewarding only new
    maxima caps the total per level at ~(cells * w) and gives ZERO for
    re-treading, so the only way to keep earning is to push into unseen ground →
    find the stairs → descend (depth pays 1.0 >> a level's ~0.4 frontier). The
    per-level max is tracked in info['rstate'] (persisted across the episode by
    the env) and reset on each depth change."""
    r = default_reward(prev, cur, info)
    st = info.get("rstate")
    if st is None or prev is None or cur is None or cur.done \
            or cur.stats.game_has_ended:
        return r
    if prev.stats.depth != cur.stats.depth:
        st["maxexp"] = 0          # fresh level — reset the frontier
    cnt = explored_count(cur)
    if cnt > st.get("maxexp", 0):
        r += w * (cnt - st.get("maxexp", 0))
        st["maxexp"] = cnt
    return r


_G_DOWN_STAIRS, _G_PLAYER = 151, 152


def _scan_map(frame):
    """One pass over the dungeon-map window: returns (explored_cell_count,
    player_screen_xy or None, downstairs_screen_xy or None). Reads glyphs off
    the raw frame. Glyph 32=unknown; 151=down-stairs; 152=player."""
    raw = frame.raw
    n = 0
    player = stairs = None
    for x in range(_MX, _MX + _MW):
        base = MAP_OFF + x * ROWS * 8
        for y in range(_MY, _MY + _MH):
            g = raw[base + y * 8] | (raw[base + y * 8 + 1] << 8)
            if g != 32:
                n += 1
                if g == _G_PLAYER:
                    player = (x, y)
                elif g == _G_DOWN_STAIRS:
                    stairs = (x, y)
    return n, player, stairs


def stairs_reward(prev, cur, info, w_front: float = 0.0002,
                  w_near: float = 0.01) -> float:
    """The bootstrap for macro-free play: explore to FIND the down-stairs, then
    move TOWARD them, then descend. Three non-farmable, fairness-safe terms over
    the depth spine:
      - frontier: pay for pushing the level's explored-cell count to a new max
        (find the stairs);
      - attraction: once the down-stairs glyph is visible, pay for getting
        CLOSER to it than ever before this level (min-distance, so oscillating
        can't farm);
      - depth (default_reward): +1.0 for descending (stepping onto the stairs).
    Diagnosed need: the manual-BC policy explores but never descends (descent
    prob ~0 -> no PPO signal); this gives a continuous gradient all the way onto
    the staircase. All state in info['rstate'], reset per level."""
    r = default_reward(prev, cur, info)
    st = info.get("rstate")
    if st is None or prev is None or cur is None or cur.done \
            or cur.stats.game_has_ended:
        return r
    if prev.stats.depth != cur.stats.depth:
        st["maxexp"] = 0
        st["smin"] = None           # fresh level: reset frontier + stair distance
    n, player, stairs = _scan_map(cur)
    if n > st.get("maxexp", 0):
        r += w_front * (n - st.get("maxexp", 0))
        st["maxexp"] = n
    if stairs is not None and player is not None:
        d = abs(player[0] - stairs[0]) + abs(player[1] - stairs[1])
        smin = st.get("smin")
        if smin is None:
            st["smin"] = d          # first sighting: anchor distance, no reward
        elif d < smin:
            r += w_near * (smin - d)   # reward only genuine approach
            st["smin"] = d
    return r


# exact displayGlyph enum values (Rogue.h) for on-floor items and stairs
_ITEM_GLYPHS = np.array([130, 202, 229, 174, 203, 210, 228, 201, 204, 173, 148,
                         135, 138], dtype=np.uint16)  # potion..gold..key
_STAIRS_GLYPHS = np.array([149, 151], dtype=np.uint16)  # up / down stairs


def discover_reward(prev, cur, info, w_cell: float = 0.002, w_item: float = 0.06,
                    w_stairs: float = 0.03, step_cost: float = 0.0005) -> float:
    """Intrinsic discovery reward — NO depth/oracle/distance signal.

    Reward each NEWLY-revealed map square (a cell that goes from unknown to known
    this episode-on-this-level), with a bonus for FINDING an item or stairs the
    first time (revealing its tile). A small per-step cost means a fully-explored
    level yields nothing more, so the only way to keep earning is to DESCEND — a
    fresh level is a whole new map of unrevealed squares. Descending itself is
    never rewarded; depth emerges from the discovery drive. Count-based (a per-
    level revealed-mask in info['rstate']) so re-treading / re-seeing earns zero —
    not farmable, and tied to FINDING items (not holding them), so it never
    discourages using items. Fairness-safe: reads only the on-screen glyph grid."""
    st = info.get("rstate")
    if st is None or cur is None or cur.done or cur.stats.game_has_ended:
        return 0.0
    raw = cur.raw
    a = np.frombuffer(raw, np.uint8, COLS * ROWS * 8, MAP_OFF).reshape(COLS, ROWS, 8)
    g = (a[:, :, 0].astype(np.uint16) | (a[:, :, 1].astype(np.uint16) << 8))
    win = g[_MX:_MX + _MW, _MY:_MY + _MH]          # dungeon-map window
    known_now = win != 32
    # (re)start the revealed-mask on the first step or on a depth change
    if "mask" not in st or (prev is not None
                            and prev.stats.depth != cur.stats.depth):
        st["mask"] = known_now.copy()
        return -step_cost
    fresh = known_now & ~st["mask"]
    r = -step_cost
    n = int(fresh.sum())
    if n:
        fg = win[fresh]
        n_item = int(np.isin(fg, _ITEM_GLYPHS).sum())
        n_stair = int(np.isin(fg, _STAIRS_GLYPHS).sum())
        r += (w_cell * (n - n_item - n_stair) + w_item * n_item
              + w_stairs * n_stair)
        st["mask"] = st["mask"] | fresh
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
    "frontier": frontier_reward,
    "stairs": stairs_reward,
    "discover": discover_reward,
}
