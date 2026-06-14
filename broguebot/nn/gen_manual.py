"""Generate a MANUAL-move imitation dataset from the scripted bot.

The scripted bot plays well (depth ~3-6) but navigates with the game's macro
commands (autoexplore `x`, travel `>`), which jump many cells per IPC step. The
patched binary (BB_MANUAL_EXPORT, hook in Movement.c playerMoves) records the
display frame + direction for EVERY single-cell move the bot makes under the
hood — so the bot's optimal autoexplore/travel/melee becomes a stream of per-turn
(frame, manual move) pairs. Cloning this gives a strong, low-multimodality manual
navigator (cf. "NetHack is Hard to Hack": imitate a strong symbolic teacher),
unlike the dithering val-0.30 human-keystroke clone.

Each episode's brogue process writes its own export file (set per-episode via the
env var before env.reset spawns it); we convert it to the standard trajlog .npz
(frames/actions/rewards/meta) so train_bc --data data/manual works unchanged. The
8 directions map to the 8 move actions; non-move actions (item use, rest) aren't
exported — moves+melee are the bulk of play and all we need for macro-free
navigate+descend+fight.

  .venv/bin/python -m broguebot.nn.gen_manual --out data/manual --episodes 300 --seed 1
"""

import argparse
import json
import os
import time

import numpy as np

from ..env import ACTION_INDEX, BrogueEnv, wipe_gamedata
from ..ipc import FRAME_SIZE, Frame
from .scripted_actor import ScriptedActor

# Brogue direction enum order (Movement.c directionKeys / nbDirs):
# 0 UP, 1 DOWN, 2 LEFT, 3 RIGHT, 4 UPLEFT, 5 DOWNLEFT, 6 UPRIGHT, 7 DOWNRIGHT
DIR2ACTION = np.array(
    [ACTION_INDEX[n] for n in ("move_n", "move_s", "move_w", "move_e",
                               "move_nw", "move_sw", "move_ne", "move_se")],
    dtype=np.int16)
_NB = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
_REC = FRAME_SIZE + 1  # one obs frame + one direction byte


def convert(binpath: str):
    """Read a manual-export file and filter to clean moves.

    The hook fires on every move ATTEMPT, including blocked ones (the scripted
    bot sometimes gets stuck bumping a wall — e.g. 1818 identical moves). Keep a
    record only if it produced a real single-cell displacement (clean move) OR
    the target cell held a monster (melee attack, which doesn't displace). Drop
    wall-bumps so we never teach the policy to get stuck.

    Returns (frames, actions, rewards, blocked_frac) or None.
    """
    with open(binpath, "rb") as fh:
        data = fh.read()
    n = len(data) // _REC
    if n == 0:
        return None
    raw = [data[i * _REC:i * _REC + FRAME_SIZE] for i in range(n)]
    dirs = [data[i * _REC + FRAME_SIZE] for i in range(n)]
    pos = [(f.stats.px, f.stats.py) for f in (Frame(r) for r in raw)]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dx, dy = _NB[dirs[i]]
        tgt = (pos[i][0] + dx, pos[i][1] + dy)
        if i + 1 < n and pos[i + 1] == tgt:
            continue                              # clean displacement -> keep
        if any(m.x == tgt[0] and m.y == tgt[1] for m in Frame(raw[i]).monsters):
            continue                              # melee into a monster -> keep
        keep[i] = False                           # wall-bump / blocked -> drop
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return None
    frames = np.stack([np.frombuffer(raw[i], np.uint8) for i in idx])
    actions = DIR2ACTION[np.asarray(dirs, dtype=np.uint8)[idx]]
    return (frames, actions, np.zeros(len(idx), dtype=np.float32),
            1.0 - len(idx) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manual")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--flee-hp", type=int, default=35)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--gamedata", default="gamedata/manual")
    ap.add_argument("--tmp", default="/tmp/bbmanual")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    wipe_gamedata(args.gamedata)
    env = BrogueEnv(args.gamedata, max_steps=args.max_steps)
    actor = ScriptedActor(flee_hp=args.flee_hp)
    t0 = time.time()
    totmoves = 0
    depths = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        binp = os.path.join(args.tmp, f"ep_{seed}.bin")
        try:
            os.remove(binp)
        except FileNotFoundError:
            pass
        os.environ["BB_MANUAL_EXPORT"] = binp   # read by the brogue subprocess
        frame = env.reset(seed)
        actor.reset()
        done = False
        info = {}
        while not done:
            frame, _r, done, info = env.step(actor(frame))
        # the brogue process has emitted DONE and exited -> binp is complete
        conv = convert(binp) if os.path.exists(binp) else None
        moves = 0
        skipped = ""
        if conv is not None:
            frames, actions, rewards, blocked = conv
            moves = len(actions)
            # one direction dominating => the bot was stuck (wall-bump loop or a
            # prolonged losing fight, e.g. 1818 south-attacks) => skip the junk
            dom = np.bincount(actions).max() / max(1, moves)
            if blocked > 0.5 or moves < 20 or dom > 0.5:
                skipped = (f" SKIP(blocked={blocked:.0%},moves={moves},"
                           f"dom={dom:.0%})")
            else:
                meta = json.dumps({"seed": seed, "depth": info.get("depth", 1),
                                   "truncated": bool(info.get("truncated")),
                                   "result": info.get("history_result", "?")})
                np.savez_compressed(
                    os.path.join(args.out, f"ep_{seed}.npz"),
                    frames=frames, actions=actions, rewards=rewards,
                    meta=np.frombuffer(meta.encode(), dtype=np.uint8))
                totmoves += moves
                depths.append(info.get("depth", 1))
        try:
            os.remove(binp)
        except FileNotFoundError:
            pass
        if ep == 0 or (ep + 1) % 10 == 0:
            md = np.mean(depths) if depths else 0
            sps = totmoves / (time.time() - t0)
            print(f"ep {ep + 1}/{args.episodes} seed={seed} moves={moves} "
                  f"depth={info.get('depth')} | kept {len(depths)} "
                  f"mean_depth {md:.2f} ({sps:.0f} moves/s){skipped}", flush=True)
    env.close()
    md = np.mean(depths) if depths else 0
    print(f"\ndone: {len(depths)} eps, {totmoves} moves, mean depth {md:.2f}")


if __name__ == "__main__":
    main()
