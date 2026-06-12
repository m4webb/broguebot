"""Extract (frame, action) imitation data from human .broguerec recordings.

Runs the IPC binary in native playback+export mode (the bbReplayExport patch)
on each recording: the game replays the human's exact inputs and emits the
display frame the human saw plus the action they took. Keystroke frames become
(frame, action) BC pairs (action = env action index); the ~1% mouse events are
skipped as labels while the native replay keeps the game state faithful.

  python -m broguebot.nn.replay_extract --recordings data/human_recordings \
      --out data/human_bc

Gives a teacher that reaches depth 26 — far beyond the scripted bot's ~6.
"""

import argparse
import os
import struct
import subprocess
import time

from ..ipc import Frame, FRAME_SIZE
from .scripted_actor import CODE_TO_ACTION
from .trajlog import TrajectoryWriter

BIN = "vendor/BrogueCE-1.15.1/bin/brogue"
EVENT_SIZE = 11
KEYSTROKE = 0


def _readn(fd, n):
    buf = b""
    while len(buf) < n:
        c = os.read(fd, n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def extract_one(rec_path, writer, gamedata):
    """Replay one recording, writing (frame, action) pairs. Returns
    (n_pairs, max_depth, n_skipped)."""
    out_r, out_w = os.pipe()
    in_r, in_w = os.pipe()                       # dummy; playback never reads
    env = dict(os.environ, BROGUE_IPC_OUT=str(out_w), BROGUE_IPC_IN=str(in_r),
               BROGUE_REPLAY_EXPORT="1")
    proc = subprocess.Popen(
        [os.path.abspath(BIN), "-vn", os.path.abspath(rec_path)],
        cwd=gamedata, env=env, pass_fds=(out_w, in_r),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.close(out_w)
    os.close(in_r)
    writer.start_episode(0)
    n = maxdepth = skipped = 0
    try:
        while True:
            fb = _readn(out_r, FRAME_SIZE)
            if fb is None:
                break
            frame = Frame(fb)
            if frame.done:
                break
            ev = _readn(out_r, EVENT_SIZE)
            if ev is None:
                break
            etype = ev[0]
            p1 = struct.unpack_from("<i", ev, 1)[0]
            maxdepth = max(maxdepth, frame.stats.deepest)
            if etype == KEYSTROKE and p1 in CODE_TO_ACTION:
                writer.record(fb, CODE_TO_ACTION[p1], 0.0)
                n += 1
            else:
                skipped += 1                     # mouse / unmappable key
    finally:
        os.close(out_r)
        os.close(in_w)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    writer.end_episode({"depth": maxdepth, "source": os.path.basename(rec_path)})
    return n, maxdepth, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/human_recordings")
    ap.add_argument("--out", default="data/human_bc")
    ap.add_argument("--gamedata", default="gamedata/replay")
    args = ap.parse_args()
    os.makedirs(args.gamedata, exist_ok=True)
    recs = sorted(f for f in os.listdir(args.recordings)
                  if f.endswith(".broguerec"))
    writer = TrajectoryWriter(args.out)
    t0 = time.time()
    tot = 0
    depths = []
    for i, r in enumerate(recs):
        n, d, sk = extract_one(os.path.join(args.recordings, r), writer,
                               args.gamedata)
        tot += n
        depths.append(d)
        print(f"[{i+1}/{len(recs)}] {r[:28]:28} pairs={n:5d} depth={d:2d} "
              f"skipped={sk} | total={tot} ({tot/(time.time()-t0):.0f}/s)",
              flush=True)
    print(f"\ndone: {len(recs)} recordings, {tot} (frame,action) pairs, "
          f"depths {min(depths)}-{max(depths)} mean {sum(depths)/len(depths):.1f}")


if __name__ == "__main__":
    main()
