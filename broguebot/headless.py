"""Headless tuning harness: run many sandboxed games fast, collect stats."""

import os
import random
import time

from . import game
from .brain import BotConfig, Status
from .game import Runner
from .tmux import BOT_BIN, Pane, run as tmux


def ensure_worker_session(name: str) -> Pane:
    try:
        tmux("kill-session", "-t", name)
    except Exception:
        pass
    tmux("new-session", "-d", "-s", name, "-x", "100", "-y", "34",
         "sleep infinity")
    tmux("set-option", "-t", name, "remain-on-exit", "on")
    return Pane(f"{name}:0.0")


def tune_parallel(games: int, workers: int, label: str = "", seed=None):
    """Split a batch across several tmux sessions running concurrently."""
    import subprocess
    import sys
    per = [games // workers + (1 if i < games % workers else 0)
           for i in range(workers)]
    procs = []
    for i, n in enumerate(per):
        if n == 0:
            continue
        cmd = [sys.executable, "-m", "broguebot", "tune", "-n", str(n),
               "--session", f"bbtune{i}", "--label", label]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        procs.append(subprocess.Popen(cmd, cwd=game.ROOT))
    rc = 0
    for p in procs:
        rc |= p.wait()
    return rc


def tune(games: int, session: str = "bbtune", seed=None,
         gamedata: str | None = None, label: str = ""):
    gamedata = gamedata or os.path.join(game.GAMEDATA, session)
    # the patched binary + a private pty beats a tmux pane by an order of
    # magnitude: output is read straight off the master fd and "game wants
    # input" is a blocking read on a FIFO, no subprocesses per action
    use_pty = os.path.exists(BOT_BIN)
    if use_pty:
        from .ptyhost import PtyPane
        pane = PtyPane()
    else:
        pane = ensure_worker_session(session)
    cfg = BotConfig(headless=True, act_delay=0.0,
                    log_path=os.path.join(game.LOGS, f"decisions-{session}.jsonl"))
    status = Status()
    runner = Runner(pane, cfg, status, seed=seed, gamedata=gamedata,
                    label=label)
    rng = random.SystemRandom()
    results = []
    for i in range(games):
        # brogue seeds new games from the clock: parallel workers started in
        # the same second would all play the identical dungeon
        if seed is None:
            runner.seed = rng.randrange(1, 2**31)
        t0 = time.time()
        rec = runner.run_episode()
        results.append(rec)
        print(f"[{session}] game {i + 1}/{games}: {rec['result']:7s} "
              f"depth {rec['depth']:2d}  cause={rec['cause'][:40]:40s} "
              f"actions={rec['actions']:4d}  {time.time() - t0:5.0f}s",
              flush=True)
    if use_pty:
        pane.kill()
    else:
        tmux("kill-session", "-t", session)
    return results
