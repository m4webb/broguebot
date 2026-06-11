"""Episode lifecycle: start a sandboxed game, run the brain, record results.

All games run with their working directory inside the bot's sandbox
(gamedata/), so the user's real high scores, saves and recordings in
~/.local/share/brogue-ce are never touched.
"""

import json
import os
import time

from . import screen as S
from .brain import Brain, BotConfig, EpisodeOver, Status
from .tmux import BOT_BIN, Pane, brogue_command

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMEDATA = os.path.join(ROOT, "gamedata")
LOGS = os.path.join(ROOT, "logs")
RUNS_FILE = os.path.join(LOGS, "runs.jsonl")


def ensure_dirs():
    os.makedirs(GAMEDATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)


def prune_recordings(gamedata: str = GAMEDATA, keep: int = 20):
    """Cap bot-game recordings in the sandbox so disk doesn't fill up."""
    recs = [os.path.join(gamedata, f) for f in os.listdir(gamedata)
            if f.endswith(".broguerec")]
    recs.sort(key=os.path.getmtime)
    for path in recs[:-keep]:
        os.unlink(path)
    # brogue auto-saves when its process is killed (pane respawn): litter
    for f in os.listdir(gamedata):
        if f.startswith("LastGame") and f.endswith(".broguesave"):
            os.unlink(os.path.join(gamedata, f))


def last_history_record(gamedata: str, since_ts: float) -> dict | None:
    """Ground truth from Brogue's own run history (written at game end)."""
    path = os.path.join(gamedata, "BrogueRunHistory.txt")
    try:
        rows = [ln.split("\t") for ln in
                open(path).read().strip().splitlines()]
        seed, ts, result, killer, *rest = rows[-1]
        if float(ts) < since_ts - 5:
            return None  # stale entry from an earlier game
        return {"seed": int(seed), "history_result": result,
                "killer": killer, "depth": int(rest[3]),
                "game_turns": int(rest[4])}
    except Exception:
        return None


class Runner:
    """Runs episodes back to back in a single tmux pane."""

    def __init__(self, pane: Pane, cfg: BotConfig, status: Status,
                 seed: int | None = None, wizard: bool = False,
                 gamedata: str = GAMEDATA, label: str = ""):
        self.label = label
        ensure_dirs()
        self.pane = pane
        self.cfg = cfg
        self.status = status
        self.seed = seed
        self.wizard = wizard
        self.gamedata = gamedata
        os.makedirs(gamedata, exist_ok=True)
        if os.path.exists(BOT_BIN) and not cfg.ready_file:
            cfg.ready_file = os.path.join(gamedata, "ready.flag")
        self.brain = Brain(pane, cfg, status)

    def start_game(self, fresh: bool = True):
        """(Re)spawn brogue in the pane unless a live game is already there."""
        if fresh or self.pane.is_dead() or not self.game_on_screen():
            if self.brain.sync:
                self.brain.sync.reset()
            self.pane.respawn(
                brogue_command(self.seed, self.wizard,
                               ready_file=self.cfg.ready_file or None),
                self.gamedata)
            if self.cfg.ready_file and self.brain.sync is None:
                self.brain.sync = self.pane.make_sync(self.cfg.ready_file)
                self.brain.sync_misses = 0
            if self.brain.sync:
                self.brain.sync.wait(1, 20.0)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                snap = S.parse(self.pane.capture_stable(timeout=2.0))
                if snap.mode == S.MODE_GAME:
                    break
                time.sleep(0.3)
        self.brain.reset_episode()

    def game_on_screen(self) -> bool:
        try:
            return S.parse(self.pane.capture()).mode == S.MODE_GAME
        except Exception:
            return False

    def run_episode(self, adopt: bool = False) -> dict:
        """Play one game to completion. Returns the result record."""
        self.start_game(fresh=not adopt)
        self.status.game_num += 1
        started = time.time()
        result, cause = "aborted", ""
        try:
            while True:
                if self.status.state == "quit":
                    result, cause = "aborted", "user quit"
                    break
                if self.status.state == "paused":
                    time.sleep(0.2)
                    continue
                if self.status.state == "stepping":
                    self.status.state = "paused"
                self.brain.step()
                if self.brain.turn >= self.cfg.max_actions:
                    result, cause = "stuck", "max actions reached"
                    break
        except EpisodeOver as eo:
            result, cause = eo.result, eo.cause
        record = {
            "label": self.label,
            "ts": started,
            "duration": round(time.time() - started, 1),
            "seed": self.seed,
            "result": result,
            "cause": cause,
            "depth": self.brain.max_depth,
            "actions": self.brain.turn,
        }
        hist = last_history_record(self.gamedata, started)
        if hist:
            record.update(hist)
            if result == "died" and hist.get("killer"):
                record["cause"] = hist["killer"]
        with open(RUNS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.status.result = f"game {self.status.game_num}: {result} " \
                             f"d{record['depth']} ({cause})"
        prune_recordings(self.gamedata)
        return record


def summarize(path: str = RUNS_FILE, last: int | None = None) -> str:
    if not os.path.exists(path):
        return "no runs recorded yet"
    runs = [json.loads(ln) for ln in open(path) if ln.strip()]
    if last:
        runs = runs[-last:]
    if not runs:
        return "no runs recorded yet"
    depths = [r["depth"] for r in runs]
    by_result = {}
    causes = {}
    for r in runs:
        by_result[r["result"]] = by_result.get(r["result"], 0) + 1
        if r["result"] == "died":
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    lines = [
        f"runs: {len(runs)}   avg depth: {sum(depths) / len(depths):.2f}   "
        f"max depth: {max(depths)}",
        "results: " + ", ".join(f"{k}={v}" for k, v in
                                sorted(by_result.items())),
        "deaths:",
    ]
    for cause, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:3d}  {cause}")
    return "\n".join(lines)
