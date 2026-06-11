"""Curses dashboard pane: shows the bot's reasoning, controls pause/resume.

Runs in its own tmux pane next to the game. The bot plays in a background
thread; the dashboard renders its Status and handles control keys:

  space  pause / resume (take over the game in the left pane while paused)
  n      single-step one decision while paused
  +/-    slow down / speed up the bot
  q      quit the bot (the game keeps running for you)
"""

import curses
import threading
import time
import traceback

from .brain import BotConfig, Status
from .game import Runner
from .tmux import Pane


def bot_loop(runner: Runner, status: Status, errors: list):
    adopt = True  # first episode adopts the game the launcher started
    while status.state != "quit":
        try:
            runner.run_episode(adopt=adopt)
        except Exception:
            errors.append(traceback.format_exc())
            time.sleep(2)
        adopt = False
        if status.state != "quit":
            time.sleep(1.5)


def hp_bar(pct: int, width: int) -> str:
    fill = max(0, min(width, round(width * pct / 100)))
    return "#" * fill + "-" * (width - fill)


def run_dashboard(stdscr, runner: Runner, status: Status, cfg: BotConfig):
    curses.curs_set(0)
    curses.use_default_colors()
    for i, fg in enumerate((curses.COLOR_GREEN, curses.COLOR_YELLOW,
                            curses.COLOR_RED, curses.COLOR_CYAN), start=1):
        curses.init_pair(i, fg, -1)
    GREEN, YELLOW, RED, CYAN = (curses.color_pair(i) for i in range(1, 5))
    stdscr.nodelay(True)

    errors: list = []
    thread = threading.Thread(target=bot_loop, args=(runner, status, errors),
                              daemon=True)
    thread.start()

    scroll = 0
    while status.state != "quit":
        key = stdscr.getch()
        if key == ord(" "):
            if status.state == "paused":
                runner.brain.on_resume()
                status.state = "running"
            else:
                status.state = "paused"
            scroll = 0
        elif key == ord("n") and status.state == "paused":
            status.state = "stepping"
        elif key == ord("q"):
            status.state = "quit"
            break
        elif key in (ord("+"), ord("=")):
            cfg.act_delay = max(0.0, cfg.act_delay - 0.15)
        elif key == ord("-"):
            cfg.act_delay = min(3.0, cfg.act_delay + 0.15)
        elif key == curses.KEY_UP:
            scroll = min(len(status.decisions), scroll + 1)
        elif key == curses.KEY_DOWN:
            scroll = max(0, scroll - 1)

        h, w = stdscr.getmaxyx()
        stdscr.erase()

        def put(y, x, text, attr=0):
            if 0 <= y < h:
                stdscr.addnstr(y, x, text, max(0, w - x - 1), attr)

        state = status.state.upper()
        st_attr = GREEN if state == "RUNNING" else YELLOW
        put(0, 1, "BROGUE BOT", curses.A_BOLD | CYAN)
        put(0, 13, state, curses.A_BOLD | st_attr)
        put(1, 1, f"game #{status.game_num}  depth {status.depth}  "
                  f"turn {status.turn}  delay {cfg.act_delay:.1f}s")
        hp_attr = GREEN if status.hp_pct > 60 else \
            (YELLOW if status.hp_pct > 30 else RED)
        put(2, 1, f"hp {status.hp_pct:3d}% [{hp_bar(status.hp_pct, max(4, w - 12))}]",
            hp_attr)
        put(3, 1, f"threats: {status.threats or '-'}",
            RED if status.threats else 0)
        put(4, 1, f"goal: {status.goal}   {status.result}")
        put(5, 0, "-" * (w - 1))

        log_h = h - 8
        items = list(status.decisions)
        if scroll:
            items = items[:len(items) - scroll]
        for i, line in enumerate(items[-log_h:]):
            put(6 + i, 1, line)

        if errors:
            put(h - 3, 1, "BOT ERROR (see logs): " +
                errors[-1].strip().splitlines()[-1], RED)
        put(h - 2, 0, "-" * (w - 1))
        help_line = "[space] pause/resume  [n] step  [+/-] speed  " \
                    "[up/down] scroll  [q] quit bot"
        if status.state == "paused":
            help_line = "PAUSED - play in the left pane; " \
                        "[space] hands control back to the bot"
        put(h - 1, 1, help_line, CYAN)
        stdscr.refresh()
        time.sleep(0.1)

    thread.join(timeout=10)


def main(pane_target: str, seed=None, wizard=False, delay=0.4,
         gamedata=None, log_path=""):
    from . import game
    cfg = BotConfig(headless=False, act_delay=delay, log_path=log_path)
    status = Status()
    runner = Runner(Pane(pane_target), cfg, status, seed=seed, wizard=wizard,
                    gamedata=gamedata or game.GAMEDATA)
    try:
        curses.wrapper(run_dashboard, runner, status, cfg)
    except KeyboardInterrupt:
        status.state = "quit"
    print("bot stopped; the game pane is all yours "
          "(it is still sandboxed in gamedata/).")
