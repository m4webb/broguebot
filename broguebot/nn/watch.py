"""Watch a trained policy play, rendered live to the terminal.

The IPC frame carries the game's full 100x34 display buffer (the same thing a
human sees: sidebar + messages + map, with colors), so we just re-render it with
ANSI truecolor as the policy steps. Shows depth / HP / the action just taken.

  .venv/bin/python -m broguebot.nn.watch --ckpt runs/discover_ppo3/ppo_best.pt \
      --seed 7 --games 3 --delay 0.12 --device cpu

Live controls (single keypress, no Enter):
  n next game   b back a game   r restart   space pause   f fast (no delay)
  +/- speed     q quit
Replays are deterministic (torch seeded per game), so b/r show the same game.
Live per-step reward + episode return are shown under the map (--reward picks
which reward fn; default `discover`, the one PPO trains on).

Tips: --device cpu avoids fighting a training run for the GPU; --delay sets the
speed (0.05 fast, 0.3 slow); --temperature 0.4 for steadier (greedier) play.
Needs a truecolor terminal (most are).
"""

import argparse
import contextlib
import select
import sys
import termios
import time
import tty

import torch

from ..env import ACTIONS, BrogueEnv, wipe_gamedata
from ..ipc import COLS, ROWS
from .featurize import batch as batch_feats, featurize
from .model import BroguePolicy, Config
from .rewards import REWARDS
from .scripted_actor import GLYPH_CHAR


@contextlib.contextmanager
def cbreak_stdin():
    """Put the terminal in cbreak mode so single keypresses are read without
    Enter and without echo; restore on exit. No-op if stdin isn't a tty."""
    if not sys.stdin.isatty():
        yield lambda: None
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def poll_key():
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

        yield poll_key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _glyph_char(g: int) -> str:
    if g in GLYPH_CHAR:
        return GLYPH_CHAR[g]
    if 32 <= g < 127:
        return chr(g)
    return ' '


def render(frame) -> str:
    """The frame's display buffer as one ANSI-colored string (34 rows)."""
    rows = []
    for y in range(ROWS):
        cells = []
        for x in range(COLS):
            g, fg, bg = frame.cell(x, y)
            ch = _glyph_char(g)
            fr, fgc, fb = (int(c * 2.55) for c in fg)
            br, bgc, bb = (int(c * 2.55) for c in bg)
            cells.append(
                f"\033[38;2;{fr};{fgc};{fb};48;2;{br};{bgc};{bb}m{ch}")
        rows.append("".join(cells) + "\033[0m")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/discover_ppo3/ppo_best.pt")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.12, help="seconds/step")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--reward", default="discover", choices=list(REWARDS),
                    help="reward fn to display live (default: discover)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device)
    model = BroguePolicy(getattr(Config, ck["config"])()).to(args.device).eval()
    model.load_state_dict(ck["model"])
    if ck.get("disabled"):
        model.set_disabled_actions(ck["disabled"])
    print(f"loaded {args.ckpt} (config {ck['config']}, "
          f"disabled {ck.get('disabled')})")
    print("controls: [n]ext  [b]ack  [r]estart  [space]pause  "
          "[f]ast  [+/-]speed  [q]uit")
    time.sleep(1.2)

    wipe_gamedata("gamedata/watch")
    env = BrogueEnv("gamedata/watch", reward_fn=REWARDS[args.reward],
                    max_steps=args.max_steps)
    delay = [args.delay]  # mutable so keys can adjust it live

    def play_game(game, poll_key):
        """Run one game; return 'next' | 'prev' | 'restart' | 'quit'."""
        seed = args.seed + game
        torch.manual_seed(seed)  # deterministic replay of this game
        frame = env.reset(seed)
        hidden = model.initial_state(1, args.device)
        done = False
        steps = 0
        last = "-"
        ret = 0.0
        rstep = 0.0
        info = {}
        paused = False
        fast = False
        sys.stdout.write("\033[2J")
        while not done:
            st = frame.stats
            sys.stdout.write("\033[H" + render(frame))
            sys.stdout.write(
                f"\n\033[0m seed {seed} | game {game + 1}/{args.games} | "
                f"step {steps} | depth {st.depth} (deepest {st.deepest}) | "
                f"hp {st.hp_pct}% | last: {last}\n"
                f" reward {rstep:+.3f}  return {ret:+.2f}"
                f"{'  [PAUSED]' if paused else ''}        ")
            sys.stdout.flush()

            # handle keys (block here while paused)
            while True:
                key = poll_key()
                if key == 'q':
                    return 'quit'
                if key == 'n':
                    return 'next'
                if key == 'b':
                    return 'prev'
                if key == 'r':
                    return 'restart'
                if key == ' ':
                    paused = not paused
                elif key == 'f':
                    fast = not fast
                elif key in ('+', '='):
                    delay[0] = max(0.0, delay[0] - 0.03)
                elif key in ('-', '_'):
                    delay[0] += 0.03
                if not paused:
                    break
                sys.stdout.write("\033[H" + render(frame) + "\n\033[0m")
                sys.stdout.write(
                    f"\n\033[0m seed {seed} | game {game + 1}/{args.games} | "
                    f"step {steps} | depth {st.depth} (deepest {st.deepest}) | "
                    f"hp {st.hp_pct}% | last: {last}\n"
                    f" reward {rstep:+.3f}  return {ret:+.2f}  [PAUSED]    ")
                sys.stdout.flush()
                time.sleep(0.05)

            if not fast:
                time.sleep(delay[0])
            obs = {k: torch.as_tensor(v, device=args.device)
                   for k, v in batch_feats([featurize(frame.raw)]).items()}
            with torch.no_grad():
                logits, _v, _a, hidden = model(obs, hidden)
                act = torch.distributions.Categorical(
                    logits=logits / args.temperature).sample().item()
            last = ACTIONS[act][0]
            frame, rstep, done, info = env.step(act)
            ret += rstep
            steps += 1
        print(f"\n\033[0m=== game {game + 1}: depth {frame.stats.deepest}, "
              f"{steps} steps, return {ret:+.2f}, "
              f"{info.get('history_result', '?')} ===")
        if not sys.stdin.isatty():
            return 'next'  # piped/non-interactive: don't wait for a keypress
        # wait for a nav key at the end (don't auto-advance)
        while True:
            key = poll_key()
            if key == 'b':
                return 'prev'
            if key == 'r':
                return 'restart'
            if key == 'q':
                return 'quit'
            if key in ('n', ' ', '\n', '\r', None):
                if key is not None:
                    return 'next'
            time.sleep(0.05)

    try:
        with cbreak_stdin() as poll_key:
            game = 0
            while 0 <= game < args.games:
                nav = play_game(game, poll_key)
                if nav == 'quit':
                    break
                elif nav == 'prev':
                    game = max(0, game - 1)
                elif nav == 'restart':
                    continue
                else:
                    game += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("\033[0m\nstopped.")


if __name__ == "__main__":
    main()
