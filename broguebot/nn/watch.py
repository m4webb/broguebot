"""Watch a trained policy play, rendered live to the terminal.

The IPC frame carries the game's full 100x34 display buffer (the same thing a
human sees: sidebar + messages + map, with colors), so we just re-render it with
ANSI truecolor as the policy steps. Shows depth / HP / the action just taken.

  .venv/bin/python -m broguebot.nn.watch --ckpt runs/discover_ppo3/ppo_best.pt \
      --seed 7 --games 3 --delay 0.12 --device cpu

Tips: --device cpu avoids fighting a training run for the GPU; --delay sets the
speed (0.05 fast, 0.3 slow); --temperature 0.4 for steadier (greedier) play.
Ctrl-C to quit. Needs a truecolor terminal (most are).
"""

import argparse
import sys
import time

import torch

from ..env import ACTIONS, BrogueEnv, wipe_gamedata
from ..ipc import COLS, ROWS
from .featurize import batch as batch_feats, featurize
from .model import BroguePolicy, Config
from .scripted_actor import GLYPH_CHAR


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

    wipe_gamedata("gamedata/watch")
    env = BrogueEnv("gamedata/watch", max_steps=args.max_steps)
    try:
        for game in range(args.games):
            seed = args.seed + game
            frame = env.reset(seed)
            hidden = model.initial_state(1, args.device)
            done = False
            steps = 0
            last = "-"
            info = {}
            sys.stdout.write("\033[2J")
            while not done:
                st = frame.stats
                sys.stdout.write("\033[H" + render(frame))
                sys.stdout.write(
                    f"\n\033[0m seed {seed} | game {game + 1}/{args.games} | "
                    f"step {steps} | depth {st.depth} (deepest {st.deepest}) | "
                    f"hp {st.hp_pct}% | last: {last}      ")
                sys.stdout.flush()
                time.sleep(args.delay)
                obs = {k: torch.as_tensor(v, device=args.device)
                       for k, v in batch_feats([featurize(frame.raw)]).items()}
                with torch.no_grad():
                    logits, _v, _a, hidden = model(obs, hidden)
                    act = torch.distributions.Categorical(
                        logits=logits / args.temperature).sample().item()
                last = ACTIONS[act][0]
                frame, _r, done, info = env.step(act)
                steps += 1
            print(f"\n\033[0m=== game {game + 1}: depth {frame.stats.deepest}, "
                  f"{steps} steps, {info.get('history_result', '?')} ===")
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\033[0m\nstopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
