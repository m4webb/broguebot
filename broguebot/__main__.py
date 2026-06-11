"""CLI: python3 -m broguebot <command>"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(prog="broguebot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="dashboard + bot (run inside tmux pane)")
    p.add_argument("--pane", required=True, help="tmux target of game pane")
    p.add_argument("--seed", type=int)
    p.add_argument("--wizard", action="store_true")
    p.add_argument("--delay", type=float, default=0.4,
                   help="pause before each bot action (watchability)")

    t = sub.add_parser("tune", help="run headless games for tuning")
    t.add_argument("-n", "--games", type=int, default=10)
    t.add_argument("--session", default="bbtune")
    t.add_argument("--seed", type=int)
    t.add_argument("--label", default="", help="config label for stats")
    t.add_argument("--workers", type=int, default=1,
                   help="parallel game sessions")

    sub.add_parser("stats", help="summarize recorded runs")

    pr = sub.add_parser("probe", help="parse a pane and dump state (debug)")
    pr.add_argument("--pane", required=True)

    args = ap.parse_args()

    if args.cmd == "play":
        from . import game
        from .dashboard import main as dash_main
        log = os.path.join(game.LOGS, "decisions-play.jsonl")
        dash_main(args.pane, seed=args.seed, wizard=args.wizard,
                  delay=args.delay, log_path=log)
    elif args.cmd == "tune":
        if args.workers > 1:
            from .headless import tune_parallel
            tune_parallel(args.games, args.workers, label=args.label,
                          seed=args.seed)
        else:
            from .headless import tune
            tune(args.games, session=args.session, seed=args.seed,
                 label=args.label)
        from .game import summarize
        print()
        print(summarize(last=args.games))
    elif args.cmd == "stats":
        from .game import summarize
        print(summarize())
    elif args.cmd == "probe":
        from . import screen as S
        from .tmux import Pane
        snap = S.parse(Pane(args.pane).capture())
        print(f"mode={snap.mode} depth={snap.depth} hp={snap.hp_pct}% "
              f"str={snap.strength} armor={snap.armor} "
              f"nutrition_lost={snap.nutrition_lost}")
        print("player:", snap.player, "at", snap.player_pos)
        for e in snap.entities:
            print("entity:", e)
        print("messages:", snap.messages)
        print("flavor:", repr(snap.flavor))
        print("prompt:", repr(snap.prompt))
        print("items:", snap.item_lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
