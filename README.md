# brogue-bot

An AI that plays [Brogue CE](https://github.com/tmewett/BrogueCE) in terminal
mode, with a live overlay showing every decision it makes. You can pause it,
take over the game yourself, and hand control back at any time.

## Safety: your scores are untouched

Every game the bot plays runs with its working directory inside `gamedata/`,
so high scores, saves and recordings land there — **never** in your personal
`~/.local/share/brogue-ce`. Recordings of recent bot games are kept in
`gamedata/` (capped at 20) if you want to replay its deaths.

## Usage

```sh
bin/brogue-bot                 # launch game + AI dashboard in tmux
bin/brogue-bot play --seed 42  # play a specific seed
bin/brogue-bot play --delay 0  # full speed (default 0.4s/action pacing)
bin/brogue-bot tune -n 20      # headless batch of games (for tuning)
bin/brogue-bot stats           # aggregate results of recorded runs
```

The launcher creates (or re-attaches to) a tmux session named `broguebot`:
the game runs in the left pane, the dashboard in the right pane.

### Dashboard controls

| key      | action                                                    |
|----------|-----------------------------------------------------------|
| `space`  | pause / resume the AI                                     |
| `n`      | single-step one decision while paused                     |
| `+` / `-`| speed up / slow down the bot                              |
| `↑` / `↓`| scroll the decision log                                   |
| `q`      | quit the bot — the game stays running for you             |

While paused, click into the left pane (mouse is enabled) and play normally.
Press `space` in the dashboard to hand the game back; the bot re-reads the
screen and carries on from whatever state you left it in.

## How it works

- **Interface** (`tmux.py`): the game runs in a tmux pane; the bot reads it
  with `capture-pane` and acts with `send-keys`. No game modification at all.
- **Parser** (`screen.py`): turns the ncurses screen into structured state —
  sidebar entities (with health %), map grid, messages, dialogs/popups.
- **Brain** (`brain.py`): priority-driven decisions: emergencies (heal/flee)
  → combat (bump attacks, chokepoints) → survival (eat, rest) → logistics
  (equip upgrades, identify consumables safely) → progress (free captive
  allies, auto-explore, descend). It builds on Brogue's own auto-explore,
  travel and rest commands, adding judgement on top.
- **Tuning** (`headless.py`): unattended batches; every run is appended to
  `logs/runs.jsonl`, every decision to `logs/decisions-*.jsonl`.

## Requirements

brogue-ce (terminal build at `/opt/brogue-ce/brogue`), tmux, Python 3.10+
(stdlib only). Terminal of at least 102x34.
