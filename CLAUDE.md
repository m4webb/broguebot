# Brogue Bot — project state & handoff

The goal: a state-of-the-art transformer policy that plays Brogue CE 1.15.1,
trained with RL (warm-started by behavioral cloning), on a strictly fair
observation channel. This file is the handoff between the laptop session
(where everything below was built) and the Windows/WSL2 desktop session
(where training happens, on an RTX 5070 12GB).

## Hard constraints (do not relax)

- **Fairness contract**: the agent may only see what a human player sees.
  The IPC frame is built from the game's own display buffer (post
  field-of-view), unidentified items expose no true kind/enchant, HP and
  nutrition are quantized to the sidebar bar's 20-cell resolution, only
  `canSeeMonster()` monsters are included. Never add omniscient fields.
- All games run sandboxed in `gamedata/` (cwd of the spawned process).
  Never touch a personal `~/.local/share/brogue-ce`.
- The vendor tree (`vendor/BrogueCE-1.15.1`) carries deliberate patches —
  see below. Keep them when upgrading Brogue.

## Architecture

- `vendor/.../src/platform/ipc-platform.c` — headless platform backend.
  Selected at runtime when env vars `BROGUE_IPC_OUT`/`BROGUE_IPC_IN` hold
  pipe fd numbers. Per input request: writes one packed ~29KB binary frame
  (header, stats, ≤24 monsters, ≤26 items with display-name strings, last
  4 messages, full 100×34 glyph+RGB grid), then blocks reading a uint16
  keycode. At game over it emits a DONE frame and exits the process
  (`notifyEvent` hook + `serverMode=true` skip the death screens), so one
  process == one episode. Build:
  `make -C vendor/BrogueCE-1.15.1 GRAPHICS=NO TERMINAL=YES IPC=YES bin/brogue`
- `broguebot/ipc.py` — frame parser (lazy), `BrogueIPC` process wrapper.
  Protocol constants here MUST mirror ipc-platform.c (FRAME_SIZE=29094, v1).
- `broguebot/env.py` — `BrogueEnv` (gym semantics; reward = depth progress
  + small gold bonus − step cost, pluggable) and threaded `VectorEnv`.
  **Action space = 55 raw keycodes** (the game's own keyboard: 8 moves +
  8 shift-runs, rest/search/explore/stairs, all item verbs, all letters,
  digits, Space/Enter(=10!)/Escape/Tab/Shift-Tab/Backspace). No macros:
  targeting, menus, quantity prompts are operated key-by-key by the policy.
- `broguebot/nn/` — `featurize.py` (numpy frame→tensors), `model.py`
  (conv stem → transformer encoder + entity tokens → GRU memory → policy/
  value/aux heads; `Config.small()`≈2M for CPU checks, `Config.base()`≈21M
  for the 5070), `trajlog.py` (raw-frame episode recording + `collect()`),
  `train_bc.py`, `train_ppo.py` (recurrent PPO; `--init bc.pt` = warm
  start as initialization only, NO KL tether — deliberate, so optimized
  play can diverge from any teacher), `evaluate.py` (fixed 200-seed suite).
- Legacy (still works, useful as a data teacher / baseline): scripted bot
  in `broguebot/brain.py` + tmux/pty plumbing (`tmux.py`, `ptyhost.py`,
  `game.py`, `headless.py`, dashboard). It reaches ~depth 3-6; caustic gas
  is its top killer. `bin/brogue-bot` runs the watchable overlay.

## Verified numbers (laptop, 8-thread i7-8650U)

- env: ~1,600 steps/s single, ~2,400/s with 8 threaded envs (GIL-bound;
  use multiple VectorEnv processes for RL at scale)
- policy-in-loop on CPU: ~110 steps/s small config (model-bound)
- BC and PPO trainers: smoke-tested end to end; checkpoints interchange

## Where to pick up (desktop session)

1. Smoke: `nvidia-smi` in WSL2; build vendor binary; venv with CUDA torch;
   `python -m broguebot.nn.train_ppo --config base --device cuda --envs 16
   --updates 20` should print healthy steps/s.
2. Measure env throughput here; if GIL-bound, shard into N processes ×
   VectorEnv(M) feeding one learner (the model API already separates
   encode/memory for this).
3. Warm-start decision: generate scripted-bot trajectories through the IPC
   env for UI mechanics (needs a small adapter driving `Brain` from frames,
   not yet written), BC on them, then PPO with `--init`. ALWAYS run a
   cold-start PPO control alongside; if warm-start underperforms, drop it.
4. Real training: reward shaping experiments (exploration bonus, kill
   credit), entropy schedule, eval every N updates on the fixed seed suite.
5. Known gaps / ideas: no Ctrl-modifier in the protocol (shift-run covers
   running); mouse excluded by design; messages enter the net as one hash
   bucket (upgrade: small text encoder); GRU memory is the v1 — consider
   Transformer-XL on the 5070; frame could carry a fairness-safe
   per-cell gas/water tint channel if RGB proves noisy to learn from.

## Conventions

- Python 3.12+; venv at `.venv`; no system pip installs.
- `gamedata/`, `logs/`, `runs/`, `data/`, `.venv/` are gitignored scratch.
- Decision/run logs are JSONL (`logs/runs.jsonl` for the scripted bot).
