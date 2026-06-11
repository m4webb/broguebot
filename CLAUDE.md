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
  Both trainers take `--amp` (bf16 autocast, default on), `--grad-checkpoint`
  (recompute the encoder in backward, default on), `--chunk N` (frames per
  encoder mini-batch in the unroll, default 128) and `--compile` (torch.compile
  the encoder, ~1.2x, default off). `model.encode_frames()` implements the
  chunked+checkpointed sequence encode that `unroll`/`masked_unroll` call —
  see the 12GB memory note below.
- `broguebot/nn/scripted_actor.py` — the warm-start data teacher. Renders the
  IPC frame's full 100x34 display buffer (sidebar+messages+map, all
  displayGlyph values) to ASCII and runs it through `screen.parse()`, so the
  legacy `Brain` drives the IPC env unchanged — monster names come from the
  sidebar, item menus/confirms are detected the same way the terminal bot saw
  them. `ScriptedActor` is a one-keycode-per-step `actor(frame)->action` for
  `trajlog.collect`: it queues the Brain's multi-key macros and answers
  prompt frames from the Action's hints. Inventory is read straight from the
  IPC item records. `broguebot/nn/gen_scripted.py` is the CLI that records
  episodes to a BC dataset (reaches depth ~3-4, ~200 steps/s).
- Legacy (still works, useful as a data teacher / baseline): scripted bot
  in `broguebot/brain.py` + tmux/pty plumbing (`tmux.py`, `ptyhost.py`,
  `game.py`, `headless.py`, dashboard). It reaches ~depth 3-6; caustic gas
  is its top killer. `bin/brogue-bot` runs the watchable overlay.

## Verified numbers (laptop, 8-thread i7-8650U)

- env: ~1,600 steps/s single, ~2,400/s with 8 threaded envs (GIL-bound;
  use multiple VectorEnv processes for RL at scale)
- policy-in-loop on CPU: ~110 steps/s small config (model-bound)
- BC and PPO trainers: smoke-tested end to end; checkpoints interchange

## Verified numbers (desktop, RTX 5070 12GB, WSL2, torch 2.12+cu130)

- raw VectorEnv: ~2,950 steps/s at 8 envs (faster than the laptop; env is
  NOT the bottleneck here)
- full base-config PPO (bf16 + grad-checkpoint, chunk=128): **~267 steps/s**,
  ~6.2GB VRAM, 98–99% GPU util. Throughput is compute-bound on the
  sequential GRU unroll, not memory — it plateaus ~280 sps as envs×segment
  grow. Good default: `--envs 64 --segment 128` (peak ~5.7GB).
- peak VRAM is set by `--chunk` (one encoder mini-batch's backward
  recompute), ~independent of envs×segment: chunk 128→~5GB, 256→~9.4GB,
  512→18GB (spills, do not use). envs×segment only grows the stored obs
  tensors (int64 glyph buffer dominates), e.g. 64×256 → 9.5GB.

## ⚠️ The 12GB memory wall (READ before scaling base config)

- base config's training unroll encodes the whole envs×segment rollout
  through the 8-layer transformer; without help the retained attention/FF
  activations need ~18GB. The fix (already wired): `encode_frames()` splits
  the flattened batch into `--chunk`-sized mini-batches AND gradient-
  checkpoints each, so backprop peak ≈ one chunk's forward. **Keep
  `--chunk ≤ 256`; 512 overflows.** bf16 autocast halves activations and
  speeds up the Blackwell card.
- WSL2 GPU has system-memory fallback ON (Task Manager shows "12GB
  dedicated + 8GB shared"). CUDA silently spills past 12GB into the 8GB
  shared pool over PCIe instead of OOM-ing — runs 20–50× slower, looks like
  a hang, not a crash. We confirmed it (allocated 15GB on a 12GB card).
  Flipping NVIDIA Control Panel → CUDA Sysmem Fallback Policy → "Prefer No
  Sysmem Fallback" did NOT take effect (likely needs `wsl --shutdown`). We
  don't rely on it: keep peak under 12GB via `--chunk` and detect spills by
  watching wall-time (a spilling update is ~20× slower than an in-VRAM one).

## Where to pick up (desktop session)

1. ✅ DONE. Smoke passed: GPU/binary/venv all good; base-config PPO trains
   on CUDA. Found + fixed a 12GB memory wall (bf16 + chunked grad-
   checkpointing — see the memory-wall section above). Healthy run now:
   `python -m broguebot.nn.train_ppo --config base --device cuda --envs 64
   --segment 128` → ~267 steps/s at ~6GB VRAM.
2. ✅ DONE (mostly). Env throughput ~2,950 steps/s — not the bottleneck.
   The learner (sequential GRU unroll) caps end-to-end at ~280 sps. If we
   want more, the lever is the unroll, not env sharding: batch the GRU/heads
   over time or move to the Transformer-XL block noted below. Multi-process
   VectorEnv sharding only helps once the learner is faster.
3. Warm-start: adapter is BUILT (`nn/scripted_actor.py` + `nn/gen_scripted.py`,
   validated end-to-end — drives the Brain over IPC to depth ~3-4). Next:
   generate a dataset (`python -m broguebot.nn.gen_scripted --out data/scripted
   --episodes 500 --seed 1`), `train_bc` on it, then PPO with `--init bc.pt`.
   ALWAYS run a cold-start PPO control alongside; if warm-start underperforms,
   drop it. (Teacher polish later if it helps: gas/hazard color detection is
   currently stubbed in the adapter — the Brain falls back to message/HP cues,
   which likely costs it a level or two vs the terminal bot's depth 3-6.)
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
