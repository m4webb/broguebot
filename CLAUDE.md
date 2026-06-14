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

## Results & findings (training sessions 2–3)

**Best policy so far: scripted-warm-start PPO = avg depth ~3.64** (50-game fixed-
seed eval, max 6). Pipeline: `gen_scripted` (500 eps) → `train_bc` (acc ~65%,
eval 2.80) → `train_ppo --init` (3.64). Cold-start PPO FAILS completely (stuck
~depth 1: sparse depth reward is never reached from random) — so warm-start is
**essential**, not just a speedup. PPO lifts the clone above the teacher.

The 3.64 **plateau is a combat-tactics ceiling** (reached by ~update 40, flat to
300+). It dies to mid-tier monsters across depths 2–6 — biggest cluster is eels
at depth 2 (water ambush; submerged eels are fairly invisible, so avoiding them
needs MEMORY of a past sighting — the GRU v1 likely can't hold that). Gas is
largely solved by PPO. Greedy eval (temp 0.5) = 3.40, so 3.64 isn't a sampling
artifact.

**What was tried and did NOT beat 3.64:**
- Reward shaping (`rewards.py`, `--reward`): explore (bonus-hacks: farms the
  cell-reveal proxy, depth falls), hp/damage-penalty (3.20, over-cautious),
  deep/progressive (≤3.5, unstable). 3.64 (depth-only) is a balanced optimum;
  shaping pushes it off-balance. Shape on EVENTS, not farmable per-step states.
- More training: 500-update run = 3.54, no headroom.
- DAgger (refine scripted teacher): augmented-BC went to 2.55 (worse base).
- **Human winning-recording imitation** (big effort, NEGATIVE result): downloaded
  32 CE 1.15.1 WINS from WebBrogue (`api/games?variant=BROGUECEV151&result=2`,
  `api/recordings/<id>`), patched the IPC binary to export (frame,action) during
  native `.broguerec` playback (`bbReplayExport` in ipc-platform.c, hooked in
  IO.c `nextBrogueEvent`; `replay_extract.py`). Extracted 569k pairs reaching
  depth 26. But keystroke-BC = depth 1 (pure) / 1.77 (mixed with scripted), both
  WORSE than scripted-BC 2.80. CAUSE: action-style mismatch — humans navigate
  manually (raw hjkl, `>` ~0.2% of keys, no auto-explore); our policy relies on
  the game's `x`/`>` commands. The data is real winning play but doesn't transfer
  to our action vocabulary. Tooling kept (`replay_extract.py`, the C patch) in
  case a future approach can use it.

**★ MEMORY is the diagnosed bottleneck (session 3, measured not guessed).** Audited
the model/training for bugs — all CORRECT (model overfits a tiny set fine;
training-time recurrent unroll reproduces the collection policy to 2e-5; value
head varies sensibly). The real issue: the 3.64 policy is **96% REACTIVE** — it
picks the same action with vs without its hidden state 96% of the time, i.e.
barely uses memory. Root cause: BC teacher-forces short 32–48-step windows from
a ZERO hidden state, so the GRU never learns long-range memory. On a game needing
memory (eel-near-water recall, layout), a near-memoryless policy plateaus.
Long-window (128) BC raised memory-use (96%→82% reactive) and imitation (65%→81%)
but OVERFIT (eval 2.62 < 2.80). PPO from that confident base collapses at the
normal LR (the value head is untrained by BC → random value fn) — `--lr 1e-4`
stabilizes it and reaches **3.42** (just under 3.64).

**★★ MEMORY DIRECTION CLOSED — rigorous NEGATIVE result (day3).** Pursued the
memory hypothesis to the end and it does NOT pay off on mean depth:
- **Value-head BC** (value-regression loss during BC, `--value-coef`) → fixed the
  PPO-from-confident-base collapse (stable full-LR PPO). vh_ppo eval 3.38.
- **Stateful BC** (`--stateful`: streaming truncated-BPTT, hidden carried across
  windows; `train_bc.py`) → produced the cleanest base yet: memory-USING (54%
  reactive vs 96%), value-calibrated, NON-overfit (held-out val acc 0.557 via
  `--val-frac`). PPO from it was stable + climbed fast (3.18@upd50 vs vh's 1.82).
- BUT the rigorous **200-game head-to-head (same seeds)**: scripted-PPO **3.710**
  vs memory-PPO **3.535** (fair budget + `ppo_best.pt` best-checkpoint selection;
  consistent ~0.18 gap). Memory made the policy HIGHER-VARIANCE — higher ceiling
  (max 8–9 vs 7) but a HEAVY early-death tail (45/200 = 22.5% die at depth 2).
  On the MEAN, the reactive policy's consistency wins. Reducing reactivity raised
  the ceiling, not the mean. **So "96% reactive" was a real property but NOT the
  lever.** Memory closed as negative — like reward-shaping & human-imitation.

**The REAL ceiling = the depth-2 death tail (death analysis: eels #1 at 18/200,
then gas).** Two more directions tried overnight (day3 night), BOTH NEGATIVE:
- **Entropy annealing** (`--entropy-final`, anneal 0.01→0.001) — day3 found the
  constant bonus over-explores late. Annealing DID raise training depth to a new
  high (smoothed 3.86, momentary 4.06) but eval was 3.57/3.52 < 3.71. Confirms a
  persistent **train≠eval gap**: higher training depth (random seeds) doesn't
  transfer to the fixed-seed eval. Annealing is a real training improvement, not
  an eval lever. (Feature kept — good default for future long runs.)
- **Eel event-reward** (`--reward eel`, penalize eel-attack damage) — targeted the
  #1 killer. Fully-trained eval 3.65 with depth-2 eel deaths 17 vs scripted's 18 =
  **UNCHANGED**. Did NOT teach eel-avoidance, just preserved the base. Eels are
  likely IRREDUCIBLE here: submerged/invisible until they strike + burst (near
  one-shot) damage, so no learnable avoid-action from the frame at decision time;
  avoiding all water costs depth.

**EXHAUSTED (all < or ≈ 3.71, none beat it):** reward shaping (explore/hp/deep/
survival/dense/eel), human-keystroke imitation, memory (stateful BC + value-head),
entropy annealing, deeper/longer training. The 3.71 ceiling is very well
characterized. **Genuinely untried** (need real new capability, not tuning):
1. **Better/deeper BC teacher** — scripted bot caps at depth 3–6; the BC base may
   cap PPO. A stronger teacher or self-play curriculum could lift the floor.
2. **Transformer-XL** memory (vs GRU) — though memory helped the ceiling not the
   mean, so low priority.
3. Accept 3.71 as the result and consolidate.
BEST policy remains **scripted-PPO 3.71** (`runs/ppo_warm/ppo.pt`).

**⚠️ Host RAM is only 16GB (WSL2 gets ~15GB cap, but Windows needs ~11GB → WSL
must stay under ~5GB or the VM OOM-crashes the whole session).** Keep training
lean: PPO `--envs 32` (not 64), and re-chunk any huge episodes before BC (human
recordings are ~14k frames each → `logs/day2/rechunk_human.py` splits them into
~512-frame pieces; never load the raw `data/human_bc`). A RAM watchdog
(`logs/day2/ram_watchdog.sh`) kills training if WSL available RAM drops below
10GB, converting an OOM into a recoverable kill.

## Conventions

- Python 3.12+; venv at `.venv`; no system pip installs.
- `gamedata/`, `logs/`, `runs/`, `data/`, `.venv/` are gitignored scratch.
- Decision/run logs are JSONL (`logs/runs.jsonl` for the scripted bot).
