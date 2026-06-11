"""The decision engine: turns a parsed Snapshot into keystrokes.

Strategy outline (priority order each turn):
  1. emergencies   - heal/flee/desperation-quaff at low HP
  2. combat        - bump-attack adjacent, approach or hold chokepoints
  3. survival      - eat when hungry, rest when hurt and safe
  4. logistics     - equip upgrades, read/quaff-identify consumables safely
  5. progress      - free captives (allies), auto-explore, descend

The bot leans on Brogue's own building blocks (auto-explore `x`, travel to
stairs `>`, rest `Z`) and adds judgement on top.
"""

import collections
import json
import re
import time
from dataclasses import dataclass, field

from . import knowledge as K
from . import screen as S
from .items import Inventory, armor_tier, weapon_tier
from .tmux import Pane

DIR_KEYS = {(-1, 0): "k", (1, 0): "j", (0, -1): "h", (0, 1): "l",
            (-1, -1): "y", (-1, 1): "u", (1, -1): "b", (1, 1): "n"}

NO_EXPLORE_MSGS = ("no path for further exploration",
                   "explored everything",
                   "nothing else to explore")
NO_STAIRS_MSGS = ("no path to", "i see no way")
BLOCKED_EXPLORE_MSGS = ("monsters nearby", "adversaries", "enemies nearby",
                        "not while")


@dataclass
class Action:
    keys: list
    reason: str
    kind: str = "misc"
    literal: bool = False
    select_letter: str | None = None    # answer for an expected item prompt
    confirm_hint: str | None = None     # 'y'/'n' override for expected confirm


@dataclass
class BotConfig:
    headless: bool = False
    act_delay: float = 0.0        # extra pause before each action (watchability)
    settle_timeout: float = 6.0
    max_actions: int = 6000       # per episode safety valve
    flee_hp: int = 35
    rest_hp: int = 75
    descend_hp: int = 85
    quaff_id_depth: int = 2       # min depth to start quaff-identifying
    log_path: str = ""
    ready_file: str = ""          # sentinel file of the patched binary


@dataclass
class Status:
    """Shared state the dashboard renders. All writes are single assignments."""
    state: str = "running"        # running | paused | stepping | quit
    goal: str = "starting"
    last_action: str = ""
    depth: int = 0
    hp_pct: int = 100
    turn: int = 0
    game_num: int = 0
    threats: str = ""
    result: str = ""
    decisions: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=400))


class EpisodeOver(Exception):
    def __init__(self, result: str, cause: str = ""):
        self.result, self.cause = result, cause


class Brain:
    def __init__(self, pane: Pane, cfg: BotConfig, status: Status | None = None):
        self.pane = pane
        self.cfg = cfg
        self.status = status or Status()
        self.log_file = open(cfg.log_path, "a") if cfg.log_path else None
        self.sync = pane.make_sync(cfg.ready_file) if cfg.ready_file else None
        self.sync_misses = 0
        self.episode = 0
        self.reset_episode()

    def send_await(self, *keys: str, literal: bool = False,
                   timeout: float = 8.0):
        """Send keys and wait until brogue asks for input again."""
        if self.sync:
            base = self.sync.count()
            self.pane.send(*keys, literal=literal)
            # the sentinel is emitted only when brogue goes idle with an
            # empty input queue: one increment == everything consumed
            if self.sync.wait(base + 1, timeout):
                self.sync_misses = 0
                return
            # sentinel didn't come (menu screens bypass the input loop, or a
            # stock binary is running): fall back to screen stability
            self.sync_misses += 1
            if self.sync_misses >= 4:
                self.sync = None  # stock binary: stop paying the timeouts
        else:
            self.pane.send(*keys, literal=literal)
        self.pane.capture_stable(timeout=2.0)

    # ------------------------------------------------------------ state

    STARTING_KIT = [("a", "; Some food"), ("b", "( A +0 dagger <12> (in hand)"),
                    ("c", "( 15 +0 darts <10>"),
                    ("d", "[ +0 leather armor [3]<10> (worn)")]

    def reset_episode(self):
        self.episode += 1
        self.prev_msg_window = []
        # the starting kit is constant; afterwards we track the pack from
        # "You now have ..." messages so the inventory screen (whose button
        # loop bypasses the ready sentinel) never needs to be opened
        self.inv = Inventory.from_lines(self.STARTING_KIT)
        self.inv_dirty = False
        self.depth = 1
        self.max_depth = 1
        self.explore_done = False
        self.reads_this_level = 0
        self.quaffs_this_level = 0
        self.descend_attempts = 0
        self.search_rounds = 0
        self.turn = 0
        self.recent = collections.deque(maxlen=10)   # screen hashes
        self.blocked = {}                            # (pos, key) -> expiry turn
        self.last_explore_blocked = False
        self.last_reason = ""
        self.reason_repeats = 0
        self.banned_reasons = set()                  # actions that keep failing
        self.prev_keys = []                          # keys sent last action
        self.stairs_blocked = False
        self.choke_turns = 0
        self.chase = None                            # (pos, name, expiry turn)
        self.water_danger_until = -1                 # turn until water is hot
        self.throw_turns = collections.deque(maxlen=8)
        self.explore_refused = False
        self.appr = {}              # glyph -> (best dist seen, no-progress count)
        self.target_cooldown = {}   # glyph -> turn until we may chase it again
        self.explore_refusals = 0
        self.travel_refused = False
        self.frontier_mode = False
        self.frontier_exhausted = False
        self.frontier_steps_used = 0
        self.chasm_mode = False
        self.gas_until = -1
        self.gas_dir = None
        self.prev_hp = 100
        self.poisoned = False
        self.last_stand_until = -1

    def on_resume(self):
        """User may have played manually: forget level-local assumptions."""
        self.inv_dirty = True
        self.explore_done = False
        self.descend_attempts = 0
        self.search_rounds = 0
        self.recent.clear()
        self.blocked.clear()

    # ------------------------------------------------------------ logging

    def report(self, snap, action: Action):
        st = self.status
        st.goal = action.kind
        st.last_action = action.reason
        st.depth = self.depth
        st.hp_pct = snap.hp_pct if snap else 0
        st.turn = self.turn
        threats = [f"{m.name}({100 - m.hp_lost}%)" for m in snap.hostiles] \
            if snap else []
        st.threats = ", ".join(threats)[:60]
        line = f"[d{self.depth} hp{st.hp_pct:3d}%] {action.reason}"
        st.decisions.append(line)
        if self.log_file:
            self.log_file.write(json.dumps({
                "t": time.time(), "episode": self.episode,
                "turn": self.turn, "depth": self.depth,
                "hp": st.hp_pct, "kind": action.kind, "reason": action.reason,
                "keys": action.keys, "threats": threats,
                "pos": snap.player_pos if snap else None,
            }) + "\n")
            self.log_file.flush()

    # ------------------------------------------------------------ main step

    def step(self) -> S.Snapshot:
        """One decide+act cycle. Raises EpisodeOver at game end."""
        snap = self.settled_snapshot()
        snap = self.resolve_interrupts(snap)
        self.read_true_hp(snap)
        self.observe(snap)
        action = self.decide(snap)
        self.turn += 1
        self.report(snap, action)
        self.act(action)
        return snap

    def settled_snapshot(self, settle: int = 2,
                         timeout: float | None = None) -> S.Snapshot:
        if self.sync:
            # the ready handshake already guarantees a finished render
            return S.parse(self.pane.capture())
        snap = S.parse(self.pane.capture_stable(
            timeout=timeout or self.cfg.settle_timeout, settle=settle))
        return snap

    def read_true_hp(self, snap: S.Snapshot):
        """The sidebar (-N%) label is a transient damage flash, not state.

        Measure the actual fill of the colored Health bar instead (rows 0-2
        of the sidebar: player name, Health bar, Nutrition bar).
        """
        try:
            rows = self.pane.capture_colors(0, 2)
        except Exception:
            return
        for raw in rows:
            plain = S.SGR_RE.sub("", raw)
            if "Health" in plain[:S.SIDEBAR_W]:
                frac = S.bar_fill_fraction(raw)
                if frac is not None:
                    snap.hp_pct = max(1, round(frac * 100))
            elif "Nutrition" in plain[:S.SIDEBAR_W]:
                frac = S.bar_fill_fraction(raw)
                if frac is not None:
                    snap.nutrition_lost = round((1 - frac) * 100)

    def act(self, action: Action):
        if self.cfg.act_delay:
            time.sleep(self.cfg.act_delay)
        self.prev_action = action
        self.prev_keys = list(action.keys)
        self.track_consumption(action)
        # multi-turn commands (explore, travel, long rest) keep acting on
        # their own; pressing another key would interrupt them mid-run
        multi_turn = any(k in ("x", "C-x", ">", "<", "Z") for k in action.keys)
        if action.keys:
            if self.sync:
                self.send_await(*action.keys, literal=action.literal,
                                timeout=60.0 if multi_turn else 8.0)
            else:
                self.pane.send(*action.keys, literal=action.literal)
        if self.sync:
            snap = S.parse(self.pane.capture())
        elif multi_turn:
            snap = self.settled_snapshot(settle=10, timeout=45.0)
        else:
            snap = self.settled_snapshot()
        self.resolve_interrupts(snap, context=action)

    # ------------------------------------------------------------ dialogs

    def resolve_interrupts(self, snap: S.Snapshot,
                           context: Action | None = None) -> S.Snapshot:
        """Clear popups/prompts until we're back at the normal game screen."""
        for _ in range(15):
            if self.pane.is_dead():
                raise EpisodeOver("crashed", "brogue exited")
            if snap.mode == S.MODE_DEAD:
                self.finish_death(snap)
            if snap.mode == S.MODE_VICTORY:
                raise EpisodeOver("won", "escaped with the Amulet")
            if snap.mode == S.MODE_MENU:
                raise EpisodeOver("menu", "returned to main menu")
            if snap.mode == S.MODE_GAME:
                return snap
            if snap.mode == S.MODE_MORE:
                self.send_await("Space")
            elif snap.mode == S.MODE_CONFIRM:
                self.read_true_hp(snap)  # decisions like diving depend on it
                answer = self.answer_confirm(snap, context)
                if answer == "n" and context and len(context.keys) == 1 \
                        and context.keys[0] in DIR_KEYS.values() \
                        and getattr(self, "last_pos", None):
                    # a declined move means that step is a hazard: remember
                    self.blocked[(self.last_pos, context.keys[0])] = \
                        self.turn + 200
                self.send_await(answer)
            elif snap.mode == S.MODE_SELECT:
                letter = context.select_letter if context else None
                if letter:
                    self.send_await(letter, literal=True)
                    context = None  # consume it
                else:
                    letter = self.answer_select(snap)
                    if letter:
                        self.send_await(letter, literal=True)
                    else:
                        self.send_await("Escape")
            else:  # unknown screen: try to back out
                self.send_await("Escape")
            snap = self.settled_snapshot()
        return snap

    def answer_confirm(self, snap: S.Snapshot, context: Action | None) -> str:
        p = (snap.prompt or snap.text()).lower()
        if context and context.confirm_hint:
            return context.confirm_hint
        if "dive" in p or "depths" in p:
            # chasm dive: a level descended for a little fall damage
            return "y" if snap.hp_pct >= 65 else "n"
        no_words = ("lava", "burn", "flames", "chasm", "fall", "pit",
                    "trap", "gas", "web", "acid", "spikes",
                    "quit", "abandon", "save", "discard the old",
                    "your ally", "the captive")
        if any(w in p for w in no_words):
            return "n"
        return "y"

    def answer_select(self, snap: S.Snapshot) -> str | None:
        """Pick an item for an unexpected selection prompt."""
        p = (snap.prompt or "").lower()
        if "enchant" in p:
            target = (self.inv.equipped_armor() or self.inv.equipped_weapon())
            if target:
                return target.letter
            items = snap.item_lines
            return items[0][0] if items else None
        if "identify" in p:
            unknown = self.inv.unknown_potions() + self.inv.unknown_scrolls()
            if unknown:
                return unknown[0].letter
            items = snap.item_lines
            return items[0][0] if items else None
        return None

    def finish_death(self, snap: S.Snapshot):
        """Walk the post-death screens, keep the replay, report the cause."""
        seen = []
        for _ in range(12):
            seen.append(snap.text())
            if S.F_SAVE_RECORDING.search(snap.text()):
                # save the .broguerec for replay. Brogue exits right after
                # (no sentinel comes), so wait for process death, not ready
                self.pane.send("Enter")
                deadline = time.monotonic() + 5.0
                while not self.pane.is_dead() and time.monotonic() < deadline:
                    time.sleep(0.02)
                break
            self.send_await("Space", timeout=4.0)
            snap = self.settled_snapshot()
        raise EpisodeOver("died", self.death_cause("\n".join(seen)))

    def death_cause(self, text: str) -> str:
        m = re.search(r"[Kk]illed by ([^.\n]+?) {2,}", text) or \
            re.search(r"[Kk]illed by ([^.\n]+)", text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        if re.search(r"you starve|starvation", text, re.I):
            return "starvation"
        m = re.search(r"[Dd]ied (?:at|on) depth \d+", text)
        if m:
            return "unknown (" + m.group(0) + ")"
        return "unknown"

    # ------------------------------------------------------------ observation

    def observe(self, snap: S.Snapshot):
        if snap.depth and snap.depth != self.depth:
            self.depth = snap.depth
            self.max_depth = max(self.max_depth, self.depth)
            self.explore_done = False
            self.reads_this_level = 0
            self.quaffs_this_level = 0
            self.descend_attempts = 0
            self.search_rounds = 0
            self.blocked.clear()
            self.appr.clear()
            self.target_cooldown.clear()
            self.explore_refusals = 0
            self.frontier_mode = False
            self.frontier_exhausted = False
            self.frontier_steps_used = 0
            self.chasm_mode = False
        # a single-step move that didn't move us (and wasn't an attack or
        # door-bump) is being refused somehow: avoid that direction briefly
        prev = getattr(self, "prev_action", None)
        prev_pos = getattr(self, "last_pos", None)
        if prev and len(prev.keys) == 1 and prev.keys[0] in DIR_KEYS.values() \
                and not prev.reason.startswith(("attack", "free captive")) \
                and snap.player_pos is not None \
                and snap.player_pos == prev_pos:
            self.blocked[(snap.player_pos, prev.keys[0])] = self.turn + 8
        self.last_pos = snap.player_pos
        msgs = " ".join(snap.messages).lower()
        # the message window persists for many turns; danger triggers must
        # only see lines that newly appeared this action. Repeats grow an
        # "(xN)" suffix, so a plain set-difference catches re-occurrences.
        prev_window = set(getattr(self, "prev_msg_window", ()))
        fresh = " ".join(l for l in snap.messages
                         if l.strip() and l not in prev_window).lower()
        self.prev_msg_window = list(snap.messages)
        # message window persists across actions/levels: only trust outcome
        # messages that directly follow the command that produces them
        explored = any(k in ("x", "C-x") for k in self.prev_keys)
        if explored and any(s in msgs for s in NO_EXPLORE_MSGS):
            self.explore_done = True
        if explored:
            # explore that moved us nowhere is being refused (enemies in view)
            self.explore_refused = (snap.player_pos == prev_pos
                                    and not self.explore_done)
            self.explore_refusals = self.explore_refusals + 1 \
                if self.explore_refused else 0
        for m in re.finditer(r"you now have (.+?) \(([a-z])\)[.,]?",
                             msgs):
            from .items import parse_item
            self.inv.items[m.group(2)] = parse_item(m.group(2), m.group(1))
        water_foe = any("eel" in m.name.lower() or "kraken" in m.name.lower()
                        or "bog monster" in m.name.lower()
                        for m in snap.hostiles)
        if water_foe or re.search(r"\beel\b|\bkraken\b", fresh):
            self.water_danger_until = self.turn + 12
        # gas clouds are invisible in plain text (background color only):
        # react to the choke messages and flavor text instead
        env_text = fresh + " " + snap.flavor.lower()
        if re.search(r"caustic|you (?:choke|cough)|burning|searing|noxious"
                     r"|eating at your flesh|scald|burns you", env_text):
            self.gas_until = max(self.gas_until, self.turn + 5)
        self.poisoned = bool(snap.player and any(
            "poison" in s.lower() for s in snap.player.statuses))
        # universal reflex: HP dropping with nothing adjacent means invisible
        # environmental damage; get out of whatever we're standing in.
        # Poison ticks are NOT environmental: fleeing can't outrun them.
        drop = self.prev_hp - snap.hp_pct
        if drop >= 8 and snap.player_pos is not None and not self.poisoned:
            pr, pc = snap.player_pos
            adjacent_foe = any(
                max(abs(r - pr), abs(c - pc)) == 1
                for m in snap.hostiles for r, c in self.glyph_cells(snap, m.glyph))
            if not adjacent_foe:
                self.gas_until = max(self.gas_until, self.turn + 4)
        self.prev_hp = snap.hp_pct
        self.last_explore_blocked = explored and \
            any(s in msgs for s in BLOCKED_EXPLORE_MSGS)
        self.stairs_blocked = ">" in self.prev_keys and \
            any(s in msgs for s in NO_STAIRS_MSGS)
        # '>' is silently ignored while an enemy is in view: no message, no
        # movement. Treat that as "must deal with the monsters first".
        self.travel_refused = (">" in self.prev_keys
                               and snap.player_pos == prev_pos
                               and not self.stairs_blocked)
        # stuck detection
        self.recent.append(hash(snap.text()))

    def is_stuck(self) -> bool:
        return (len(self.recent) == self.recent.maxlen
                and len(set(self.recent)) <= 2)

    # ------------------------------------------------------------ inventory

    def track_consumption(self, action: Action):
        """Mirror inventory changes implied by our own actions."""
        def consume(letter):
            it = self.inv.items.get(letter)
            if it is None:
                return
            it.count -= 1
            if it.count <= 0:
                del self.inv.items[letter]

        if action.keys[:1] == ["a"] and action.select_letter:
            consume(action.select_letter)
        elif action.keys[:1] == ["t"] and len(action.keys) > 1:
            consume(action.keys[1])
        elif action.keys[:1] == ["e"] and action.select_letter:
            it = self.inv.items.get(action.select_letter)
            if it:
                for other in self.inv.all(it.category):
                    other.equipped = False
                it.equipped = True

    def refresh_inventory(self):
        self.send_await("i")
        snap = self.settled_snapshot()
        if snap.item_lines:
            self.inv = Inventory.from_lines(snap.item_lines)
        self.send_await("Escape")
        if not self.sync:
            self.pane.capture_stable(timeout=1.5)
        self.inv_dirty = False

    # ------------------------------------------------------------ decisions

    def decide(self, snap: S.Snapshot) -> Action:
        if self.is_stuck():
            self.recent.clear()
            return self.unstick(snap)

        hostiles = snap.hostiles
        infos = [K.monster_info(m.name) for m in hostiles]
        if self.travel_refused and hostiles:
            # stairs travel is pinned by visible enemies: chase them down
            self.target_cooldown.clear()
            self.explore_refused = True
        # turrets/statues etc. we can't chase
        active = [(m, i) for m, i in zip(hostiles, infos)
                  if not ({"static", "avoid", "water"} & i["flags"])]

        act = (self.gas_retreat(snap)
               or self.water_retreat(snap)
               or self.emergency(snap, hostiles, infos)
               or self.combat(snap, active, list(zip(hostiles, infos)))
               or self.pursue(snap, active)
               or self.upkeep(snap, hostiles)
               or self.progress(snap, hostiles))

        # an action that repeats without effect is broken: ban it this level
        # (emergencies are exempt: fleeing gas looks repetitive but is vital)
        if act.reason == self.last_reason and act.kind in ("logistics",
                                                           "survival"):
            self.reason_repeats += 1
            if self.reason_repeats >= 3:
                self.banned_reasons.add(act.reason)
                self.reason_repeats = 0
                return Action(["Escape"], f"ban looping action: {act.reason}",
                              kind="recover")
        else:
            self.reason_repeats = 0
        self.last_reason = act.reason
        if act.reason in self.banned_reasons:
            return self.progress(snap, hostiles)
        return act

    # --- emergencies

    def hazard_bgs(self, snap):
        """Background colors of the map region (gas/fire tint cells)."""
        try:
            rows = self.pane.capture_colors(snap.map_row0,
                                            snap.map_row0 + len(snap.grid) - 1)
        except Exception:
            return None
        grid = S.bg_grid(rows)

        def cell(r, c):
            sc = c + S.MAP_COL0
            if r < len(grid) and sc < len(grid[r]):
                return grid[r][sc]
            return (0, 0, 0)
        return cell

    def gas_retreat(self, snap) -> Action | None:
        """We're taking damage from a cloud: use colors to walk out of it."""
        if self.turn > self.gas_until or snap.player_pos is None:
            return None
        bg = self.hazard_bgs(snap)
        pr, pc = snap.player_pos
        if bg is not None:
            mine = bg(pr, pc)
            if sum(mine) < 50:
                return None  # standing on clean ground already
            if mine[2] > 60 and mine[2] > mine[0] + mine[1]:
                return None  # blue tint is water, not a cloud
            # BFS to the nearest cell with a dark (untinted) background
            seen = {(pr, pc)}
            queue = collections.deque([((pr, pc), None)])
            while queue:
                pos, first = queue.popleft()
                if first and sum(bg(*pos)) < 50:
                    return Action([first], "escape cloud (color path)",
                                  kind="emergency")
                for (dr, dc), key in DIR_KEYS.items():
                    np = (pos[0] + dr, pos[1] + dc)
                    if np in seen or not self.can_step(snap, *pos, dr, dc):
                        continue
                    ch = snap.grid[np[0]][np[1]] if np[0] < len(snap.grid) \
                        and np[1] < len(snap.grid[np[0]]) else "#"
                    if ch in S.MONSTER_GLYPHS or ch == "~":
                        continue
                    seen.add(np)
                    queue.append((np, first or key))
        # colors unavailable: fall back to committed-direction marching
        if self.last_pos == snap.player_pos:
            self.gas_dir = None  # last flee step didn't move: try another way
        if self.gas_dir and self.can_step(snap, pr, pc, *self.gas_dir[0]):
            d, key = self.gas_dir
            return Action([key], "flee gas cloud", kind="emergency")
        for (dr, dc), key in DIR_KEYS.items():
            if self.blocked.get(((pr, pc), key), 0) > self.turn:
                continue
            if self.can_step(snap, pr, pc, dr, dc) and \
                    snap.grid[pr + dr][pc + dc] not in S.MONSTER_GLYPHS:
                self.gas_dir = ((dr, dc), key)
                return Action([key], "flee gas cloud", kind="emergency")
        return None

    def near_water(self, snap, r, c, radius=1) -> bool:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < len(snap.grid) and 0 <= cc < len(snap.grid[rr]) \
                        and snap.grid[rr][cc] == "~":
                    return True
        return False

    def water_retreat(self, snap) -> Action | None:
        """An eel just struck (or lurks nearby): get well clear of the shore."""
        if self.turn > self.water_danger_until or snap.player_pos is None:
            return None
        pr, pc = snap.player_pos
        if not self.near_water(snap, pr, pc, radius=1):
            return None
        # BFS to the nearest cell with NO water within 2 (out of eel reach);
        # in a swamp that may require wading, which beats standing still
        for allow_wading, radius in ((False, 2), (True, 1)):
            seen = {(pr, pc)}
            queue = collections.deque([((pr, pc), None)])
            while queue:
                pos, first = queue.popleft()
                if first and not self.near_water(snap, *pos, radius=radius):
                    return Action([first], "retreat from water (eel!)",
                                  kind="emergency")
                for (dr, dc), key in DIR_KEYS.items():
                    np = (pos[0] + dr, pos[1] + dc)
                    if np in seen or not self.can_step(snap, *pos, dr, dc):
                        continue
                    ch = snap.grid[np[0]][np[1]]
                    if ch in S.MONSTER_GLYPHS:
                        continue
                    if ch == "~" and not allow_wading:
                        continue
                    seen.add(np)
                    queue.append((np, first or key))
        return None

    def emergency(self, snap, hostiles, infos) -> Action | None:
        hp = snap.hp_pct
        in_gas = self.turn <= self.gas_until
        # poison keeps ticking whatever we do: heal before it finishes us
        if self.poisoned and hp < 50:
            heal = self.inv.potion(*K.EMERGENCY_POTIONS)
            if heal:
                self.inv_dirty = True
                return Action(["a"], f"poisoned: quaff {heal.name}",
                              kind="emergency", select_letter=heal.letter)
        if hp > self.cfg.flee_hp or not (hostiles or in_gas or self.poisoned):
            return None
        heal = self.inv.potion(*K.EMERGENCY_POTIONS)
        if heal:
            self.inv_dirty = True
            return Action(["a"], f"emergency: quaff {heal.name}",
                          kind="emergency", select_letter=heal.letter)
        charm = next((c for c in self.inv.all("charm")
                      if "health" in c.name), None)
        if charm:
            return Action(["a"], "emergency: health charm",
                          kind="emergency", select_letter=charm.letter)
        if hp <= 22:
            # desperation: any unknown consumable might be the way out
            if self.inv.unknown_potions():
                pot = self.inv.unknown_potions()[0]
                self.inv_dirty = True
                return Action(["a"], f"desperation: quaff {pot.text}",
                              kind="emergency", select_letter=pot.letter,
                              confirm_hint="y")
            if self.inv.unknown_scrolls():
                sc = self.inv.unknown_scrolls()[0]
                self.inv_dirty = True
                return Action(["a"], f"desperation: read {sc.text}",
                              kind="emergency", select_letter=sc.letter,
                              confirm_hint="y")
        # retreat a step if nothing is adjacent yet; if cornered, commit to
        # the fight instead of thrashing between retreat and approach
        if hostiles and self.turn > getattr(self, "last_stand_until", -1):
            step = self.step_away(snap)
            if step:
                return Action([step], "retreat: low hp", kind="emergency")
            self.last_stand_until = self.turn + 12
        return None  # cornered: fall through and fight

    # --- combat

    def combat(self, snap, active, everyone=None) -> Action | None:
        if snap.player_pos is None or not (active or everyone):
            return None
        targets = self.monster_cells(snap, active)
        pr, pc = snap.player_pos
        in_corr = self.in_corridor(snap)

        def asleep(m):
            return "sleeping" in " ".join(m.statuses).lower()

        # outnumbered in the open: fall back so they queue up single file
        awake_near = [(m, i) for m, i, pos in targets
                      if not asleep(m) and
                      max(abs(pos[0] - pr), abs(pos[1] - pc)) <= 7]
        # retreating from double-speed hunters just gives them free hits
        any_fast = any("fast" in i["flags"] for m, i in awake_near)
        outnumbered = len(awake_near) >= 3 and not in_corr and not any_fast

        # adjacent enemy: hit the most wounded one (sleepers too: sneak attack)
        adj = [(m, i, pos) for m, i, pos in targets
               if max(abs(pos[0] - pr), abs(pos[1] - pc)) == 1]
        if adj:
            self.choke_turns = 0
            m, i, pos = max(adj, key=lambda t: t[0].hp_lost)
            # bloats explode into gas when killed adjacent: pop from range
            if "pop" in i["flags"] and self.inv.all("thrown"):
                step = self.step_away(snap)
                if step:
                    return Action([step], f"back off from {m.name} (pops)",
                                  kind="combat")
            # striking a splitter in the open breeds a swarm: back off instead
            if not in_corr and ("splits" in i["flags"] or outnumbered):
                step = self.chokepoint_step(snap) or self.step_away(snap)
                if step:
                    return Action([step], f"fall back from {m.name}",
                                  kind="combat")
            key = DIR_KEYS[(pos[0] - pr, pos[1] - pc)]
            return Action([key], f"attack {m.name}", kind="combat")

        if outnumbered:
            step = self.chokepoint_step(snap) or self.step_away(snap)
            if step:
                return Action([step],
                              f"fall back ({len(awake_near)} foes, open ground)",
                              kind="combat")

        ranged = self.ranged_attack(snap, everyone or active)
        if ranged:
            return ranged
        if not targets:
            return None

        # let an actively hunting enemy come to us in a corridor (a few turns)
        hunting = any("hunting" in " ".join(m.statuses).lower()
                      for m, i, pos in targets)
        shore_peril = self.turn <= self.water_danger_until and \
            self.near_water(snap, pr, pc, radius=2)
        if hunting and self.in_corridor(snap) and self.choke_turns < 3 \
                and not shore_peril:
            self.choke_turns += 1
            return Action(["z"], "hold chokepoint", kind="combat")
        self.choke_turns = 0
        # approach the nearest reachable enemy (sleepers are free kills);
        # distant wanderers aren't worth a cross-level chase unless they
        # block exploration entirely
        best = None
        for m, i, pos in targets:
            st = " ".join(m.statuses).lower()
            if "fleeing" in st:
                continue  # not worth chasing
            if "splits" in i["flags"]:
                continue  # never start a fight with a jelly
            if "pop" in i["flags"] and self.inv.all("thrown"):
                continue  # bloats are dart targets, not melee targets
            if self.target_cooldown.get(m.glyph, 0) > self.turn:
                continue  # chasing this one went nowhere; let it be a while
            if self.turn <= self.water_danger_until and \
                    self.near_water(snap, *pos, radius=1):
                continue  # shoreline bait while an eel lurks
            path = self.path_step(snap, pos)
            if path:
                step, dist = path
                if dist > 8 and not ("sleeping" in st or "hunting" in st
                                     or self.explore_refused):
                    continue
                # kill summoners first or their minions never stop coming
                score = dist - (100 if "summoner" in i["flags"] else 0)
                if best is None or score < best[2]:
                    best = (m, step, score)
        if best:
            m, step, dist = best
            # a chase that never closes distance gets abandoned for a while
            bd, fails = self.appr.get(m.glyph, (dist, -1))
            fails = fails + 1 if dist >= bd else 0
            self.appr[m.glyph] = (min(bd, dist), fails)
            if fails >= 6:
                self.target_cooldown[m.glyph] = self.turn + 100
                self.appr.pop(m.glyph, None)
                self.chase = None
                return None
            # remember the target: it may drop out of view while we walk
            for mm, ii, pos in targets:
                if mm is m:
                    self.chase = (pos, m.name, self.turn + 40)
                    break
            verb = "sneak up on" if asleep(m) else "approach"
            return Action([step], f"{verb} {m.name} (dist {dist})",
                          kind="combat")
        return None  # nobody reachable: keep exploring/progressing

    def ranged_attack(self, snap, pairs) -> Action | None:
        """Throw darts at foes we can't or shouldn't melee yet."""
        thrown = self.inv.all("thrown")
        if not thrown or snap.player_pos is None:
            return None
        pr, pc = snap.player_pos
        best = None
        for m, i in pairs:
            st = " ".join(m.statuses).lower()
            if any(s in st for s in ("sleeping", "submerged", "fleeing")):
                continue
            worth = "water" in i["flags"] or "pop" in i["flags"] or \
                (i["threat"] >= 4 and "hunting" in st)
            if not worth:
                continue
            for pos in self.glyph_cells(snap, m.glyph):
                d = max(abs(pos[0] - pr), abs(pos[1] - pc))
                if 2 <= d <= 7 and (best is None or d < best[1]):
                    best = (m, d)
        if best:
            # if 8 recent throws haven't resolved the fight, stop wasting ammo
            if len(self.throw_turns) == self.throw_turns.maxlen and \
                    self.turn - self.throw_turns[0] < 12:
                return None
            m, d = best
            it = thrown[0]
            self.inv_dirty = True
            self.throw_turns.append(self.turn)
            return Action(["t", it.letter, "Tab", "Enter"],
                          f"throw dart at {m.name} (dist {d})", kind="combat")
        return None

    def pursue(self, snap, active) -> Action | None:
        """Keep walking toward the last seen position of an out-of-view foe."""
        if active or not self.chase or snap.player_pos is None:
            return None
        pos, name, expiry = self.chase
        pr, pc = snap.player_pos
        arrived = max(abs(pos[0] - pr), abs(pos[1] - pc)) <= 1
        if self.turn > expiry or arrived:
            self.chase = None
            return None
        path = self.path_step(snap, pos)
        if not path:
            self.chase = None
            return None
        return Action([path[0]], f"pursue {name} (last seen)", kind="combat")

    # --- survival / logistics

    def upkeep(self, snap, hostiles) -> Action | None:
        msgs = " ".join(snap.messages).lower()
        hungry = snap.nutrition_lost >= 80 or "hungry" in msgs \
            or "starving" in msgs or "famished" in msgs
        if hungry:
            food = self.inv.food()
            if food:
                self.inv_dirty = True
                return Action(["a"], f"eat {food.text}", kind="survival",
                              select_letter=food.letter)
        if hostiles:
            return None

        # drink obviously-good potions immediately
        for name in ("strength", "life"):
            pot = self.inv.potion(name)
            if pot:
                self.inv_dirty = True
                return Action(["a"], f"quaff potion of {name}",
                              kind="logistics", select_letter=pot.letter)

        # equipment upgrades
        up = self.equip_upgrade(snap)
        if up:
            return up

        # identify chores, only when healthy and quiet
        if snap.hp_pct >= 80:
            if self.inv.unknown_scrolls() and self.reads_this_level < 2:
                sc = self.inv.unknown_scrolls()[0]
                self.reads_this_level += 1
                self.inv_dirty = True
                return Action(["a"], f"read-ID {sc.text}", kind="logistics",
                              select_letter=sc.letter, confirm_hint="y")
            if (snap.hp_pct >= 90 and self.depth >= self.cfg.quaff_id_depth
                    and self.quaffs_this_level < 1
                    and self.inv.unknown_potions()
                    and self.open_area(snap)):
                pot = self.inv.unknown_potions()[0]
                self.quaffs_this_level += 1
                self.inv_dirty = True
                return Action(["a"], f"quaff-ID {pot.text}", kind="logistics",
                              select_letter=pot.letter, confirm_hint="y")

        if snap.hp_pct < self.cfg.rest_hp:
            return Action(["Z"], f"rest (hp {snap.hp_pct}%)", kind="survival")
        return None

    def equip_upgrade(self, snap) -> Action | None:
        cur_a = self.inv.equipped_armor()
        cur_t = armor_tier(cur_a) if cur_a else 0
        for it in self.inv.all("armor"):
            if it.equipped:
                continue
            if armor_tier(it) > cur_t and (it.str_req or 0) <= snap.strength + 1:
                self.inv_dirty = True
                return Action(["e"], f"wear {it.text}", kind="logistics",
                              select_letter=it.letter, confirm_hint="y")
        cur_w = self.inv.equipped_weapon()
        cur_wt = weapon_tier(cur_w) if cur_w else 0
        for it in self.inv.all("weapon"):
            if it.equipped:
                continue
            if weapon_tier(it) > cur_wt and (it.str_req or 0) <= snap.strength + 1:
                self.inv_dirty = True
                return Action(["e"], f"wield {it.text}", kind="logistics",
                              select_letter=it.letter, confirm_hint="y")
        return None

    # --- progress

    def progress(self, snap, hostiles) -> Action | None:
        # free captives for allies
        captive = next((e for e in snap.entities if e.captive), None)
        if captive and not hostiles and snap.player_pos:
            cells = self.glyph_cells(snap, captive.glyph)
            if cells:
                pr, pc = snap.player_pos
                pos = cells[0]
                if max(abs(pos[0] - pr), abs(pos[1] - pc)) == 1:
                    key = DIR_KEYS[(pos[0] - pr, pos[1] - pc)]
                    return Action([key], f"free captive {captive.name}",
                                  kind="progress", confirm_hint="y")
                path = self.path_step(snap, pos)
                if path:
                    return Action([path[0]],
                                  f"approach captive {captive.name}",
                                  kind="progress")

        # repeated refusals (an unreachable foe pins the explore command):
        # give up on full exploration and head down instead
        if not self.explore_done and self.explore_refusals < 4:
            if self.last_explore_blocked and hostiles:
                return Action(["z"], "wait: explore blocked by foes",
                              kind="progress")
            key = "C-x" if self.cfg.headless else "x"
            return Action([key], "auto-explore", kind="progress")

        # level finished: heal up, then take the stairs
        if snap.hp_pct < self.cfg.descend_hp and not hostiles:
            return Action(["Z"], f"rest before stairs (hp {snap.hp_pct}%)",
                          kind="survival")
        # a chasm is a free ride to the next depth when stairs are missing
        if self.chasm_mode:
            if snap.hp_pct < self.cfg.descend_hp and not hostiles:
                return Action(["Z"], f"rest before diving (hp {snap.hp_pct}%)",
                              kind="survival")
            step = self.chasm_step(snap)
            if step:
                return Action([step], "jump into the chasm (descend)",
                              kind="progress", confirm_hint="y")
            self.chasm_mode = False

        if self.frontier_mode:
            self.frontier_steps_used += 1
            if self.frontier_steps_used > 80:
                # a frontier walk this long is going in circles: give up
                self.frontier_mode = False
                self.frontier_exhausted = True
            step = self.frontier_step(snap) if self.frontier_mode else None
            if step:
                return Action([step], "push toward unexplored area (swim)",
                              kind="progress")
            # frontier reached: new ground is visible, explore it normally
            self.frontier_mode = False
            self.explore_done = False
            self.explore_refusals = 0
            return Action(["C-x" if self.cfg.headless else "x"],
                          "explore new ground", kind="progress")
        if self.stairs_blocked or self.descend_attempts >= 3:
            self.descend_attempts = 0
            if self.chasm_step(snap):
                self.chasm_mode = True
                self.blocked.clear()  # earlier declined jumps don't count now
                return self.progress(snap, hostiles)
            # stairs never seen: often they're across deep water that
            # auto-explore refuses to swim; push the frontier ourselves
            if not self.frontier_exhausted and self.frontier_step(snap):
                self.frontier_mode = True
                return self.progress(snap, hostiles)
            # otherwise search for secret doors, then re-explore
            self.search_rounds += 1
            self.explore_done = self.search_rounds > 4
            if self.search_rounds > 6:
                raise EpisodeOver("stuck", f"no way down from d{self.depth}")
            return Action(["s", "s", "s", "s", "s"],
                          f"search for hidden paths (round {self.search_rounds})",
                          kind="progress")
        self.descend_attempts += 1
        return Action([">"], "descend", kind="progress")

    # --- stuck recovery

    def unstick(self, snap) -> Action:
        self.explore_done = False
        self.inv_dirty = True
        if snap.player_pos:
            pr, pc = snap.player_pos
            for (dr, dc), key in DIR_KEYS.items():
                if self.blocked.get(((pr, pc), key), 0) > self.turn:
                    continue
                if self.can_step(snap, pr, pc, dr, dc):
                    return Action(["Escape", key], "unstick: sidestep",
                                  kind="recover")
        return Action(["Escape", "Space"], "unstick: clear ui", kind="recover")

    # ------------------------------------------------------------ map utils

    def walkable(self, snap, r, c) -> bool:
        if r < 0 or r >= len(snap.grid):
            return False
        row = snap.grid[r]
        if c < 0 or c >= len(row):
            return False
        ch = row[c]
        return ch in S.WALKABLE or ch in S.MONSTER_GLYPHS

    def can_step(self, snap, r, c, dr, dc) -> bool:
        """Brogue forbids cutting corners: diagonals need both sides open."""
        if not self.walkable(snap, r + dr, c + dc):
            return False
        if dr and dc:
            return self.walkable(snap, r + dr, c) and \
                self.walkable(snap, r, c + dc)
        return True

    def in_corridor(self, snap) -> bool:
        if snap.player_pos is None:
            return False
        r, c = snap.player_pos
        open_n = sum(self.walkable(snap, r + dr, c + dc)
                     for dr, dc in DIR_KEYS)
        return open_n <= 3

    def glyph_cells(self, snap, glyph) -> list:
        out = []
        for r, row in enumerate(snap.grid):
            for c, ch in enumerate(row):
                if ch == glyph:
                    out.append((r, c))
        return out

    def monster_cells(self, snap, pairs):
        """[(entity, info, (r,c))] for each monster we can locate on the map."""
        out = []
        for m, i in pairs:
            for pos in self.glyph_cells(snap, m.glyph):
                out.append((m, i, pos))
        return out

    def path_step(self, snap, goal) -> tuple[str, int] | None:
        """BFS first-step key and distance from player to goal cell."""
        start = snap.player_pos
        if start is None:
            return None
        if start == goal:
            return None
        seen = {start}
        queue = collections.deque([(start, None, 0)])
        while queue:
            pos, first, dist = queue.popleft()
            for (dr, dc), key in DIR_KEYS.items():
                if first is None and \
                        self.blocked.get((start, key), 0) > self.turn:
                    continue  # we tried this step and declined a hazard
                np = (pos[0] + dr, pos[1] + dc)
                if np in seen or not self.can_step(snap, pos[0], pos[1],
                                                   dr, dc):
                    continue
                seen.add(np)
                nfirst = first or key
                if np == goal:
                    return nfirst, dist + 1
                if snap.grid[np[0]][np[1]] not in S.MONSTER_GLYPHS:
                    queue.append((np, nfirst, dist + 1))
        return None

    def chasm_step(self, snap) -> str | None:
        """First step toward the nearest chasm cell (':') to dive down it."""
        if snap.player_pos is None:
            return None
        cells = self.glyph_cells(snap, ":")
        if not cells:
            return None
        pr, pc = snap.player_pos
        best = None
        for pos in cells:
            d = max(abs(pos[0] - pr), abs(pos[1] - pc))
            if best is None or d < best[0]:
                best = (d, pos)
        path = self.path_step(snap, best[1])
        if path:
            return path[0]
        return None

    def frontier_step(self, snap) -> str | None:
        """Step toward unexplored space, swimming if that's the only way.

        Auto-explore refuses to cross deep water, so levels whose stairs sit
        beyond a lake dead-end it. We swim: BFS that treats '~' as passable
        (unless an eel was seen recently) toward any cell bordering the
        unknown.
        """
        start = snap.player_pos
        if start is None or self.turn <= self.water_danger_until:
            return None

        def wet_walkable(r, c):
            if r < 0 or r >= len(snap.grid):
                return False
            row = snap.grid[r]
            if c < 0 or c >= len(row):
                return False
            return row[c] in S.WALKABLE or row[c] == "~"

        def borders_unknown(r, c):
            for dr, dc in DIR_KEYS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < len(snap.grid):
                    row = snap.grid[rr]
                    if cc >= len(row) or (cc >= 0 and row[cc] == " "):
                        return True
            return False

        seen = {start}
        queue = collections.deque([(start, None)])
        while queue:
            pos, first = queue.popleft()
            # only LAND bordering the unknown is a real frontier: water at
            # the map edge "borders unknown" forever without revealing anything
            if first and snap.grid[pos[0]][pos[1]] != "~" \
                    and borders_unknown(*pos):
                return first
            for (dr, dc), key in DIR_KEYS.items():
                if first is None and \
                        self.blocked.get((start, key), 0) > self.turn:
                    continue
                np = (pos[0] + dr, pos[1] + dc)
                if np in seen or not wet_walkable(*np):
                    continue
                if dr and dc and not (wet_walkable(pos[0] + dr, pos[1])
                                      and wet_walkable(pos[0], pos[1] + dc)):
                    continue
                if snap.grid[np[0]][np[1]] in S.MONSTER_GLYPHS:
                    continue
                seen.add(np)
                queue.append((np, first or key))
        return None

    def open_area(self, snap, radius: int = 2, need: int = 12) -> bool:
        """Is there room to run if this spot erupts in gas or flames?"""
        if snap.player_pos is None:
            return False
        pr, pc = snap.player_pos
        n = sum(1 for dr in range(-radius, radius + 1)
                for dc in range(-radius, radius + 1)
                if self.walkable(snap, pr + dr, pc + dc)
                and snap.grid[pr + dr][pc + dc] != "~")
        return n >= need

    def open_neighbors(self, snap, r, c) -> int:
        return sum(self.walkable(snap, r + dr, c + dc) for dr, dc in DIR_KEYS)

    def chokepoint_step(self, snap) -> str | None:
        """First step toward the nearest corridor cell away from monsters."""
        start = snap.player_pos
        if start is None:
            return None
        mcells = []
        for m in snap.hostiles:
            mcells += self.glyph_cells(snap, m.glyph)
        if not mcells:
            return None

        def mdist(r, c):
            return min(max(abs(r - mr), abs(c - mc)) for mr, mc in mcells)

        cur_md = mdist(*start)
        seen = {start}
        queue = collections.deque([(start, None, 0)])
        while queue:
            pos, first, dist = queue.popleft()
            if dist > 8:
                break
            if first and self.open_neighbors(snap, *pos) <= 3 \
                    and mdist(*pos) >= max(cur_md, 2):
                return first
            for (dr, dc), key in DIR_KEYS.items():
                if first is None and \
                        self.blocked.get((start, key), 0) > self.turn:
                    continue
                np = (pos[0] + dr, pos[1] + dc)
                if np in seen or not self.can_step(snap, pos[0], pos[1],
                                                   dr, dc):
                    continue
                if snap.grid[np[0]][np[1]] in S.MONSTER_GLYPHS:
                    continue
                seen.add(np)
                queue.append((np, first or key, dist + 1))
        return None

    def step_away(self, snap) -> str | None:
        """Pick a walkable step that increases distance from nearest hostile."""
        if snap.player_pos is None:
            return None
        cells = []
        for m in snap.hostiles:
            cells += self.glyph_cells(snap, m.glyph)
        if not cells:
            return None
        pr, pc = snap.player_pos

        def near(r, c):
            return min(max(abs(r - mr), abs(c - mc)) for mr, mc in cells)

        cur = near(pr, pc)
        best = None
        for (dr, dc), key in DIR_KEYS.items():
            r, c = pr + dr, pc + dc
            if self.can_step(snap, pr, pc, dr, dc) and \
                    snap.grid[r][c] not in S.MONSTER_GLYPHS:
                d = near(r, c)
                if d > cur and (best is None or d > best[0]):
                    best = (d, key)
        return best[1] if best else None
