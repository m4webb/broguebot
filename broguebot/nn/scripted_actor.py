"""Drive the scripted Brain from IPC frames to generate BC training data.

The Brain (broguebot/brain.py) was written against a tmux terminal: it reads
a parsed text Snapshot (screen.parse) and sends multi-key macros, resolving
menus by re-capturing the screen. The IPC env is one-keycode-per-step.

Key insight: the IPC frame's grid IS the game's full 100x34 display buffer —
sidebar (with monster names), message rows, map, depth bar, and any menu
overlay, all stored as Brogue displayGlyph values. So we render the whole
buffer to ASCII and feed it to screen.parse() exactly as the terminal bot
did: mode detection, item menus, and monster names all come for free.

  * `frame_to_snapshot` = render full buffer -> screen.parse, then override
    HP/nutrition with the reliable quantized values from the stats block.
  * `ScriptedActor` is a stateful `actor(frame) -> action_index` for
    trajlog.collect(): when idle it runs Brain.decide() and queues the
    Action's keycodes one per step; it drains those planned keys first (they
    include targeting/selection sequences), and only diverts to answering an
    unexpected prompt (MORE / confirm / select) when the queue is empty.

Inventory is taken straight from the IPC item records (richer and always
correct) so the Brain never needs to open the inventory screen.
"""

import collections

from .. import screen as S
from ..brain import Brain, BotConfig, EpisodeOver
from ..items import Inventory, parse_item
from ..ipc import COLS, ROWS
from ..env import KEYCODES, ACTION_INDEX

# Brogue displayGlyph enum (exact values from Rogue.h) -> ASCII char. Values
# <128 are already ASCII (sidebar/message text, punctuation) and pass through;
# only the >=128 enum entries need a table. We map to chars consistent with
# screen.WALKABLE / MONSTER_GLYPHS so the parsed Snapshot drives the Brain.
GLYPH_CHAR = {
    130: '!',   # G_POTION
    131: '"',   # G_GRASS
    132: '#',   # G_WALL
    134: "'",   # G_OPEN_DOOR
    135: '*',   # G_GOLD
    136: '+',   # G_CLOSED_DOOR
    137: ',',   # G_RUBBLE
    138: '-',   # G_KEY
    139: '~',   # G_BOG
    148: ';',   # G_FOOD
    149: '<',   # G_UP_STAIRS
    150: '=',   # G_VENT
    151: '>',   # G_DOWN_STAIRS
    152: '@',   # G_PLAYER
    173: '[',   # G_ARMOR
    174: '/',   # G_STAFF
    175: ',',   # G_WEB (walkable-ish; NOT ':' to avoid chasm confusion)
    194: ',',   # G_ALTAR
    195: '~',   # G_LIQUID (water/lava ambiguity, as in the terminal bot)
    196: '.',   # G_FLOOR
    197: ':',   # G_CHASM (Brain dives toward ':')
    198: '^',   # G_TRAP
    199: '#',   # G_FIRE (impassable for pathing)
    200: '"',   # G_FOLIAGE
    201: ',',   # G_AMULET
    202: '?',   # G_SCROLL
    203: '=',   # G_RING
    204: ')',   # G_WEAPON
    205: '#',   # G_TURRET (static; obstacle)
    206: '#',   # G_TOTEM
    209: "'",   # G_DOORWAY
    210: '%',   # G_CHARM
    211: '#',   # G_WALL_TOP
    226: '.',   # G_FLOOR_ALT
    228: '*',   # G_GEM
    229: '/',   # G_WAND
    230: '#',   # G_GRANITE
    231: '.',   # G_CARPET
    232: '+',   # G_CLOSED_IRON_DOOR
    233: "'",   # G_OPEN_IRON_DOOR
    234: '#', 235: '#', 236: '#', 237: '#',  # torch/crystal/portcullis/barricade
    238: '#', 239: '#', 240: '#',            # statue/cracked statue/closed cage
    241: "'",   # G_OPEN_CAGE
    246: '=',   # G_BRIDGE
    247: ',',   # G_BONES
    249: "'",   # G_ASHES
    250: '=',   # G_BEDROLL
}
# Monster glyphs -> their catalog display letter (must be in MONSTER_GLYPHS so
# the parser's map scan can locate them and match the sidebar entry).
_MONSTER_CHAR = {
    153: 'B', 154: 'C', 155: 'D', 156: 'F', 157: 'G', 158: 'H', 159: 'I',
    160: 'J', 161: 'K', 162: 'L', 163: 'N', 164: 'O', 165: 'P', 166: 'R',
    167: 'S', 168: 'T', 169: 'U', 170: 'V', 171: 'W', 172: 'Z', 176: 'a',
    177: 'b', 178: 'c', 179: 'd', 180: 'e', 181: 'f', 182: 'g', 183: 'i',
    184: 'j', 185: 'k', 186: 'm', 187: 'p', 188: 'r', 189: 's', 190: 't',
    191: 'v', 192: 'w', 193: 'P', 212: 'd', 213: 'd', 214: 'g', 215: 'g',
    216: 'O', 220: 'Y', 222: 'M', 227: 'u', 133: '&',
}
GLYPH_CHAR.update(_MONSTER_CHAR)

# key-name (as the Brain emits) -> keycode, then keycode -> env action index
KEY_TO_CODE = {"Space": 32, "Enter": 10, "Escape": 27, "Tab": 9,
               "Backspace": 127, "C-x": ord('x')}
CODE_TO_ACTION = {code: i for i, code in enumerate(KEYCODES)}


def glyph_to_char(g: int) -> str:
    if g < 128:
        return chr(g) if 32 <= g < 127 else ' '
    return GLYPH_CHAR.get(g, '.')   # unmapped feature -> assume open ground


def key_to_action(key: str) -> int | None:
    """Map one Brain key (e.g. 'k', '>', 'Space') to an env action index."""
    code = KEY_TO_CODE.get(key)
    if code is None and len(key) == 1:
        code = ord(key)
    return CODE_TO_ACTION.get(code) if code is not None else None


def render_screen(frame) -> list[str]:
    """The full 100x34 display buffer rendered to ASCII lines."""
    return ["".join(glyph_to_char(frame.cell(x, y)[0]) for x in range(COLS))
            .rstrip() for y in range(ROWS)]


def frame_to_snapshot(frame) -> S.Snapshot:
    """screen.parse on the rendered buffer, with reliable HP/nutrition from
    the stats block (the sidebar's -N% label is a transient damage flash)."""
    snap = S.parse(render_screen(frame))
    st = frame.stats
    snap.hp_pct = st.hp_pct
    snap.nutrition_lost = round((1 - st.nutrition_q / 20) * 100)
    if snap.player:
        snap.player.hp_lost = 100 - st.hp_pct
    return snap


def inventory_from_frame(frame) -> Inventory:
    """Rebuild the Brain's Inventory from the IPC item records (their `name`
    is itemName(), the same player-visible string the terminal showed)."""
    inv = Inventory()
    for it in frame.items:
        if it.category == "gold":
            continue
        parsed = parse_item(it.letter, it.name)
        parsed.count = it.quantity or parsed.count
        parsed.equipped = it.equipped
        if it.str_req:
            parsed.str_req = it.str_req
        inv.items[it.letter] = parsed
    return inv


class _StubPane:
    """The Brain constructor wants a Pane; we drive decisions ourselves. Only
    capture_colors is reached (via gas/hazard checks); returning nothing makes
    those degrade to the Brain's message/HP-based fallbacks."""
    def make_sync(self, ready_file):
        return None

    def capture_colors(self, *a, **k):
        return []


class ScriptedActor:
    """Stateful `actor(frame) -> action_index` driving the Brain over IPC.

    One key per call. Planned keys from the current Action drain first (they
    carry selection letters and targeting sequences); an unexpected prompt is
    answered only when nothing is queued, using the Action's hints or the
    Brain's answer_confirm / answer_select.
    """

    def __init__(self, flee_hp: int = 35):
        cfg = BotConfig(headless=False, flee_hp=flee_hp)
        self.brain = Brain(_StubPane(), cfg)
        self.pending = collections.deque()   # queued action indices
        self.context = None                   # current Action (prompt hints)
        self._select_used = False
        self.done_reason = None

    def reset(self):
        self.brain.reset_episode()
        self.pending.clear()
        self.context = None
        self._select_used = False
        self.done_reason = None

    def __call__(self, frame) -> int:
        snap = frame_to_snapshot(frame)
        self.brain.inv = inventory_from_frame(frame)
        self.brain.inv_dirty = False

        # 1) drain the current action's planned keys (selection/targeting)
        if self.pending:
            return self.pending.popleft()

        # 2) an unexpected prompt with nothing queued: answer it
        if snap.mode == S.MODE_MORE:
            return ACTION_INDEX["ack"]
        if snap.mode == S.MODE_CONFIRM:
            ans = (self.context.confirm_hint if self.context else None) \
                or self.brain.answer_confirm(snap, self.context)
            return _code_action(ans)
        if snap.mode == S.MODE_SELECT:
            letter = None
            if self.context and self.context.select_letter \
                    and not self._select_used:
                letter, self._select_used = self.context.select_letter, True
            else:
                letter = self.brain.answer_select(snap)
            return _code_action(letter) if letter else ACTION_INDEX["cancel"]

        # 3) idle in normal play: make a new decision
        try:
            self.brain.observe(snap)
            action = self.brain.decide(snap)
        except EpisodeOver as e:
            self.done_reason = e.result
            return ACTION_INDEX["ack"]
        self._record_action(snap, action)
        for key in action.keys:
            idx = key_to_action(key)
            if idx is not None:
                self.pending.append(idx)
        return self.pending.popleft() if self.pending else ACTION_INDEX["ack"]

    def _record_action(self, snap, action):
        """Mirror the bookkeeping Brain.act() normally does."""
        self.context = action
        self._select_used = False
        self.brain.prev_action = action
        self.brain.prev_keys = list(action.keys)
        self.brain.track_consumption(action)
        self.brain.last_pos = snap.player_pos
        self.brain.turn += 1


def _code_action(key) -> int:
    idx = key_to_action(key if isinstance(key, str) else chr(key))
    return idx if idx is not None else ACTION_INDEX["ack"]
