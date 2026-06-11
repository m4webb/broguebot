"""Client for the brogue IPC platform (vendor ipc-platform.c).

One observation frame arrives on a pipe every time the game wants input;
one little-endian uint16 keycode goes back. Frames carry only
player-visible information (see the fairness contract in ipc-platform.c).
"""

import os
import struct
import subprocess

from .tmux import BOT_BIN

# must mirror ipc-platform.c
MAGIC = 0x42424631
VERSION = 1
TYPE_OBS = 0
TYPE_DONE = 1
COLS, ROWS = 100, 34
MAX_MONSTERS, MAX_ITEMS, NUM_MESSAGES, MSG_LEN = 24, 26, 4, 99
BAR_CELLS = 20

HEADER = struct.Struct("<IBBHI")
STATS = struct.Struct("<HHHHhhIIIIBBBBB3xHH")
MONSTER = struct.Struct("<HBBBBBx")
ITEM = struct.Struct("<BBBBbBbx40s")

STATS_OFF = HEADER.size                          # 12
MON_OFF = STATS_OFF + STATS.size                 # count byte, then records
ITEM_OFF = MON_OFF + 1 + MAX_MONSTERS * MONSTER.size
MSG_OFF = ITEM_OFF + 1 + MAX_ITEMS * ITEM.size
MAP_OFF = MSG_OFF + NUM_MESSAGES * (1 + MSG_LEN)
FRAME_SIZE = MAP_OFF + COLS * ROWS * 8

# creature states (Rogue.h enum creatureStates)
SLEEPING, HUNTING, WANDERING, FLEEING, ALLY = range(5)

# item categories by bit (Rogue.h enum itemCategory)
CATEGORIES = ["food", "weapon", "armor", "potion", "scroll", "staff",
              "wand", "ring", "charm", "gold", "amulet", "gem", "key"]

# status effects by bit (Rogue.h enum statusCondition)
STATUSES = ["searching", "donning", "weakened", "telepathic",
            "hallucinating", "levitating", "slowed", "hasted", "confused",
            "burning", "paralyzed", "poisoned", "stuck", "nauseous",
            "discordant", "immune_to_fire", "explosion_immunity",
            "nutrition", "enters_level_in", "enraged", "magical_fear",
            "entranced", "darkness", "lifespan_remaining", "shielded",
            "invisible", "aggravating"]


class Stats:
    __slots__ = ("depth", "deepest", "hp_q", "nutrition_q", "strength",
                 "armor", "gold", "player_turns", "absolute_turns",
                 "status_mask", "px", "py", "game_has_ended",
                 "weapon_letter", "armor_letter", "stealth_range")

    def __init__(self, raw: bytes):
        (self.depth, self.deepest, self.hp_q, self.nutrition_q,
         self.strength, self.armor, self.gold, self.player_turns,
         self.absolute_turns, self.status_mask, self.px, self.py,
         self.game_has_ended, self.weapon_letter, self.armor_letter,
         self.stealth_range, _pad) = STATS.unpack_from(raw, STATS_OFF)

    @property
    def hp_pct(self) -> int:
        return round(self.hp_q * 100 / BAR_CELLS)

    def statuses(self) -> set:
        return {name for i, name in enumerate(STATUSES)
                if self.status_mask & (1 << i)}


class Monster:
    __slots__ = ("glyph", "x", "y", "hp_q", "state", "captive")

    def __init__(self, rec: tuple):
        self.glyph, self.x, self.y, self.hp_q, self.state, flags = rec
        self.captive = bool(flags & 1)


class Item:
    __slots__ = ("letter", "category", "kind", "equipped", "identified",
                 "enchant", "quantity", "str_req", "name")

    def __init__(self, rec: tuple):
        letter, cat_bit, kind, flags, ench, qty, sreq, name = rec
        self.letter = chr(letter)
        self.category = CATEGORIES[cat_bit] if cat_bit < len(CATEGORIES) \
            else f"cat{cat_bit}"
        self.kind = None if kind == 255 else kind
        self.equipped = bool(flags & 1)
        self.identified = bool(flags & 2)
        self.enchant = None if ench == -128 else ench
        self.quantity = qty
        self.str_req = sreq
        self.name = name.split(b"\0")[0].decode("ascii", "replace")


class Frame:
    """Parsed lazily: stats are cheap and always needed; monsters, items
    and messages only materialize on access (keeps the env's hot loop off
    the GIL as much as possible)."""

    def __init__(self, raw: bytes):
        magic, self.type, version, _, self.seq = HEADER.unpack_from(raw, 0)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"bad frame magic={magic:08x} ver={version}")
        self.raw = raw
        self.stats = Stats(raw)
        self._monsters = self._items = self._messages = None

    @property
    def monsters(self) -> list:
        if self._monsters is None:
            raw = self.raw
            self._monsters = [
                Monster(MONSTER.unpack_from(raw, MON_OFF + 1 + i * MONSTER.size))
                for i in range(raw[MON_OFF])]
        return self._monsters

    @property
    def items(self) -> list:
        if self._items is None:
            raw = self.raw
            self._items = [
                Item(ITEM.unpack_from(raw, ITEM_OFF + 1 + i * ITEM.size))
                for i in range(raw[ITEM_OFF])]
        return self._items

    @property
    def messages(self) -> list:
        if self._messages is None:
            self._messages = []
            for i in range(NUM_MESSAGES):
                off = MSG_OFF + i * (1 + MSG_LEN)
                ln = self.raw[off]
                if ln:
                    self._messages.append(
                        self.raw[off + 1:off + 1 + ln].decode("ascii", "replace"))
        return self._messages

    @property
    def done(self) -> bool:
        return self.type == TYPE_DONE

    def cell(self, x: int, y: int) -> tuple:
        """(glyph, (fr,fg,fb), (br,bg,bb)) at screen position. Colors 0-100."""
        off = MAP_OFF + (x * ROWS + y) * 8
        g = self.raw[off] | (self.raw[off + 1] << 8)
        return (g, tuple(self.raw[off + 2:off + 5]),
                tuple(self.raw[off + 5:off + 8]))

    def glyph_rows(self) -> list[list[int]]:
        """Row-major [y][x] grid of glyph ids."""
        raw = self.raw
        return [[raw[MAP_OFF + (x * ROWS + y) * 8]
                 | (raw[MAP_OFF + (x * ROWS + y) * 8 + 1] << 8)
                 for x in range(COLS)] for y in range(ROWS)]


class BrogueIPC:
    """A single brogue process driven over pipes."""

    def __init__(self, gamedata: str, seed: int | None = None,
                 wizard: bool = False, binary: str = BOT_BIN):
        os.makedirs(gamedata, exist_ok=True)
        out_r, out_w = os.pipe()
        in_r, in_w = os.pipe()
        env = dict(os.environ,
                   BROGUE_IPC_OUT=str(out_w), BROGUE_IPC_IN=str(in_r))
        argv = [binary, "-n", "-E"]
        if wizard:
            argv.append("-W")
        if seed is not None:
            argv += ["-s", str(seed)]
        self.proc = subprocess.Popen(
            argv, cwd=gamedata, env=env, pass_fds=(out_w, in_r),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(out_w)
        os.close(in_r)
        self._out = out_r
        self._in = in_w

    def read_frame(self) -> Frame | None:
        buf = b""
        while len(buf) < FRAME_SIZE:
            chunk = os.read(self._out, FRAME_SIZE - len(buf))
            if not chunk:
                return None  # process exited without a done frame
            buf += chunk
        return Frame(buf)

    def send_key(self, key) -> None:
        if isinstance(key, str):
            key = ord(key)
        os.write(self._in, struct.pack("<H", key))

    def step(self, key) -> Frame | None:
        """Send one key, return the next frame (None on process exit)."""
        try:
            self.send_key(key)
        except BrokenPipeError:
            pass  # game just exited; its final DONE frame may still be queued
        return self.read_frame()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        for fd in (self._out, self._in):
            try:
                os.close(fd)
            except OSError:
                pass
