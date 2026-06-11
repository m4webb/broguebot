"""Parse a captured Brogue CE 1.15 ncurses screen into structured state.

Layout (terminal mode, >=100x34):
  cols 0-19   sidebar: player block, then one block per visible entity
  col  20     separator
  rows 0-2    message area (cols 21+)
  rows 3..N   dungeon map (cols 21-99)
  row  N+1    flavor text line
  row  N+2    button bar containing "-- Depth: D --"

Popups (help, item descriptions, confirmations) draw over the map and
interleave their footer text with whatever was underneath, so all dialog
markers are matched with gap-tolerant "fuzzy" regexes.
"""

import re
from dataclasses import dataclass, field

SIDEBAR_W = 20
MAP_COL0 = 21

# map glyphs the player can step on for short tactical walks.
# Deliberately excludes ~ and = (water/lava ambiguity in plain text).
WALKABLE = set('."\',<>+&:;!?=%*$()[]/^o')  # o: also a monster; handled by caller
MONSTER_GLYPHS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

MODE_GAME = "game"
MODE_MORE = "more"          # "press space or click to continue"
MODE_CONFIRM = "confirm"    # yes/no question
MODE_SELECT = "select"      # item-selection / cancellable prompt
MODE_MENU = "menu"          # main menu
MODE_DEAD = "dead"
MODE_VICTORY = "victory"
MODE_UNKNOWN = "unknown"


SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")


def bar_fill_fraction(line: str, width: int = SIDEBAR_W) -> float | None:
    """Fraction of a sidebar bar row whose background is 'filled'.

    Brogue renders HP/nutrition as background-colored bars; the text label
    is unreliable (the (-N%) suffix is a transient damage flash). Filled
    cells use a bright background, empty cells a near-black one.
    """
    bg = (0, 0, 0)
    col = 0
    filled = empty = 0
    pos = 0
    for m in SGR_RE.finditer(line):
        text = line[pos:m.start()]
        for _ch in text:
            if col >= width:
                break
            s = sum(bg)
            if s > 90:
                filled += 1
            elif s > 0:
                empty += 1
            col += 1
        pos = m.end()
        params = m.group(1).split(";")
        i = 0
        while i < len(params):
            p = params[i]
            if p == "48" and i + 4 < len(params) and params[i + 1] == "2":
                bg = (int(params[i + 2] or 0), int(params[i + 3] or 0),
                      int(params[i + 4] or 0))
                i += 5
            elif p in ("0", "", "49"):
                bg = (0, 0, 0)
                i += 1
            else:
                i += 1
    for _ch in line[pos:]:
        if col >= width:
            break
        s = sum(bg)
        if s > 90:
            filled += 1
        elif s > 0:
            empty += 1
        col += 1
    total = filled + empty
    if total < 5:
        return None  # not a bar row
    return filled / total


def bg_grid(sgr_lines: list[str]) -> list[list[tuple]]:
    """Per-cell background colors from a colored capture."""
    out = []
    for line in sgr_lines:
        row = []
        bg = (0, 0, 0)
        pos = 0
        for m in SGR_RE.finditer(line):
            for _ch in line[pos:m.start()]:
                row.append(bg)
            pos = m.end()
            params = m.group(1).split(";")
            i = 0
            while i < len(params):
                p = params[i]
                if p == "48" and i + 4 < len(params) and params[i + 1] == "2":
                    bg = (int(params[i + 2] or 0), int(params[i + 3] or 0),
                          int(params[i + 4] or 0))
                    i += 5
                elif p in ("0", "", "49"):
                    bg = (0, 0, 0)
                    i += 1
                else:
                    i += 1
        for _ch in line[pos:]:
            row.append(bg)
        out.append(row)
    return out


def fuzzy(phrase: str) -> re.Pattern:
    """Regex matching phrase chars in order with up to 2 junk chars between.

    Brogue popup chrome interleaves with underlying screen text in terminal
    mode, e.g. '--opressdspacenorlclickutobcontinueu--'.
    """
    letters = [c for c in phrase.lower() if not c.isspace()]
    return re.compile(".{0,2}".join(re.escape(c) for c in letters))


F_PRESS_SPACE = fuzzy("press space or click")
F_CONTINUE = fuzzy("to continue")
F_YESNO = re.compile(r"\(y/n", re.I)
F_DEPTH = re.compile(r"-- Depth: (\d+) --")
# exact substrings: fuzzy matching here would false-positive ("you discover")
F_DEAD = [re.compile(p, re.I) for p in
          (r"\byou die\b", r"killed by", r"\byou starve\b",
           r"died (?:at|on) depth", r"rest in peace", r"save recording as")]
F_VICTORY = [re.compile(p, re.I) for p in
             (r"escape.{1,3}with the amulet", r"\bvictory\b")]
F_SAVE_RECORDING = re.compile(r"save recording as", re.I)
F_MENU = [re.compile(r"New Game"), re.compile(r"\bQuit\b")]
F_CANCEL_PROMPT = fuzzy("esc> to cancel")
F_INV_FOOTER = fuzzy("for more info --")
# popup dialogs render No/Yes buttons instead of a (y/n) message prompt
F_BUTTON_ROW = re.compile(r"\bN.?o\b.{2,44}\bY.?e.?s\b")
# item prompts: "Apply what?", "Enchant what? (a-z; ...)" — esc text optional
F_ITEM_PROMPT = re.compile(r"\bwhat\?|\(a-z[;,)]", re.I)
F_HEALTH_LOST = re.compile(r"Health\s*\(-(\d+)%\)")
F_NUTRITION_LOST = re.compile(r"Nutrition\s*\(-(\d+)%\)")
ITEM_LINE = re.compile(r"([a-z])\).(\S.*?)\s*$")
ENTITY_START = re.compile(r"^(\S): (.+)$")


@dataclass
class Entity:
    glyph: str
    name: str
    hp_lost: int = 0          # percent of health missing
    statuses: list = field(default_factory=list)
    is_player: bool = False

    @property
    def hostile(self) -> bool:
        if self.is_player:
            return False
        low = " ".join(self.statuses).lower() + " " + self.name.lower()
        return not any(s in low for s in
                       ("ally", "captive", "shackled", "discordant friend"))

    @property
    def captive(self) -> bool:
        low = " ".join(self.statuses).lower() + " " + self.name.lower()
        return "captive" in low or "shackled" in low


@dataclass
class Snapshot:
    lines: list
    mode: str = MODE_UNKNOWN
    depth: int = 0
    hp_pct: int = 100
    nutrition_lost: int = 0
    strength: int = 0
    armor: int = 0
    player: Entity | None = None
    entities: list = field(default_factory=list)   # non-player sidebar entries
    messages: list = field(default_factory=list)
    flavor: str = ""
    grid: list = field(default_factory=list)        # map region rows (strings)
    map_row0: int = 3
    player_pos: tuple | None = None                 # (row, col) in grid coords
    item_lines: list = field(default_factory=list)  # [(letter, text)]
    prompt: str = ""

    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def monsters(self):
        return [e for e in self.entities if e.glyph in MONSTER_GLYPHS]

    @property
    def hostiles(self):
        return [e for e in self.monsters if e.hostile]


def parse(lines: list[str]) -> Snapshot:
    snap = Snapshot(lines=list(lines))
    whole = "\n".join(lines)
    whole_low = whole.lower()

    # locate the button bar / depth indicator
    depth_row = None
    for i, ln in enumerate(lines):
        m = F_DEPTH.search(ln)
        if m:
            depth_row = i
            snap.depth = int(m.group(1))
    in_game_ui = depth_row is not None

    # message area + flavor
    if in_game_ui:
        snap.messages = [ln[MAP_COL0:].strip() for ln in lines[:3]]
        snap.flavor = lines[depth_row - 1][MAP_COL0:].strip() if depth_row >= 1 else ""
        snap.grid = [ln[MAP_COL0:].rstrip() for ln in lines[3:depth_row - 1]]
        snap.map_row0 = 3
        for r, row in enumerate(snap.grid):
            c = row.find("@")
            if c >= 0:
                snap.player_pos = (r, c)
                break

    # sidebar entities
    cur = None
    for ln in lines[: depth_row if depth_row else len(lines)]:
        side = ln[:SIDEBAR_W].rstrip()
        if not side:
            cur = None
            continue
        m = ENTITY_START.match(side)
        if m:
            cur = Entity(glyph=m.group(1), name=m.group(2).strip())
            if cur.glyph == "@":
                cur.is_player = True
                snap.player = cur
            else:
                snap.entities.append(cur)
            continue
        if cur is None:
            continue
        stripped = side.strip()
        hm = F_HEALTH_LOST.search(side)
        if hm:
            cur.hp_lost = int(hm.group(1))
            continue
        nm = F_NUTRITION_LOST.search(side)
        if nm and cur.is_player:
            snap.nutrition_lost = int(nm.group(1))
            continue
        sm = re.match(r"^\((.+)\)$", stripped)
        if sm:
            cur.statuses.append(sm.group(1))
            continue
        tm = re.match(r"Str: (\d+)\s+Armor: (-?\d+)", stripped)
        if tm and cur.is_player:
            snap.strength, snap.armor = int(tm.group(1)), int(tm.group(2))
            continue
        if stripped.startswith(("Health", "Nutrition", "Stealth")):
            continue
        # wrapped continuation of the entity name
        cur.name += " " + stripped

    if snap.player:
        snap.hp_pct = 100 - snap.player.hp_lost

    # Item lines from inventory/selection popups. The popup is transparent
    # (map bleeds through its spaces) so anchor on geometry instead: the
    # "x)" tokens of real item entries all align on one column.
    candidates = []
    for ln in lines:
        for m in ITEM_LINE.finditer(ln[MAP_COL0:]):
            candidates.append((m.start(1), m.group(1), m.group(2)))
    if candidates:
        cols = {}
        for col, letter, text in candidates:
            cols.setdefault(col, []).append((letter, text))
        col, best = max(cols.items(), key=lambda kv: len(kv[1]))
        if len(best) > 1 or (candidates and len(cols) == 1):
            seen_letters = set()
            for letter, text in best:
                if letter not in seen_letters:
                    seen_letters.add(letter)
                    snap.item_lines.append((letter, text))

    # ------------------------------------------------------------- mode
    # prompts always occupy the most recent message line; older lines are
    # history and must not be mistaken for a live prompt
    last_msg = next((m for m in reversed(snap.messages) if m.strip()), "")
    if any(p.search(whole_low) for p in F_DEAD):
        snap.mode = MODE_DEAD
    elif any(p.search(whole_low) for p in F_VICTORY):
        snap.mode = MODE_VICTORY
    elif F_PRESS_SPACE.search(whole_low) or F_CONTINUE.search(whole_low) \
            and not in_game_ui:
        snap.mode = MODE_MORE
    elif F_YESNO.search(last_msg) or F_YESNO.search(snap.flavor):
        snap.mode = MODE_CONFIRM
        snap.prompt = last_msg if F_YESNO.search(last_msg) else snap.flavor
    elif any(F_BUTTON_ROW.search(ln) for ln in lines):
        snap.mode = MODE_CONFIRM
        btn_row = next(i for i, ln in enumerate(lines)
                       if F_BUTTON_ROW.search(ln))
        for j in range(btn_row - 1, max(-1, btn_row - 5), -1):
            if "?" in lines[j]:
                snap.prompt = lines[j].strip()
                break
        else:
            snap.prompt = "unknown dialog"
    elif F_ITEM_PROMPT.search(last_msg) or F_CANCEL_PROMPT.search(last_msg.lower()):
        snap.mode = MODE_SELECT
        snap.prompt = last_msg
    elif F_INV_FOOTER.search(whole_low):
        # a bare inventory screen left open eats every game key
        snap.mode = MODE_SELECT
        snap.prompt = ""
    elif in_game_ui:
        # a popup footer can coexist with the depth row
        if F_PRESS_SPACE.search(whole_low):
            snap.mode = MODE_MORE
        else:
            snap.mode = MODE_GAME
    elif all(p.search(whole_low) for p in F_MENU):
        snap.mode = MODE_MENU
    else:
        snap.mode = MODE_UNKNOWN
    return snap
