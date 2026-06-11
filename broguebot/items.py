"""Inventory item parsing for Brogue CE."""

import re
from dataclasses import dataclass, field

from . import knowledge as K

# Brogue popups render spaces transparently: the map shows through them
# ("leather#armor", "(in#hand)"). Match keywords with any char for a space.


def kw_search(phrase: str, text: str):
    return re.search(phrase.replace(" ", "."), text)


NAME_RE = r"([a-z']+(?:.[a-z']+)*)"


def clean_name(raw: str) -> str:
    """Normalize a name that may contain bleed-through junk for spaces."""
    return re.sub(r"[^a-z']+", " ", raw).strip()

CATEGORIES = ("food", "potion", "scroll", "weapon", "thrown", "armor", "wand",
              "staff", "ring", "charm", "amulet", "gem", "key", "gold", "other")

FOOD_WORDS = ("ration of food", "food ration", "mango")
THROWN_WORDS = ("dart", "javelin", "incendiary dart")


@dataclass
class Item:
    letter: str
    text: str
    category: str = "other"
    name: str = ""            # canonical name when identifiable
    count: int = 1
    str_req: int | None = None
    armor_val: int | None = None
    equipped: bool = False
    identified: bool = False  # for potions/scrolls: do we know what it is

    def __repr__(self):
        return f"<{self.letter}) {self.text}>"


def parse_item(letter: str, text: str) -> Item:
    text = re.sub(r"^[^\w\s].", "", text.strip())  # drop leading glyph char
    it = Item(letter=letter, text=text)
    low = it.text.lower()

    m = re.match(r"(\d+)\D", low)
    if m:
        it.count = int(m.group(1))
    sm = re.search(r"<(\d+)>", low)
    if sm:
        it.str_req = int(sm.group(1))
    am = re.search(r"\[(\d+)\]", low)
    if am:
        it.armor_val = int(am.group(1))
    it.equipped = bool(re.search(
        r"\((?:being.)?worn\)|\(in..?.?hand\)?|\(equipped", low))

    def has(*words):
        return any(kw_search(w, low) for w in words)

    if has("gold piece"):
        it.category = "gold"
    elif has("amulet of yendor"):
        it.category = "amulet"
    elif has(*THROWN_WORDS):
        it.category = "thrown"
    elif has(*FOOD_WORDS):
        it.category = "food"
    elif "potion" in low:
        it.category = "potion"
        m = re.search(r"potions?.of." + NAME_RE, low)
        if m:
            it.identified = True
            it.name = clean_name(m.group(1))
    elif "scroll" in low:
        it.category = "scroll"
        m = re.search(r"scrolls?.of." + NAME_RE, low)
        if m:
            it.identified = True
            it.name = clean_name(m.group(1))
    elif "wand" in low:
        it.category = "wand"
        it.identified = bool(kw_search("wand of", low))
    elif "staff" in low:
        it.category = "staff"
        it.identified = bool(kw_search("staff of", low))
    elif "ring" in low:
        it.category = "ring"
        it.identified = bool(kw_search("ring of", low))
    elif "charm" in low:
        it.category = "charm"
        it.identified = bool(kw_search("charm of", low))
        m = re.search(r"charm.of." + NAME_RE, low)
        if m:
            it.name = clean_name(m.group(1))
    elif has("lumenstone", "gem"):
        it.category = "gem"
    elif has("key"):
        it.category = "key"
    else:
        for name in K.ARMORS:
            if kw_search(name, low):
                it.category, it.name = "armor", name
                return it
        for name in sorted(K.WEAPONS, key=len, reverse=True):
            if kw_search(name, low):
                it.category, it.name = "weapon", name
                return it
    return it


def armor_tier(it: Item) -> int:
    if it.armor_val is not None:
        return it.armor_val
    return K.ARMORS.get(it.name, (0, 0))[0]


def weapon_tier(it: Item) -> int:
    return K.WEAPONS.get(it.name, (0, 0))[0]


def enchant_bonus(it: Item) -> int:
    m = re.search(r"([+-]\d+)\s", it.text)
    return int(m.group(1)) if m else 0


@dataclass
class Inventory:
    items: dict = field(default_factory=dict)   # letter -> Item

    @classmethod
    def from_lines(cls, lines):
        inv = cls()
        for letter, text in lines:
            inv.items[letter] = parse_item(letter, text)
        return inv

    def all(self, category=None):
        out = list(self.items.values())
        if category:
            out = [i for i in out if i.category == category]
        return out

    def equipped_armor(self):
        return next((i for i in self.all("armor") if i.equipped), None)

    def equipped_weapon(self):
        return next((i for i in self.all("weapon") if i.equipped), None)

    def potion(self, *names):
        """First identified potion matching any given name."""
        for want in names:
            for it in self.all("potion"):
                if it.identified and want in it.name:
                    return it
        return None

    def unknown_potions(self):
        return [i for i in self.all("potion") if not i.identified]

    def unknown_scrolls(self):
        return [i for i in self.all("scroll") if not i.identified]

    def food(self):
        items = self.all("food")
        return items[0] if items else None
