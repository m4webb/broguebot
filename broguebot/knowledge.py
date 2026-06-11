"""Static game knowledge for Brogue CE 1.15: monsters, items, danger ratings."""

# ---------------------------------------------------------------- monsters

# threat: rough danger score for melee engagement (player power grows ~ depth).
# flags:
#   water    - only fights in water; never wade near it
#   splits   - splits when struck (pink jelly)
#   steals   - grabs an item and flees; don't chase
#   corrode  - attacking it with a weapon corrodes the weapon
#   ranged   - attacks from range (turrets, spellcasters)
#   static   - cannot move (turrets, totems); ignore unless in its line
#   pack     - travels in groups; expect friends
#   avoid    - not worth engaging at all
MONSTERS = {
    "rat":              dict(threat=1),
    "kobold":           dict(threat=1),
    "jackal":           dict(threat=2, flags={"pack", "fast"}),
    "eel":              dict(threat=8, flags={"water", "avoid"}),
    "monkey":           dict(threat=2, flags={"steals"}),
    "pit bloat":        dict(threat=1, flags={"pop"}),
    "acid mound":       dict(threat=3, flags={"corrode"}),
    "centipede":        dict(threat=4),
    "ogre":             dict(threat=7),
    "fungus":           dict(threat=1, flags={"static"}),
    "bloat":            dict(threat=1, flags={"pop"}),
    "spider":           dict(threat=8, flags={"ranged"}),
    "arrow turret":     dict(threat=5, flags={"ranged", "static"}),
    "vampire bat":      dict(threat=4),
    "wraith":           dict(threat=9),
    "zombie":           dict(threat=5),
    "troll":            dict(threat=12),
    "goblin":           dict(threat=4, flags={"pack"}),
    "goblin conjurer":  dict(threat=6, flags={"pack", "ranged", "summoner"}),
    "spectral blade":   dict(threat=2),
    "goblin mystic":    dict(threat=5, flags={"pack"}),
    "goblin warchief":  dict(threat=8, flags={"pack"}),
    "pink jelly":       dict(threat=6, flags={"splits"}),
    "toad":             dict(threat=1),
    "will-o-the-wisp":  dict(threat=6),
    "mound":            dict(threat=3),
    "bog monster":      dict(threat=7, flags={"water"}),
    "ogre shaman":      dict(threat=9, flags={"ranged"}),
    "naga":             dict(threat=10, flags={"water"}),
    "salamander":       dict(threat=11),
    "dar blademaster":  dict(threat=13, flags={"pack"}),
    "dar priestess":    dict(threat=11, flags={"pack", "ranged"}),
    "dar battlemage":   dict(threat=12, flags={"pack", "ranged"}),
    "wisp":             dict(threat=4),
    "specter":          dict(threat=12),
    "vampire":          dict(threat=14),
    "flamedancer":      dict(threat=12, flags={"ranged"}),
    "lich":             dict(threat=18, flags={"ranged"}),
    "phantom":          dict(threat=11),
    "imp":              dict(threat=10, flags={"steals"}),
    "fury":             dict(threat=13, flags={"pack"}),
    "revenant":         dict(threat=13),
    "tentacle horror":  dict(threat=16),
    "golem":            dict(threat=15),
    "dragon":           dict(threat=20),
    "spark turret":     dict(threat=7, flags={"ranged", "static"}),
    "flame turret":     dict(threat=8, flags={"ranged", "static"}),
    "guardian":         dict(threat=0, flags={"static", "avoid"}),
    "kraken":           dict(threat=14, flags={"water", "avoid"}),
    "mirror totem":     dict(threat=0, flags={"static", "avoid"}),
    "sentinel":         dict(threat=6, flags={"ranged", "static"}),
    "horror":           dict(threat=16),
}


def monster_info(name: str) -> dict:
    """Look up a monster by sidebar name (longest keyword match)."""
    n = name.lower()
    best = None
    for key, info in MONSTERS.items():
        if key in n and (best is None or len(key) > len(best[0])):
            best = (key, info)
    if best:
        return {"name": best[0], "threat": best[1]["threat"],
                "flags": set(best[1].get("flags", set()))}
    return {"name": n, "threat": 6, "flags": set()}  # unknown: assume mid danger


# ---------------------------------------------------------------- equipment

# name -> (tier score, strength requirement). Tier is a rough "how good".
ARMORS = {
    "leather armor": (3, 10),
    "scale mail":    (4, 11),
    "chain mail":    (5, 13),
    "banded mail":   (7, 15),
    "splint mail":   (9, 17),
    "plate armor":   (11, 19),
}

WEAPONS = {
    "dagger":     (2, 12),
    "whip":       (3, 14),
    "rapier":     (5, 15),
    "sword":      (7, 14),
    "mace":       (9, 16),
    "axe":        (8, 15),
    "spear":      (6, 13),
    "broadsword": (13, 19),
    "war axe":    (12, 19),
    "hammer":     (14, 20),
    "war pike":   (11, 18),
}

# potions/scrolls we recognize once identified
GOOD_POTIONS = {"healing", "extra healing", "strength", "life", "telepathy",
                "levitation", "detect magic", "haste self", "fire immunity",
                "speed", "descent", "invisibility"}
BAD_POTIONS = {"incineration", "darkness", "hallucination", "creeping death",
               "caustic gas", "confusion", "paralysis"}

EMERGENCY_POTIONS = ("extra healing", "healing")  # quaff order at low HP

# scrolls that are clearly fine to read while standing safely
GOOD_SCROLLS = {"enchanting", "identify", "remove curse", "protect armor",
                "protect weapon", "recharging", "magic mapping", "sanctity",
                "negation", "shattering", "teleportation", "discord",
                "cause fear"}
BAD_SCROLLS = {"aggravate monsters", "summon monster"}
