"""Frame -> numpy tensors for the policy network.

Operates on the raw frame bytes (broguebot.ipc.Frame.raw), vectorized with
numpy so batching whole trajectories is cheap. No torch dependency: the
same code featurizes during data generation, training and play.

Outputs per frame:
  glyphs   int16  (ROWS, COLS)        display glyph id per cell
  colors   float32 (6, ROWS, COLS)    fg+bg RGB scaled to [0,1]
  stats    float32 (STATS_DIM,)       scalars + status one-hots
  items    int16   (26, ITEM_FEATS)   per-slot categorical features
  monsters float32 (24, MON_FEATS)    glyph id + relative pos + hp + state
  msg_hash int64   ()                 bucket of the latest message text
"""

import hashlib

import numpy as np

from ..ipc import (BAR_CELLS, COLS, ITEM, ITEM_OFF, MAP_OFF, MAX_ITEMS,
                   MAX_MONSTERS, MON_OFF, MONSTER, MSG_LEN, MSG_OFF,
                   NUM_MESSAGES, ROWS, STATS, STATS_OFF, STATUSES)

GLYPH_VOCAB = 512        # display glyphs are <128 ascii + enums up to ~340
MSG_BUCKETS = 4096
NAME_BUCKETS = 2048      # item display-name hash (distinguishes flavors)

N_STATUS = len(STATUSES)
# depth, hp, nutrition, strength, armor, gold(log), stealth, turns(log),
# px, py, equipped-weapon?, equipped-armor?
STATS_DIM = 12 + N_STATUS
ITEM_FEATS = 6           # category, kind+1, name_bucket, equipped, enchant_known, qty
MON_FEATS = 6            # glyph, dx, dy, hp_frac, state, captive

_cell_dtype = np.dtype([("glyph", "<u2"), ("fg", "u1", 3), ("bg", "u1", 3)])
_mon_dtype = np.dtype([("glyph", "<u2"), ("x", "u1"), ("y", "u1"),
                       ("hp_q", "u1"), ("state", "u1"), ("flags", "u1"),
                       ("pad", "u1")])


def _name_bucket(name: bytes) -> int:
    name = name.split(b"\0")[0]
    if not name:
        return 0
    return 1 + int.from_bytes(hashlib.blake2b(name, digest_size=4).digest(),
                              "little") % (NAME_BUCKETS - 1)


def featurize(raw: bytes) -> dict:
    out = {}

    cells = np.frombuffer(raw, dtype=_cell_dtype,
                          count=COLS * ROWS, offset=MAP_OFF)
    cells = cells.reshape(COLS, ROWS)            # column-major in the frame
    glyphs = cells["glyph"].T.astype(np.int16)   # -> (ROWS, COLS)
    np.clip(glyphs, 0, GLYPH_VOCAB - 1, out=glyphs)
    out["glyphs"] = glyphs
    fg = cells["fg"].transpose(1, 0, 2).astype(np.float32) / 100.0
    bg = cells["bg"].transpose(1, 0, 2).astype(np.float32) / 100.0
    out["colors"] = np.concatenate([fg, bg], axis=2).transpose(2, 0, 1)

    (depth, deepest, hp_q, nut_q, strength, armor, gold, pturns, aturns,
     status_mask, px, py, ended, wl, al, stealth, _pad) = \
        STATS.unpack_from(raw, STATS_OFF)
    stats = np.zeros(STATS_DIM, dtype=np.float32)
    stats[0] = depth / 26.0
    stats[1] = hp_q / BAR_CELLS
    stats[2] = nut_q / BAR_CELLS
    stats[3] = strength / 30.0
    stats[4] = armor / 30.0
    stats[5] = np.log1p(gold) / 10.0
    stats[6] = min(stealth, 40) / 40.0
    stats[7] = np.log1p(pturns) / 12.0
    stats[8] = px / COLS
    stats[9] = py / ROWS
    stats[10] = 1.0 if wl else 0.0
    stats[11] = 1.0 if al else 0.0
    for i in range(N_STATUS):
        if status_mask & (1 << i):
            stats[12 + i] = 1.0
    out["stats"] = stats

    items = np.zeros((MAX_ITEMS, ITEM_FEATS), dtype=np.int16)
    n_items = raw[ITEM_OFF]
    for i in range(min(n_items, MAX_ITEMS)):
        letter, cat_bit, kind, flags, ench, qty, sreq, name = \
            ITEM.unpack_from(raw, ITEM_OFF + 1 + i * ITEM.size)
        slot = letter - ord("a")
        if not 0 <= slot < MAX_ITEMS:
            continue
        items[slot, 0] = cat_bit + 1                  # 0 = empty slot
        items[slot, 1] = (kind + 1) if kind != 255 else 0
        items[slot, 2] = _name_bucket(name)
        items[slot, 3] = flags & 1                    # equipped
        items[slot, 4] = 0 if ench == -128 else min(max(ench + 16, 0), 32)
        items[slot, 5] = min(qty, 99)
    out["items"] = items

    monsters = np.zeros((MAX_MONSTERS, MON_FEATS), dtype=np.float32)
    n_mon = raw[MON_OFF]
    if n_mon:
        recs = np.frombuffer(raw, dtype=_mon_dtype,
                             count=min(n_mon, MAX_MONSTERS),
                             offset=MON_OFF + 1)
        k = len(recs)
        monsters[:k, 0] = np.clip(recs["glyph"], 0, GLYPH_VOCAB - 1)
        monsters[:k, 1] = (recs["x"].astype(np.float32) - px) / 40.0
        monsters[:k, 2] = (recs["y"].astype(np.float32) - py) / 20.0
        monsters[:k, 3] = recs["hp_q"].astype(np.float32) / BAR_CELLS
        monsters[:k, 4] = recs["state"]
        monsters[:k, 5] = recs["flags"] & 1
    out["monsters"] = monsters

    # newest message only; the encoder embeds its hash bucket
    last = b""
    for i in range(NUM_MESSAGES):
        off = MSG_OFF + i * (1 + MSG_LEN)
        if raw[off]:
            last = raw[off + 1:off + 1 + raw[off]]
    out["msg_hash"] = np.int64(
        0 if not last else
        1 + int.from_bytes(hashlib.blake2b(last, digest_size=4).digest(),
                           "little") % (MSG_BUCKETS - 1))
    return out


def batch(features: list[dict]) -> dict:
    """Stack per-frame feature dicts into batched arrays."""
    return {k: np.stack([f[k] for f in features]) for k in features[0]}
