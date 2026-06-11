"""Transformer policy for Brogue.

Per frame: a conv stem turns the glyph/color grid into spatial tokens;
stats, inventory slots, visible monsters and the last message join as
extra tokens; a transformer encoder mixes them and a CLS token reads out
the frame summary. A GRU carries memory across timesteps (item identities,
explored layout, recent events) — recurrent state is explicit, so the
same module serves BC (teacher-forced over sequences) and RL (carried
hidden state). Swappable for a Transformer-XL block on bigger hardware.

Heads: action logits over env.ACTIONS, value estimate, plus an auxiliary
next-depth/hp prediction to densify gradients.
"""

import torch
import torch.nn as nn

from ..env import NUM_ACTIONS
from .featurize import (GLYPH_VOCAB, ITEM_FEATS, MON_FEATS, MSG_BUCKETS,
                        NAME_BUCKETS, STATS_DIM)


class Config:
    def __init__(self, d_model=128, n_layers=4, n_heads=4, glyph_dim=32,
                 gru_dim=256, dropout=0.0):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.glyph_dim = glyph_dim
        self.gru_dim = gru_dim
        self.dropout = dropout

    @classmethod
    def small(cls):       # laptop CPU: ~1.5M params
        return cls()

    @classmethod
    def base(cls):        # RTX 5070: ~15M params
        return cls(d_model=384, n_layers=8, n_heads=8, glyph_dim=48,
                   gru_dim=768)


class BroguePolicy(nn.Module):
    ROWS, COLS = 34, 100

    def __init__(self, cfg: Config | None = None):
        super().__init__()
        self.cfg = cfg = cfg or Config.small()
        d = cfg.d_model

        self.glyph_emb = nn.Embedding(GLYPH_VOCAB, cfg.glyph_dim)
        in_ch = cfg.glyph_dim + 6  # + fg/bg rgb
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, d, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1),
        )
        self.h_tokens = (self.ROWS + 3) // 4   # 9
        self.w_tokens = (self.COLS + 3) // 4   # 25
        n_spatial = self.h_tokens * self.w_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, n_spatial, d))

        self.stats_proj = nn.Linear(STATS_DIM, d)
        self.item_cat = nn.Embedding(16, d // 4)
        self.item_kind = nn.Embedding(64, d // 4)
        self.item_name = nn.Embedding(NAME_BUCKETS, d // 4)
        self.item_scalar = nn.Linear(3, d // 4)
        self.slot_emb = nn.Parameter(torch.zeros(1, 26, d))
        self.mon_glyph = nn.Embedding(GLYPH_VOCAB, d // 2)
        self.mon_scalar = nn.Linear(MON_FEATS - 1, d - d // 2)
        self.msg_emb = nn.Embedding(MSG_BUCKETS, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=4 * d, dropout=cfg.dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)

        self.memory = nn.GRUCell(d, cfg.gru_dim)
        self.post = nn.Sequential(nn.Linear(cfg.gru_dim + d, d), nn.GELU())
        self.policy_head = nn.Linear(d, NUM_ACTIONS)
        self.value_head = nn.Linear(d, 1)
        self.aux_head = nn.Linear(d, 2)  # predict Δdepth>0, Δhp next step

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.gru_dim, device=device)

    def encode_frame(self, obs: dict) -> torch.Tensor:
        """obs: featurize.batch() arrays as tensors. Returns (B, d)."""
        B = obs["glyphs"].shape[0]
        g = self.glyph_emb(obs["glyphs"].long())        # B,R,C,gd
        img = torch.cat([g.permute(0, 3, 1, 2),
                         obs["colors"].float()], dim=1)
        sp = self.stem(img)                             # B,d,9,25
        sp = sp.flatten(2).transpose(1, 2) + self.pos_emb

        stats = self.stats_proj(obs["stats"].float()).unsqueeze(1)

        it = obs["items"].long()
        item_tok = torch.cat([
            self.item_cat(it[..., 0].clamp(0, 15)),
            self.item_kind(it[..., 1].clamp(0, 63)),
            self.item_name(it[..., 2]),
            self.item_scalar(it[..., 3:6].float()),
        ], dim=-1) + self.slot_emb

        mon = obs["monsters"].float()
        mon_tok = torch.cat([
            self.mon_glyph(mon[..., 0].long().clamp(0, GLYPH_VOCAB - 1)),
            self.mon_scalar(mon[..., 1:]),
        ], dim=-1)

        msg = self.msg_emb(obs["msg_hash"].long()).unsqueeze(1)
        cls = self.cls.expand(B, -1, -1)

        x = torch.cat([cls, stats, msg, item_tok, mon_tok, sp], dim=1)
        x = self.encoder(x)
        return x[:, 0]                                  # CLS readout

    def forward(self, obs: dict, hidden: torch.Tensor):
        """One timestep. Returns (logits, value, aux, new_hidden)."""
        z = self.encode_frame(obs)
        hidden = self.memory(z, hidden)
        h = self.post(torch.cat([hidden, z], dim=-1))
        return (self.policy_head(h), self.value_head(h).squeeze(-1),
                self.aux_head(h), hidden)

    def unroll(self, obs_seq: dict, hidden: torch.Tensor):
        """Teacher-forced unroll for BC/RL training.

        obs_seq arrays are (B, T, ...); returns logits (B, T, A), values
        (B, T), final hidden.
        """
        B, T = obs_seq["glyphs"].shape[:2]
        flat = {k: v.reshape(B * T, *v.shape[2:]) for k, v in obs_seq.items()}
        z = self.encode_frame(flat).reshape(B, T, -1)
        logits, values = [], []
        for t in range(T):
            hidden = self.memory(z[:, t], hidden)
            h = self.post(torch.cat([hidden, z[:, t]], dim=-1))
            logits.append(self.policy_head(h))
            values.append(self.value_head(h).squeeze(-1))
        return torch.stack(logits, 1), torch.stack(values, 1), hidden


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
