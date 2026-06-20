"""Random Network Distillation (RND) intrinsic-motivation reward.

A single, game-agnostic "interestingness" signal: a fixed RANDOM target network
maps each on-screen observation to a k-dim embedding; a PREDICTOR network is
trained online to match it. The reward is the predictor's error -> high on novel
observations (predictor hasn't learned them), ~0 on familiar ones ("boring").
Information-theoretically the error approximates surprise / bits of new info.

We feed the STABLE game-state observation: the glyph grid + the (already
non-animated, see --no-effects/trueColorMode) RGB, so color encodes gas/terrain/
monsters without flicker. Depth is appended so a new dungeon level — maximally
novel — is salient. No game-specific reward is encoded; descent, exploration,
combat, item use all surface as novelty on their own.

Burda et al. 2018, "Exploration by Random Network Distillation". This is the
lifelong-novelty term; NGU/Agent57's episodic k-NN novelty is a future upgrade.
"""

import torch
import torch.nn as nn

from ..ipc import COLS, ROWS
from .featurize import GLYPH_VOCAB


class _RunningMeanStd:
    """Welford running mean/variance for reward (and obs) normalization."""

    def __init__(self, shape=(), device="cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x):
        x = x.reshape(-1, *self.mean.shape) if self.mean.shape else x.reshape(-1)
        bmean = x.mean(0)
        bvar = x.var(0, unbiased=False)
        bcount = x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean = self.mean + delta * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        self.var = (m_a + m_b + delta.pow(2) * self.count * bcount / tot) / tot
        self.count = tot

    @property
    def std(self):
        return self.var.clamp_min(1e-8).sqrt()


class _RNDNet(nn.Module):
    """Conv stem over (glyph-embedding + 6 RGB channels), depth appended at the
    MLP, -> k-dim embedding. Target and predictor share this architecture."""

    def __init__(self, d_emb: int = 16, k: int = 128):
        super().__init__()
        self.emb = nn.Embedding(GLYPH_VOCAB, d_emb)
        cin = d_emb + 6
        self.conv = nn.Sequential(
            nn.Conv2d(cin, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, cin, ROWS, COLS)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat + 1, 256), nn.ReLU(), nn.Linear(256, k))

    def forward(self, glyphs, colors, depth):
        # glyphs (N,ROWS,COLS) long; colors (N,6,ROWS,COLS); depth (N,1)
        e = self.emb(glyphs).permute(0, 3, 1, 2)
        x = self.conv(torch.cat([e, colors], dim=1)).flatten(1)
        return self.head[1:](torch.cat([x, depth], dim=1))


class RND:
    """Owns the target/predictor nets, predictor optimizer, and reward
    normalization. reward() returns a normalized novelty scalar per observation;
    distill() trains the predictor on the rollout's observations."""

    def __init__(self, device, k: int = 128, lr: float = 1e-4):
        self.device = device
        self.target = _RNDNet(k=k).to(device).eval()
        self.predictor = _RNDNet(k=k).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.ret_rms = _RunningMeanStd(device=device)

    def _split(self, obs):
        glyphs = obs["glyphs"].long()
        colors = obs["colors"].float()
        depth = obs["stats"][..., :1].float()  # stats[0] = depth/26
        return glyphs, colors, depth

    @torch.no_grad()
    def reward_raw(self, obs):
        """Raw per-observation novelty = mean-squared predictor error (un-scaled)."""
        g, c, d = self._split(obs)
        return (self.predictor(g, c, d) - self.target(g, c, d)).pow(2).mean(1)

    @torch.no_grad()
    def normalize(self, rews, gamma):
        """Scale raw novelty by the running std of its DISCOUNTED RETURNS (the
        RND "reward forward filter"), so intrinsic returns stay ~O(1) and don't
        blow up the value head. Not mean-centred — reward stays non-negative."""
        run = torch.zeros(rews.shape[0], device=rews.device)
        ret = torch.empty_like(rews)
        for t in range(rews.shape[1]):
            run = run * gamma + rews[:, t]
            ret[:, t] = run
        self.ret_rms.update(ret.reshape(-1))
        return rews / self.ret_rms.std

    def distill(self, obs):
        """One predictor-update step toward the (frozen) target on these obs."""
        g, c, d = self._split(obs)
        with torch.no_grad():
            t = self.target(g, c, d)
        loss = (self.predictor(g, c, d) - t).pow(2).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return loss.item()

    def state_dict(self):
        return {"target": self.target.state_dict(),
                "predictor": self.predictor.state_dict(),
                "opt": self.opt.state_dict()}

    def load_state_dict(self, sd):
        self.target.load_state_dict(sd["target"])
        self.predictor.load_state_dict(sd["predictor"])
        self.opt.load_state_dict(sd["opt"])
