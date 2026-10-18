"""The run value network: P(run outcome | state), shared by every decision the agent makes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import features as F
from . import gamedata as gd


def schema_hash() -> str:
    """Changes whenever the encoding would make an old checkpoint read garbage."""
    parts = [",".join(F.SCALAR_NAMES), str(len(gd.card_vocab())), str(len(gd.relic_vocab())),
             str(len(gd.potion_vocab())), str(len(gd.encounter_vocab())), str(F.MAX_DECK), str(F.MAX_RELICS)]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


class ValueNet(nn.Module):
    """Deep-sets encoder over deck/relics/potions + MLP trunk, three heads.

    score  E[final run score] in [0, 1] -- the quantity decisions maximise
    win    P(win the run)
    floor  E[final floor / total floors]
    """

    def __init__(self, d: int = 64, hidden: int = 256):
        super().__init__()
        self.cfg = {"d": d, "hidden": hidden}
        n_cards, n_relics = len(gd.card_vocab()), len(gd.relic_vocab())
        n_potions, n_enc = len(gd.potion_vocab()), len(gd.encounter_vocab())
        self.card = nn.Embedding(n_cards, d, padding_idx=0)
        self.upgrade = nn.Embedding(2, d)
        self.card_phi = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.relic = nn.Embedding(n_relics, d, padding_idx=0)
        self.potion = nn.Embedding(n_potions, 16, padding_idx=0)
        self.encounter = nn.Embedding(n_enc, 32)
        self.boss = nn.Embedding(n_enc, 32)
        trunk_in = 2 * d + d + 16 + 32 + 32 + F.N_SCALARS
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.heads = nn.Linear(hidden // 2, 3)
        with torch.no_grad():
            self.card.weight.mul_(0.3)
            self.relic.weight.mul_(0.3)
            # Zero-initialised heads: an untrained network reproduces the hand-written
            # prior below exactly, so learning starts from a sensible agent instead of a
            # random one and only has to learn corrections.
            self.heads.weight.zero_()
            self.heads.bias.zero_()
        self._prior_idx = [F.SCALAR_NAMES.index(n) for n in ("floor", "hp_frac", "potions_held")]

    def forward(self, deck, deck_upg, relics, potions, encounter, boss, scalars):
        mask = (deck > 0).unsqueeze(-1).float()
        e = self.card(deck) + self.upgrade(deck_upg) * mask
        e = self.card_phi(e) * mask
        count = mask.sum(1).clamp(min=1.0)
        deck_sum = e.sum(1) / 20.0
        deck_mean = e.sum(1) / count
        relic_sum = self.relic(relics).sum(1) / 5.0
        potion_sum = self.potion(potions).sum(1)
        x = torch.cat([deck_sum, deck_mean, relic_sum, potion_sum, self.encounter(encounter), self.boss(boss), scalars], dim=1)
        out = self.heads(self.trunk(x))
        return out + self.prior_logits(scalars)  # logits: score, win, floor

    def prior_logits(self, scalars):
        """The heuristic the agent starts from: progress so far, plus HP, plus potions."""
        floor, hp_frac, potions = (scalars[:, i] for i in self._prior_idx)
        score = (0.5 * floor + 0.15 * hp_frac + 0.03 * potions).clamp(0.01, 0.99)
        floor_frac = (floor + 0.1 * hp_frac).clamp(0.01, 0.99)
        win = torch.full_like(score, 0.01)
        return torch.logit(torch.stack([score, win, floor_frac], dim=1))

    # -- numpy convenience -----------------------------------------------------------

    @staticmethod
    def tensors(batch: F.Encoded, device="cpu"):
        return (
            torch.as_tensor(batch.deck, device=device),
            torch.as_tensor(batch.deck_upg, device=device),
            torch.as_tensor(batch.relics, device=device),
            torch.as_tensor(batch.potions, device=device),
            torch.as_tensor(batch.encounter, device=device),
            torch.as_tensor(batch.boss, device=device),
            torch.as_tensor(batch.scalars, device=device),
        )

    @torch.no_grad()
    def predict(self, batch: F.Encoded) -> np.ndarray:
        """Expected final score for each state in the batch."""
        self.eval()
        out = self(*self.tensors(batch))
        return torch.sigmoid(out[:, 0]).numpy()

    @torch.no_grad()
    def predict_all(self, batch: F.Encoded) -> np.ndarray:
        self.eval()
        return torch.sigmoid(self(*self.tensors(batch))).numpy()

    # -- persistence -------------------------------------------------------------------

    def save(self, path: str | Path, **meta) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"state": self.state_dict(), "cfg": self.cfg, "schema": schema_hash(), "meta": meta}, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> ValueNet:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("schema") != schema_hash():
            raise ValueError(
                f"{path} was trained on a different state encoding ({blob.get('schema')} vs {schema_hash()}); retrain it."
            )
        net = cls(**blob["cfg"])
        net.load_state_dict(blob["state"])
        net.meta = blob.get("meta", {})
        return net
