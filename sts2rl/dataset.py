"""Training data: recorded run states, their outcomes, and TD(lambda) targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import features as F
from . import sim
from .agent import RunRecord


def records_to_arrays(records: list[RunRecord]) -> dict[str, np.ndarray] | None:
    states, run_id, step, score, won, floor, v_pred = [], [], [], [], [], [], []
    for r_i, rec in enumerate(records):
        if not rec.states:
            continue
        n = len(rec.states)
        states.extend(rec.states)
        run_id.extend([r_i] * n)
        step.extend(range(n))
        score.extend([rec.score] * n)
        won.extend([float(rec.won)] * n)
        floor.extend([min(1.0, rec.final_floor / sim.TOTAL_FLOORS)] * n)
        v_pred.extend(rec.v_pred)
    if not states:
        return None
    out = F.Encoded.stack(states).to_dict("x_")
    # Ids fit in int16 (and flags in int8); the network widens them per batch. This keeps
    # a replay window of hundreds of thousands of states in a few hundred MB.
    for key, dtype in (("x_deck", np.int16), ("x_deck_upg", np.int8), ("x_relics", np.int16),
                       ("x_potions", np.int16), ("x_encounter", np.int16), ("x_boss", np.int16)):
        out[key] = out[key].astype(dtype)
    out.update(
        run_id=np.array(run_id, np.int64),
        step=np.array(step, np.int64),
        score=np.array(score, np.float32),
        won=np.array(won, np.float32),
        floor=np.array(floor, np.float32),
        v_pred=np.array(v_pred, np.float32),
    )
    return out


def save_shard(path: Path, records: list[RunRecord]) -> int:
    arrays = records_to_arrays(records)
    if arrays is None:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return len(arrays["score"])


class Replay:
    """The most recent shards, concatenated, with run ids made globally unique."""

    def __init__(self, shard_paths: list[Path]):
        parts = []
        offset = 0
        for p in shard_paths:
            with np.load(p) as z:
                d = {k: z[k] for k in z.files}
            d["run_id"] = d["run_id"] + offset
            offset = int(d["run_id"].max()) + 1
            parts.append(d)
        keys = parts[0].keys() if parts else []
        self.data = {k: np.concatenate([p[k] for p in parts]) for k in keys}

    def __len__(self) -> int:
        return len(self.data.get("score", []))

    @property
    def x(self) -> F.Encoded:
        return F.Encoded.from_dict(self.data, "x_")

    def td_targets(self, values: np.ndarray, lam: float) -> np.ndarray:
        """TD(lambda) returns, computed backwards through each run.

        `values` are the current network's predictions for every stored state. The return
        of a run's last recorded state is its final score; earlier states blend the next
        state's value with the next state's return.
        """
        run_id, score = self.data["run_id"], self.data["score"]
        g = np.empty_like(score)
        n = len(score)
        nxt_return = 0.0
        for i in range(n - 1, -1, -1):
            last = i == n - 1 or run_id[i + 1] != run_id[i]
            if last:
                g[i] = score[i]
            else:
                g[i] = (1 - lam) * values[i + 1] + lam * nxt_return
            nxt_return = g[i]
        return g


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
