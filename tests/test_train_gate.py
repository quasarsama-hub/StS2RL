"""The training loop's champion bookkeeping, with self-play, training and evaluation faked.

A fake checkpoint is a JSON file holding a quality q; a run's validation score on each
seed is a fixed per-seed base plus q. So which checkpoint is better is known exactly, and
each test checks only what the loop does with that knowledge: who the champion is, when
it changes, and when cached results may be reused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import sts2rl.train as T
from sts2rl import paths
from sts2rl.agent import RunRecord
from sts2rl.evaluate import summarize


class World:
    def __init__(self):
        self.schedule: list[float] = []  # quality latest.pt gets after training iteration i
        self.errors: dict[str, set] = {}  # model name ("heuristic", "best.pt", ...) -> seeds that error
        self.selfplay: list[str | None] = []
        self.evals: list[str] = []


def quality(path) -> float:
    return 0.0 if path is None else json.loads(Path(path).read_text())["q"]


def records(world: World, path, seeds, errors=()):
    q = quality(path)
    out = []
    for i, s in enumerate(seeds):
        r = RunRecord(seed=s)
        r.score = 0.1 + 0.15 * ((i * 37) % 64) / 64 + q
        r.final_floor = int(10 + 30 * r.score)
        r.acts_cleared = int(r.score > 0.2)
        if s in errors:
            r.death_encounter = "ERROR: RuntimeError: no legal action"
        out.append(r)
    return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World()

    class Pool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def play(self, seeds, cfg, model_path, base_seed=0):
            w.selfplay.append(Path(model_path).name if model_path else None)
            return records(w, model_path, seeds)

    def run_eval(model_path, n_runs, cfg, workers, offset=0, pool=None):
        seeds = [f"EVAL{i:06d}" for i in range(offset, offset + n_runs)]
        name = Path(model_path).name if model_path else "heuristic"
        w.evals.append(f"{name}:q={quality(model_path)}")
        recs = records(w, model_path, seeds, w.errors.get(name, set()))
        return summarize(recs, cfg.max_act), recs

    class Net:
        def __init__(self, hidden=256, q=0.0, meta=None):
            self.q, self.meta = q, meta or {}

        @classmethod
        def load(cls, path):
            blob = json.loads(Path(path).read_text())
            return cls(q=blob["q"], meta=blob.get("meta", {}))

        def save(self, path, **meta):
            Path(path).write_text(json.dumps({"q": self.q, "meta": {"iteration": meta.get("iteration")}}))

    def train_value(model, replay, *, epochs, batch, lr, lam, seed):
        model.q = w.schedule[seed] if seed < len(w.schedule) else model.q
        return {"states": 1, "val_loss": 0.5, "val_baseline": 0.6}

    def save_shard(path, recs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        return len(recs)

    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(T, "Pool", Pool)
    monkeypatch.setattr(T, "run_eval", run_eval)
    monkeypatch.setattr(T, "ValueNet", Net)
    monkeypatch.setattr(T, "train_value", train_value)
    monkeypatch.setattr(T, "save_shard", save_shard)
    monkeypatch.setattr(T, "Replay", lambda window: None)
    w.dir = tmp_path / "x"
    return w


def train(*extra):
    sys.argv = ["train", "--name", "x", "--runs-per-iter", "2", *extra]
    T.main()


def champion(world) -> dict:
    return json.loads((world.dir / "champion.json").read_text())


def test_better_candidates_are_promoted_and_worse_ones_are_not(world):
    world.schedule = [0.1, 0.05, 0.2]
    train("--iterations", "3")
    assert champion(world)["iteration"] == 2 and quality(world.dir / "best.pt") == 0.2


def test_a_champion_engine_error_does_not_block_every_later_promotion(world):
    world.errors = {"heuristic": {"EVAL000005"}}
    world.schedule = [0.1, 0.2]
    train("--iterations", "2")
    assert champion(world)["iteration"] == 1


def test_continuing_from_a_supplied_checkpoint_keeps_it_champion_over_a_worse_candidate(world):
    world.dir.mkdir(parents=True)
    (world.dir / "latest.pt").write_text(json.dumps({"q": 0.3, "meta": {"iteration": 8}}))
    world.schedule = [0.2]  # the first fine-tune is worse than where it started
    train("--iterations", "1")
    assert champion(world)["model"] == "best.pt" and champion(world)["iteration"] == 8
    assert quality(world.dir / "best.pt") == 0.3


def test_champion_self_play_uses_the_heuristic_while_the_heuristic_is_champion(world):
    world.schedule = [0.0, 0.0]  # never better: no promotion
    train("--iterations", "2", "--selfplay-model", "champion")
    assert world.selfplay == [None, None]


def test_cached_champion_results_are_reused_only_for_the_same_code(world, monkeypatch):
    world.schedule = [0.0]
    train("--iterations", "1")
    train("--iterations", "1")
    assert world.evals.count("heuristic:q=0.0") == 1  # resumed: cache reused
    monkeypatch.setattr(T, "code_fingerprint", lambda: "a different build")
    train("--iterations", "1")
    assert world.evals.count("heuristic:q=0.0") == 2  # re-evaluated


def test_a_replaced_best_pt_is_re_evaluated(world):
    world.schedule = [0.1]
    train("--iterations", "1")
    (world.dir / "best.pt").write_text(json.dumps({"q": 0.5, "meta": {"iteration": 0}}))
    train("--iterations", "0")
    assert "best.pt:q=0.5" in world.evals


def test_an_interrupted_promotion_test_is_finished_on_resume(world):
    world.schedule = [0.2]
    train("--iterations", "1")
    # Drop iteration 0's metrics row and champion, as if it was killed during evaluation.
    lines = (world.dir / "metrics.jsonl").read_text().splitlines()
    (world.dir / "metrics.jsonl").write_text("\n".join(l for l in lines if json.loads(l)["iter"] != 0) + "\n")
    (world.dir / "champion.json").unlink()
    (world.dir / "best.pt").unlink()
    train("--iterations", "0")
    rows = [json.loads(l) for l in (world.dir / "metrics.jsonl").read_text().splitlines()]
    assert any(r.get("resumed_gate") and r["gate"]["promoted"] for r in rows)
    assert champion(world)["iteration"] == 0


def test_training_goes_back_to_the_champion_after_iterations_without_a_promotion(world):
    world.dir.mkdir(parents=True)
    (world.dir / "latest.pt").write_text(json.dumps({"q": 0.3, "meta": {"iteration": 8}}))
    world.schedule = [0.2, 0.15, 0.1]  # drifting below the champion
    train("--iterations", "3")
    assert quality(world.dir / "latest.pt") == 0.3
    rows = [json.loads(l) for l in (world.dir / "metrics.jsonl").read_text().splitlines()]
    assert [r.get("reverted_to_champion") for r in rows if r["iter"] >= 0] == [None, None, 8]
    assert world.selfplay == ["best.pt"] * 3  # the champion plays self-play by default


def test_revert_after_zero_keeps_training_the_drifting_network(world):
    world.dir.mkdir(parents=True)
    (world.dir / "latest.pt").write_text(json.dumps({"q": 0.3, "meta": {"iteration": 8}}))
    world.schedule = [0.2, 0.15, 0.1]
    train("--iterations", "3", "--revert-after", "0")
    assert quality(world.dir / "latest.pt") == 0.1
