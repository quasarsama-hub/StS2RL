"""Check curriculum lineage, seed separation and attribution without a game."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import (
    ExperimentConfig, atomic_json, commit_checkpoint, read_json, sha256, verified_checkpoint,
)
from spire2_rl.study import SEEDS_PER_SPLIT, evaluation_seeds, seed_for, summarize


@pytest.fixture
def workflow():
    script = Path(__file__).resolve().parents[1] / "scripts" / "learn_act1.py"
    spec = importlib.util.spec_from_file_location("act1_workflow_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SchemaOnlyEnv(gym.Env):
    """Constructible public schema; mocked collection never plays this game."""

    observation_space = gym.spaces.Dict({"visible": gym.spaces.Box(0, 1, (1,), dtype=np.float32)})
    action_space = gym.spaces.Discrete(2)
    scope = "fullrun"

    def __init__(self, config):
        self.backend_metadata = {"schema": "public-fixture", "max_acts": config.max_acts}
        self.raw_env = SimpleNamespace(_mgr=None)

    def reset(self, *, seed=None, options=None):
        return {"visible": np.zeros(1, dtype=np.float32)}, {}

    def action_masks(self):
        return np.ones(2, dtype=bool)


class StoredPolicy:
    """Small saved tensors and optimizer state for orchestration, not PPO math."""

    def __init__(self, num_timesteps=0):
        self.num_timesteps = num_timesteps
        self.policy = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.policy.weight.fill_(1)
            self.policy.bias.zero_()
        self.policy.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

    def save(self, path):
        torch.save({"weights": self.policy.state_dict(), "optimizer": self.policy.optimizer.state_dict(),
                    "num_timesteps": self.num_timesteps}, path)

    @classmethod
    def load(cls, path, **kwargs):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(saved["num_timesteps"])
        model.policy.load_state_dict(saved["weights"])
        model.policy.optimizer.load_state_dict(saved["optimizer"])
        return model


def bytes_in(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def episode_rows(seeds, label, *, floor=8, outcome="death"):
    return [{"seed": seed, "policy": label, "floor": floor, "hp": 0, "outcome": outcome}
            for seed in seeds]


@pytest.fixture
def source_setup(workflow, monkeypatch, tmp_path):
    source, upstream = tmp_path / "source", tmp_path / "upstream"
    config = ExperimentConfig(n_envs=2, n_steps=4, batch_size=4, n_epochs=1)
    def factory(config, upstream):
        return SchemaOnlyEnv(config)
    monkeypatch.setattr(runner, "_make", factory)
    monkeypatch.setattr(runner, "provenance", lambda upstream: {"core_source": "unchanged"})
    with workflow.selection["finite_selection_runner"]():
        manifest = runner.initialize(source, config, upstream)
    model = StoredPolicy(160768)
    model.policy(torch.ones(1, 1)).sum().backward()
    model.policy.optimizer.step()
    state = commit_checkpoint(model, source, [SimpleNamespace(next_index=2), SimpleNamespace(next_index=3)],
                              manifest["fingerprint"])
    atomic_json(source / "seed-cursors" / "worker-0.json", {"next_index": 10})
    atomic_json(source / "seed-cursors" / "worker-1.json", {"next_index": 13})
    atomic_json(source / "champion.json", {"old_claim": True})
    (source / "episodes.jsonl").write_text('{"outcome":"death","seed":1}\n', encoding="utf-8")
    monkeypatch.setattr(workflow, "MaskablePPO", StoredPolicy)
    def fresh_policy(source_model, config, vec):
        fresh = StoredPolicy()
        fresh.policy.load_state_dict(source_model.policy.state_dict(), strict=True)
        vec._seeds = [None] * config.n_envs
        return fresh
    monkeypatch.setitem(workflow.selection, "fresh_policy", fresh_policy)
    monkeypatch.setattr(workflow, "FairTeacher", lambda: SimpleNamespace(select=lambda env: 1))
    calls = {"collection": [], "cloning": []}
    def collect(env, teacher, seeds, output):
        assert env.backend_metadata["curriculum"] == "act1-v1"
        assert all(0 < seed <= SEEDS_PER_SPLIT for seed in seeds)
        calls["collection"].append(list(seeds))
        output.write_text(json.dumps({"split": "train", "seeds": seeds}), encoding="utf-8")
        rows = episode_rows(seeds, "teacher")
        rows[0].update(outcome="win", floor=15)
        return {"dataset": output.name, "sha256": sha256(output), "split": "train", "seeds": seeds,
                "gameplay": summarize(rows), "samples": len(seeds)}
    monkeypatch.setattr(workflow, "collect_demonstrations", collect)
    def clone(model, dataset, **kwargs):
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        data = json.loads(dataset.read_text())
        assert data["split"] == "train"
        calls["cloning"].append({"seeds": data["seeds"], **kwargs})
        with torch.no_grad():
            model.policy.weight.add_(2)
        return {"method": "supervised_imitation", "gradient_updates": kwargs["epochs"],
                "new_ppo_steps": 0, "dataset_sha256": sha256(dataset)}
    monkeypatch.setattr(workflow, "behavior_clone", clone)
    recipe = {**workflow.default_recipe(), "demonstration_episodes": 8, "bc_epochs": 2}
    return SimpleNamespace(source=source, upstream=upstream, config=config, manifest=manifest,
                           state=state, recipe=recipe, calls=calls, factory=factory)


def test_seed_plan_skips_durable_frontier_and_keeps_ppo_after_training_demos(workflow, tmp_path):
    config = ExperimentConfig(n_envs=2)
    source = tmp_path / "source"
    atomic_json(source / "seed-cursors" / "worker-0.json", {"next_index": 10})
    atomic_json(source / "seed-cursors" / "worker-1.json", {"next_index": 13})
    before = bytes_in(source)
    plan = workflow.seed_plan(source, {"next_indices": [2, 3]}, config, 5)
    assert plan["source_highwater_indices"] == [10, 13]
    assert plan["demonstration_index_start"] == 14 and plan["demonstration_index_end_exclusive"] == 19
    assert plan["demonstration_seeds"] == [seed_for(i, "train", 7) for i in range(14, 19)]
    assert plan["rl_next_indices"] == [20, 21]
    validation = evaluation_seeds("validation", 100, 7, start=400)
    assert set(plan["demonstration_seeds"]).isdisjoint(validation)
    assert bytes_in(source) == before


@pytest.mark.parametrize("bad_plan", ["wrong_rank_below_checkpoint", "negative", "exhausted", "too_few"])
def test_seed_plan_rejects_invalid_or_exhausted_accounting(workflow, tmp_path, bad_plan):
    source = tmp_path / "source"
    state, count = {"next_indices": [2, 3]}, 5
    if bad_plan == "exhausted":
        state = {"next_indices": [SEEDS_PER_SPLIT - 2, SEEDS_PER_SPLIT - 1]}
    elif bad_plan == "too_few":
        count = 4
    else:
        atomic_json(source / "seed-cursors" / "worker-0.json",
                    {"next_index": 1 if bad_plan == "wrong_rank_below_checkpoint" else -2})
    with pytest.raises(ValueError):
        workflow.seed_plan(source, state, ExperimentConfig(n_envs=2), count)


def test_creation_preserves_source_and_separates_imitation_from_ppo(workflow, source_setup, tmp_path):
    old = source_setup
    before = bytes_in(old.source)
    root = tmp_path / "act1"
    lineage = workflow.create_experiment(old.source, root, old.upstream, old.recipe)
    assert bytes_in(old.source) == before and runner._make is old.factory
    manifest = read_json(root / "experiment.json")
    assert manifest["fingerprint"] != old.manifest["fingerprint"]
    assert manifest["config"]["scope"] == "fullrun" and manifest["config"]["max_acts"] == 1
    assert manifest["config"]["learning_rate"] == old.recipe["ppo_learning_rate"]
    assert lineage["source_checkpoint"] == old.state and lineage["source_checkpoint"]["num_timesteps"] == 160768
    assert lineage["cloning"]["method"] == "supervised_imitation"
    assert lineage["cloning"]["new_ppo_steps"] == 0
    assert lineage["seed_plan"]["rl_next_indices"] == [22, 23]
    assert old.calls["collection"] == [lineage["seed_plan"]["demonstration_seeds"]]
    assert old.calls["cloning"][0]["seeds"] == old.calls["collection"][0]
    original = StoredPolicy.load(old.source / old.state["model"])
    imported = StoredPolicy.load(root / lineage["imported_checkpoint"]["model"])
    cloned = StoredPolicy.load(root / lineage["cloned_checkpoint"]["model"])
    assert original.policy.optimizer.state
    assert imported.num_timesteps == cloned.num_timesteps == 0
    assert not imported.policy.optimizer.state and not cloned.policy.optimizer.state
    torch.testing.assert_close(imported.policy.weight, original.policy.weight, rtol=0, atol=0)
    torch.testing.assert_close(cloned.policy.weight, original.policy.weight + 2, rtol=0, atol=0)
    assert lineage["imported_checkpoint"]["sha256"] != lineage["cloned_checkpoint"]["sha256"]
    assert verified_checkpoint(root)[1] == lineage["cloned_checkpoint"]
    assert not (root / "champion.json").exists() and not (root / "episodes.jsonl").exists()
    assert (root / lineage["demonstrations"]["dataset"]).exists()


def test_failed_cloning_leaves_no_partial_target_and_can_retry(workflow, source_setup, tmp_path, monkeypatch):
    old = source_setup
    before = bytes_in(old.source)
    root = tmp_path / "act1"
    clone = workflow.behavior_clone
    def fail(*args, **kwargs):
        raise RuntimeError("cloning failed")
    monkeypatch.setattr(workflow, "behavior_clone", fail)
    with pytest.raises(RuntimeError, match="cloning failed"):
        workflow.create_experiment(old.source, root, old.upstream, old.recipe)
    assert not root.exists() and not list(tmp_path.glob(".act1-initializing-*"))
    assert bytes_in(old.source) == before and runner._make is old.factory
    monkeypatch.setattr(workflow, "behavior_clone", clone)
    lineage = workflow.create_experiment(old.source, root, old.upstream, old.recipe)
    assert old.calls["collection"][0] == old.calls["collection"][1] == lineage["seed_plan"]["demonstration_seeds"]


def test_act1_context_requires_new_scope_and_restores_factory(workflow, source_setup, tmp_path):
    old = source_setup
    with pytest.raises(ValueError, match="max_acts=1"):
        with workflow.act1_runner(old.recipe):
            runner._make(old.config, old.upstream)
    assert runner._make is old.factory
    with workflow.selection["finite_selection_runner"]():
        assert runner.initialize(old.source, old.config, old.upstream) == old.manifest


def assessment(manifest, state, label, count=100, start=400, *, floor=8, outcome="death"):
    config = ExperimentConfig(**manifest["config"])
    rows = episode_rows(evaluation_seeds("validation", count, config.experiment_seed, start=start),
                        label, floor=floor, outcome=outcome)
    return {"stage": label, "fingerprint": manifest["fingerprint"], "checkpoint": state,
            "split": "validation", "seed_start": start, "episodes": rows,
            "gameplay": summarize(rows), "inference": "deterministic", "victory_scope": "act1"}


def test_assessment_uses_explicit_stage_checkpoint_and_never_training_seeds(workflow, source_setup,
                                                                         tmp_path, monkeypatch):
    old, root = source_setup, tmp_path / "act1"
    lineage = workflow.create_experiment(old.source, root, old.upstream, old.recipe)
    manifest = read_json(root / "experiment.json")
    weights, seeds_seen = [], []
    def rollout(env, seeds, *, model, label):
        weights.append(model.policy.weight.detach().clone())
        seeds_seen.append(list(seeds))
        return episode_rows(seeds, label)
    monkeypatch.setattr(runner, "_rollout", rollout)
    collection_count = len(old.calls["collection"])
    with workflow.act1_runner(old.recipe):
        for label, state in (("before_imitation", lineage["imported_checkpoint"]),
                             ("after_imitation_before_rl", lineage["cloned_checkpoint"])):
            report = workflow.assess(root, old.upstream, manifest, state, label=label,
                                     episodes=7, validation_start=400)
            workflow.validate_assessment(report, manifest, state, label, 7, 400)
            assert report["victory_scope"] == "act1"
    assert seeds_seen[0] == seeds_seen[1] == evaluation_seeds("validation", 7, 7, start=400)
    assert set(seeds_seen[0]).isdisjoint(old.calls["collection"][0])
    assert len(old.calls["collection"]) == collection_count
    torch.testing.assert_close(weights[1], weights[0] + 2, rtol=0, atol=0)


@pytest.mark.parametrize("tamper", ["scope", "stage", "seed", "summary", "checkpoint"])
def test_assessment_rejects_stale_stage_or_raw_evidence(workflow, tamper):
    config = ExperimentConfig(max_acts=1)
    manifest = {"config": asdict(config), "fingerprint": "act1"}
    state = {"model": "imported.zip", "sha256": "source", "fingerprint": "act1", "num_timesteps": 0}
    valid = assessment(manifest, state, "before_imitation", count=7)
    workflow.validate_assessment(valid, manifest, state, "before_imitation", 7, 400)
    changed = deepcopy(valid)
    if tamper == "scope":
        changed["victory_scope"] = "fullrun"
    elif tamper == "stage":
        changed["stage"] = "after_imitation_before_rl"
    elif tamper == "seed":
        changed["episodes"][0]["seed"] = seed_for(0, "train", 7)
    elif tamper == "summary":
        changed["gameplay"]["win_rate"] = 1
    else:
        changed["checkpoint"]["sha256"] = "another-policy"
    with pytest.raises(ValueError):
        workflow.validate_assessment(changed, manifest, state, "before_imitation", 7, 400)


def test_cache_recovers_unpinned_complete_report_and_rejects_later_tampering(workflow, source_setup,
                                                                         tmp_path, monkeypatch):
    old, root = source_setup, tmp_path / "act1"
    lineage = workflow.create_experiment(old.source, root, old.upstream, old.recipe)
    manifest = read_json(root / "experiment.json")
    state = lineage["imported_checkpoint"]
    target = root / "before_imitation.json"
    report = assessment(manifest, state, "before_imitation", count=7)
    atomic_json(target, report)
    def unexpected_assess(*args, **kwargs):
        pytest.fail("A valid completed report must recover without new games")
    monkeypatch.setattr(workflow, "assess", unexpected_assess)
    restored = workflow.cached_assessment(root, old.upstream, manifest, lineage,
                                          "before_imitation", state, 7, 400)
    assert restored == report
    assert lineage["before_imitation_sha256"] == sha256(target)
    changed = deepcopy(report)
    changed["gameplay"]["mean_floor"] = 999
    atomic_json(target, changed)
    with pytest.raises(ValueError, match="checksum"):
        workflow.cached_assessment(root, old.upstream, manifest, lineage, "before_imitation", state, 7, 400)


@pytest.fixture
def progress_setup(workflow, source_setup, tmp_path, monkeypatch):
    old, root = source_setup, tmp_path / "act1"
    lineage = workflow.create_experiment(old.source, root, old.upstream, old.recipe)
    manifest = read_json(root / "experiment.json")
    calls = {"rollouts": [], "training": []}
    def rollout(env, seeds, *, model, label):
        calls["rollouts"].append((label, list(seeds)))
        return episode_rows(seeds, label, floor=5 if label == "before_imitation" else 8,
                            outcome="win" if label == "teacher_act1" else "death")
    monkeypatch.setattr(runner, "_rollout", rollout)
    def train(root, config, upstream, *, steps):
        calls["training"].append(steps)
        path, latest = verified_checkpoint(root, manifest["fingerprint"])
        model = StoredPolicy.load(path)
        model.num_timesteps = 80
        commit_checkpoint(model, root, [SimpleNamespace(next_index=i) for i in latest["next_indices"]],
                          manifest["fingerprint"])
        return {"status": "completed", "total_steps": 80}
    monkeypatch.setattr(runner, "train", train)
    return SimpleNamespace(old=old, root=root, manifest=manifest, lineage=lineage, calls=calls)


def test_progress_keeps_imitation_gains_separate_and_reruns_only_remaining_ppo(workflow, progress_setup):
    setup = progress_setup
    first = workflow.progress(setup.old.source, setup.root, setup.old.upstream,
                              target_steps=75, episodes=100, recipe=setup.old.recipe)
    second = workflow.progress(setup.old.source, setup.root, setup.old.upstream,
                               target_steps=75, episodes=100, recipe=setup.old.recipe)
    assert setup.calls["training"] == [75]
    assert first["new_rl_steps"] == second["new_rl_steps"] == 80
    assert second["training"]["status"] == "target_already_reached"
    assert first["imitation_comparison"]["promote"] is True
    assert first["additional_rl_comparison"]["floor_delta"] == 0
    assert first["additional_rl_establishes_more_act1_clears"] is False
    assert first["victory_scope"] == "act1" and first["source_rl_steps"] == 160768
    assert len(setup.old.calls["collection"]) == len(setup.old.calls["cloning"]) == 1
    expected = evaluation_seeds("validation", 100, 7, start=400)
    assert all(seeds == expected for _, seeds in setup.calls["rollouts"])
    assert [label for label, _ in setup.calls["rollouts"]] == [
        "before_imitation", "after_imitation_before_rl", "teacher_act1", "after_act1_rl", "after_act1_rl",
    ]


@pytest.mark.parametrize("reason", ["no_imitation_gain", "no_teacher_win", "incomplete"])
def test_progress_does_not_allocate_ppo_without_its_evidence_gate(workflow, progress_setup, monkeypatch, reason):
    setup = progress_setup
    if reason == "no_teacher_win":
        lineage = read_json(setup.root / "act1-lineage.json")
        lineage["demonstrations"]["gameplay"]["outcomes"]["win"] = 0
        atomic_json(setup.root / "act1-lineage.json", lineage)
    def rollout(env, seeds, *, model, label):
        floor = 5 if label == "before_imitation" or reason == "no_imitation_gain" else 8
        rows = episode_rows(seeds, label, floor=floor)
        if reason == "incomplete" and label == "teacher_act1":
            rows[0]["outcome"] = "aborted"
        return rows
    monkeypatch.setattr(runner, "_rollout", rollout)
    result = workflow.progress(setup.old.source, setup.root, setup.old.upstream,
                               target_steps=75, episodes=100, recipe=setup.old.recipe)
    expected = {"no_imitation_gain": "stopped_imitation_did_not_establish_improvement",
                "no_teacher_win": "stopped_teacher_produced_no_act1_successes",
                "incomplete": "stopped_incomplete_evaluation"}[reason]
    assert result["status"] == expected and result["new_rl_steps"] == 0
    assert not setup.calls["training"] and verified_checkpoint(setup.root)[1]["num_timesteps"] == 0
    assert result["victory_scope"] == "act1" and result["source_rl_steps"] == 160768
