"""Bounded exploration can be justified without granting a champion claim."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
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
from spire2_rl.study import evaluation_seeds, seed_for, summarize


@pytest.fixture
def trial():
    script = Path(__file__).resolve().parents[1] / "scripts" / "act1_rl_trial.py"
    spec = importlib.util.spec_from_file_location("act1_trial_under_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SchemaOnlyEnv(gym.Env):
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
    """Use real tensors, saved optimizer moments and checkpoint publication."""

    def __init__(self, num_timesteps=0, weight=1):
        self.num_timesteps = num_timesteps
        self.policy = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.policy.weight.fill_(weight)
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


def rows(count=100, *, label="policy", floor=8, wins=0, start=400):
    seeds = evaluation_seeds("validation", count, 7, start=start)
    return [{"seed": seed, "policy": label, "floor": 15 if i < wins else floor,
             "hp": 0, "outcome": "win" if i < wins else "death"}
            for i, seed in enumerate(seeds)]


def assessment(manifest, state, label, count=100, start=400, *, floor=8, wins=0):
    episodes = rows(count, label=label, floor=floor, wins=wins, start=start)
    return {"stage": label, "fingerprint": manifest["fingerprint"], "checkpoint": state,
            "split": "validation", "seed_start": start, "episodes": episodes,
            "gameplay": summarize(episodes), "inference": "deterministic", "victory_scope": "act1"}


def test_one_lost_win_allows_only_exploration_without_rewriting_promotion(trial):
    gate = trial.exploration_gate({"episodes": rows(floor=5, wins=1)}, {"episodes": rows(floor=8)})
    assert gate["eligible"] and gate["comparison"]["floor_delta_ci95"][0] > 0
    assert gate["comparison"]["win_delta"] == -0.01
    assert gate["comparison"]["win_delta_ci95"][0] < 0 <= gate["comparison"]["win_delta_ci95"][1]
    assert gate["comparison"]["promote"] is gate["original_promotion_rule_passed"] is False
    assert gate["clear_rate_noninferiority_claim"] == "not made"


def test_established_clear_rate_loss_blocks_exploration_even_with_floor_gain(trial):
    gate = trial.exploration_gate({"episodes": rows(floor=1, wins=20)}, {"episodes": rows(floor=8)})
    assert gate["comparison"]["floor_delta_ci95"][0] > 0
    assert gate["comparison"]["win_delta_ci95"][1] < 0
    assert not gate["eligible"]
    assert any("clear-rate decrease" in reason for reason in gate["reasons"])


def test_zero_observed_clears_does_not_become_a_noninferiority_claim(trial):
    gate = trial.exploration_gate({"episodes": rows(floor=5)}, {"episodes": rows(floor=8)})
    assert gate["eligible"] and gate["comparison"]["win_delta_ci95"] == [0, 0]
    assert gate["original_promotion_rule_passed"] is True
    assert gate["clear_rate_noninferiority_claim"] == "not made"


@pytest.mark.parametrize("failure", ["insufficient", "timeout", "aborted", "no_floor_gain"])
def test_exploration_still_requires_clean_sufficient_positive_evidence(trial, failure):
    count = 99 if failure == "insufficient" else 100
    before = rows(count, floor=5)
    cloned = rows(count, floor=5 if failure == "no_floor_gain" else 8)
    if failure in {"timeout", "aborted"}:
        cloned[0]["outcome"] = failure
    assert not trial.exploration_gate({"episodes": before}, {"episodes": cloned})["eligible"]


@pytest.mark.parametrize("invalid", [{"target_steps": 0}, {"episodes": 99}, {"validation_start": 499}])
def test_invalid_trial_plan_is_rejected_before_creating_an_experiment(trial, tmp_path, invalid):
    plan = {"target_steps": 75, "episodes": 100, "validation_start": 500, **invalid}
    root = tmp_path / "trial"
    with pytest.raises(ValueError, match="positive bounded budget"):
        trial.progress(tmp_path / "absent-source", root, tmp_path / "upstream", **plan)
    assert not root.exists()


@pytest.fixture
def source_setup(trial, monkeypatch, tmp_path):
    source, upstream = tmp_path / "source", tmp_path / "upstream"
    config = ExperimentConfig(max_acts=1, n_envs=2, n_steps=4, batch_size=4, n_epochs=1,
                              learning_rate=0.0001)
    recipe = trial.act1["default_recipe"]()
    def factory(config, upstream):
        return SchemaOnlyEnv(config)
    monkeypatch.setattr(runner, "_make", factory)
    monkeypatch.setattr(runner, "provenance", lambda upstream: {"core_source": "unchanged"})
    with trial.act1["act1_runner"](recipe):
        manifest = runner.initialize(source, config, upstream)
    states = []
    for weight in (1, 3):
        model = StoredPolicy(weight=weight)
        # A warm start must discard moments even if the source has accumulated them.
        model.policy(torch.ones(1, 1)).sum().backward()
        model.policy.optimizer.step()
        states.append(commit_checkpoint(model, source,
                                        [SimpleNamespace(next_index=22), SimpleNamespace(next_index=23)],
                                        manifest["fingerprint"]))
    atomic_json(source / "seed-cursors" / "worker-0.json", {"next_index": 30})
    atomic_json(source / "seed-cursors" / "worker-1.json", {"next_index": 33})
    atomic_json(source / "champion.json", {"source_only_claim": True})
    (source / "episodes.jsonl").write_text('{"outcome":"death"}\n', encoding="utf-8")
    (source / "demonstrations.npz").write_bytes(b"already-collected training data")
    lineage = {"config": asdict(config), "recipe": recipe,
               "source_checkpoint": {"num_timesteps": 160768},
               "imported_checkpoint": states[0], "cloned_checkpoint": states[1],
               "demonstrations": {"dataset": "demonstrations.npz",
                                  "sha256": sha256(source / "demonstrations.npz"),
                                  "gameplay": {"outcomes": {"win": 1}}}}
    for label, state, floor, wins in (("before_imitation", states[0], 5, 1),
                                      ("after_imitation_before_rl", states[1], 8, 0)):
        target = source / f"{label}.json"
        atomic_json(target, assessment(manifest, state, label, floor=floor, wins=wins))
        lineage[f"{label}_sha256"] = sha256(target)
    atomic_json(source / "act1-lineage.json", lineage)
    monkeypatch.setattr(trial, "MaskablePPO", StoredPolicy)
    calls = {"fresh_weights": []}
    def fresh_policy(model, config, vec):
        calls["fresh_weights"].append(model.policy.weight.detach().clone())
        fresh = StoredPolicy()
        fresh.policy.load_state_dict(model.policy.state_dict(), strict=True)
        vec._seeds = [None] * config.n_envs
        return fresh
    monkeypatch.setitem(trial.selection, "fresh_policy", fresh_policy)
    def forbidden(*args, **kwargs):
        pytest.fail("Exploration must not collect demonstrations, clone again or promote a champion")
    monkeypatch.setitem(trial.act1, "collect_demonstrations", forbidden)
    monkeypatch.setitem(trial.act1, "behavior_clone", forbidden)
    # runpy returns a namespace copy; guard the old creator's real globals too.
    old_globals = trial.act1["create_experiment"].__globals__
    monkeypatch.setitem(old_globals, "collect_demonstrations", forbidden)
    monkeypatch.setitem(old_globals, "behavior_clone", forbidden)
    monkeypatch.setattr(runner, "promote", forbidden)
    return SimpleNamespace(source=source, upstream=upstream, config=config, manifest=manifest,
                           lineage=lineage, states=states, calls=calls, factory=factory,
                           plan={"target_steps": 75, "episodes": 100, "validation_start": 500})


def test_eligible_source_recomputes_raw_evidence_without_changing_source(trial, source_setup):
    old = source_setup
    before = bytes_in(old.source)
    lineage, manifest, config, gate = trial.eligible_source(old.source, old.upstream)
    assert lineage == old.lineage and manifest == old.manifest and config == old.config
    assert gate["eligible"] and not gate["original_promotion_rule_passed"]
    assert bytes_in(old.source) == before and runner._make is old.factory


@pytest.mark.parametrize("tamper", ["checksum", "repinned_summary", "repinned_training_seed", "new_latest"])
def test_source_gate_does_not_trust_pins_summaries_or_a_later_checkpoint(trial, source_setup, tamper):
    old = source_setup
    target = old.source / "after_imitation_before_rl.json"
    if tamper == "new_latest":
        commit_checkpoint(StoredPolicy(num_timesteps=8), old.source,
                          [SimpleNamespace(next_index=30), SimpleNamespace(next_index=33)],
                          old.manifest["fingerprint"])
    else:
        report = read_json(target)
        if tamper == "repinned_training_seed":
            report["episodes"][0]["seed"] = seed_for(0, "train", 7)
            report["gameplay"] = summarize(report["episodes"])
        else:
            report["gameplay"]["mean_floor"] = 999
        atomic_json(target, report)
        if tamper.startswith("repinned"):
            lineage = read_json(old.source / "act1-lineage.json")
            lineage["after_imitation_before_rl_sha256"] = sha256(target)
            atomic_json(old.source / "act1-lineage.json", lineage)
    before = bytes_in(old.source)
    with pytest.raises(ValueError):
        trial.eligible_source(old.source, old.upstream)
    assert bytes_in(old.source) == before and runner._make is old.factory


def test_trial_copies_both_frozen_policies_with_new_identity_and_durable_cursors(trial, source_setup, tmp_path):
    old, root = source_setup, tmp_path / "trial"
    before = bytes_in(old.source)
    lineage = trial.create_trial(old.source, root, old.upstream, old.plan)
    manifest = read_json(root / "experiment.json")
    assert bytes_in(old.source) == before and runner._make is old.factory
    assert manifest["fingerprint"] != old.manifest["fingerprint"]
    assert lineage["source_fingerprint"] == old.manifest["fingerprint"]
    assert lineage["source_rl_steps"] == 160768 and lineage["training_seed_indices"] == [30, 33]
    assert lineage["exploration_entry"]["eligible"] and not lineage["exploration_entry"]["original_promotion_rule_passed"]
    for new_key, old_state in zip(("incumbent_checkpoint", "initial_checkpoint"), old.states):
        state = lineage[new_key]
        original = StoredPolicy.load(old.source / old_state["model"])
        copied = StoredPolicy.load(root / state["model"])
        assert state["fingerprint"] == manifest["fingerprint"] and state["next_indices"] == [30, 33]
        assert state["num_timesteps"] == copied.num_timesteps == 0
        assert original.policy.optimizer.state and not copied.policy.optimizer.state
        for name, value in original.policy.state_dict().items():
            torch.testing.assert_close(copied.policy.state_dict()[name], value, rtol=0, atol=0)
    assert verified_checkpoint(root)[1] == lineage["initial_checkpoint"]
    assert len(old.calls["fresh_weights"]) == 2
    assert not (root / "demonstrations.npz").exists() and not (root / "champion.json").exists()
    with trial.act1["act1_runner"](old.lineage["recipe"]):
        assert runner.initialize(old.source, old.config, old.upstream) == old.manifest
        with pytest.raises(ValueError, match="configuration|fingerprint"):
            runner.initialize(root, old.config, old.upstream)
    assert runner._make is old.factory


def test_failed_second_transplant_cleans_target_and_restores_source_context(trial, source_setup,
                                                                         tmp_path, monkeypatch):
    old, root = source_setup, tmp_path / "trial"
    before = bytes_in(old.source)
    fresh = trial.selection["fresh_policy"]
    attempts = []
    def fail_second(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 2:
            raise RuntimeError("second transplant failed")
        return fresh(*args, **kwargs)
    monkeypatch.setitem(trial.selection, "fresh_policy", fail_second)
    with pytest.raises(RuntimeError, match="second transplant failed"):
        trial.create_trial(old.source, root, old.upstream, old.plan)
    assert not root.exists() and not list(tmp_path.glob(".trial-initializing-*"))
    assert bytes_in(old.source) == before and runner._make is old.factory
    monkeypatch.setitem(trial.selection, "fresh_policy", fresh)
    assert trial.create_trial(old.source, root, old.upstream, old.plan)["initial_checkpoint"]["num_timesteps"] == 0


def test_wrong_rank_durable_cursor_is_rejected_even_below_checkpoint(trial, source_setup):
    old = source_setup
    atomic_json(old.source / "seed-cursors" / "worker-0.json", {"next_index": 1})
    with pytest.raises(ValueError, match="wrong worker rank"):
        trial.source_indices(old.source, old.states[1], old.config)


@pytest.mark.parametrize("start,count", [(400, 101), (401, 100)])
def test_trial_seed_schedule_must_be_disjoint_from_actual_source_evidence(trial, source_setup,
                                                                       tmp_path, start, count):
    old, root = source_setup, tmp_path / "trial"
    lineage = read_json(old.source / "act1-lineage.json")
    for label, state, floor, wins in (("before_imitation", old.states[0], 5, 1),
                                      ("after_imitation_before_rl", old.states[1], 8, 0)):
        target = old.source / f"{label}.json"
        atomic_json(target, assessment(old.manifest, state, label, count, start, floor=floor, wins=wins))
        lineage[f"{label}_sha256"] = sha256(target)
    atomic_json(old.source / "act1-lineage.json", lineage)
    before = bytes_in(old.source)
    assert trial.eligible_source(old.source, old.upstream)[3]["eligible"]
    with pytest.raises(ValueError, match="overlap"):
        trial.create_trial(old.source, root, old.upstream, old.plan)
    assert not root.exists() and not old.calls["fresh_weights"]
    assert bytes_in(old.source) == before and runner._make is old.factory


@pytest.fixture
def progress_setup(trial, source_setup, tmp_path, monkeypatch):
    old, root = source_setup, tmp_path / "trial"
    calls = {"assessments": [], "training": []}
    def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
        trial.act1["checked_state"](root, state, manifest["fingerprint"])
        calls["assessments"].append((label, episodes, validation_start, deepcopy(state)))
        floor = {"incumbent_reference": 5, "before_rl": 8, "after_exploratory_rl": 9}[label]
        return assessment(manifest, state, label, episodes, validation_start, floor=floor)
    monkeypatch.setitem(trial.act1, "assess", assess)
    def train(root, config, upstream, *, steps):
        calls["training"].append(steps)
        manifest = read_json(root / "experiment.json")
        path, latest = verified_checkpoint(root, manifest["fingerprint"])
        model = StoredPolicy.load(path)
        model.num_timesteps += 80  # PPO may complete its rollout slightly beyond the target.
        commit_checkpoint(model, root, [SimpleNamespace(next_index=i + config.n_envs)
                                        for i in latest["next_indices"]], manifest["fingerprint"])
        return {"status": "completed", "total_steps": model.num_timesteps}
    monkeypatch.setattr(runner, "train", train)
    return SimpleNamespace(old=old, root=root, calls=calls)


def test_resume_never_repeats_full_budget_and_compares_rl_against_both_frozen_policies(trial, progress_setup):
    setup, old = progress_setup, progress_setup.old
    before = bytes_in(old.source)
    first = trial.progress(old.source, setup.root, old.upstream, **old.plan)
    second = trial.progress(old.source, setup.root, old.upstream, **old.plan)
    assert setup.calls["training"] == [75]
    assert first["new_rl_steps"] == second["new_rl_steps"] == 80
    assert second["training"]["status"] == "target_already_reached"
    assert first["source_rl_steps"] == 160768
    assert first["additional_rl_comparison"]["floor_delta"] == 1
    assert first["incumbent_comparison"]["floor_delta"] == 4
    assert first["champion_updated"] is second["champion_updated"] is False
    assert first["additional_rl_comparison"]["promote"] is True  # Even this never authorizes promotion.
    stages = [call[0] for call in setup.calls["assessments"]]
    assert stages == ["before_rl", "incumbent_reference", "after_exploratory_rl", "after_exploratory_rl"]
    assert all(call[1:3] == (100, 500) for call in setup.calls["assessments"])
    assert setup.calls["assessments"][0][3] != setup.calls["assessments"][1][3]
    assert setup.calls["assessments"][2][3]["num_timesteps"] == 80
    seeds = evaluation_seeds("validation", 100, 7, start=500)
    report = read_json(Path(first["evaluation_report"]))
    assert [row["seed"] for row in report["episodes"]] == seeds
    assert set(seeds).isdisjoint(evaluation_seeds("validation", 100, 7, start=400))
    assert bytes_in(old.source) == before and not (setup.root / "champion.json").exists()
    assert runner._make is old.factory


@pytest.mark.parametrize("changed", [{"target_steps": 76}, {"episodes": 101}, {"validation_start": 501}])
def test_resume_refuses_a_changed_budget_or_evaluation_schedule(trial, progress_setup, changed):
    setup, old = progress_setup, progress_setup.old
    trial.progress(old.source, setup.root, old.upstream, **old.plan)
    with pytest.raises(ValueError, match="budget and evaluation schedule are fixed"):
        trial.progress(old.source, setup.root, old.upstream, **{**old.plan, **changed})
    assert setup.calls["training"] == [75]


def test_reference_cache_recovers_valid_unpinned_commit_and_rejects_later_changes(trial, source_setup,
                                                                            tmp_path, monkeypatch):
    old, root = source_setup, tmp_path / "trial"
    lineage = trial.create_trial(old.source, root, old.upstream, old.plan)
    manifest = read_json(root / "experiment.json")
    state = lineage["initial_checkpoint"]
    target = root / "before_rl.json"
    report = assessment(manifest, state, "before_rl", start=500)
    atomic_json(target, report)
    def no_new_games(*args, **kwargs):
        pytest.fail("A completed valid reference must recover without new episodes")
    monkeypatch.setitem(trial.act1, "assess", no_new_games)
    assert trial.cached_reference(root, old.upstream, manifest, lineage, "before_rl", state) == report
    assert lineage["before_rl_sha256"] == sha256(target)
    report["episodes"][0]["outcome"] = "win"
    atomic_json(target, report)
    with pytest.raises(ValueError, match="checksum changed"):
        trial.cached_reference(root, old.upstream, manifest, lineage, "before_rl", state)


def test_incomplete_reference_stops_before_any_ppo(trial, progress_setup):
    setup, old = progress_setup, progress_setup.old
    lineage = trial.create_trial(old.source, setup.root, old.upstream, old.plan)
    manifest = read_json(setup.root / "experiment.json")
    report = assessment(manifest, lineage["initial_checkpoint"], "before_rl", start=500)
    report["episodes"][0]["outcome"] = "aborted"
    report["gameplay"] = summarize(report["episodes"])
    target = setup.root / "before_rl.json"
    atomic_json(target, report)
    lineage["before_rl_sha256"] = sha256(target)
    atomic_json(setup.root / "trial-lineage.json", lineage)
    with pytest.raises(ValueError, match="no further PPO started"):
        trial.progress(old.source, setup.root, old.upstream, **old.plan)
    assert not setup.calls["training"]
    assert verified_checkpoint(setup.root)[1]["num_timesteps"] == 0
