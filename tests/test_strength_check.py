"""A frozen strength diagnostic must never become another training workflow."""

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
    ExperimentConfig, atomic_json, commit_checkpoint, read_json, sha256,
)
from spire2_rl.study import evaluation_seeds, paired_compare, summarize


@pytest.fixture
def checker():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_act1_trial.py"
    spec = importlib.util.spec_from_file_location("strength_check_under_test", script)
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
    def __init__(self, num_timesteps=0, weight=1):
        self.num_timesteps = num_timesteps
        self.policy = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.policy.weight.fill_(weight)
            self.policy.bias.zero_()

    def save(self, path):
        torch.save({"weights": self.policy.state_dict(), "num_timesteps": self.num_timesteps}, path)

    @classmethod
    def load(cls, path, **kwargs):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(saved["num_timesteps"])
        model.policy.load_state_dict(saved["weights"])
        return model


def bytes_in(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def rows(seeds, label, *, floor=8, wins=0):
    return [{"seed": seed, "policy": label, "outcome": "win" if i < wins else "death",
             "floor": 15 if i < wins else floor, "hp": 0} for i, seed in enumerate(seeds)]


def diagnostic(protocol, seeds, *, wins=0):
    candidate = rows(seeds, "frozen-rl-candidate", floor=9, wins=wins)
    baseline = rows(seeds, "frozen-cloned-initializer", floor=8)
    return {"protocol": protocol, "candidate_episodes": candidate, "baseline_episodes": baseline,
            "candidate": summarize(candidate), "baseline": summarize(baseline),
            "comparison": paired_compare(candidate, baseline), "training_record": {"status": "read"}}


@pytest.fixture
def saved_trial(checker, monkeypatch, tmp_path):
    root, source, upstream = tmp_path / "trial", tmp_path / "source", tmp_path / "upstream"
    config = ExperimentConfig(max_acts=1, n_envs=2, n_steps=4, batch_size=4, n_epochs=1)
    recipe = checker.act1["default_recipe"]()
    plan = {"target_steps": 80000, "episodes": 100, "validation_start": 500}
    def factory(config, upstream):
        return SchemaOnlyEnv(config)
    monkeypatch.setattr(runner, "_make", factory)
    monkeypatch.setattr(runner, "provenance", lambda upstream: {"core_source": "unchanged"})
    wrappers = [SimpleNamespace(next_index=20), SimpleNamespace(next_index=21)]
    with checker.act1["act1_runner"](recipe):
        source_manifest = runner.initialize(source, config, upstream)
    source_states = [commit_checkpoint(StoredPolicy(weight=weight), source, wrappers, source_manifest["fingerprint"])
                     for weight in (1, 3)]
    source_lineage = {"config": asdict(config), "recipe": recipe, "source_checkpoint": {"num_timesteps": 160768},
                      "imported_checkpoint": source_states[0], "cloned_checkpoint": source_states[1]}
    seeds = evaluation_seeds("validation", 100, config.experiment_seed, start=400)
    for label, state in zip(("before_imitation", "after_imitation_before_rl"), source_states):
        episodes = rows(seeds, label)
        report = {"stage": label, "fingerprint": source_manifest["fingerprint"], "checkpoint": state,
                  "split": "validation", "seed_start": 400, "episodes": episodes,
                  "gameplay": summarize(episodes), "inference": "deterministic", "victory_scope": "act1"}
        target = source / f"{label}.json"
        atomic_json(target, report)
        source_lineage[f"{label}_sha256"] = sha256(target)
    atomic_json(source / "act1-lineage.json", source_lineage)
    with checker.trial["trial_runner"](recipe, plan):
        manifest = runner.initialize(root, config, upstream)
    baseline = commit_checkpoint(StoredPolicy(weight=3), root, wrappers, manifest["fingerprint"])
    current = commit_checkpoint(StoredPolicy(num_timesteps=80952, weight=4), root, wrappers,
                                manifest["fingerprint"])
    lineage = {"config": asdict(config), "recipe": recipe, "plan": plan,
               "source_experiment": str(source), "source_fingerprint": source_manifest["fingerprint"],
               "source_imported_checkpoint": source_states[0], "source_cloned_checkpoint": source_states[1],
               "initial_checkpoint": baseline}
    atomic_json(root / "trial-lineage.json", lineage)
    (root / "episodes.jsonl").write_text('{"session":"one","outcome":"death","floor":8}\n', encoding="utf-8")
    (root / "sessions.jsonl").write_text('{"session":"one","status":"completed"}\n', encoding="utf-8")
    monkeypatch.setattr(checker, "MaskablePPO", StoredPolicy)
    calls = []
    def rollout(env, seeds, *, model, label):
        assert env.backend_metadata["trial_plan"] == plan
        calls.append((label, list(seeds), model.num_timesteps, float(model.policy.weight.item())))
        return rows(seeds, label, floor=9 if model.num_timesteps else 8)
    monkeypatch.setattr(runner, "_rollout", rollout)
    def forbidden(*args, **kwargs):
        pytest.fail("A strength check must not train, imitate, or promote")
    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "promote", forbidden)
    old_globals = checker.act1["create_experiment"].__globals__
    monkeypatch.setitem(old_globals, "collect_demonstrations", forbidden)
    monkeypatch.setitem(old_globals, "behavior_clone", forbidden)
    monkeypatch.setitem(checker.trial, "create_trial", forbidden)
    return SimpleNamespace(root=root, source=source, upstream=upstream, config=config, recipe=recipe,
                           plan=plan, lineage=lineage, manifest=manifest, baseline=baseline, current=current,
                           calls=calls, factory=factory)


def test_check_freezes_exact_models_uses_matched_fresh_batches_and_preserves_training(checker, saved_trial):
    old = saved_trial
    original, source_original = bytes_in(old.root), bytes_in(old.source)
    summary = checker.check(old.root, old.upstream)
    seeds = evaluation_seeds("validation", 1000, 7, start=700)
    assert len(old.calls) == 40
    for index in range(0, len(old.calls), 2):
        current, baseline = old.calls[index:index + 2]
        assert current[0] == "frozen-rl-candidate" and baseline[0] == "frozen-cloned-initializer"
        assert current[1] == baseline[1] == seeds[(index // 2) * 50:(index // 2 + 1) * 50]
        assert current[2:] == (80952, 4) and baseline[2:] == (0, 3)
    report_path = Path(summary["report_path"])
    report = read_json(report_path)
    protocol = report["protocol"]
    assert protocol["candidate_checkpoint"] == old.current and protocol["baseline_checkpoint"] == old.baseline
    assert protocol["fingerprint"] == old.manifest["fingerprint"]
    assert protocol["episodes"] == 1000 and protocol["seed_start"] == 700
    assert protocol["inference"] == "deterministic" and protocol["victory_scope"] == "act1"
    checker.validate_report(report, protocol, seeds)
    assert summary["rl_floor_gain"] == 1 and summary["rl_floor_gain_ci95"] == [1, 1]
    assert summary["establishes_positive_floor_gain"] is True
    assert summary["establishes_positive_act1_clear_gain"] is False
    assert summary["additional_training_steps"] == 0 and summary["champion_updated"] is False
    after = bytes_in(old.root)
    assert all(after[name] == value for name, value in original.items())
    assert set(after) - set(original) == {str(report_path.relative_to(old.root)),
                                         str(report_path.with_name("protocol.json").relative_to(old.root))}
    assert bytes_in(old.source) == source_original and runner._make is old.factory
    second = checker.check(old.root, old.upstream)
    assert second == summary and len(old.calls) == 40  # Reuse evidence, do not draw another sample.


@pytest.mark.parametrize("previous", ["trial_plan", "actual_source"])
def test_fresh_seeds_checks_prior_actual_seed_sets_not_only_minimum_start(checker, saved_trial, previous):
    old = saved_trial
    lineage = deepcopy(old.lineage)
    if previous == "trial_plan":
        lineage["plan"]["validation_start"] = 950
    else:
        source_lineage = read_json(old.source / "act1-lineage.json")
        for label in ("before_imitation", "after_imitation_before_rl"):
            target = old.source / f"{label}.json"
            report = read_json(target)
            report["seed_start"] = 950
            report["episodes"] = rows(evaluation_seeds("validation", 100, 7, start=950), label)
            report["gameplay"] = summarize(report["episodes"])
            atomic_json(target, report)
            source_lineage[f"{label}_sha256"] = sha256(target)
        atomic_json(old.source / "act1-lineage.json", source_lineage)
    with pytest.raises(ValueError, match="overlap"):
        checker.fresh_seeds(old.root, lineage, old.config, 1000, 700)
    assert not old.calls


@pytest.mark.parametrize("missing", ["before_imitation", "after_imitation_before_rl"])
def test_freshness_requires_both_original_source_evidence_files(checker, saved_trial, missing):
    old = saved_trial
    (old.source / f"{missing}.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        checker.fresh_seeds(old.root, old.lineage, old.config, 1000, 700)


@pytest.mark.parametrize("invalid", [{"episodes": 999}, {"start": 699}])
def test_check_rejects_a_smaller_or_stale_plan_before_touching_files(checker, tmp_path, invalid):
    root = tmp_path / "absent"
    with pytest.raises(ValueError, match="at least 1000"):
        checker.check(root, tmp_path / "upstream", **invalid)
    assert not root.exists()


@pytest.mark.parametrize("tamper", ["incomplete", "interrupted", "fingerprint", "candidate_checksum", "baseline_checksum"])
def test_check_rejects_inexact_or_unfinished_frozen_states_before_any_rollout(checker, saved_trial, tamper):
    old = saved_trial
    if tamper == "candidate_checksum":
        (old.root / old.current["model"]).write_bytes(b"different candidate")
    elif tamper == "baseline_checksum":
        (old.root / old.baseline["model"]).write_bytes(b"different baseline")
    else:
        current = deepcopy(old.current)
        if tamper == "incomplete":
            current["num_timesteps"] = 79999
        elif tamper == "interrupted":
            current["interrupted"] = True
        else:
            current["fingerprint"] = "different-backend"
        atomic_json(old.root / "latest.json", current)
    with pytest.raises(ValueError):
        checker.check(old.root, old.upstream)
    assert not old.calls and runner._make is old.factory


def test_a_model_diagnostic_cannot_silently_redraw_a_different_sample(checker, saved_trial):
    old = saved_trial
    summary = checker.check(old.root, old.upstream)
    with pytest.raises(ValueError, match="already frozen"):
        checker.check(old.root, old.upstream, start=1700)
    assert len(old.calls) == 40 and Path(summary["report_path"]).exists()


def test_checkpoint_mutation_during_evaluation_prevents_publishing_evidence(checker, saved_trial, monkeypatch):
    old = saved_trial
    rollout = runner._rollout
    mutated = []
    def mutate_once(*args, **kwargs):
        result = rollout(*args, **kwargs)
        if not mutated:
            path = old.root / old.current["model"]
            with path.open("ab") as stream:
                stream.write(b"changed while evaluating")
            mutated.append(True)
        return result
    monkeypatch.setattr(runner, "_rollout", mutate_once)
    with pytest.raises(ValueError, match="checksum changed during evaluation"):
        checker.check(old.root, old.upstream)
    assert not list((old.root / "diagnostics").rglob("report.json"))
    assert runner._make is old.factory


@pytest.mark.parametrize("tamper", ["summary", "comparison", "seed", "policy", "checkpoint"])
def test_cached_raw_evidence_is_validated_instead_of_trusting_stored_results(checker, tamper):
    seeds = evaluation_seeds("validation", 100, 7, start=700)
    protocol = {"candidate_checkpoint": {"sha256": "frozen"}}
    report = diagnostic(protocol, seeds)
    checker.validate_report(report, protocol, seeds)
    changed = deepcopy(report)
    if tamper == "summary":
        changed["candidate"]["mean_floor"] = 99
    elif tamper == "comparison":
        changed["comparison"]["floor_delta"] = 99
    elif tamper == "seed":
        changed["candidate_episodes"][0]["seed"] = seeds[-1]
    elif tamper == "policy":
        changed["baseline_episodes"][0]["policy"] = "another model"
    else:
        changed["protocol"]["candidate_checkpoint"]["sha256"] = "another checkpoint"
    with pytest.raises(ValueError):
        checker.validate_report(changed, protocol, seeds)


def test_one_new_clear_with_zero_lower_bound_does_not_establish_a_clear_rate_gain(checker, tmp_path):
    seeds = evaluation_seeds("validation", 100, 7, start=700)
    report = diagnostic({"candidate_checkpoint": {}}, seeds, wins=1)
    summary = checker.view(report, tmp_path / "report.json")
    assert summary["rl_act1_clear_rate_gain"] == 0.01
    assert summary["rl_act1_clear_rate_gain_ci95"][0] == 0
    assert summary["establishes_positive_act1_clear_gain"] is False
    assert summary["establishes_positive_floor_gain"] is True
    assert summary["champion_updated"] is False


@pytest.mark.parametrize("gains,losses,probability,supports", [
    (4, 0, 0.125, False), (6, 0, 0.03125, True), (0, 6, 0.03125, False), (0, 0, 1, False),
])
def test_exact_clear_evidence_handles_rare_events_and_direction(checker, gains, losses, probability, supports):
    seeds = evaluation_seeds("validation", 1000, 7, start=700)
    candidate = rows(seeds, "frozen-rl-candidate", wins=gains)
    baseline = rows(seeds, "frozen-cloned-initializer", wins=losses)
    evidence = checker.exact_win_test(candidate, list(reversed(baseline)))
    assert evidence["candidate_only_clears"] == gains and evidence["baseline_only_clears"] == losses
    assert evidence["discordant_pairs"] == gains + losses
    assert evidence["exact_two_sided_p"] == probability
    assert evidence["supports_more_clears"] is supports
    assert "No clear-rate equivalence or noninferiority claim" in evidence["note"]


@pytest.mark.parametrize("wins,established", [(4, False), (6, True)])
def test_view_uses_exact_paired_clear_support_when_bootstrap_is_positive(checker, tmp_path, wins, established):
    seeds = evaluation_seeds("validation", 1000, 7, start=700)
    report = diagnostic({"candidate_checkpoint": {}}, seeds, wins=wins)
    assert report["comparison"]["win_delta_ci95"][0] > 0
    summary = checker.view(report, tmp_path / "report.json")
    assert summary["paired_clear_evidence"]["candidate_only_clears"] == wins
    assert summary["establishes_positive_act1_clear_gain"] is established
    assert summary["champion_updated"] is False and summary["additional_training_steps"] == 0


def test_exact_clear_test_rejects_unmatched_seed_sets(checker):
    seeds = evaluation_seeds("validation", 100, 7, start=700)
    candidate = rows(seeds, "frozen-rl-candidate")
    baseline = rows(seeds, "frozen-cloned-initializer")
    with pytest.raises(ValueError, match="matched seed sets"):
        checker.exact_win_test(candidate, baseline[:-1])


@pytest.mark.parametrize("incomplete", ["timeout", "aborted"])
def test_incomplete_evaluation_keeps_errors_visible_and_blocks_gain_claims(checker, tmp_path, incomplete):
    seeds = evaluation_seeds("validation", 100, 7, start=700)
    report = diagnostic({"candidate_checkpoint": {}}, seeds)
    report["candidate_episodes"][0]["outcome"] = incomplete
    report["candidate"] = summarize(report["candidate_episodes"])
    report["comparison"] = paired_compare(report["candidate_episodes"], report["baseline_episodes"])
    summary = checker.view(report, tmp_path / "report.json")
    assert summary["after_rl"]["episodes"] == 100 and summary["after_rl"]["act1_clears"] == 0
    assert summary["after_rl"]["timeouts"] == (incomplete == "timeout")
    assert summary["after_rl"]["aborted"] == (incomplete == "aborted")
    assert summary["establishes_positive_floor_gain"] is False
    assert summary["establishes_positive_act1_clear_gain"] is False


def test_training_record_separates_session_boundary_discards_from_exception_aborts(checker, tmp_path):
    root = tmp_path / "trial"
    root.mkdir()
    episodes = [{"session": "one", "outcome": "win"}, {"session": "one", "outcome": "death"},
                {"session": "one", "outcome": "timeout"},
                {"session": "one", "outcome": "aborted", "reason": "training budget ended; unfinished episode discarded"},
                {"session": "two", "outcome": "aborted", "reason": "KeyboardInterrupt: "},
                {"session": "two", "outcome": "aborted", "reason": "RuntimeError: invalid action"}]
    sessions = [{"session": "one", "status": "completed", "added_steps": 80000},
                {"session": "two", "status": "interrupted", "checkpoint": {"num_timesteps": 80952}}]
    (root / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n", encoding="utf-8")
    (root / "sessions.jsonl").write_text("\n".join(json.dumps(row) for row in sessions) + "\n", encoding="utf-8")
    record = checker.training_record(root, ExperimentConfig(), {"num_timesteps": 80952})
    assert record["status"] == "read"
    assert record["training_outcomes"] == {"win": 1, "death": 1, "timeout": 1}
    assert record["budget_boundary_discards"] == 1 and record["other_aborted_episodes"] == 2
    assert record["other_abort_reasons"] == {"KeyboardInterrupt: ": 1, "RuntimeError: invalid action": 1}
    assert record["session_statuses"] == {"completed": 1, "interrupted": 1}
    assert record["interrupted_checkpoint_steps"] == [80952]
    assert record["latest_decisions"] == 80952 and record["rollout_size"] == 1024
    assert record["latest_remainder_within_rollout"] == 80952 % 1024
    assert "Training policies change" in record["note"] and "not held-out strength" in record["note"]


def test_malformed_training_journal_is_reported_without_poisoning_frozen_gameplay(checker, tmp_path):
    (tmp_path / "episodes.jsonl").write_text('{"outcome":\n', encoding="utf-8")
    record = checker.training_record(tmp_path, ExperimentConfig(), {"num_timesteps": 80952})
    assert record["status"] == "unreadable" and "JSONDecodeError" in record["error"]


@pytest.mark.parametrize("invalid", ["episode_outcome", "session_status"])
def test_unknown_journal_outcomes_are_disclosed_instead_of_omitted_from_totals(checker, tmp_path, invalid):
    if invalid == "episode_outcome":
        target, value = tmp_path / "episodes.jsonl", {"outcome": "unsupported-result"}
    else:
        target, value = tmp_path / "sessions.jsonl", {"status": "unsupported-status"}
    target.write_text(json.dumps(value) + "\n", encoding="utf-8")
    record = checker.training_record(tmp_path, ExperimentConfig(), {"num_timesteps": 80952})
    assert record["status"] == "unreadable" and record["error"]
