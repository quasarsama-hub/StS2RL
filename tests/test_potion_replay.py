"""Frozen failure replays must diagnose transport without changing a learner."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, read_json, sha256
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds, paired_compare, summarize

from test_strength_check import checker, saved_trial, StoredPolicy, bytes_in, rows


def quick_compare(candidate, baseline):
    # These tests concern exact evidence/model transport, rather than precision.
    return paired_compare(candidate, baseline, bootstrap_samples=100)


@pytest.fixture
def potion_checker(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_potion_failures.py"
    spec = importlib.util.spec_from_file_location("potion_replay_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MaskablePPO", StoredPolicy)
    monkeypatch.setitem(module.checker["validate_report"].__globals__, "paired_compare", quick_compare)
    return module


def failure_row(seed, policy="frozen-rl-candidate"):
    return {"seed": seed, "policy": policy, "outcome": "aborted", "steps": 40,
            "floor": 4, "hp": 30.0, "return": 0,
            "error": (f"SimulatorError: Emulator step failed (seed={seed}, step=40, action=1): "
                      "RuntimeError: Engine rejected advertised legal pick_potion")}


def original_result(expected):
    return {**expected, "error_stage": "game_step", "capacity_mask_decisions": 0,
            "last_decisions": [{"named_action": {"action": "pick_potion"},
                                "potion_inventory": {"max_slots": 3,
                                                     "slot_potion_ids": ["A", "B", "C"],
                                                     "free_slots": [],
                                                     "pick_action_slots": [1],
                                                     "pick_action_mask": [True]}}]}


@pytest.fixture
def failure_trial(potion_checker, saved_trial, monkeypatch):
    old = saved_trial
    seeds = evaluation_seeds("validation", 1000, old.config.experiment_seed, start=700)
    protocol = {"fingerprint": old.manifest["fingerprint"], "candidate_checkpoint": old.current,
                "baseline_checkpoint": old.baseline, "split": "validation", "episodes": 1000,
                "seed_start": 700, "inference": "deterministic", "victory_scope": "act1",
                "checker_sha256": sha256(Path(__file__).resolve().parents[1] / "scripts" / "check_act1_trial.py")}
    candidate = rows(seeds, "frozen-rl-candidate", floor=9)
    baseline = rows(seeds, "frozen-cloned-initializer", floor=8)
    candidate[0] = failure_row(seeds[0])
    report = {"protocol": protocol, "candidate_episodes": candidate, "baseline_episodes": baseline,
              "candidate": summarize(candidate), "baseline": summarize(baseline),
              "comparison": quick_compare(candidate, baseline), "training_record": {"status": "read"}}
    path = old.root / "diagnostics" / f"strength-{old.current['sha256'][:16]}" / "report.json"
    atomic_json(path, report)
    calls = []

    def replay(env, model, seed, output):
        repaired = isinstance(env, PotionCapacityMask)
        calls.append((seed, "capacity_mask" if repaired else "original", model.num_timesteps,
                      float(model.policy.weight.item()), Path(output)))
        key = "candidate_episodes" if model.num_timesteps else "baseline_episodes"
        expected = next(row for row in report[key] if row["seed"] == seed)
        result = original_result(expected)
        if repaired:
            result.update(outcome="death", error=None, error_stage=None, steps=100,
                          floor=7, hp=0.0, capacity_mask_decisions=1)
        atomic_json(Path(output) / "fixture-replay.json", result)
        return result

    monkeypatch.setattr(potion_checker, "replay", replay)
    return SimpleNamespace(old=old, report=report, path=path, seeds=seeds, calls=calls, replay=replay)


def save_report(fixture):
    report = fixture.report
    report["candidate"] = summarize(report["candidate_episodes"])
    report["baseline"] = summarize(report["baseline_episodes"])
    report["comparison"] = quick_compare(report["candidate_episodes"], report["baseline_episodes"])
    atomic_json(fixture.path, report)


def test_replays_only_failure_with_same_saved_model_and_preserves_old_evidence(potion_checker, failure_trial):
    fixture = failure_trial
    before = bytes_in(fixture.old.root)
    source_before = bytes_in(fixture.old.source)
    factory = runner._make
    result = potion_checker.check(fixture.old.root, fixture.old.upstream)
    assert [(seed, mode, steps, weight) for seed, mode, steps, weight, _ in fixture.calls] == [
        (fixture.seeds[0], "original", 80952, 4),
        (fixture.seeds[0], "capacity_mask", 80952, 4)]
    assert fixture.calls[0][-1] != fixture.calls[1][-1]
    assert result["repair_verified"] is True
    assert result["failed_evaluation_runs"] == 1
    assert result["additional_training_steps"] == 0 and result["champion_updated"] is False
    assert result["runs"][0]["confirmed_full_capacity_failure"] is True
    assert result["runs"][0]["repair_completed_run"] is True
    assert read_json(Path(result["summary_path"])) == result
    after = bytes_in(fixture.old.root)
    assert all(after[name] == data for name, data in before.items())
    assert bytes_in(fixture.old.source) == source_before
    assert runner._make is factory


def test_baseline_failures_replay_the_baseline_weights_not_candidate(potion_checker, failure_trial):
    fixture = failure_trial
    fixture.report["baseline_episodes"][1] = failure_row(fixture.seeds[1], "frozen-cloned-initializer")
    save_report(fixture)
    result = potion_checker.check(fixture.old.root, fixture.old.upstream, fixture.path)
    assert result["failed_evaluation_runs"] == 2 and result["repair_verified"]
    assert [(seed, mode, steps, weight) for seed, mode, steps, weight, _ in fixture.calls] == [
        (fixture.seeds[0], "original", 80952, 4), (fixture.seeds[0], "capacity_mask", 80952, 4),
        (fixture.seeds[1], "original", 0, 3), (fixture.seeds[1], "capacity_mask", 0, 3)]


def test_no_failures_cannot_verify_a_repair_or_load_models(potion_checker, failure_trial, monkeypatch):
    fixture = failure_trial
    fixture.report["candidate_episodes"][0] = rows([fixture.seeds[0]], "frozen-rl-candidate", floor=9)[0]
    save_report(fixture)

    def forbidden(*args, **kwargs):
        pytest.fail("A diagnostic with no failures must not load or replay any policy")

    monkeypatch.setattr(potion_checker, "MaskablePPO", SimpleNamespace(load=forbidden))
    monkeypatch.setattr(potion_checker, "replay", forbidden)
    result = potion_checker.check(fixture.old.root, fixture.old.upstream)
    assert result["failed_evaluation_runs"] == 0 and result["runs"] == []
    assert result["repair_verified"] is False


@pytest.mark.parametrize("tamper", ["candidate_checksum", "baseline_checksum", "report_statistics", "report_model", "checker_source", "inference"])
def test_invalid_frozen_evidence_rejected_before_replay(potion_checker, failure_trial, tamper):
    fixture = failure_trial
    if tamper.endswith("checksum"):
        state = fixture.old.current if tamper == "candidate_checksum" else fixture.old.baseline
        (fixture.old.root / state["model"]).write_bytes(b"changed frozen model")
    elif tamper == "report_statistics":
        fixture.report["candidate"]["mean_floor"] += 1
        atomic_json(fixture.path, fixture.report)
    else:
        if tamper == "report_model":
            fixture.report["protocol"]["baseline_checkpoint"] = deepcopy(fixture.old.current)
        elif tamper == "checker_source":
            fixture.report["protocol"]["checker_sha256"] = "wrong checker bytes"
        else:
            fixture.report["protocol"]["inference"] = "stochastic"
        atomic_json(fixture.path, fixture.report)
    with pytest.raises(ValueError):
        potion_checker.check(fixture.old.root, fixture.old.upstream)
    assert fixture.calls == []


@pytest.mark.parametrize("mutation", ["original_error", "original_step", "not_full", "repair_error", "repair_timeout", "mask_unused"])
def test_unreproduced_or_unrepaired_failure_never_counts_as_verified(potion_checker, failure_trial, monkeypatch, mutation):
    fixture = failure_trial

    def replay(env, model, seed, output):
        result = fixture.replay(env, model, seed, output)
        repaired = isinstance(env, PotionCapacityMask)
        if not repaired:
            if mutation == "original_error":
                result["error"] = "different engine failure"
            elif mutation == "original_step":
                result["steps"] += 1
            elif mutation == "not_full":
                result["last_decisions"][-1]["potion_inventory"]["free_slots"] = [0]
        else:
            if mutation == "repair_error":
                result["outcome"], result["error"] = "aborted", "unexpected later error"
            elif mutation == "repair_timeout":
                result["outcome"] = "timeout"
            elif mutation == "mask_unused":
                result["capacity_mask_decisions"] = 0
        return result

    monkeypatch.setattr(potion_checker, "replay", replay)
    result = potion_checker.check(fixture.old.root, fixture.old.upstream)
    assert result["repair_verified"] is False
    assert len(fixture.calls) == 2


@pytest.mark.parametrize("target", ["report", "candidate", "baseline"])
def test_changed_source_or_model_during_replay_rejected(potion_checker, failure_trial, monkeypatch, target):
    fixture = failure_trial

    def replay(env, model, seed, output):
        result = fixture.replay(env, model, seed, output)
        if isinstance(env, PotionCapacityMask):
            path = fixture.path if target == "report" else fixture.old.root / (
                fixture.old.current if target == "candidate" else fixture.old.baseline)["model"]
            path.write_bytes(path.read_bytes() + b"changed bytes")
        return result

    monkeypatch.setattr(potion_checker, "replay", replay)
    with pytest.raises(ValueError, match="checksum changed"):
        potion_checker.check(fixture.old.root, fixture.old.upstream)


def test_exact_original_full_inventory_pick_failure_is_verified(potion_checker):
    expected = failure_row(902704693)
    assert potion_checker.verified_full_capacity_failure(original_result(expected), expected) is True


@pytest.mark.parametrize("tamper", ["outcome", "seed", "steps", "floor", "hp", "error", "logging_error", "no_tail", "wrong_action", "free_slot", "negative_capacity", "bool_capacity", "missing_inventory", "empty_pick_slots", "empty_pick_mask", "unequal_mask_slots", "not_advertised", "missing_inventory_slot", "empty_inventory_slot"])
def test_capacity_failure_requires_exact_game_step_and_visible_contract(potion_checker, tamper):
    expected = failure_row(902704693)
    result = original_result(expected)
    inventory = result["last_decisions"][-1]["potion_inventory"]
    if tamper in {"seed", "steps", "floor", "hp"}:
        result[tamper] += 1
    elif tamper in {"outcome", "error"}:
        result[tamper] = "different"
    elif tamper == "logging_error":
        result["error_stage"] = "diagnostic_logging"
    elif tamper == "no_tail":
        result["last_decisions"] = []
    elif tamper == "wrong_action":
        result["last_decisions"][-1]["named_action"]["action"] = "skip_potion"
    elif tamper == "free_slot":
        inventory["free_slots"] = [0]
    elif tamper == "negative_capacity":
        inventory["max_slots"] = -1
    elif tamper == "bool_capacity":
        inventory["max_slots"] = True
    elif tamper == "missing_inventory":
        del result["last_decisions"][-1]["potion_inventory"]
    elif tamper == "empty_pick_slots":
        inventory["pick_action_slots"] = []
    elif tamper == "empty_pick_mask":
        inventory["pick_action_mask"] = []
    elif tamper == "unequal_mask_slots":
        inventory["pick_action_mask"] = [True, True]
    elif tamper == "not_advertised":
        inventory["pick_action_mask"] = [False]
    elif tamper == "missing_inventory_slot":
        inventory["slot_potion_ids"] = ["A", "B"]
    elif tamper == "empty_inventory_slot":
        inventory["slot_potion_ids"] = ["A", None, "C"]
    assert potion_checker.verified_full_capacity_failure(result, expected) is False


def test_real_trace_records_same_prefix_and_only_the_masked_capacity_intervention(potion_checker, monkeypatch, tmp_path):
    class PublicRewardEnv(gym.Env):
        scope = "fullrun"
        action_space = gym.spaces.Discrete(2)
        observation_space = gym.spaces.Dict({"visible": gym.spaces.Box(0, 2, (1,), dtype=np.float32)})
        backend_metadata = {"fixture": "public-reward-transport"}

        def __init__(self):
            self.raw_env = self
            player = SimpleNamespace(max_potion_slots=3,
                                     potions=[SimpleNamespace(potion_id=name) for name in ("A", "B", "C")])
            self._mgr = SimpleNamespace(phase="EVENT", PHASE_COMBAT="COMBAT",
                                        run_state=SimpleNamespace(player=player))
            self.position = 0

        def public_slots(self):
            return ([{"action": "continue"}] if self.position == 0 else
                    [{"action": "skip_potion"}, {"action": "pick_potion"}])

        def action_masks(self):
            return np.array([True, self.position > 0], dtype=bool)

        def observation(self):
            return {"visible": np.array([self.position], dtype=np.float32)}

        def info(self):
            return {"phase": self._mgr.phase, "floor": 4, "hp": 30.0,
                    "outcome": "running", "action_mask": self.action_masks()}

        def reset(self, *, seed=None, options=None):
            self.position = 0
            self._mgr.phase = "EVENT"
            return self.observation(), self.info()

        def step(self, action):
            if self.position == 0:
                assert action == 0
                self.position = 1
                self._mgr.phase = "CARD_REWARD"
                return self.observation(), 0.0, False, False, self.info()
            if action == 1:
                raise RuntimeError("Engine rejected advertised legal pick_potion")
            assert action == 0
            self.position = 2
            return self.observation(), -1.0, True, False, {**self.info(), "outcome": "death", "hp": 0.0}

    class PublicMaskedPolicy:
        @staticmethod
        def predict(obs, *, action_masks, deterministic):
            assert deterministic
            return int(np.flatnonzero(action_masks)[-1]), None

    monkeypatch.setattr(potion_checker, "_describe", lambda env, action: {
        "named_action": dict(env.raw_env.public_slots()[action])})
    original_dir, repaired_dir = tmp_path / "original", tmp_path / "capacity_mask"
    original_dir.mkdir()
    repaired_dir.mkdir()
    model = PublicMaskedPolicy()
    original = potion_checker.replay(PublicRewardEnv(), model, 902704693, original_dir)
    repaired = potion_checker.replay(PotionCapacityMask(PublicRewardEnv()), model, 902704693, repaired_dir)
    assert original["outcome"] == "aborted" and original["error_stage"] == "game_step"
    assert original["steps"] == 1 and repaired["steps"] == 2
    assert original["capacity_mask_decisions"] == 0 and repaired["capacity_mask_decisions"] == 1
    assert original["last_decisions"][0] == repaired["last_decisions"][0]
    original_last, repaired_last = original["last_decisions"][-1], repaired["last_decisions"][-1]
    assert original_last["named_action"] == {"action": "pick_potion"}
    assert repaired_last["named_action"] == {"action": "skip_potion"}
    assert original_last["potion_inventory"]["slot_potion_ids"] == ["A", "B", "C"]
    assert original_last["potion_inventory"]["free_slots"] == []
    assert original_last["potion_inventory"]["pick_action_mask"] == [True]
    assert repaired_last["potion_inventory"]["pick_action_mask"] == [False]
    assert repaired["outcome"] == "death" and repaired["error"] is None
    expected = {key: original[key] for key in ("seed", "outcome", "steps", "floor", "hp", "error")}
    assert potion_checker.verified_full_capacity_failure(original, expected) is True
