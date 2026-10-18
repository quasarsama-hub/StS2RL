"""Frozen-policy event repair checks keep weights, evidence and budgets fixed."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, read_json, sha256
from spire2_rl.study import evaluation_seeds

from test_act1_cycle_audit import audit_setup, bytes_in, cycle_auditor


@pytest.fixture(scope="module")
def event_checker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_cycle_event_isolation.py"
    spec = importlib.util.spec_from_file_location("event_check_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mechanics_evidence():
    return {"neow_original": {"shared_event_model": True, "matches_solo": False,
                              "solo_expected_reward": [{"reward": "seed42"}],
                              "interleaved_actual_reward": [{"reward": "seed43"}]},
            "neow_isolated": {"shared_event_model": False, "matches_solo": True,
                              "solo_expected_reward": [{"reward": "seed42"}],
                              "interleaved_actual_reward": [{"reward": "seed42"}]},
            "tablet_original": {"shared_event_model": True, "matches": False,
                                "max_hp_before": [80, 80], "max_hp_after": [77, 74]},
            "tablet_isolated": {"shared_event_model": False, "matches": True,
                                "max_hp_before": [80, 80], "max_hp_after": [77, 77]},
            "global_registry_restored": True, "passed": True}


@pytest.fixture
def event_setup(event_checker, cycle_auditor, audit_setup, monkeypatch):
    checker, setup = event_checker, audit_setup
    # Reuse small serialized evidence, not the expensive completed-cycle fixture.
    monkeypatch.setattr(checker, "cycles", cycle_auditor.cycles)
    monkeypatch.setattr(checker, "strength_checker", cycle_auditor.strength_checker)
    monkeypatch.setitem(checker.behavior, "strength_evidence", cycle_auditor.strength_evidence)
    monkeypatch.setattr(checker, "MaskablePPO", cycle_auditor.MaskablePPO)
    schema, actions = ("public-hand-and-state", 12), ("fixed-public-slots", 512)
    for model in setup.model_states.values():
        model.observation_space, model.action_space = schema, actions
    calls = {"rollouts": [], "envs": [], "proofs": 0, "proof": mechanics_evidence(),
             "isolated_outcome": "death", "mutate": None, "change_space": False}

    class Env:
        def __init__(self):
            self.scope, self.observation_space, self.action_space = "fullrun", schema, actions
            self.backend_metadata = {"backend": "python-emulator", "legacy_interface": "unchanged"}
            self.isolated, self.closed = False, False
            calls["envs"].append(self)

        def close(self):
            self.closed = True

    class Isolated:
        def __init__(self, env):
            self.env = env
            self.scope, self.observation_space, self.action_space = env.scope, env.observation_space, env.action_space
            if calls["change_space"]:
                self.action_space = ("changed-action-slots", 42)
            self.backend_metadata = deepcopy(env.backend_metadata)
            self.backend_metadata["event_state_isolation"] = "private-registry-per-run-v1"
            self.backend_metadata["event_isolation_sha256"] = sha256(Path(checker.__file__).parents[1] / "src/spire2_rl/event_isolation.py")
            self.backend_metadata["event_registry_count"] = 68
            self.isolated = True

        def close(self):
            self.env.close()

    def make(config, upstream):
        assert setup.calls["factory"] == 1
        assert config is setup.parent["config"] and upstream == setup.upstream
        return Env()

    def proof(upstream):
        assert upstream == setup.upstream
        calls["proofs"] += 1
        return deepcopy(calls["proof"])

    def rollout(env, seeds, *, model, label):
        isolated = label == "same-policy-isolated"
        assert env.isolated is isolated
        assert ("event_state_isolation" in env.backend_metadata) is isolated
        calls["rollouts"].append((label, list(seeds), model))
        if calls["mutate"]:
            calls["mutate"]()
        return [{"seed": seed, "policy": label,
                 "outcome": calls["isolated_outcome"] if isolated else "death",
                 "floor": 11 if isolated else 10, "hp": 0,
                 "steps": 40, "return": -1,
                 "error": "game error" if isolated and calls["isolated_outcome"] == "aborted" else None}
                for seed in seeds]

    monkeypatch.setattr(checker, "IsolatedEvents", Isolated)
    monkeypatch.setattr(checker, "mechanics_probe", proof)
    monkeypatch.setattr(runner, "_make", make)
    monkeypatch.setattr(runner, "_rollout", rollout)
    return SimpleNamespace(source=setup, calls=calls)


def run(checker, setup, **kwargs):
    source = setup.source
    return checker.check(source.series, source.upstream, cycle=2,
                         episodes=kwargs.pop("episodes", 2), **kwargs)


def test_repair_replays_exact_same_frozen_model_and_preserves_source(event_checker, event_setup):
    checker, setup = event_checker, event_setup
    source = setup.source
    protected = bytes_in(source.root)
    original_state = {path: deepcopy(model.policy_state) for path, model in source.model_states.items()}
    summary = run(checker, setup)
    assert summary["stage"] == "cycle_event_isolation_check" and summary["status"] == "verified"
    assert summary["same_frozen_checkpoint"] == source.parent["state"]
    assert summary["validation_start"] == 7700
    assert summary["additional_training_steps"] == 0
    assert summary["champion_updated"] is False and summary["training_gate_unchanged"] is True
    original, isolated = setup.calls["rollouts"]
    assert original[0] == "same-policy-original" and isolated[0] == "same-policy-isolated"
    assert original[1] == isolated[1] == evaluation_seeds("validation", 2, 7, start=7700)
    assert original[2] is isolated[2] and original[2].num_timesteps == 323640
    assert summary["matching_episode_outcomes"] == 2 and summary["matching_episode_floors"] == 0
    assert all(bytes_in(source.root)[path] == data for path, data in protected.items())
    for path, values in original_state.items():
        for key, value in values.items():
            torch.testing.assert_close(source.model_states[path].policy_state[key], value, rtol=0, atol=0)
    assert setup.calls["proofs"] == 1 and all(env.closed for env in setup.calls["envs"])
    assert not (source.series / "cycle-0003").exists()
    report = read_json(Path(summary["report_path"]))
    assert report["protocol"]["scope"] == "same-weights event-instance isolation check"
    assert report["protocol"]["repair_sha256"] == sha256(Path(checker.__file__).parents[1] / "src/spire2_rl/event_isolation.py")
    identity = report["protocol"]["isolated_backend"]
    assert identity["legacy_interface"] == "unchanged"
    assert identity["event_state_isolation"] == "private-registry-per-run-v1"
    assert identity["event_isolation_sha256"] == report["protocol"]["repair_sha256"]
    assert identity["event_registry_count"] == 68
    assert report["protocol"]["automatic_training"] is False


def test_cached_event_check_is_read_only_and_does_not_replay(event_checker, event_setup):
    first = run(event_checker, event_setup)
    published = bytes_in(event_setup.source.root)
    second = run(event_checker, event_setup)
    assert first == second and bytes_in(event_setup.source.root) == published
    assert len(event_setup.calls["rollouts"]) == 2 and event_setup.calls["proofs"] == 1
    assert len(event_setup.source.calls["loaded"]) == 1


@pytest.mark.parametrize("change", [{"episodes": 3}, {"start": 7701}])
def test_event_check_rejects_new_sample_after_publication(event_checker, event_setup, change):
    run(event_checker, event_setup)
    published = bytes_in(event_setup.source.root)
    with pytest.raises(ValueError, match="fixed|original|overlap"):
        run(event_checker, event_setup, **change)
    assert bytes_in(event_setup.source.root) == published and len(event_setup.calls["rollouts"]) == 2


@pytest.mark.parametrize("tamper", ["checkpoint", "label", "seed", "statistics"])
def test_event_raw_report_validation_recomputes_evidence(event_checker, event_setup, tamper):
    result = run(event_checker, event_setup)
    report = read_json(Path(result["report_path"]))
    protocol = deepcopy(report["protocol"])
    if tamper == "checkpoint":
        report["protocol"]["checkpoint"]["sha256"] = "different-model"
    elif tamper == "label":
        report["isolated_episodes"][0]["policy"] = "same-policy-original"
    elif tamper == "seed":
        report["original_episodes"][0]["seed"] += 1
    else:
        report["isolated_gameplay"]["mean_floor"] += 5
    with pytest.raises(ValueError, match="evidence|source|schedule"):
        event_checker.validate_report(report, protocol, evaluation_seeds("validation", 2, 7, start=7700))


def test_cached_statistics_tampering_rejected_with_updated_outer_hash(event_checker, event_setup):
    result = run(event_checker, event_setup)
    path = Path(result["report_path"])
    report = read_json(path)
    report["isolated_gameplay"]["mean_floor"] = 99
    atomic_json(path, report)
    completion_path = path.with_name("completion.json")
    completion = read_json(completion_path)
    completion["report_sha256"] = sha256(path)
    atomic_json(completion_path, completion)
    with pytest.raises(ValueError, match="statistics.*evidence"):
        run(event_checker, event_setup)
    assert len(event_setup.calls["rollouts"]) == 2


@pytest.mark.parametrize("outcome", ["aborted", "timeout"])
def test_incomplete_repaired_runs_do_not_verify_repair(event_checker, event_setup, outcome):
    event_setup.calls["isolated_outcome"] = outcome
    summary = run(event_checker, event_setup)
    assert summary["status"] == "inspection_required" and summary["mechanics"]["passed"] is True
    assert summary["isolated"]["aborted" if outcome == "aborted" else "timeouts"] == 2
    assert summary["additional_training_steps"] == 0 and summary["training_gate_unchanged"] is True


def test_failed_mechanics_proof_does_not_verify_repair(event_checker, event_setup):
    event_setup.calls["proof"]["tablet_isolated"]["matches"] = False
    event_setup.calls["proof"]["tablet_isolated"]["max_hp_after"][1] = 74
    event_setup.calls["proof"]["passed"] = False
    summary = run(event_checker, event_setup)
    assert summary["status"] == "inspection_required"
    assert summary["isolated"]["aborted"] == summary["isolated"]["timeouts"] == 0


def test_inconsistent_cached_mechanics_pass_flag_is_rejected(event_checker, event_setup):
    result = run(event_checker, event_setup)
    path = Path(result["report_path"])
    report = read_json(path)
    report["mechanics"]["neow_isolated"]["matches_solo"] = False
    report["mechanics"]["neow_isolated"]["interleaved_actual_reward"] = [{"reward": "wrong-seed"}]
    # Keep passed=True: cache validation must derive success from the raw probe.
    atomic_json(path, report)
    completion_path = path.with_name("completion.json")
    completion = read_json(completion_path)
    completion["report_sha256"] = sha256(path)
    atomic_json(completion_path, completion)
    with pytest.raises(ValueError, match="mechanics|proof|evidence"):
        run(event_checker, event_setup)
    assert len(event_setup.calls["rollouts"]) == 2


@pytest.mark.parametrize("change", ["journal", "counter", "spaces"])
def test_changed_source_or_public_model_contract_stops_check(event_checker, event_setup, change):
    source = event_setup.source
    if change == "journal":
        event_setup.calls["mutate"] = lambda: (source.root / "episodes.jsonl").write_text("changed training journal\n")
    elif change == "counter":
        source.model_states[source.root / source.parent["state"]["model"]].num_timesteps += 1
    else:
        event_setup.calls["change_space"] = True
    with pytest.raises(ValueError, match="files|counter|spaces"):
        run(event_checker, event_setup)
    assert not (source.series / "cycle-0003").exists()
    assert all(env.closed for env in event_setup.calls["envs"])
