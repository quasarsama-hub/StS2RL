"""Fast behavior-audit checks without rebuilding PPO or large strength fixtures."""

from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, read_json, sha256
from spire2_rl.study import evaluation_seeds


@pytest.fixture(scope="module")
def cycle_auditor():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_act1_cycle.py"
    spec = importlib.util.spec_from_file_location("behavior_audit_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bytes_in(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file() and path.name != ".lock"}


def audit_row(seed, label, *, outcome="death", errors=None):
    return {
        "seed": seed, "policy": label, "outcome": outcome,
        "floor": 17 if outcome == "win" else 10, "hp": 20 if outcome == "win" else 0,
        "steps": 30, "return": 1 if outcome == "win" else -1,
        "error": "simulator failure" if outcome == "aborted" else None,
        "error_stage": "game_step" if outcome == "aborted" else None,
        "act1_cleared": outcome == "win", "phase_decisions": {"combat": 30},
        "metrics": {
            "end_turns": 3, "end_turns_with_playable_cards": 1,
            "unused_energy_sum": 3, "combat_wins": 2,
            "potions_left_at_death": 1 if outcome == "death" else None,
            "legal_potion_slots_before_death": 1 if outcome == "death" else None,
        },
        "audit": {
            "macro_action_counts": {"move": 2, "rest_option": 1},
            "map_room_choices": {"MONSTER": 1, "ELITE": 1},
            "picked_card_counts": {"BASH": 1}, "rest_option_counts": {"HEAL": 1},
            "combat_entries": [{"room_type": "ELITE", "hp": 35},
                               {"room_type": "BOSS", "hp": 20}],
            "death_context": {"room_type": "ELITE", "hp_entering_fight": 35}
                             if outcome == "death" else None,
            "flagged_decisions": [{"kind": "end_turn_with_cards_or_energy", "step": 5}],
            "observer_errors": errors or [],
        },
    }


@pytest.fixture
def audit_setup(cycle_auditor, monkeypatch, tmp_path):
    """Use real diagnostic hashes and stub the already-tested learning integration."""
    audit = cycle_auditor
    series, upstream = tmp_path / "cycles", tmp_path / "upstream"
    root = series / "cycle-0002"
    root.mkdir(parents=True)
    fingerprint = "fixed-cycle-interface"
    initial, final = [], []
    model_states = {}
    for label, counter, destination in (("previous", 242744, initial), ("current", 323640, final)):
        path = root / "checkpoints" / f"{label}.zip"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"frozen {label} PPO weights and Adam".encode())
        state = {"model": str(path.relative_to(root)), "num_timesteps": counter,
                 "sha256": sha256(path), "fingerprint": fingerprint, "interrupted": False}
        destination.append(state)
        model_states[path] = SimpleNamespace(num_timesteps=counter,
                                            policy_state={"policy": torch.tensor([counter]),
                                                          "adam_moment": torch.tensor([0.7])})
    initial, final = initial[0], final[0]
    lineage = {"cycle": 2, "initial_checkpoint": initial,
               "source_state": initial, "plan": {"episodes": 1000, "validation_start": 3700}}
    atomic_json(root / "cycle-lineage.json", lineage)
    atomic_json(root / "cycle-progress.json", {"next_cycle_eligible": False})
    atomic_json(root / "latest.json", final)
    (root / "episodes.jsonl").write_text('{"outcome":"death"}\n')
    (root / "sessions.jsonl").write_text('{"status":"completed"}\n')
    atomic_json(root / "champion.json", {"sha256": "old-champion"})
    parent = {"lineage": lineage, "state": final,
              "manifest": {"fingerprint": fingerprint}, "base_factory": {"source_lineage": {}},
              "config": SimpleNamespace(max_acts=1, scope="fullrun", experiment_seed=7),
              "before": {"seed_start": 3700, "episodes": []},
              "after": {"seed_start": 3700, "episodes": []},
              "source_files": {str(path): sha256(path) for path in root.rglob("*") if path.is_file()}}
    strength_dir = root / "diagnostics" / f"cycle-strength-{final['sha256'][:16]}"
    protocol = {"cycle": 2, "fingerprint": fingerprint, "candidate_checkpoint": final,
                "baseline_checkpoint": initial, "episodes": 3000, "seed_start": 4700,
                "split": "validation", "inference": "deterministic", "victory_scope": "act1",
                "checker_sha256": sha256(Path(audit.__file__).with_name("check_act1_cycle.py"))}
    atomic_json(strength_dir / "protocol.json", protocol)
    atomic_json(strength_dir / "report.json", {"protocol": protocol, "comparison": {"excluded_pairs": 0}})
    atomic_json(strength_dir / "completion.json", {
        "protocol_sha256": sha256(strength_dir / "protocol.json"),
        "report_sha256": sha256(strength_dir / "report.json")})
    calls = {"played": [], "loaded": [], "factory": 0, "closed": 0,
             "outcomes": {}, "observer_errors": {}, "mutate": None}

    def validate_strength(report, expected, seeds):
        assert report["protocol"] == expected == protocol
        assert seeds == evaluation_seeds("validation", 3000, 7, start=4700)

    monkeypatch.setitem(audit.strength_checker, "validate_report", validate_strength)
    monkeypatch.setitem(audit.strength_checker["checker"], "fresh_seeds",
                        lambda root, original, config, count, start:
                        evaluation_seeds("validation", count, config.experiment_seed, start=start))
    monkeypatch.setitem(audit.cycles, "load_parent", lambda expected_root, expected_upstream: parent)

    @contextmanager
    def factory(base, plan, source):
        assert base == parent["base_factory"] and plan == lineage["plan"] and source == initial
        calls["factory"] += 1
        try:
            yield
        finally:
            calls["factory"] -= 1

    monkeypatch.setitem(audit.cycles, "cycle_runner", factory)

    class Policy:
        @classmethod
        def load(cls, path, **kwargs):
            assert kwargs == {"device": "cpu"}
            calls["loaded"].append(Path(path))
            return model_states[Path(path)]

    class Teacher:
        def __init__(self):
            self.bot = SimpleNamespace(use_lookahead=False)

    class Env:
        def close(self):
            calls["closed"] += 1

    def make(config, path):
        assert calls["factory"] == 1 and config is parent["config"] and path == upstream
        return Env()

    def play(env, seeds, *, model, label, controller=None):
        assert isinstance(env, audit.CycleAuditObserver)
        calls["played"].append((label, list(seeds), model, controller))
        if calls["mutate"]:
            calls["mutate"]()
        outcomes = calls["outcomes"].get(label, ["win" if label == "current" else "death", "death"])
        return [audit_row(seed, label, outcome=outcomes[index % len(outcomes)],
                          errors=calls["observer_errors"].get(label)) for index, seed in enumerate(seeds)]

    def forbidden(*args, **kwargs):
        pytest.fail("A behavior audit must not train or promote")

    monkeypatch.setattr(audit, "MaskablePPO", Policy)
    monkeypatch.setattr(audit, "FairTeacher", Teacher)
    monkeypatch.setattr(audit, "play_controller", play)
    monkeypatch.setattr(runner, "_make", make)
    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "promote", forbidden)
    return SimpleNamespace(series=series, root=root, upstream=upstream, parent=parent,
                           calls=calls, model_states=model_states, strength_dir=strength_dir)


def run(audit, setup, **kwargs):
    return audit.audit(setup.series, setup.upstream, cycle=2, episodes=kwargs.pop("episodes", 2), **kwargs)


def test_five_frozen_controllers_share_fresh_schedule_and_preserve_learning(cycle_auditor, audit_setup):
    audit, setup = cycle_auditor, audit_setup
    protected = bytes_in(setup.root)
    states = {path: deepcopy(model.policy_state) for path, model in setup.model_states.items()}
    summary = run(audit, setup)
    assert summary["stage"] == "act1_cycle_behavior_audit" and summary["status"] == "completed"
    assert summary["validation_start"] == 7700 and summary["episodes_per_controller"] == 2
    assert summary["additional_training_steps"] == 0
    assert summary["champion_updated"] is False and summary["training_gate_unchanged"] is True
    seeds = evaluation_seeds("validation", 2, 7, start=7700)
    calls = setup.calls["played"]
    assert [call[0] for call in calls] == list(audit.LABELS)
    assert all(call[1] == seeds for call in calls)
    assert calls[0][2].num_timesteps == 323640 and calls[1][2].num_timesteps == 242744
    assert calls[0][3] is calls[1][3] is None
    assert calls[2][2] is calls[3][2] is calls[4][2] is None
    for call in calls[3:]:
        assert call[3].model is calls[0][2] and call[3].mode == call[0]
    assert all(bytes_in(setup.root)[path] == value for path, value in protected.items())
    for path, original in states.items():
        for name, value in original.items():
            torch.testing.assert_close(setup.model_states[path].policy_state[name], value, rtol=0, atol=0)
    assert setup.calls["closed"] == 1 and setup.calls["factory"] == 0
    assert not (setup.series / "cycle-0003").exists()
    report = read_json(Path(summary["report_path"]))
    assert report["protocol"]["candidate_checkpoint"] == setup.parent["state"]
    assert report["protocol"]["baseline_checkpoint"] == setup.parent["lineage"]["initial_checkpoint"]
    assert report["protocol"]["use_lookahead"] is False


def test_completed_audit_repeats_without_replaying_or_rewriting(cycle_auditor, audit_setup):
    first = run(cycle_auditor, audit_setup)
    published = bytes_in(audit_setup.root)
    second = run(cycle_auditor, audit_setup)
    assert second == first and bytes_in(audit_setup.root) == published
    assert len(audit_setup.calls["played"]) == 5 and len(audit_setup.calls["loaded"]) == 2


@pytest.mark.parametrize("changed", [{"episodes": 3}, {"start": 7701}])
def test_published_audit_schedule_is_fixed(cycle_auditor, audit_setup, changed):
    run(cycle_auditor, audit_setup)
    original = bytes_in(audit_setup.root)
    with pytest.raises(ValueError, match="fixed|original|overlap"):
        run(cycle_auditor, audit_setup, **changed)
    assert bytes_in(audit_setup.root) == original and len(audit_setup.calls["played"]) == 5


@pytest.mark.parametrize("phase,pending", [("combat", False), ("combat", True), ("event", True), ("map", False)])
def test_hybrids_switch_only_on_ordinary_combat_and_pass_actual_public_observation(
        cycle_auditor, phase, pending):
    obs = {"hand": np.array([1, 2]), "public_state": np.array([3])}
    masks = np.array([False, True, True, False])
    predictions, teacher_calls = [], []

    class Model:
        def predict(self, supplied_obs, *, action_masks, deterministic):
            assert supplied_obs is obs and action_masks is masks and deterministic is True
            predictions.append(True)
            return np.int64(1), None

    class Teacher:
        def select(self, supplied_env):
            assert supplied_env is env
            teacher_calls.append(True)
            return 2

    combat = SimpleNamespace(pending_choice=object() if pending else None)
    mgr = SimpleNamespace(phase=phase, PHASE_COMBAT="combat", get_combat_state=lambda: combat)
    env = SimpleNamespace(raw_env=SimpleNamespace(_mgr=mgr), current_observation=obs,
                          action_masks=lambda: masks)
    ordinary = phase == "combat" and not pending
    assert cycle_auditor.HybridController(Model(), Teacher(), "teacher_combat").select(env) == (2 if ordinary else 1)
    assert cycle_auditor.HybridController(Model(), Teacher(), "teacher_noncombat").select(env) == (1 if ordinary else 2)
    assert len(predictions) == len(teacher_calls) == 1
    with pytest.raises(ValueError, match="Unknown"):
        cycle_auditor.HybridController(Model(), Teacher(), "all_choices")


@pytest.mark.parametrize("tamper", ["protocol", "label", "seed", "clear", "summary"])
def test_raw_report_validation_recomputes_all_controller_evidence(cycle_auditor, audit_setup, tamper):
    summary = run(cycle_auditor, audit_setup)
    report = read_json(Path(summary["report_path"]))
    protocol = deepcopy(report["protocol"])
    seeds = evaluation_seeds("validation", 2, 7, start=7700)
    if tamper == "protocol":
        report["protocol"]["candidate_checkpoint"]["sha256"] = "replaced-policy"
    elif tamper == "label":
        report["episodes"]["current"][0]["policy"] = "teacher_combat"
    elif tamper == "seed":
        report["episodes"]["previous"][0]["seed"] += 1
    elif tamper == "clear":
        report["episodes"]["current"][0]["act1_cleared"] = False
    else:
        report["controllers"]["teacher_noncombat"]["boss_arrivals"] += 1
    with pytest.raises(ValueError, match="evidence|controllers|models"):
        cycle_auditor.validate_report(report, protocol, seeds)


def test_cached_summary_tampering_rejected_even_with_updated_completion(cycle_auditor, audit_setup):
    result = run(cycle_auditor, audit_setup)
    path = Path(result["report_path"]).with_name("summary.json")
    summary = read_json(path)
    summary["controllers"]["current"]["gameplay"]["act1_clears"] += 1
    atomic_json(path, summary)
    completion_path = path.with_name("completion.json")
    completion = read_json(completion_path)
    completion["summary_sha256"] = sha256(path)
    atomic_json(completion_path, completion)
    with pytest.raises(ValueError, match="summary.*raw"):
        run(cycle_auditor, audit_setup)
    assert len(audit_setup.calls["played"]) == 5


@pytest.mark.parametrize("error", ["aborted", "timeout", "observer"])
def test_incomplete_gameplay_and_observer_failures_are_disclosed(cycle_auditor, audit_setup, error):
    if error == "observer":
        audit_setup.calls["observer_errors"]["current"] = [{"stage": "snapshot", "error": "missing public field"}]
    else:
        audit_setup.calls["outcomes"]["current"] = [error, "death"]
    summary = run(cycle_auditor, audit_setup)
    assert summary["status"] == "diagnostic_errors"
    current = summary["controllers"]["current"]
    if error == "observer":
        assert current["observer_error_runs"] == 2
    else:
        assert current["gameplay"]["aborted" if error == "aborted" else "timeouts"] == 1
    assert summary["training_gate_unchanged"] is True and summary["additional_training_steps"] == 0


@pytest.mark.parametrize("mutation", ["missing_strength", "changed_strength", "frozen_model", "wrong_counter", "journal"])
def test_incompatible_frozen_evidence_stops_without_training(cycle_auditor, audit_setup, mutation):
    setup = audit_setup
    if mutation == "missing_strength":
        (setup.strength_dir / "completion.json").unlink()
    elif mutation == "changed_strength":
        (setup.strength_dir / "report.json").write_text("{}")
    elif mutation == "frozen_model":
        (setup.root / setup.parent["state"]["model"]).write_bytes(b"replaced frozen policy")
    elif mutation == "wrong_counter":
        setup.model_states[next(path for path in setup.model_states if path.name == "current.zip")].num_timesteps += 1
    else:
        setup.calls["mutate"] = lambda: (setup.root / "episodes.jsonl").write_text("changed journal\n")
    with pytest.raises(ValueError, match="strength|counter|files|checkpoint"):
        run(cycle_auditor, setup)
    assert not (setup.series / "cycle-0003").exists()
    if mutation != "journal":
        assert not setup.calls["played"]


def test_act1_wins_and_arrival_metrics_derive_from_completed_rows(cycle_auditor):
    rows = [audit_row(1, "current", outcome="win"), audit_row(2, "current"),
            audit_row(3, "current", outcome="aborted")]
    summary = cycle_auditor.summarize_rows(rows)
    assert summary["act1_clears"] == summary["gameplay"]["outcomes"]["win"] == 1
    assert summary["metrics_episodes"] == 2 and summary["gameplay"]["aborted_rate"] == pytest.approx(1 / 3)
    assert summary["death_room_counts"] == {"ELITE": 1}
    assert summary["elite_arrivals"] == summary["boss_arrivals"] == 3
    assert summary["mean_hp_entering_elites"] == 35 and summary["mean_hp_entering_bosses"] == 20


def test_play_controller_preserves_game_outcome_and_copies_each_observer_episode(cycle_auditor, monkeypatch):
    env = SimpleNamespace(episode_audit=None)
    episode_data = []

    def play(supplied_env, seeds, *, model, label, controller):
        seed, = seeds
        assert supplied_env is env
        outcome = "win" if seed == 1 else "death"
        row = audit_row(seed, label, outcome=outcome)
        row["act1_cleared"] = False  # Frozen historical helper counts act transitions.
        env.episode_audit = row.pop("audit")
        episode_data.append(env.episode_audit)
        return [row]

    monkeypatch.setitem(cycle_auditor.auditor, "run_policy", play)
    rows = cycle_auditor.play_controller(env, [1, 2], model=object(), label="current")
    assert [row["act1_cleared"] for row in rows] == [True, False]
    episode_data[0]["macro_action_counts"]["move"] = 99
    assert rows[0]["audit"]["macro_action_counts"]["move"] == 2
