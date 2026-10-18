"""Frozen numbered-cycle diagnostics preserve training and fix their evidence."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, read_json, sha256
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds, paired_compare, summarize

from test_act1_continuation import (
    ContinuationPolicy, continuation, load_script, progress_setup, repaired_source, rows,
)
from test_act1_cycles import cycles, cycle_setup, run_cycle
from test_act1_trial import bytes_in


@pytest.fixture
def cycle_checker():
    return load_script("check_act1_cycle.py")


@pytest.fixture
def saved_cycle(cycle_checker, cycles, cycle_setup, monkeypatch):
    setup = cycle_setup
    run_cycle(cycles, setup)
    run_cycle(cycles, setup, 2)
    root = setup.series / "cycle-0002"
    parent = cycles.load_parent(root, setup.upstream)
    monkeypatch.setattr(cycle_checker, "MaskablePPO", ContinuationPolicy)
    calls = []
    outcomes = {"candidate": "death", "baseline": "death"}

    def rollout(env, seeds, *, model, label):
        assert isinstance(env, PotionCapacityMask)
        assert env.backend_metadata["rl_cycle_plan"] == parent["lineage"]["plan"]
        assert env.backend_metadata["rl_cycle_source_checkpoint"] == parent["lineage"]["source_state"]
        calls.append((label, list(seeds), model.num_timesteps))
        candidate = label == "frozen-cycle-candidate"
        result = rows(seeds, label, floor=11 if candidate else 10)
        if result and outcomes["candidate" if candidate else "baseline"] != "death":
            result[0]["outcome"] = outcomes["candidate" if candidate else "baseline"]
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("A frozen cycle check must not train, imitate, or promote")

    monkeypatch.setattr(runner, "_rollout", rollout)
    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "promote", forbidden)
    return SimpleNamespace(setup=setup, root=root, series=setup.series,
                           upstream=setup.upstream, parent=parent, calls=calls,
                           outcomes=outcomes)


def check(cycle_checker, saved, **kwargs):
    return cycle_checker.check(saved.series, saved.upstream, cycle=2,
                               episodes=kwargs.pop("episodes", 1000), **kwargs)


def test_frozen_check_uses_exact_cycle_states_fresh_paired_batches_and_preserves_all_training(
        cycle_checker, saved_cycle):
    saved = saved_cycle
    original_series = bytes_in(saved.series)
    original_source = bytes_in(saved.setup.source)
    factory = runner._make
    summary = check(cycle_checker, saved)
    seeds = evaluation_seeds("validation", 1000, 7, start=4700)
    assert len(saved.calls) == 40
    for index in range(0, len(saved.calls), 2):
        candidate, baseline = saved.calls[index:index + 2]
        assert candidate[0] == "frozen-cycle-candidate"
        assert baseline[0] == "frozen-cycle-baseline"
        assert candidate[1] == baseline[1] == seeds[(index // 2) * 50:(index // 2 + 1) * 50]
        assert candidate[2] == saved.parent["state"]["num_timesteps"] == 81192
        assert baseline[2] == saved.parent["lineage"]["initial_checkpoint"]["num_timesteps"] == 81112
    report_path = Path(summary["report_path"])
    report = read_json(report_path)
    protocol = report["protocol"]
    assert protocol["candidate_checkpoint"] == saved.parent["state"]
    assert protocol["baseline_checkpoint"] == saved.parent["lineage"]["initial_checkpoint"]
    assert protocol["fingerprint"] == saved.parent["manifest"]["fingerprint"]
    assert protocol["episodes"] == 1000 and protocol["seed_start"] == 4700
    assert protocol["inference"] == "deterministic" and protocol["victory_scope"] == "act1"
    cycle_checker.validate_report(report, protocol, seeds)
    assert summary["stage"] == "frozen_cycle_strength_check" and summary["cycle"] == 2
    assert summary["rl_floor_gain"] == 1 and summary["rl_floor_gain_ci95"] == [1, 1]
    assert summary["establishes_positive_floor_gain"] is True
    assert summary["establishes_positive_act1_clear_gain"] is False
    assert summary["additional_training_steps"] == 0
    assert summary["champion_updated"] is False and summary["training_gate_unchanged"] is True
    current_series = bytes_in(saved.series)
    assert all(current_series[name] == value for name, value in original_series.items())
    assert bytes_in(saved.setup.source) == original_source
    assert runner._make is factory
    assert not (saved.series / "cycle-0003").exists()
    assert saved.setup.calls["training"] == [
        (saved.series / "cycle-0001", 75), (saved.root, 75)]


def test_repeated_command_reuses_frozen_evidence_byte_for_byte(cycle_checker, saved_cycle):
    saved = saved_cycle
    first = check(cycle_checker, saved)
    published = bytes_in(saved.series)
    second = check(cycle_checker, saved)
    assert first == second
    assert bytes_in(saved.series) == published
    assert len(saved.calls) == 40


@pytest.mark.parametrize("change", [{"episodes": 1001}, {"start": 4701}])
def test_existing_diagnostic_refuses_changed_sample_or_start(cycle_checker, saved_cycle, change):
    saved = saved_cycle
    check(cycle_checker, saved)
    published = bytes_in(saved.series)
    with pytest.raises(ValueError, match="fixed|original|protocol|different|changed"):
        check(cycle_checker, saved, **change)
    assert bytes_in(saved.series) == published and len(saved.calls) == 40


@pytest.mark.parametrize("start", [400, 500, 700, 1700, 2700, 3700])
def test_explicit_overlap_with_original_or_numbered_evaluation_is_rejected(
        cycle_checker, saved_cycle, start):
    saved = saved_cycle
    original = bytes_in(saved.series)
    with pytest.raises(ValueError, match="overlap|fresh|used"):
        check(cycle_checker, saved, start=start)
    assert bytes_in(saved.series) == original and not saved.calls


def test_fresh_schedule_reserves_other_numbered_cycle_diagnostic_protocols(
        cycle_checker, saved_cycle):
    saved = saved_cycle
    reservation = saved.series / "cycle-0001" / "diagnostics" / "another-checker" / "protocol.json"
    atomic_json(reservation, {"split": "validation", "episodes": 250, "seed_start": 5000})
    seeds, start = cycle_checker.fresh_schedule(saved.root, saved.parent, 1000)
    assert start >= 5250
    assert seeds == evaluation_seeds("validation", 1000, 7, start=start)
    with pytest.raises(ValueError, match="overlap|fresh|used"):
        cycle_checker.fresh_schedule(saved.root, saved.parent, 1000, start=4900)


def test_fresh_schedule_reserves_planned_future_cycle_before_reports_exist(
        cycle_checker, saved_cycle):
    saved = saved_cycle
    future = saved.series / "cycle-0003" / "cycle-lineage.json"
    atomic_json(future, {"plan": {"episodes": 1000, "validation_start": 4700}})
    seeds, start = cycle_checker.fresh_schedule(saved.root, saved.parent, 1000)
    assert start == 5700
    assert seeds == evaluation_seeds("validation", 1000, 7, start=5700)
    with pytest.raises(ValueError, match="overlap|fresh|used"):
        cycle_checker.fresh_schedule(saved.root, saved.parent, 1000, start=4700)


@pytest.mark.parametrize("outcome", ["timeout", "aborted"])
def test_unclean_frozen_run_prevents_both_positive_strength_claims(
        cycle_checker, saved_cycle, outcome):
    saved = saved_cycle
    saved.outcomes["candidate"] = outcome
    summary = check(cycle_checker, saved)
    assert summary["after_rl"]["timeouts" if outcome == "timeout" else "aborted"] > 0
    assert summary["establishes_positive_floor_gain"] is False
    assert summary["establishes_positive_act1_clear_gain"] is False
    assert summary["training_gate_unchanged"] is True
    assert read_json(saved.root / "cycle-progress.json")["next_cycle_eligible"] is True
    assert not (saved.series / "cycle-0003").exists()


@pytest.mark.parametrize("tamper", ["candidate_model", "baseline_model", "training_summary", "training_raw"])
def test_changed_training_evidence_or_frozen_model_rejected_before_rollout(
        cycle_checker, saved_cycle, tamper):
    saved = saved_cycle
    if tamper.endswith("model"):
        state = saved.parent["state"] if tamper == "candidate_model" else saved.parent["lineage"]["initial_checkpoint"]
        (saved.root / state["model"]).write_bytes(b"changed frozen weights")
    else:
        path = saved.root / ("cycle-progress.json" if tamper == "training_summary" else "after_rl.json")
        report = read_json(path)
        if tamper == "training_summary":
            report["after_rl"]["mean_floor"] += 1
        else:
            report["episodes"][0]["floor"] += 1
        atomic_json(path, report)
    with pytest.raises(ValueError):
        check(cycle_checker, saved)
    assert not saved.calls


@pytest.mark.parametrize("tamper", ["summary", "label", "seed", "order", "protocol", "paired", "clears"])
def test_validation_recomputes_raw_episodes_and_exact_frozen_protocol(
        cycle_checker, saved_cycle, tamper):
    saved = saved_cycle
    summary = check(cycle_checker, saved)
    report = read_json(Path(summary["report_path"]))
    protocol = deepcopy(report["protocol"])
    seeds = evaluation_seeds("validation", 1000, 7, start=protocol["seed_start"])
    if tamper == "summary":
        report["candidate"]["mean_floor"] += 1
    elif tamper == "label":
        report["candidate_episodes"][0]["policy"] = "different-policy"
    elif tamper == "seed":
        report["candidate_episodes"][0]["seed"] = -1
    elif tamper == "order":
        report["baseline_episodes"][:2] = report["baseline_episodes"][1::-1]
    elif tamper == "protocol":
        report["protocol"]["candidate_checkpoint"]["sha256"] = "different"
    elif tamper == "paired":
        report["comparison"]["floor_delta_ci95"] = [2, 2]
    else:
        report["paired_clear_evidence"]["supports_more_clears"] = True
    with pytest.raises(ValueError):
        cycle_checker.validate_report(report, protocol, seeds)


def test_cached_report_tampering_is_rejected_even_with_refreshed_outer_hash(
        cycle_checker, saved_cycle):
    saved = saved_cycle
    summary = check(cycle_checker, saved)
    report_path = Path(summary["report_path"])
    report = read_json(report_path)
    report["candidate"]["mean_floor"] += 1
    atomic_json(report_path, report)
    completion_path = report_path.with_name("completion.json")
    completion = read_json(completion_path)
    for key, value in completion.items():
        if key.endswith("sha256") and value != sha256(report_path):
            # Only report checksum entries are adjusted; the raw consistency check must still reject.
            if "report" in key:
                completion[key] = sha256(report_path)
    atomic_json(completion_path, completion)
    with pytest.raises(ValueError, match="raw|summary|report|evidence|checksum"):
        check(cycle_checker, saved)
    assert len(saved.calls) == 40


def test_model_counter_mismatch_is_rejected_even_when_checkpoint_files_are_valid(
        cycle_checker, saved_cycle, monkeypatch):
    saved = saved_cycle

    class WrongCounter(ContinuationPolicy):
        @classmethod
        def load(cls, path, **kwargs):
            model = super().load(path, **kwargs)
            model.num_timesteps += 1
            return model

    monkeypatch.setattr(cycle_checker, "MaskablePPO", WrongCounter)
    with pytest.raises(ValueError, match="counter"):
        check(cycle_checker, saved)
    assert not saved.calls


@pytest.mark.parametrize("episodes,unclean", [(1000, False), (999, False), (1000, True)])
def test_view_uses_exact_clear_test_and_requires_enough_clean_pairs(
        cycle_checker, tmp_path, episodes, unclean):
    seeds = evaluation_seeds("validation", episodes, 7, start=4700)
    candidate = rows(seeds, "frozen-cycle-candidate", floor=11)
    baseline = rows(seeds, "frozen-cycle-baseline", floor=10)
    for row in candidate[:20]:
        row["outcome"] = "win"
    for row in baseline[:5]:
        row["outcome"] = "win"
    if unclean:
        candidate[-1]["outcome"] = "aborted"
    report = {"protocol": {"cycle": 2, "candidate_checkpoint": {"sha256": "frozen"}, "seed_start": 4700},
              "candidate": summarize(candidate), "baseline": summarize(baseline),
              "comparison": paired_compare(candidate, baseline),
              "paired_clear_evidence": cycle_checker.checker["exact_win_test"](candidate, baseline)}
    result = cycle_checker.view(report, tmp_path / "report.json")
    expected = episodes == 1000 and not unclean
    assert result["establishes_positive_floor_gain"] is expected
    assert result["establishes_positive_act1_clear_gain"] is expected
    assert result["paired_clear_evidence"]["supports_more_clears"] is True
    assert result["additional_training_steps"] == 0 and result["training_gate_unchanged"] is True


def test_rollout_failure_keeps_fixed_protocol_for_same_sample_retry(
        cycle_checker, saved_cycle, monkeypatch):
    saved = saved_cycle
    original_rollout = runner._rollout
    original_files = bytes_in(saved.series)

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted frozen evaluation")

    monkeypatch.setattr(runner, "_rollout", interrupted)
    with pytest.raises(RuntimeError, match="interrupted frozen"):
        check(cycle_checker, saved)
    protocols = list(saved.root.glob("diagnostics/cycle-strength-*/protocol.json"))
    assert len(protocols) == 1
    assert read_json(protocols[0])["seed_start"] == 4700
    assert not protocols[0].with_name("report.json").exists()
    assert all(bytes_in(saved.series)[name] == value for name, value in original_files.items())
    monkeypatch.setattr(runner, "_rollout", original_rollout)
    result = check(cycle_checker, saved)
    assert result["validation_start"] == 4700 and len(saved.calls) == 40


@pytest.mark.parametrize("invalid", [{"cycle": 0}, {"cycle": -1}, {"episodes": 999}, {"start": -1}])
def test_invalid_diagnostic_arguments_never_create_an_experiment(cycle_checker, tmp_path, invalid):
    series = tmp_path / "absent"
    with pytest.raises(ValueError):
        cycle_checker.check(series, tmp_path / "upstream", **{"cycle": 2, "episodes": 1000, **invalid})
    assert not series.exists()
