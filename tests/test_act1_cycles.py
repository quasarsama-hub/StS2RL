"""Numbered PPO cycles keep their budgets, learned state and evidence fixed."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, commit_checkpoint, read_json, sha256, verified_checkpoint
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds, paired_compare, summarize

from test_act1_continuation import (
    ContinuationPolicy, assessment, assert_optimizer_equal, continuation, load_script,
    progress_setup, repaired_source, rows,
)
from test_act1_trial import StoredPolicy, bytes_in


@pytest.fixture
def cycles():
    return load_script("act1_cycles.py")


@pytest.fixture
def cycle_setup(cycles, continuation, progress_setup, monkeypatch, tmp_path):
    """Build a completed repaired parent using real serialized PPO Adam moments."""
    setup, old = progress_setup, progress_setup.old
    result = continuation.progress(old.source, setup.destination, old.upstream,
                                   repair_check=old.repair_path, **old.plan)
    assert result["establishes_positive_floor_gain"]
    source = setup.destination
    # Seed reservations may advance beyond the last published checkpoint.
    atomic_json(source / "seed-cursors" / "worker-0.json", {"next_index": 40})
    atomic_json(source / "seed-cursors" / "worker-1.json", {"next_index": 45})
    monkeypatch.setattr(cycles, "MaskablePPO", ContinuationPolicy)
    calls = {"training": [], "assessments": [], "after_floor": 11}

    def forbidden(*args, **kwargs):
        pytest.fail("A PPO cycle must not clone, reset an optimizer or promote")

    monkeypatch.setitem(cycles.trial["selection"], "fresh_policy", forbidden)
    creation_globals = cycles.trial["act1"]["create_experiment"].__globals__
    monkeypatch.setitem(creation_globals, "collect_demonstrations", forbidden)
    monkeypatch.setitem(creation_globals, "behavior_clone", forbidden)

    def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
        continuation.trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
        calls["assessments"].append((Path(root), label, episodes, validation_start, deepcopy(state)))
        seeds = evaluation_seeds("validation", episodes, 7, start=validation_start)
        return assessment(manifest, state, label, seeds, validation_start,
                          floor=10 if label == "before_rl" else calls["after_floor"])

    monkeypatch.setattr(cycles, "assess", assess)

    def train(root, config, upstream, *, steps):
        calls["training"].append((Path(root), steps))
        manifest = read_json(root / "experiment.json")
        path, latest = verified_checkpoint(root, manifest["fingerprint"])
        model = StoredPolicy.load(path)
        rollout = config.n_envs * config.n_steps
        model.num_timesteps += ((steps + rollout - 1) // rollout) * rollout
        model.policy(torch.ones(1, 1)).sum().backward()
        model.policy.optimizer.step()
        commit_checkpoint(model, root, [SimpleNamespace(next_index=i + config.n_envs)
                                       for i in latest["next_indices"]], manifest["fingerprint"])
        return {"status": "completed", "total_steps": model.num_timesteps}

    monkeypatch.setattr(runner, "train", train)
    return SimpleNamespace(source=source, series=tmp_path / "cycles", upstream=old.upstream,
                           old=old, config=old.config, calls=calls)


def run_cycle(cycles, setup, number=1, **kwargs):
    return cycles.progress(setup.source, setup.series, setup.upstream, cycle=number,
                           new_steps=kwargs.pop("new_steps", 75),
                           episodes=kwargs.pop("episodes", 1000), **kwargs)


def test_first_cycle_preserves_weights_optimizer_counter_and_reserved_cursor(cycles, cycle_setup):
    setup = cycle_setup
    source_before = bytes_in(setup.source)
    ancestor_before = bytes_in(setup.old.source)
    parent = cycles.load_parent(setup.source, setup.upstream)
    root = setup.series / "cycle-0001"
    lineage = cycles.create_cycle(setup.source, root, setup.upstream, cycle=1,
                                 base_source=setup.source, series=setup.series,
                                 new_steps=75, episodes=1000)
    manifest = read_json(root / "experiment.json")
    imported = lineage["initial_checkpoint"]
    assert lineage["source_state"] == parent["state"]
    assert lineage["base_factory"] == parent["base_factory"]
    assert lineage["plan"] == {"new_steps": 75, "episodes": 1000, "validation_start": 2700}
    assert imported["num_timesteps"] == parent["state"]["num_timesteps"] == 81032
    assert lineage["target_total_steps"] == 81107
    assert imported["next_indices"] == [40, 45]
    assert imported["fingerprint"] == manifest["fingerprint"] != parent["manifest"]["fingerprint"]
    original = StoredPolicy.load(parent["checkpoint"])
    copied = StoredPolicy.load(root / imported["model"])
    for name, value in original.policy.state_dict().items():
        torch.testing.assert_close(copied.policy.state_dict()[name], value, rtol=0, atol=0)
    assert_optimizer_equal(original, copied)
    assert bytes_in(setup.source) == source_before and bytes_in(setup.old.source) == ancestor_before
    assert not (root / "champion.json").exists() and not (root / "demonstrations.npz").exists()
    assert setup.calls["training"] == []


def test_completed_cycle_repeats_without_spending_more_budget(cycles, cycle_setup):
    setup = cycle_setup
    before = bytes_in(setup.source)
    first = run_cycle(cycles, setup)
    root = setup.series / "cycle-0001"
    after = bytes_in(root)
    second = run_cycle(cycles, setup)
    assert setup.calls["training"] == [(root, 75)]
    assert first["new_rl_steps"] == second["new_rl_steps"] == 80
    assert first["champion_updated"] is second["champion_updated"] is False
    assert first["establishes_positive_floor_gain"] is second["establishes_positive_floor_gain"] is True
    assert verified_checkpoint(root)[1]["num_timesteps"] == 81112
    assert bytes_in(setup.source) == before
    assert bytes_in(root) == after
    assert len(setup.calls["assessments"]) == 2
    assert all(call[2:4] == (1000, 2700) for call in setup.calls["assessments"])


@pytest.mark.parametrize("artifact", ["cycle-evidence.json", "cycle-progress.json"])
def test_completed_artifact_tampering_is_rejected_without_new_training(cycles, cycle_setup, artifact):
    setup = cycle_setup
    run_cycle(cycles, setup)
    root = setup.series / "cycle-0001"
    path = root / artifact
    value = read_json(path)
    if artifact == "cycle-evidence.json":
        value["training"]["total_steps"] -= 1
    else:
        value["next_cycle_eligible"] = False
    atomic_json(path, value)
    with pytest.raises(ValueError, match="changed|checksum|evidence|summary"):
        run_cycle(cycles, setup)
    assert setup.calls["training"] == [(root, 75)]


def test_unpublished_summary_recovers_without_rewriting_training_evidence(cycles, cycle_setup):
    setup = cycle_setup
    first = run_cycle(cycles, setup)
    root = setup.series / "cycle-0001"
    evidence_bytes = (root / "cycle-evidence.json").read_bytes()
    lineage_path = root / "cycle-lineage.json"
    lineage = read_json(lineage_path)
    lineage = {key: value for key, value in lineage.items()
               if not (key.endswith("_sha256") and key not in {"before_rl_sha256", "after_rl_sha256"})}
    atomic_json(lineage_path, lineage)
    (root / "cycle-progress.json").unlink()
    second = run_cycle(cycles, setup)
    assert first == second
    assert (root / "cycle-evidence.json").read_bytes() == evidence_bytes
    assert setup.calls["training"] == [(root, 75)]


def test_second_cycle_chains_latest_state_and_fresh_seeds_without_nested_factory(cycles, cycle_setup):
    setup = cycle_setup
    first = run_cycle(cycles, setup)
    first_root = setup.series / "cycle-0001"
    first_lineage = read_json(first_root / "cycle-lineage.json")
    first_before = bytes_in(first_root)
    source_before = bytes_in(setup.source)
    second = run_cycle(cycles, setup, 2)
    second_root = setup.series / "cycle-0002"
    second_lineage = read_json(second_root / "cycle-lineage.json")
    assert second_lineage["source_experiment"] == str(first_root.resolve())
    assert second_lineage["base_source"] == str(setup.source.resolve())
    assert second_lineage["source_state"] == first["checkpoint"]
    assert second_lineage["base_factory"] == first_lineage["base_factory"]
    assert second_lineage["plan"] == {"new_steps": 75, "episodes": 1000, "validation_start": 3700}
    assert second_lineage["initial_checkpoint"]["num_timesteps"] == 81112
    assert second["checkpoint"]["num_timesteps"] == 81192
    assert second["checkpoint"]["fingerprint"] != first["checkpoint"]["fingerprint"]
    assert_optimizer_equal(StoredPolicy.load(first_root / first["checkpoint"]["model"]),
                           StoredPolicy.load(second_root / second_lineage["initial_checkpoint"]["model"]))
    first_rows = read_json(first_root / "after_rl.json")["episodes"]
    second_rows = read_json(second_root / "before_rl.json")["episodes"]
    assert set(row["seed"] for row in first_rows).isdisjoint(row["seed"] for row in second_rows)
    assert bytes_in(first_root) == first_before and bytes_in(setup.source) == source_before
    assert setup.calls["training"] == [(first_root, 75), (second_root, 75)]
    assert run_cycle(cycles, setup, 2) == second
    assert len(setup.calls["training"]) == 2


def test_second_cycle_reconstructs_exactly_one_capacity_wrapper_and_restores_factory(cycles, cycle_setup):
    setup = cycle_setup
    run_cycle(cycles, setup)
    run_cycle(cycles, setup, 2)
    lineage = read_json(setup.series / "cycle-0002" / "cycle-lineage.json")
    factory = runner._make
    with pytest.raises(RuntimeError, match="inside cycle"):
        with cycles.cycle_runner(lineage["base_factory"], lineage["plan"], lineage["source_state"]):
            env = runner._make(setup.config, setup.upstream)
            layers, layer = [], env
            while hasattr(layer, "env"):
                layers.append(layer)
                layer = layer.env
            assert sum(isinstance(layer, PotionCapacityMask) for layer in layers) == 1
            assert env.backend_metadata["continuation_plan"] == setup.old.plan
            assert env.backend_metadata["trial_plan"] == setup.old.old_plan
            assert env.backend_metadata["continuation_source_checkpoint"] == setup.old.current
            assert env.backend_metadata["rl_cycle_plan"] == lineage["plan"]
            assert env.backend_metadata["rl_cycle_source_checkpoint"] == lineage["source_state"]
            assert env.observation_space == ContinuationPolicy.observation_space
            assert env.action_space == ContinuationPolicy.action_space
            env.close()
            raise RuntimeError("inside cycle")
    assert runner._make is factory


def test_real_cycle_assessment_uses_same_batched_repaired_public_interface(cycles, cycle_setup,
                                                                         monkeypatch):
    setup = cycle_setup
    root = setup.series / "cycle-0001"
    lineage = cycles.create_cycle(setup.source, root, setup.upstream, cycle=1,
                                 base_source=setup.source, series=setup.series,
                                 new_steps=75, episodes=1000)
    manifest = read_json(root / "experiment.json")
    calls = []

    def rollout(env, seeds, *, model, label):
        assert isinstance(env, PotionCapacityMask)
        assert env.backend_metadata["rl_cycle_plan"] == lineage["plan"]
        calls.append((label, list(seeds), model.num_timesteps))
        return rows(seeds, label, floor=10)

    monkeypatch.setattr(runner, "_rollout", rollout)
    monkeypatch.setitem(cycles.continued["assess"].__globals__, "MaskablePPO", ContinuationPolicy)
    with cycles.cycle_runner(lineage["base_factory"], lineage["plan"], lineage["source_state"]):
        for label in ("before_rl", "after_rl"):
            result = cycles.continued["assess"](root, setup.upstream, manifest, lineage["initial_checkpoint"],
                                               label=label, episodes=1000, validation_start=2700)
            cycles.trial["act1"]["validate_assessment"](result, manifest, lineage["initial_checkpoint"],
                                                        label, 1000, 2700)
    assert len(calls) == 40
    for before, after in zip(calls[:20], calls[20:]):
        assert before[0] == "before_rl" and after[0] == "after_rl"
        assert before[1:] == after[1:]
        assert before[2] == 81032


@pytest.mark.parametrize("number", [0, -1, 2, 3])
def test_invalid_or_missing_predecessor_never_allocates_training(cycles, cycle_setup, number):
    setup = cycle_setup
    before = bytes_in(setup.source)
    with pytest.raises(ValueError):
        run_cycle(cycles, setup, number)
    assert setup.calls["training"] == [] and bytes_in(setup.source) == before
    assert not (setup.series / f"cycle-{number:04d}").exists()


@pytest.mark.parametrize("change", [{"new_steps": 76}, {"episodes": 1001}])
def test_existing_cycle_refuses_changed_budget_or_sample_size(cycles, cycle_setup, change):
    setup = cycle_setup
    run_cycle(cycles, setup)
    with pytest.raises(ValueError, match="fixed|original"):
        run_cycle(cycles, setup, **change)
    assert len(setup.calls["training"]) == 1


def test_partial_cycle_resumes_only_remaining_budget(cycles, cycle_setup):
    setup = cycle_setup
    root = setup.series / "cycle-0001"
    lineage = cycles.create_cycle(setup.source, root, setup.upstream, cycle=1,
                                 base_source=setup.source, series=setup.series,
                                 new_steps=75, episodes=1000)
    manifest = read_json(root / "experiment.json")
    model = StoredPolicy.load(root / lineage["initial_checkpoint"]["model"])
    model.num_timesteps += 32
    commit_checkpoint(model, root, [SimpleNamespace(next_index=44), SimpleNamespace(next_index=47)],
                      manifest["fingerprint"], interrupted=True)
    result = run_cycle(cycles, setup)
    assert setup.calls["training"] == [(root, 43)]
    assert result["new_rl_steps"] == 80
    assert not result["checkpoint"]["interrupted"]


def test_failed_checkpoint_publication_cleans_staging_and_keeps_parent_retryable(cycles, cycle_setup,
                                                                               monkeypatch):
    setup = cycle_setup
    before = bytes_in(setup.source)
    root = setup.series / "cycle-0001"
    original = cycles.commit_checkpoint

    def failed(*args, **kwargs):
        raise RuntimeError("failed publication")

    monkeypatch.setattr(cycles, "commit_checkpoint", failed)
    with pytest.raises(RuntimeError, match="failed publication"):
        cycles.create_cycle(setup.source, root, setup.upstream, cycle=1,
                            base_source=setup.source, series=setup.series,
                            new_steps=75, episodes=1000)
    assert not root.exists() and not list(setup.series.glob(".cycle-0001-initializing-*"))
    assert bytes_in(setup.source) == before and setup.calls["training"] == []
    monkeypatch.setattr(cycles, "commit_checkpoint", original)
    assert cycles.create_cycle(setup.source, root, setup.upstream, cycle=1,
                               base_source=setup.source, series=setup.series,
                               new_steps=75, episodes=1000)["initial_checkpoint"]["num_timesteps"] == 81032


def test_completed_counter_with_interrupted_checkpoint_is_not_assumed_finished(cycles, cycle_setup):
    setup = cycle_setup
    root = setup.series / "cycle-0001"
    lineage = cycles.create_cycle(setup.source, root, setup.upstream, cycle=1,
                                 base_source=setup.source, series=setup.series,
                                 new_steps=75, episodes=1000)
    manifest = read_json(root / "experiment.json")
    model = StoredPolicy.load(root / lineage["initial_checkpoint"]["model"])
    model.num_timesteps += 80
    commit_checkpoint(model, root, [SimpleNamespace(next_index=44), SimpleNamespace(next_index=47)],
                      manifest["fingerprint"], interrupted=True)
    with pytest.raises(ValueError, match="Interrupted|interrupted"):
        run_cycle(cycles, setup)
    assert setup.calls["training"] == []


@pytest.mark.parametrize("offset", [-8, 84])
def test_counter_outside_fixed_cycle_budget_is_rejected_before_training(cycles, cycle_setup, offset):
    setup = cycle_setup
    root = setup.series / "cycle-0001"
    lineage = cycles.create_cycle(setup.source, root, setup.upstream, cycle=1,
                                 base_source=setup.source, series=setup.series,
                                 new_steps=75, episodes=1000)
    manifest = read_json(root / "experiment.json")
    model = StoredPolicy.load(root / lineage["initial_checkpoint"]["model"])
    model.num_timesteps += offset
    commit_checkpoint(model, root, [SimpleNamespace(next_index=44), SimpleNamespace(next_index=47)],
                      manifest["fingerprint"])
    with pytest.raises(ValueError, match="counter|budget"):
        run_cycle(cycles, setup)
    assert setup.calls["training"] == []


def test_flat_or_regressing_cycle_blocks_another_allocation(cycles, cycle_setup):
    setup = cycle_setup
    setup.calls["after_floor"] = 10
    first = run_cycle(cycles, setup)
    assert first["establishes_positive_floor_gain"] is False
    with pytest.raises(ValueError, match="gain|improvement|eligible|gate|progress"):
        run_cycle(cycles, setup, 2)
    assert len(setup.calls["training"]) == 1
    assert not (setup.series / "cycle-0002").exists()


@pytest.mark.parametrize("change", ["source_raw", "source_summary", "cycle_cached_before"])
def test_evaluation_tampering_blocks_resume_or_next_cycle(cycles, cycle_setup, change):
    setup = cycle_setup
    if change == "cycle_cached_before":
        run_cycle(cycles, setup)
        path = setup.series / "cycle-0001" / "before_rl.json"
    else:
        path = setup.source / ("after_rl.json" if change == "source_raw" else "continuation-progress.json")
    value = read_json(path)
    if change == "source_summary":
        value["after_rl"]["mean_floor"] += 1
    else:
        value["episodes"][0]["floor"] += 1
        value["gameplay"] = summarize(value["episodes"])
    atomic_json(path, value)
    before = bytes_in(setup.source)
    with pytest.raises(ValueError):
        run_cycle(cycles, setup)
    assert bytes_in(setup.source) == before
    assert len(setup.calls["training"]) == int(change == "cycle_cached_before")


def test_forged_positive_summary_flag_cannot_allocate_after_flat_raw_results(cycles, cycle_setup):
    setup = cycle_setup
    after_path = setup.source / "after_rl.json"
    after = read_json(after_path)
    for row in after["episodes"]:
        row["floor"] = 9
    after["gameplay"] = summarize(after["episodes"])
    atomic_json(after_path, after)
    lineage_path = setup.source / "capacity-lineage.json"
    lineage = read_json(lineage_path)
    lineage["after_rl_sha256"] = sha256(after_path)
    atomic_json(lineage_path, lineage)
    before = read_json(setup.source / "before_rl.json")
    comparison = paired_compare(after["episodes"], before["episodes"])
    summary_path = setup.source / "continuation-progress.json"
    summary = read_json(summary_path)
    summary["after_rl"] = cycles.trial["brief_gameplay"](after["gameplay"])
    for key, raw_key in (("rl_floor_gain", "floor_delta"), ("rl_floor_gain_ci95", "floor_delta_ci95"),
                         ("rl_act1_clear_rate_gain", "win_delta"),
                         ("rl_act1_clear_rate_gain_ci95", "win_delta_ci95")):
        summary[key] = comparison[raw_key]
    summary["paired_clear_evidence"] = cycles.checker["exact_win_test"](after["episodes"], before["episodes"])
    summary["establishes_positive_floor_gain"] = True  # Raw evidence must take precedence.
    atomic_json(summary_path, summary)
    with pytest.raises(ValueError, match="progress|gain|summary"):
        run_cycle(cycles, setup)
    assert setup.calls["training"] == [] and not (setup.series / "cycle-0001").exists()


@pytest.mark.parametrize("outcome", ["timeout", "aborted"])
def test_unclean_new_starting_evaluation_stops_before_any_ppo(cycles, cycle_setup, monkeypatch, outcome):
    setup = cycle_setup
    original = cycles.assess

    def unclean(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs["label"] == "before_rl":
            value["episodes"][0]["outcome"] = outcome
            value["gameplay"] = summarize(value["episodes"])
        return value

    monkeypatch.setattr(cycles, "assess", unclean)
    with pytest.raises(ValueError, match="no PPO|errors|timeouts|clean"):
        run_cycle(cycles, setup)
    assert setup.calls["training"] == []
    assert verified_checkpoint(setup.series / "cycle-0001")[1]["num_timesteps"] == 81032


def test_flat_zero_win_bootstrap_does_not_become_clear_equivalence(cycles):
    seeds = evaluation_seeds("validation", 1000, 7, start=2700)
    before = {"episodes": rows(seeds, "before", floor=9)}
    after = {"episodes": rows(seeds, "after", floor=10)}
    gate = cycles.continuation_gate(before, after)
    assert gate["eligible"]
    assert gate["comparison"]["win_delta_ci95"] == [0, 0]
    assert not gate["paired_clear_evidence"]["supports_more_clears"]


def test_exact_clear_rate_regression_blocks_cycles_despite_floor_gain(cycles):
    seeds = evaluation_seeds("validation", 1000, 7, start=2700)
    before_rows = rows(seeds, "before", floor=9)
    for row in before_rows[:20]:
        row["outcome"], row["floor"] = "win", 16
    before = {"episodes": before_rows}
    after = {"episodes": rows(seeds, "after", floor=10)}
    gate = cycles.continuation_gate(before, after)
    assert gate["comparison"]["floor_delta_ci95"][0] > 0
    assert not gate["eligible"]


@pytest.mark.parametrize("invalid", [{"new_steps": 0}, {"new_steps": -1}, {"episodes": 999}])
def test_invalid_cycle_budget_or_sample_is_rejected_before_creation(cycles, tmp_path, invalid):
    series = tmp_path / "series"
    with pytest.raises(ValueError):
        cycles.progress(tmp_path / "missing", series, tmp_path / "upstream", cycle=1,
                        **{"new_steps": 75, "episodes": 1000, **invalid})
    assert not series.exists()
