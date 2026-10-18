"""A repaired continuation preserves PPO state and spends only a fixed new budget."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import (
    ExperimentConfig, atomic_json, commit_checkpoint, read_json, sha256, verified_checkpoint,
)
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds, paired_compare, summarize

from test_act1_trial import SchemaOnlyEnv, StoredPolicy, bytes_in


class ContinuationPolicy(StoredPolicy):
    observation_space = SchemaOnlyEnv.observation_space
    action_space = SchemaOnlyEnv.action_space

    @classmethod
    def load(cls, path, **kwargs):
        if "env" in kwargs:
            assert kwargs["force_reset"] is True
            assert kwargs["env"].observation_space == cls.observation_space
            assert kwargs["env"].action_space == cls.action_space
        return super().load(path, **kwargs)


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"continuation_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def continuation():
    return load_script("continue_act1_rl.py")


def rows(seeds, label, *, floor=8):
    return [{"seed": seed, "policy": label, "outcome": "death", "floor": floor, "hp": 0}
            for seed in seeds]


def assessment(manifest, state, label, seeds, start, *, floor=8):
    episodes = rows(seeds, label, floor=floor)
    return {"stage": label, "fingerprint": manifest["fingerprint"], "checkpoint": state,
            "split": "validation", "seed_start": start, "episodes": episodes,
            "gameplay": summarize(episodes), "inference": "deterministic", "victory_scope": "act1"}


def saved_replay(expected, repaired, output):
    """Persist actual trace schemas, including the first capacity intervention."""
    seed = expected["seed"]
    steps = 70 if repaired else expected["steps"]
    action_path = output / f"seed-{seed}-actions.jsonl"
    observation_path = output / f"seed-{seed}-observations.jsonl"
    actions, observations = [], []
    for step in range(steps if repaired else steps + 1):
        full_pick = step == expected["steps"]
        legal = [0] if repaired and full_pick else [0, 1]
        public = {"observation": {"visible": [step]}, "legal_actions": legal}
        digest = hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        observations.append({"observation_sha256": digest, **public})
        action = 1 if full_pick and not repaired else 0
        row = {"seed": seed, "step": step, "phase": "reward", "floor": 4, "hp": 30,
               "action_slot": action, "observation_sha256": digest, "observation_visits": 1,
               "named_action": {"action": "pick_potion" if action == 1 else "skip_potion"}}
        if full_pick:
            row["potion_inventory"] = {"max_slots": 3, "slot_potion_ids": ["A", "B", "C"],
                                      "free_slots": [], "pick_action_slots": [1],
                                      "pick_action_mask": [not repaired]}
        actions.append(row)
    action_path.write_text("".join(json.dumps(row) + "\n" for row in actions), encoding="utf-8")
    observation_path.write_text("".join(json.dumps(row) + "\n" for row in observations), encoding="utf-8")
    tail = [{key: value for key, value in row.items()
             if key not in {"seed", "observation_sha256", "observation_visits"}} for row in actions[-6:]]
    return {**expected, "outcome": "death" if repaired else "aborted", "steps": steps,
            "floor": 7 if repaired else 4, "hp": 0 if repaired else 30,
            "error": None if repaired else expected["error"],
            "error_stage": None if repaired else "game_step", "final_phase": "reward",
            "last_decisions": tail, "capacity_mask_decisions": 1 if repaired else 0,
            "unique_public_observations": len(observations), "most_repeated_observation_visits": 1,
            "actions_trace": str(action_path), "observations_trace": str(observation_path)}


@pytest.fixture
def repaired_source(continuation, monkeypatch, tmp_path):
    source, ancestor, upstream = tmp_path / "trial", tmp_path / "imitation", tmp_path / "upstream"
    config = ExperimentConfig(max_acts=1, n_envs=2, n_steps=4, batch_size=4, n_epochs=1)
    recipe = continuation.trial["act1"]["default_recipe"]()
    old_plan = {"target_steps": 80000, "episodes": 100, "validation_start": 500}

    def factory(config, upstream):
        return SchemaOnlyEnv(config)

    monkeypatch.setattr(runner, "_make", factory)
    monkeypatch.setattr(runner, "provenance", lambda upstream: {"core_source": "unchanged"})
    wrappers = [SimpleNamespace(next_index=20), SimpleNamespace(next_index=21)]
    with continuation.trial["act1"]["act1_runner"](recipe):
        ancestor_manifest = runner.initialize(ancestor, config, upstream)
    ancestor_states = [commit_checkpoint(StoredPolicy(weight=weight), ancestor, wrappers,
                                        ancestor_manifest["fingerprint"]) for weight in (1, 3)]
    ancestor_lineage = {"config": asdict(config), "recipe": recipe,
                        "source_checkpoint": {"num_timesteps": 160768},
                        "imported_checkpoint": ancestor_states[0], "cloned_checkpoint": ancestor_states[1]}
    old_seeds = evaluation_seeds("validation", 100, 7, start=400)
    for label, state in zip(("before_imitation", "after_imitation_before_rl"), ancestor_states):
        target = ancestor / f"{label}.json"
        atomic_json(target, assessment(ancestor_manifest, state, label, old_seeds, 400))
        ancestor_lineage[f"{label}_sha256"] = sha256(target)
    atomic_json(ancestor / "act1-lineage.json", ancestor_lineage)
    with continuation.trial["trial_runner"](recipe, old_plan):
        manifest = runner.initialize(source, config, upstream)
    baseline = commit_checkpoint(StoredPolicy(weight=3), source, wrappers, manifest["fingerprint"])
    model = StoredPolicy(num_timesteps=80952, weight=4)
    model.policy(torch.ones(1, 1)).sum().backward()
    model.policy.optimizer.step()
    current = commit_checkpoint(model, source, wrappers, manifest["fingerprint"])
    atomic_json(source / "seed-cursors" / "worker-0.json", {"next_index": 30})
    atomic_json(source / "seed-cursors" / "worker-1.json", {"next_index": 33})
    atomic_json(source / "champion.json", {"preserved_source_claim": True})
    lineage = {"config": asdict(config), "recipe": recipe, "plan": old_plan,
               "source_experiment": str(ancestor), "source_fingerprint": ancestor_manifest["fingerprint"],
               "source_imported_checkpoint": ancestor_states[0], "source_cloned_checkpoint": ancestor_states[1],
               "initial_checkpoint": baseline}
    atomic_json(source / "trial-lineage.json", lineage)
    seeds = evaluation_seeds("validation", 1000, 7, start=700)
    candidate = rows(seeds, "frozen-rl-candidate", floor=9)
    baseline_rows = rows(seeds, "frozen-cloned-initializer", floor=8)
    candidate[0] = {"seed": seeds[0], "policy": "frozen-rl-candidate", "outcome": "aborted",
                    "steps": 40, "floor": 4, "hp": 30, "return": 0,
                    "error": (f"SimulatorError: Emulator step failed (seed={seeds[0]}, step=40, action=1): "
                              "RuntimeError: Engine rejected advertised legal pick_potion")}
    protocol = {"fingerprint": manifest["fingerprint"], "candidate_checkpoint": current,
                "baseline_checkpoint": baseline, "split": "validation", "episodes": 1000,
                "seed_start": 700, "inference": "deterministic", "victory_scope": "act1",
                "checker_sha256": sha256(Path(__file__).resolve().parents[1] / "scripts" / "check_act1_trial.py")}
    report = {"protocol": protocol, "candidate_episodes": candidate, "baseline_episodes": baseline_rows,
              "candidate": summarize(candidate), "baseline": summarize(baseline_rows),
              "comparison": paired_compare(candidate, baseline_rows), "training_record": {"status": "read"}}
    report_path = source / "diagnostics" / f"strength-{current['sha256'][:16]}" / "report.json"
    atomic_json(report_path, report)
    potion_checker = load_script("check_potion_failures.py")
    monkeypatch.setattr(potion_checker, "MaskablePPO", ContinuationPolicy)
    monkeypatch.setattr(potion_checker, "replay", lambda env, model, seed, output:
                        saved_replay(candidate[0], isinstance(env, PotionCapacityMask), Path(output)))
    result = potion_checker.check(source, upstream)
    assert result["repair_verified"]
    monkeypatch.setattr(continuation, "MaskablePPO", ContinuationPolicy)

    def forbidden(*args, **kwargs):
        pytest.fail("Continuation must not collect, clone, reset a PPO optimizer, or promote")

    monkeypatch.setattr(runner, "promote", forbidden)
    monkeypatch.setitem(continuation.trial["selection"], "fresh_policy", forbidden)
    globals_ = continuation.trial["act1"]["create_experiment"].__globals__
    monkeypatch.setitem(globals_, "collect_demonstrations", forbidden)
    monkeypatch.setitem(globals_, "behavior_clone", forbidden)
    return SimpleNamespace(source=source, ancestor=ancestor, upstream=upstream, config=config,
                           recipe=recipe, old_plan=old_plan, manifest=manifest, lineage=lineage,
                           current=current, baseline=baseline, report_path=report_path,
                           repair_path=Path(result["summary_path"]), factory=factory,
                           plan={"new_steps": 75, "episodes": 1000, "validation_start": 1700})


def assert_optimizer_equal(left, right):
    left, right = left.policy.optimizer.state_dict(), right.policy.optimizer.state_dict()
    assert left["param_groups"] == right["param_groups"]
    assert left["state"] and left["state"].keys() == right["state"].keys()
    for key, values in left["state"].items():
        assert values.keys() == right["state"][key].keys()
        for name, value in values.items():
            torch.testing.assert_close(value, right["state"][key][name], rtol=0, atol=0)


def test_verified_repair_recomputes_raw_trace_evidence_and_preserves_source(continuation, repaired_source):
    old = repaired_source
    before = bytes_in(old.source)
    evidence = continuation.validated_repair(old.source, old.lineage, old.manifest, old.current,
                                            old.repair_path)
    assert evidence["summary"]["repair_verified"] is True
    assert Path(evidence["summary_path"]) == old.repair_path
    assert evidence["strength_report"] == read_json(old.report_path)
    assert str(old.repair_path) in evidence["files"]
    assert str(old.report_path) in evidence["files"]
    assert all(sha256(Path(path)) == digest for path, digest in evidence["files"].items())
    assert bytes_in(old.source) == before and runner._make is old.factory


@pytest.mark.parametrize("tamper", ["unverified", "trained", "promoted", "original_trace",
                                    "repair_trace", "observation_trace", "runs", "protocol", "strength"])
def test_changed_or_unverified_repair_is_rejected_before_migration(continuation, repaired_source,
                                                                  tmp_path, tamper):
    old = repaired_source
    folder = old.repair_path.parent
    if tamper in {"unverified", "trained", "promoted"}:
        value = read_json(old.repair_path)
        key, replacement = {"unverified": ("repair_verified", False), "trained": ("additional_training_steps", 1),
                            "promoted": ("champion_updated", True)}[tamper]
        value[key] = replacement
        atomic_json(old.repair_path, value)
    elif tamper.endswith("trace"):
        runs = read_json(folder / "runs.json")["runs"]
        mode = "original" if tamper == "original_trace" else "capacity_mask"
        key = "observations_trace" if tamper == "observation_trace" else "actions_trace"
        path = Path(runs[0][mode][key])
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if key == "actions_trace":
            records[0]["action_slot"] = 1
        else:
            records[0]["observation_sha256"] = "changed observation proof"
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    elif tamper in {"runs", "protocol"}:
        path = folder / f"{tamper}.json"
        value = read_json(path)
        if tamper == "runs":
            value["runs"][0]["repair_completed_run"] = False
        else:
            value["repair_adapter_sha256"] = "different adapter"
        atomic_json(path, value)
    else:
        value = read_json(old.report_path)
        value["candidate"]["mean_floor"] += 1
        atomic_json(old.report_path, value)
    destination = tmp_path / "continued"
    before = bytes_in(old.source)
    with pytest.raises(ValueError):
        continuation.create_continuation(old.source, destination, old.upstream, old.plan, old.repair_path)
    assert not destination.exists() and not list(tmp_path.glob(".continued-initializing-*"))
    assert bytes_in(old.source) == before and runner._make is old.factory


def test_migration_preserves_optimizer_counter_weights_and_reserved_seed_highwater(continuation,
                                                                                 repaired_source,
                                                                                 tmp_path):
    old, destination = repaired_source, tmp_path / "continued"
    before, ancestor_before = bytes_in(old.source), bytes_in(old.ancestor)
    lineage = continuation.create_continuation(old.source, destination, old.upstream,
                                              old.plan, old.repair_path)
    manifest = read_json(destination / "experiment.json")
    assert lineage["source_state"] == old.current
    assert lineage["source_lineage"] == old.lineage
    assert lineage["plan"] == old.plan
    assert lineage["source_experiment"] == str(old.source)
    imported = lineage["initial_checkpoint"]
    assert imported["fingerprint"] == manifest["fingerprint"] != old.manifest["fingerprint"]
    assert imported["num_timesteps"] == old.current["num_timesteps"] == 80952
    assert imported["next_indices"] == [30, 33]
    assert verified_checkpoint(destination)[1] == imported
    original = StoredPolicy.load(old.source / old.current["model"])
    copied = StoredPolicy.load(destination / imported["model"])
    for name, value in original.policy.state_dict().items():
        torch.testing.assert_close(copied.policy.state_dict()[name], value, rtol=0, atol=0)
    assert_optimizer_equal(original, copied)
    assert bytes_in(old.source) == before and bytes_in(old.ancestor) == ancestor_before
    assert not (destination / "champion.json").exists() and not (destination / "demonstrations.npz").exists()
    with continuation.trial["trial_runner"](old.recipe, old.old_plan):
        assert runner.initialize(old.source, old.config, old.upstream) == old.manifest
        with pytest.raises(ValueError, match="configuration|fingerprint"):
            runner.initialize(destination, old.config, old.upstream)
    assert runner._make is old.factory


def test_public_spaces_unchanged_and_original_context_restored_even_after_failure(continuation,
                                                                               repaired_source):
    old = repaired_source
    evidence = continuation.validated_repair(old.source, old.lineage, old.manifest, old.current,
                                            old.repair_path)
    with pytest.raises(RuntimeError, match="inside continuation"):
        with continuation.continuation_runner(old.lineage, old.plan, old.current, evidence):
            env = runner._make(old.config, old.upstream)
            assert isinstance(env, PotionCapacityMask)
            assert env.observation_space == SchemaOnlyEnv.observation_space
            assert env.action_space == SchemaOnlyEnv.action_space
            assert env.backend_metadata["trial_plan"] == old.old_plan
            assert env.backend_metadata["potion_capacity_interface"] == "mask-full-inventory-v1"
            env.close()
            raise RuntimeError("inside continuation")
    assert runner._make is old.factory


def test_failed_migration_cleans_staging_and_keeps_source_retryable(continuation, repaired_source,
                                                                 tmp_path, monkeypatch):
    old, destination = repaired_source, tmp_path / "continued"
    before = bytes_in(old.source)
    real_commit = continuation.commit_checkpoint

    def fail_commit(*args, **kwargs):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(continuation, "commit_checkpoint", fail_commit)
    with pytest.raises(RuntimeError, match="publication failed"):
        continuation.create_continuation(old.source, destination, old.upstream, old.plan, old.repair_path)
    assert not destination.exists() and not list(tmp_path.glob(".continued-initializing-*"))
    assert bytes_in(old.source) == before and runner._make is old.factory
    monkeypatch.setattr(continuation, "commit_checkpoint", real_commit)
    assert continuation.create_continuation(old.source, destination, old.upstream,
                                           old.plan, old.repair_path)["initial_checkpoint"]["num_timesteps"] == 80952


@pytest.mark.parametrize("change", ["corrupt", "later_latest"])
def test_source_weights_and_verified_latest_must_match_repair_evidence(continuation, repaired_source,
                                                                    tmp_path, change):
    old, destination = repaired_source, tmp_path / "continued"
    if change == "corrupt":
        path = old.source / old.current["model"]
        path.write_bytes(path.read_bytes() + b"changed model")
    else:
        commit_checkpoint(StoredPolicy(num_timesteps=81976, weight=5), old.source,
                          [SimpleNamespace(next_index=30), SimpleNamespace(next_index=33)],
                          old.manifest["fingerprint"])
    before = bytes_in(old.source)
    with pytest.raises(ValueError):
        continuation.create_continuation(old.source, destination, old.upstream, old.plan, old.repair_path)
    assert not destination.exists() and bytes_in(old.source) == before


def test_wrong_rank_cursor_rejected_even_if_below_checkpoint(continuation, repaired_source, tmp_path):
    old = repaired_source
    atomic_json(old.source / "seed-cursors" / "worker-0.json", {"next_index": 1})
    with pytest.raises(ValueError, match="worker rank"):
        continuation.create_continuation(old.source, tmp_path / "continued", old.upstream,
                                         old.plan, old.repair_path)


def test_validation_freshness_checks_actual_imitation_rows_beyond_minimum_start(continuation,
                                                                              repaired_source,
                                                                              tmp_path):
    old = repaired_source
    lineage = read_json(old.ancestor / "act1-lineage.json")
    seeds = evaluation_seeds("validation", 100, 7, start=2000)
    for label in ("before_imitation", "after_imitation_before_rl"):
        path = old.ancestor / f"{label}.json"
        report = read_json(path)
        report["seed_start"] = 2000
        report["episodes"] = rows(seeds, label)
        report["gameplay"] = summarize(report["episodes"])
        atomic_json(path, report)
        lineage[f"{label}_sha256"] = sha256(path)
    atomic_json(old.ancestor / "act1-lineage.json", lineage)
    destination = tmp_path / "continued"
    before = bytes_in(old.source)
    with pytest.raises(ValueError, match="overlap"):
        continuation.create_continuation(old.source, destination, old.upstream, old.plan, old.repair_path)
    assert not destination.exists() and bytes_in(old.source) == before


@pytest.mark.parametrize("invalid", [{"new_steps": 0}, {"new_steps": -1}, {"episodes": 999},
                                    {"validation_start": 1699}])
def test_invalid_budget_or_schedule_does_not_create_an_experiment(continuation, tmp_path, invalid):
    plan = {"new_steps": 75, "episodes": 1000, "validation_start": 1700, **invalid}
    destination = tmp_path / "continued"
    with pytest.raises(ValueError):
        continuation.progress(tmp_path / "missing", destination, tmp_path / "upstream", **plan)
    assert not destination.exists()


@pytest.fixture
def progress_setup(continuation, repaired_source, tmp_path, monkeypatch):
    old, destination = repaired_source, tmp_path / "continued"
    calls = {"assessments": [], "training": []}

    def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
        continuation.trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
        calls["assessments"].append((label, episodes, validation_start, deepcopy(state)))
        seeds = evaluation_seeds("validation", episodes, 7, start=validation_start)
        return assessment(manifest, state, label, seeds, validation_start,
                          floor=9 if label == "before_rl" else 10)

    monkeypatch.setattr(continuation, "assess", assess)

    def train(root, config, upstream, *, steps):
        calls["training"].append(steps)
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
    return SimpleNamespace(old=old, destination=destination, calls=calls)


def test_fixed_additional_budget_and_matched_repaired_evaluations_never_repeat_training(continuation,
                                                                                    progress_setup):
    setup, old = progress_setup, progress_setup.old
    before = bytes_in(old.source)
    first = continuation.progress(old.source, setup.destination, old.upstream,
                                  repair_check=old.repair_path, **old.plan)
    second = continuation.progress(old.source, setup.destination, old.upstream,
                                   repair_check=old.repair_path, **old.plan)
    assert setup.calls["training"] == [75]
    assert first["new_rl_steps"] == second["new_rl_steps"] == 80
    assert verified_checkpoint(setup.destination)[1]["num_timesteps"] == 80952 + 80
    assert first["champion_updated"] is second["champion_updated"] is False
    assert not (setup.destination / "champion.json").exists()
    assert bytes_in(old.source) == before and runner._make is old.factory
    assessments = setup.calls["assessments"]
    assert sum(call[0] == "before_rl" for call in assessments) == 1
    assert any(call[0] == "after_rl" for call in assessments)
    assert all(call[1:3] == (1000, 1700) for call in assessments)
    assert assessments[0][3]["num_timesteps"] == 80952
    seeds = evaluation_seeds("validation", 1000, 7, start=1700)
    report = read_json(setup.destination / "before_rl.json")
    assert [row["seed"] for row in report["episodes"]] == seeds
    assert set(seeds).isdisjoint(evaluation_seeds("validation", 1000, 7, start=700))


@pytest.mark.parametrize("changed", [{"new_steps": 76}, {"episodes": 1001}, {"validation_start": 1701}])
def test_resume_refuses_changed_additional_budget_or_seed_protocol(continuation, progress_setup, changed):
    setup, old = progress_setup, progress_setup.old
    continuation.progress(old.source, setup.destination, old.upstream,
                          repair_check=old.repair_path, **old.plan)
    with pytest.raises(ValueError, match="fixed"):
        continuation.progress(old.source, setup.destination, old.upstream,
                              repair_check=old.repair_path, **{**old.plan, **changed})
    assert setup.calls["training"] == [75]


def test_partial_resume_trains_only_the_remaining_new_budget(continuation, progress_setup, monkeypatch):
    setup, old = progress_setup, progress_setup.old
    lineage = continuation.create_continuation(old.source, setup.destination, old.upstream,
                                              old.plan, old.repair_path)
    manifest = read_json(setup.destination / "experiment.json")
    model = StoredPolicy.load(setup.destination / lineage["initial_checkpoint"]["model"])
    model.num_timesteps += 32
    commit_checkpoint(model, setup.destination, [SimpleNamespace(next_index=34), SimpleNamespace(next_index=35)],
                      manifest["fingerprint"], interrupted=True)
    continuation.progress(old.source, setup.destination, old.upstream,
                          repair_check=old.repair_path, **old.plan)
    assert setup.calls["training"] == [43]


def test_modified_pinned_repair_blocks_resume_before_more_ppo(continuation, progress_setup):
    setup, old = progress_setup, progress_setup.old
    continuation.create_continuation(old.source, setup.destination, old.upstream,
                                     old.plan, old.repair_path)
    old.repair_path.write_bytes(old.repair_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum|changed"):
        continuation.progress(old.source, setup.destination, old.upstream,
                              repair_check=old.repair_path, **old.plan)
    assert setup.calls["training"] == []


def test_real_assessment_batches_both_policies_on_the_same_repaired_public_interface(continuation,
                                                                                  repaired_source,
                                                                                  tmp_path, monkeypatch):
    old, destination = repaired_source, tmp_path / "continued"
    lineage = continuation.create_continuation(old.source, destination, old.upstream, old.plan, old.repair_path)
    manifest = read_json(destination / "experiment.json")
    calls = []

    def rollout(env, seeds, *, model, label):
        assert isinstance(env, PotionCapacityMask)
        assert env.backend_metadata["trial_plan"] == old.old_plan
        assert env.backend_metadata["continuation_plan"] == old.plan
        calls.append((label, list(seeds), model.num_timesteps))
        return rows(seeds, label, floor=9)

    monkeypatch.setattr(runner, "_rollout", rollout)
    with continuation.continuation_runner(old.lineage, old.plan, old.current, lineage["repair_evidence"]):
        for label in ("before_rl", "after_rl"):
            report = continuation.assess(destination, old.upstream, manifest, lineage["initial_checkpoint"],
                                         label=label, episodes=1000, validation_start=1700)
            continuation.trial["act1"]["validate_assessment"](report, manifest, lineage["initial_checkpoint"],
                                                             label, 1000, 1700)
    assert len(calls) == 40
    for before, after in zip(calls[:20], calls[20:]):
        assert before[0] == "before_rl" and after[0] == "after_rl"
        assert before[1:] == after[1:]
        assert before[2] == 80952
    assert runner._make is old.factory


def test_unclean_repaired_starting_evaluation_blocks_any_new_training(continuation, progress_setup,
                                                                   monkeypatch):
    setup, old = progress_setup, progress_setup.old
    assess = continuation.assess

    def unclean(*args, **kwargs):
        report = assess(*args, **kwargs)
        if kwargs["label"] == "before_rl":
            report["episodes"][0]["outcome"] = "aborted"
            report["gameplay"] = summarize(report["episodes"])
        return report

    monkeypatch.setattr(continuation, "assess", unclean)
    before = bytes_in(old.source)
    with pytest.raises(ValueError, match="no PPO started"):
        continuation.progress(old.source, setup.destination, old.upstream,
                              repair_check=old.repair_path, **old.plan)
    assert setup.calls["training"] == [] and bytes_in(old.source) == before
    assert verified_checkpoint(setup.destination)[1]["num_timesteps"] == 80952


def test_cached_before_evidence_cannot_be_edited_between_resumes(continuation, progress_setup):
    setup, old = progress_setup, progress_setup.old
    continuation.progress(old.source, setup.destination, old.upstream,
                          repair_check=old.repair_path, **old.plan)
    path = setup.destination / "before_rl.json"
    report = read_json(path)
    report["episodes"][0]["floor"] += 1
    report["gameplay"] = summarize(report["episodes"])
    atomic_json(path, report)
    with pytest.raises(ValueError, match="checksum changed"):
        continuation.progress(old.source, setup.destination, old.upstream,
                              repair_check=old.repair_path, **old.plan)
    assert setup.calls["training"] == [75]
