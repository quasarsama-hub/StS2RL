"""Combined sessions retain learned state and one fixed, auditable allowance.

These tests persist real Torch policy/value tensors and Adam moments, but use
controlled training/evaluation APIs. They do not claim actual SB3 throughput
or increased playing strength; the release's runtime check covers real PPO.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import gymnasium as gym
import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import (
    ExperimentConfig, append_jsonl, atomic_json, commit_checkpoint, read_json,
    sha256, verified_checkpoint,
)
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds, summarize

from test_act1_continuation import assert_optimizer_equal, load_script
from test_act1_trial import SchemaOnlyEnv, bytes_in
from test_isolated_act1_cycles import PreservedPolicy, raw_compare


@pytest.fixture(scope="module")
def sessions():
    return load_script("run_learning_session.py")


def outcome_rows(seeds, label, *, floor=10):
    return [{"seed": seed, "policy": label, "outcome": "death", "floor": floor,
             "hp": 0.0, "steps": 100, "return": -1.0, "error": None}
            for seed in seeds]


def assessed(manifest, state, label, count, start, *, floor=10):
    episodes = outcome_rows(evaluation_seeds("validation", count, 7, start=start),
                            label, floor=floor)
    return {"stage": label, "fingerprint": manifest["fingerprint"], "checkpoint": deepcopy(state),
            "split": "validation", "seed_start": start, "episodes": episodes,
            "gameplay": summarize(episodes), "inference": "deterministic", "victory_scope": "act1"}


def tensors_equal(left, right):
    assert left.policy.state_dict().keys() == right.policy.state_dict().keys()
    for key, tensor in left.policy.state_dict().items():
        torch.testing.assert_close(tensor, right.policy.state_dict()[key], rtol=0, atol=0)
    assert_optimizer_equal(left, right)


@pytest.fixture
def setup(sessions, monkeypatch, tmp_path):
    module = sessions
    source = tmp_path / "isolated-cycles" / "cycle-0001"
    ancestor = tmp_path / "original-cycles" / "cycle-0002"
    series, upstream = tmp_path / "learning-sessions", tmp_path / "upstream"
    config = ExperimentConfig(max_acts=1)  # The production four-by-256 rollout.
    base_factory = {"source_lineage": {"retained_base": True}}
    original_plan = {"new_steps": 80000, "episodes": 1000, "validation_start": 3700}
    original_source_state = {"num_timesteps": 242744, "sha256": "ancestor-identity"}
    isolated_plan = {"new_steps": 80000, "episodes": 1000, "validation_start": 7800}
    isolated_source_state = {"num_timesteps": 323640, "sha256": "original-cycle-final"}

    @contextmanager
    def flat_original_factory(descriptor, plan, state):
        assert descriptor == base_factory
        assert plan == original_plan and state == original_source_state
        previous = runner._make
        def make(config, upstream):
            env = PotionCapacityMask(SchemaOnlyEnv(config))
            env.backend_metadata.update(rl_cycle_plan=deepcopy(plan),
                                        rl_cycle_source_checkpoint=deepcopy(state),
                                        inherited_base="unchanged")
            return env
        runner._make = make
        try:
            yield
        finally:
            runner._make = previous

    class Isolated(gym.Wrapper):
        def __init__(self, env):
            if "event_state_isolation" in env.backend_metadata:
                raise ValueError("Event isolation already installed")
            super().__init__(env)
            self.backend_metadata = deepcopy(env.backend_metadata)
            self.backend_metadata.update(event_state_isolation="private-registry-per-run-v1",
                                         event_registry_count=68,
                                         event_isolation_sha256=sha256(Path(module.__file__).parents[1] /
                                                                      "src/spire2_rl/event_isolation.py"))
        @property
        def scope(self):
            return self.env.scope
        @property
        def raw_env(self):
            return self.env.raw_env

    isolated_globals = module.isolated["factory"].__wrapped__.__globals__
    monkeypatch.setitem(isolated_globals["cycles"], "cycle_runner", flat_original_factory)
    monkeypatch.setitem(isolated_globals, "IsolatedEvents", Isolated)
    monkeypatch.setattr(module, "MaskablePPO", PreservedPolicy)
    monkeypatch.setattr(runner, "provenance", lambda upstream: {"core_source": "unchanged"})
    monkeypatch.setitem(module.cycles["continuation_gate"].__globals__, "paired_compare", raw_compare)
    # The ancient imitation ancestry is outside this controlled source fixture.
    # Reservation scanning remains real, and the tests check actual disjoint rows.
    for checker in (module.checker, module.isolated["strength_checker"]["checker"]):
        monkeypatch.setitem(checker, "fresh_seeds", lambda root, original, config, count, start:
                            evaluation_seeds("validation", count, config.experiment_seed, start=start))

    ancestor.mkdir(parents=True)
    ancestor_before = outcome_rows(evaluation_seeds("validation", 1000, 7, start=3700), "before_rl")
    ancestor_after = outcome_rows(evaluation_seeds("validation", 1000, 7, start=3700), "after_rl")
    atomic_json(ancestor / "before_rl.json", {"split": "validation", "seed_start": 3700, "episodes": ancestor_before})
    atomic_json(ancestor / "after_rl.json", {"split": "validation", "seed_start": 3700, "episodes": ancestor_after})
    atomic_json(ancestor / "cycle-lineage.json", {"plan": original_plan})
    proof = ancestor / "diagnostics" / "event-proof"
    atomic_json(proof / "protocol.json", {"split": "validation", "seed_start": 7700, "episodes": 100})
    atomic_json(proof / "report.json", {"mechanics": "retained-completed-proof"})
    source_files = {str(path): sha256(path) for path in ancestor.rglob("*") if path.is_file()}

    with module.isolated["factory"](base_factory, original_plan, original_source_state,
                                    isolated_plan, isolated_source_state):
        manifest = runner.initialize(source, config, upstream)
    initial_model = PreservedPolicy(num_timesteps=323640, weight=7)
    initial_model.policy(torch.ones(1, 1)).sum().backward()
    initial_model.policy.optimizer.step()
    initial = commit_checkpoint(initial_model, source,
                                [SimpleNamespace(next_index=4800 + rank) for rank in range(4)],
                                manifest["fingerprint"])
    final_model = PreservedPolicy(num_timesteps=404536, weight=11)
    final_model.policy(torch.ones(1, 1)).sum().backward()
    final_model.policy.optimizer.step()
    final = commit_checkpoint(final_model, source,
                              [SimpleNamespace(next_index=4840 + rank) for rank in range(4)],
                              manifest["fingerprint"])
    for rank in range(4):
        atomic_json(source / "seed-cursors" / f"worker-{rank}.json", {"next_index": 4880 + rank})
    append_jsonl(source / "episodes.jsonl", {"outcome": "death"})
    append_jsonl(source / "sessions.jsonl", {"status": "completed"})
    atomic_json(source / "champion.json", {"unchanged_source_pointer": True})
    lineage = {"cycle": 1, "series": str(source.parent), "base_source": str(ancestor),
               "source_experiment": str(ancestor), "source_state": isolated_source_state,
               "source_files": source_files, "base_factory": base_factory,
               "original_plan": original_plan, "original_source_state": original_source_state,
               "repair_evidence": {"status": "verified", "source_files": source_files},
               "config": asdict(config), "plan": isolated_plan,
               "initial_checkpoint": initial, "target_total_steps": 403640,
               "training_seed_indices": [4800, 4801, 4802, 4803],
               "parent_gate": {"eligible": True}, "optimizer": "preserved PPO Adam state"}
    before = assessed(manifest, initial, "before_rl", 1000, 7800)
    after = assessed(manifest, final, "after_rl", 1000, 7800, floor=11)

    def publish_source():
        before["gameplay"], after["gameplay"] = summarize(before["episodes"]), summarize(after["episodes"])
        atomic_json(source / "before_rl.json", before)
        atomic_json(source / "after_rl.json", after)
        lineage["before_rl_sha256"] = sha256(source / "before_rl.json")
        lineage["after_rl_sha256"] = sha256(source / "after_rl.json")
        summary, gate = module.isolated["cycle_summary"](source, config, lineage, final, before, after)
        atomic_json(source / "event-evidence.json", {"before": before, "after": after,
                    "comparison": gate["comparison"], "next_cycle_gate": gate,
                    "training": {"status": "completed", "total_steps": final["num_timesteps"]}})
        atomic_json(source / "event-progress.json", summary)
        lineage["event_evidence_sha256"] = sha256(source / "event-evidence.json")
        lineage["event_progress_sha256"] = sha256(source / "event-progress.json")
        atomic_json(source / "event-lineage.json", lineage)

    publish_source()
    calls = {"training": [], "assessments": [], "replays": [], "before_timeouts": 0,
             "after_timeouts": 0, "before_aborts": 0, "after_aborts": 0,
             "after_clears": 0, "replay_failure": None, "interrupt_once": False, "engine_abort": False}

    def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
        module.trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
        calls["assessments"].append((label, episodes, validation_start, deepcopy(state)))
        report = assessed(manifest, state, label, episodes, validation_start,
                          floor=10 if label == "before_rl" else 11)
        which = "before" if label == "before_rl" else "after"
        for row in report["episodes"][:calls[f"{which}_timeouts"]]:
            row.update(outcome="timeout", steps=4000, hp=10.0, **{"return": 0.0})
        for row in report["episodes"][calls[f"{which}_timeouts"]:
                                      calls[f"{which}_timeouts"] + calls[f"{which}_aborts"]]:
            row.update(outcome="aborted", error="RuntimeError: simulator step failed")
        if which == "after":
            start = calls["after_timeouts"] + calls["after_aborts"]
            for row in report["episodes"][start:start + calls["after_clears"]]:
                row.update(outcome="win", floor=17, **{"return": 1.0})
        report["gameplay"] = summarize(report["episodes"])
        return report

    def replay(root, config, upstream, state, row, out_dir, *, tail=40):
        assert config.max_steps == 4000 and config.max_acts == 1
        assert row["outcome"] == "timeout" and row["steps"] == config.max_steps
        expected = {key: row[key] for key in module.tracer["GAME_FIELDS"]}
        module.trial["act1"]["checked_state"](root, state,
                                                read_json(root / "experiment.json")["fingerprint"])
        calls["replays"].append({"root": str(root), "cap": config.max_steps,
                                "checkpoint_sha256": state["sha256"], "original_result": expected})
        result = {**expected, "original_result": deepcopy(expected), "replayed_result": deepcopy(expected),
                  "matches_original_result": True, "original_decision_cap": config.max_steps,
                  "checkpoint_sha256": state["sha256"], "logging_errors": [], "logging_error_count": 0,
                  "additional_training_steps": 0, "last_decisions": [], "requires_pause": False}
        if calls["replay_failure"] == "mismatch":
            result["replayed_result"]["hp"] += 1
            # A stored assertion cannot override disagreement in the raw results.
        elif calls["replay_failure"] == "engine":
            result["replayed_result"].update(outcome="aborted", error="SimulatorError: rejected legal action")
        elif calls["replay_failure"] == "logging":
            result["logging_errors"] = [{"error": "Disk logging failed"}]
            result["logging_error_count"] = 1
        elif calls["replay_failure"] == "logging_count":
            result["logging_error_count"] = 1
        elif calls["replay_failure"] == "cap":
            result["original_decision_cap"] += 1
        elif calls["replay_failure"] == "checkpoint":
            result["checkpoint_sha256"] = "different policy"
        out_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(out_dir / "summary.json", result)
        return result

    def train(root, config, upstream, *, steps):
        calls["training"].append(steps)
        manifest = read_json(root / "experiment.json")
        path, state = verified_checkpoint(root, manifest["fingerprint"])
        model = PreservedPolicy.load(path)
        rollout = config.n_envs * config.n_steps
        interrupted = calls["interrupt_once"]
        addition = 512 if interrupted else ((steps + rollout - 1) // rollout) * rollout
        model.num_timesteps += addition
        model.policy.optimizer.zero_grad()
        model.policy(torch.ones(1, 1)).sum().backward()
        model.policy.optimizer.step()
        saved = commit_checkpoint(model, root,
                                  [SimpleNamespace(next_index=i + config.n_envs) for i in state["next_indices"]],
                                  manifest["fingerprint"], interrupted=interrupted)
        if interrupted:
            calls["interrupt_once"] = False
            append_jsonl(root / "episodes.jsonl", {"outcome": "aborted", "reason": "KeyboardInterrupt: "})
            append_jsonl(root / "sessions.jsonl", {"status": "interrupted", "checkpoint": saved})
            raise KeyboardInterrupt
        append_jsonl(root / "episodes.jsonl", {"outcome": "death"})
        for rank in range(config.n_envs):
            append_jsonl(root / "episodes.jsonl", {"outcome": "aborted",
                        "reason": "training budget ended; unfinished episode discarded"})
        if calls["engine_abort"]:
            append_jsonl(root / "episodes.jsonl", {"outcome": "aborted", "reason": "SimulatorError: illegal pick_potion"})
        append_jsonl(root / "sessions.jsonl", {"status": "completed", "checkpoint": saved})
        return {"status": "completed", "total_steps": model.num_timesteps, "added_steps": addition,
                "requested_steps": steps, "checkpoint": saved}

    monkeypatch.setattr(module, "assess", assess)
    monkeypatch.setattr(runner, "train", train)
    monkeypatch.setitem(module.tracer, "replay", replay)

    def forbidden(*args, **kwargs):
        pytest.fail("Combined sessions must not imitate, reset the optimizer, or promote")
    monkeypatch.setattr(runner, "promote", forbidden)
    monkeypatch.setitem(module.trial["selection"], "fresh_policy", forbidden)
    return SimpleNamespace(module=module, source=source, ancestor=ancestor, series=series,
                           upstream=upstream, config=config, manifest=manifest, initial=initial,
                           final=final, lineage=lineage, before=before, after=after,
                           publish_source=publish_source, calls=calls, Isolation=Isolated)


def execute(setup, *, session=1, steps=240000, chunk_steps=80000, episodes=1000):
    return setup.module.progress(setup.source, setup.series, setup.upstream,
                                 session=session, steps=steps, chunk_steps=chunk_steps, episodes=episodes)


def create(setup, *, steps=240000, chunk_steps=80000, episodes=1000):
    return setup.module.create_session(setup.source, setup.series / "session-0001", setup.upstream,
                                      session=1, series=setup.series, steps=steps,
                                      chunk_steps=chunk_steps, episodes=episodes, base_source=setup.source)


def test_import_retains_action_value_adam_counter_and_seed_highwaters(setup):
    frozen_source, frozen_ancestor = bytes_in(setup.source), bytes_in(setup.ancestor)
    lineage = create(setup)
    root = setup.series / "session-0001"
    imported = lineage["initial_checkpoint"]
    assert imported["num_timesteps"] == 404536
    assert lineage["target_total_steps"] == 644536
    assert imported["next_indices"] == [4880, 4881, 4882, 4883]
    assert lineage["training_seed_indices"] == imported["next_indices"]
    assert read_json(root / "experiment.json")["config"] == asdict(setup.config)
    source_model = PreservedPolicy.load(setup.source / setup.final["model"])
    imported_model = PreservedPolicy.load(root / imported["model"])
    tensors_equal(source_model, imported_model)
    assert {key.split(".")[0] for key in imported_model.policy.state_dict()} == {"action_net", "value_net"}
    assert bytes_in(setup.source) == frozen_source and bytes_in(setup.ancestor) == frozen_ancestor
    assert not (root / "champion.json").exists() and setup.calls["training"] == []


def test_factory_reconstructs_exactly_one_event_and_potion_wrapper_and_restores_context(setup):
    lineage = create(setup)
    original = runner._make
    with pytest.raises(RuntimeError, match="exercise cleanup"):
        with setup.module.factory(lineage["descriptor"], lineage["plan"], lineage["source_state"]):
            env = runner._make(setup.config, setup.upstream)
            layers, layer = [], env
            while hasattr(layer, "env"):
                layers.append(layer)
                layer = layer.env
            assert sum(isinstance(layer, setup.Isolation) for layer in layers) == 1
            assert sum(isinstance(layer, PotionCapacityMask) for layer in layers) == 1
            assert env.observation_space == PreservedPolicy.observation_space
            assert env.action_space == PreservedPolicy.action_space
            env.close()
            raise RuntimeError("exercise cleanup")
    assert runner._make is original


def test_global_allowance_subtracts_actual_rollout_steps_and_evaluates_only_endpoints(setup):
    result = execute(setup)
    root = setup.series / "session-0001"
    assert setup.calls["training"] == [80000, 80000, 78208]
    assert verified_checkpoint(root)[1]["num_timesteps"] == 645176
    assert result["new_rl_steps"] == 240640
    assert result["cumulative_rl_steps"] == 645176
    assert [call[:3] for call in setup.calls["assessments"]] == [
        ("before_rl", 1000, 8800), ("after_rl", 1000, 8800)]
    assert result["champion_updated"] is False


def test_completed_repeat_preserves_all_bytes_and_never_allocates_another_budget(setup):
    first = execute(setup)
    frozen = bytes_in(setup.series)
    calls = deepcopy(setup.calls)
    second = execute(setup)
    assert second == first
    assert bytes_in(setup.series) == frozen
    assert setup.calls == calls


def test_final_publication_interruption_recovers_reports_without_more_games_or_training(setup):
    first = execute(setup, steps=1500, chunk_steps=700, episodes=25)
    root = setup.series / "session-0001"
    lineage = read_json(root / "session-lineage.json")
    lineage.pop("session_evidence_sha256")
    lineage.pop("session_progress_sha256")
    atomic_json(root / "session-lineage.json", lineage)
    for name in ("session-progress.json", "report.txt", "report.json", "diagnostics.zip"):
        (root / name).unlink()
    frozen = bytes_in(root)
    calls = deepcopy(setup.calls)
    # Starting another session cannot silently finalize this incomplete parent.
    with pytest.raises(ValueError, match="publishing|previous|original"):
        setup.module.load_source(root, setup.upstream)
    assert bytes_in(root) == frozen and setup.calls == calls
    recovered = execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert recovered == first and setup.calls == calls
    republished = read_json(root / "session-lineage.json")
    assert republished["session_evidence_sha256"] == sha256(root / "session-evidence.json")
    assert republished["session_progress_sha256"] == sha256(root / "session-progress.json")
    assert all((root / name).is_file() for name in ("report.txt", "report.json", "diagnostics.zip"))


@pytest.mark.parametrize("change", [{"steps": 240001}, {"chunk_steps": 81000}, {"episodes": 1001}])
def test_repeat_cannot_change_fixed_budget_or_reserved_evaluation(setup, change):
    execute(setup)
    calls = deepcopy(setup.calls)
    with pytest.raises(ValueError, match="fixed|protocol|original"):
        execute(setup, **change)
    assert setup.calls == calls


def test_interruption_resumes_only_remaining_counter_allowance(setup):
    setup.calls["interrupt_once"] = True
    with pytest.raises(KeyboardInterrupt):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert verified_checkpoint(setup.series / "session-0001")[1]["num_timesteps"] == 405048
    execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert setup.calls["training"] == [700, 700]
    assert verified_checkpoint(setup.series / "session-0001")[1]["num_timesteps"] == 406072
    assert [call[0] for call in setup.calls["assessments"]] == ["before_rl", "after_rl"]


def test_explicit_second_session_imports_previous_final_and_reserves_fresh_seeds(setup):
    first = execute(setup, steps=1500, chunk_steps=700, episodes=25)
    first_root = setup.series / "session-0001"
    frozen = bytes_in(first_root)
    second = execute(setup, session=2, steps=1024, chunk_steps=1024, episodes=25)
    second_root = setup.series / "session-0002"
    lineage = read_json(second_root / "session-lineage.json")
    assert lineage["source_state"] == first["checkpoint"]
    tensors_equal(PreservedPolicy.load(first_root / first["checkpoint"]["model"]),
                  PreservedPolicy.load(second_root / lineage["initial_checkpoint"]["model"]))
    assert second["source_rl_steps"] == first["cumulative_rl_steps"]
    assert second["validation_start"] == first["validation_start"] + 25
    first_seeds = {row["seed"] for row in read_json(first_root / "after_rl.json")["episodes"]}
    second_seeds = {row["seed"] for row in read_json(second_root / "before_rl.json")["episodes"]}
    assert first_seeds.isdisjoint(second_seeds)
    assert bytes_in(first_root) == frozen


@pytest.mark.parametrize("target", ["source_report", "source_checkpoint", "source_ancestor"])
def test_source_tampering_rejected_before_import_or_training(setup, target):
    if target == "source_report":
        path = setup.source / "after_rl.json"
    elif target == "source_checkpoint":
        path = setup.source / setup.final["model"]
    else:
        path = setup.ancestor / "diagnostics/event-proof/report.json"
    path.write_bytes(path.read_bytes() + b"\nchanged bytes")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        create(setup)
    assert not (setup.series / "session-0001").exists()
    assert setup.calls["training"] == []


def test_source_model_counter_disagreement_rejected_despite_rehashed_checkpoint(setup):
    path = setup.source / setup.final["model"]
    wrong = PreservedPolicy.load(path)
    wrong.num_timesteps += 4
    wrong.save(path)
    setup.final["sha256"] = sha256(path)
    atomic_json(setup.source / "latest.json", setup.final)
    setup.after["checkpoint"] = deepcopy(setup.final)
    setup.publish_source()
    with pytest.raises(ValueError, match="counter|timestep"):
        create(setup)
    assert setup.calls["training"] == []


def test_latest_counter_metadata_cannot_claim_unsaved_decisions(setup):
    create(setup, steps=1500, chunk_steps=700, episodes=25)
    root = setup.series / "session-0001"
    state = read_json(root / "latest.json")
    state["num_timesteps"] += 4
    atomic_json(root / "latest.json", state)
    with pytest.raises(ValueError, match="counter|timestep"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert setup.calls["training"] == []


def test_wrong_worker_rank_cursor_rejected_without_migrating_the_model(setup):
    atomic_json(setup.source / "seed-cursors/worker-0.json", {"next_index": 1})
    with pytest.raises(ValueError, match="rank|cursor"):
        create(setup)
    assert not (setup.series / "session-0001").exists()
    assert setup.calls["training"] == []


def test_cached_raw_evaluation_changes_block_more_training(setup):
    create(setup, steps=1500, chunk_steps=700, episodes=25)
    root = setup.series / "session-0001"
    lineage = read_json(root / "session-lineage.json")
    with setup.module.factory(lineage["descriptor"], lineage["plan"], lineage["source_state"]):
        manifest = runner.initialize(root, setup.config, setup.upstream)
        setup.module.cached_assessment(root, setup.upstream, manifest, lineage,
                                       "before_rl", lineage["initial_checkpoint"])
    path = root / "before_rl.json"
    raw = read_json(path)
    raw["episodes"][0]["floor"] = 99
    raw["gameplay"] = summarize(raw["episodes"])
    atomic_json(path, raw)
    with pytest.raises(ValueError, match="checksum|changed"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert setup.calls["training"] == []


@pytest.mark.parametrize("refresh_hash", [False, True])
def test_session_parent_protocol_tamper_blocks_explicit_next_session(setup, refresh_hash):
    execute(setup, steps=1500, chunk_steps=700, episodes=25)
    root = setup.series / "session-0001"
    path = root / "session-protocol.json"
    protocol = read_json(path)
    protocol["steps"] = 999999
    atomic_json(path, protocol)
    if refresh_hash:
        lineage = read_json(root / "session-lineage.json")
        lineage["protocol_sha256"] = sha256(path)
        atomic_json(root / "session-lineage.json", lineage)
    previous_calls = deepcopy(setup.calls)
    with pytest.raises(ValueError, match="protocol|plan|fixed|checksum"):
        execute(setup, session=2, steps=1024, chunk_steps=1024, episodes=25)
    assert setup.calls == previous_calls


def test_training_engine_failure_pauses_instead_of_allocating_remaining_chunks(setup):
    setup.calls["engine_abort"] = True
    with pytest.raises(ValueError, match="abort|Simulator|simulator|health"):
        execute(setup)
    assert setup.calls["training"] == [80000]
    assert [call[0] for call in setup.calls["assessments"]] == ["before_rl"]


def source_timeouts(setup, count):
    for row in setup.after["episodes"][:count]:
        row.update(outcome="timeout", hp=10.0, steps=4000, **{"return": 0.0})
    setup.publish_source()


def test_source_known_timeouts_replay_original_policy_and_cap_without_rewriting_source(setup):
    source_timeouts(setup, 2)
    assert read_json(setup.source / "event-progress.json")["status"] == "incomplete_evaluation"
    frozen = bytes_in(setup.source)
    result = execute(setup, steps=1024, chunk_steps=1024, episodes=25)
    assert result["status"] == "completed"
    assert setup.calls["training"] == [1024]
    assert len(setup.calls["replays"]) == 2
    assert all(row["root"] == str(setup.source) and row["checkpoint_sha256"] == setup.final["sha256"]
               and row["cap"] == 4000 for row in setup.calls["replays"])
    assert [row["original_result"] for row in setup.calls["replays"]] == [
        {key: row[key] for key in setup.module.tracer["GAME_FIELDS"]} for row in setup.after["episodes"][:2]]
    assert bytes_in(setup.source) == frozen


def test_five_source_timeouts_use_four_bounded_replays_and_disclose_untraced_case(setup):
    source_timeouts(setup, 5)
    frozen = bytes_in(setup.source)
    result = execute(setup, steps=1024, chunk_steps=1024, episodes=25)
    assert result["status"] == "completed" and setup.calls["training"] == [1024]
    assert len(setup.calls["replays"]) == 4
    assert result["source_replays"]["total_failed_cases"] == 5
    assert result["source_replays"]["untraced_cases"] == 1
    assert len(result["source_replays"]["runs"]) == 4
    assert bytes_in(setup.source) == frozen


def test_capped_timeouts_remain_explicit_in_all_attempt_denominator_and_do_not_prevent_training(setup):
    setup.calls["before_timeouts"] = 1
    setup.calls["after_timeouts"] = 5
    setup.calls["after_clears"] = 1
    result = execute(setup, steps=1024, chunk_steps=1024, episodes=25)
    assert result["status"] == "completed" and setup.calls["training"] == [1024]
    assert result["before_rl"]["outcomes"]["timeout"] == 1
    assert result["after_rl"]["outcomes"]["timeout"] == 5
    assert result["after_rl"]["outcomes"]["win"] == 1
    assert result["after_rl"]["episodes"] == 25
    assert result["after_rl"]["clear_rate"] == 1 / 25
    assert result["rl_act1_clear_rate_gain"] == 1 / 25
    assert len(setup.calls["replays"]) == 5  # One baseline plus four final, not five final.
    after = read_json(setup.series / "session-0001" / "after_rl.json")
    assert sum(row["outcome"] == "timeout" for row in after["episodes"]) == 5


@pytest.mark.parametrize("failure", ["mismatch", "engine", "logging", "logging_count", "cap", "checkpoint"])
def test_source_replay_mismatch_engine_or_logging_failure_pauses_before_new_allowance(setup, failure):
    source_timeouts(setup, 1)
    setup.calls["replay_failure"] = failure
    frozen = bytes_in(setup.source)
    with pytest.raises(ValueError, match="replay|logging|diagnos|mismatch|pause"):
        execute(setup, steps=1024, chunk_steps=1024, episodes=25)
    assert setup.calls["training"] == []
    assert bytes_in(setup.source) == frozen


def test_changed_cached_source_replay_evidence_blocks_training_without_replaying(setup):
    source_timeouts(setup, 1)
    lineage = create(setup, steps=1500, chunk_steps=700, episodes=25)
    root = setup.series / "session-0001"
    paths = [root / name for name in lineage["source_replay_files"] if name.endswith("/report.json")]
    path = next(path for path in paths if read_json(path).get("runs"))
    path.write_bytes(path.read_bytes() + b"\n")
    frozen_replays = deepcopy(setup.calls["replays"])
    with pytest.raises(ValueError, match="source|replay|diagnostic|checksum|changed"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert setup.calls["training"] == []
    assert setup.calls["replays"] == frozen_replays


def test_source_aborted_game_is_not_eligible_for_another_allowance(setup):
    setup.after["episodes"][0].update(outcome="aborted", error="SimulatorError: rejected legal action")
    setup.publish_source()
    frozen = bytes_in(setup.source)
    with pytest.raises(ValueError, match="abort|error|simulator"):
        create(setup)
    assert setup.calls["training"] == [] and bytes_in(setup.source) == frozen


def test_aborted_baseline_prevents_any_training_and_writes_safe_pause_bundle(setup):
    setup.calls["before_aborts"] = 1
    with pytest.raises(ValueError, match="abort|error|simulator"):
        execute(setup)
    assert setup.calls["training"] == []
    bundles = list(setup.series.rglob("diagnostics.zip"))
    assert len(bundles) == 1
    with zipfile.ZipFile(bundles[0]) as bundle:
        names = bundle.namelist()
        assert any(name.endswith("before_rl.json") for name in names)
        assert any(name.endswith("session-lineage.json") for name in names)
        assert any(name.endswith("session-protocol.json") for name in names)
        assert not any("checkpoints" in name or name.endswith("actions.jsonl") or
                       name.endswith("observations.jsonl") for name in names)
        assert bundle.testzip() is None
    assert bundles[0].stat().st_size < 500000
