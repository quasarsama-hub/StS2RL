"""Private-event PPO forks retain learned state and fixed evaluated budgets."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import ExperimentConfig, atomic_json, commit_checkpoint, read_json, sha256, verified_checkpoint
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds, summarize

from test_act1_continuation import ContinuationPolicy, assessment, assert_optimizer_equal, load_script, rows
from test_act1_trial import SchemaOnlyEnv, bytes_in
from test_cycle_event_check import mechanics_evidence


class ActionValue(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.action_net, self.value_net = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.action_net.weight.fill_(weight)
            self.action_net.bias.fill_(0.2)
            self.value_net.weight.fill_(weight + 3)
            self.value_net.bias.fill_(0.4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, value):
        return self.action_net(value) + self.value_net(value)


class PreservedPolicy(ContinuationPolicy):
    def __init__(self, num_timesteps=0, weight=4):
        self.num_timesteps, self.policy = num_timesteps, ActionValue(weight)


def raw_compare(candidate, baseline):
    """Constant paired changes need no thousands of resamples in unit fixtures."""
    assert [row["seed"] for row in candidate] == [row["seed"] for row in baseline]
    clean = [(a, b) for a, b in zip(candidate, baseline)
             if a["outcome"] in {"win", "death"} and b["outcome"] in {"win", "death"}]
    excluded = [a["seed"] for a, b in zip(candidate, baseline)
                if a["outcome"] not in {"win", "death"} or b["outcome"] not in {"win", "death"}]
    floor = sum(a["floor"] - b["floor"] for a, b in clean) / len(clean) if clean else None
    wins = sum((a["outcome"] == "win") - (b["outcome"] == "win") for a, b in clean) / len(clean) if clean else None
    return {"pairs": len(candidate), "eligible_pairs": len(clean), "excluded_pairs": len(excluded),
            "excluded_seeds": excluded, "candidate": summarize(candidate), "baseline": summarize(baseline),
            "floor_delta": floor, "floor_delta_ci95": [floor, floor] if floor is not None else None,
            "win_delta": wins, "win_delta_ci95": [wins, wins] if wins is not None else None,
            "promote": False, "reasons": [], "promotion_rule": "fixture raw paired evidence"}


@pytest.fixture(scope="module")
def isolated_cycles():
    return load_script("isolated_act1_cycles.py")


@pytest.fixture
def isolated_setup(isolated_cycles, monkeypatch, tmp_path):
    module = isolated_cycles
    source = tmp_path / "original-cycles" / "cycle-0002"
    series, upstream = tmp_path / "isolated-cycles", tmp_path / "upstream"
    config = ExperimentConfig(max_acts=1, n_envs=2, n_steps=4, batch_size=4, n_epochs=1,
                              learning_rate=0.0001)
    base = {"source_lineage": {"retained_base": True}}
    original_plan = {"new_steps": 80000, "episodes": 1000, "validation_start": 3700}
    original_source = {"num_timesteps": 242744, "sha256": "frozen-original-ancestor"}

    @contextmanager
    def original_factory(base_factory, plan, state):
        assert base_factory == base and plan == original_plan and state == original_source
        previous = runner._make
        def make(config, upstream):
            env = PotionCapacityMask(SchemaOnlyEnv(config))
            env.backend_metadata.update(rl_cycle_plan=deepcopy(plan),
                                        rl_cycle_source_checkpoint=deepcopy(state), inherited_base="unchanged")
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
            self.backend_metadata.update(
                event_state_isolation="private-registry-per-run-v1", event_registry_count=68,
                event_isolation_sha256=sha256(Path(module.__file__).parents[1] / "src/spire2_rl/event_isolation.py"))

        @property
        def scope(self):
            return self.env.scope

        @property
        def raw_env(self):
            return self.env.raw_env

    monkeypatch.setitem(module.cycles, "cycle_runner", original_factory)
    monkeypatch.setattr(module, "IsolatedEvents", Isolated)
    monkeypatch.setattr(module, "MaskablePPO", PreservedPolicy)
    monkeypatch.setattr(runner, "provenance", lambda upstream: {"core_source": "unchanged"})
    monkeypatch.setitem(module.cycles["continuation_gate"].__globals__, "paired_compare", raw_compare)
    for checker in (module.checker, module.strength_checker["checker"]):
        monkeypatch.setitem(checker, "fresh_seeds", lambda root, original, config, count, start:
                            evaluation_seeds("validation", count, config.experiment_seed, start=start))

    with original_factory(base, original_plan, original_source):
        manifest = runner.initialize(source, config, upstream)
    wrappers = [SimpleNamespace(next_index=42), SimpleNamespace(next_index=43)]
    previous = commit_checkpoint(PreservedPolicy(num_timesteps=242744), source, wrappers, manifest["fingerprint"])
    model = PreservedPolicy(num_timesteps=323640, weight=7)
    model.policy(torch.ones(1, 1)).sum().backward()
    model.policy.optimizer.step()
    final = commit_checkpoint(model, source, wrappers, manifest["fingerprint"])
    atomic_json(source / "seed-cursors" / "worker-0.json", {"next_index": 50})
    atomic_json(source / "seed-cursors" / "worker-1.json", {"next_index": 55})
    (source / "episodes.jsonl").write_text('{"outcome":"death"}\n')
    (source / "sessions.jsonl").write_text('{"status":"completed"}\n')
    atomic_json(source / "champion.json", {"unchanged_source_only": True})
    old_seeds = evaluation_seeds("validation", 1000, 7, start=3700)
    before = assessment(manifest, previous, "before_rl", old_seeds, 3700, floor=10)
    after = assessment(manifest, final, "after_rl", old_seeds, 3700, floor=10)
    atomic_json(source / "before_rl.json", before)
    atomic_json(source / "after_rl.json", after)
    source_lineage = {"cycle": 2, "plan": original_plan, "source_state": original_source,
                      "initial_checkpoint": previous, "base_factory": base, "config": asdict(config)}
    atomic_json(source / "cycle-lineage.json", source_lineage)
    atomic_json(source / "cycle-progress.json", {"next_cycle_eligible": False})
    source_files = {str(path): sha256(path) for path in source.rglob("*") if path.is_file()}
    parent = {"config": config, "manifest": manifest, "state": final,
              "checkpoint": source / final["model"], "base_factory": base, "lineage": source_lineage,
              "source_files": source_files, "before": before, "after": after,
              "gate": {"eligible": False, "reasons": ["original trial inconclusive"]}}
    monkeypatch.setitem(module.cycles, "load_parent", lambda root, upstream: deepcopy(parent))
    strength = source / "diagnostics" / f"cycle-strength-{final['sha256'][:16]}"
    atomic_json(strength / "protocol.json", {"split": "validation", "episodes": 3000, "seed_start": 4700})
    atomic_json(strength / "report.json", {"frozen_strength": True})
    atomic_json(strength / "completion.json", {"protocol_sha256": sha256(strength / "protocol.json"),
                                               "report_sha256": sha256(strength / "report.json")})
    strength_files = {str(path): sha256(path) for path in strength.glob("*.json")}
    monkeypatch.setitem(module.repair_checker["behavior"], "strength_evidence", lambda root, parent: dict(strength_files))
    proof_folder = source / "diagnostics" / f"event-isolation-{final['sha256'][:16]}"
    with original_factory(base, original_plan, original_source):
        env = Isolated(runner._make(config, upstream))
        identity = env.backend_metadata
        env.close()
    protocol = {"cycle": 2, "fingerprint": manifest["fingerprint"], "checkpoint": final,
                "split": "validation", "episodes": 100, "seed_start": 7700,
                "checker_sha256": sha256(Path(module.__file__).with_name("check_cycle_event_isolation.py")),
                "repair_sha256": sha256(Path(module.__file__).parents[1] / "src/spire2_rl/event_isolation.py"),
                "source_strength_evidence": {**source_files, **strength_files},
                "inference": "deterministic", "victory_scope": "act1",
                "scope": "same-weights event-instance isolation check", "isolated_backend": identity,
                "automatic_training": False}
    proof_seeds = evaluation_seeds("validation", 100, 7, start=7700)
    original = rows(proof_seeds, "same-policy-original", floor=10)
    repaired = rows(proof_seeds, "same-policy-isolated", floor=10)
    report = {"protocol": protocol, "mechanics": mechanics_evidence(),
              "original_episodes": original, "isolated_episodes": repaired,
              "original_gameplay": summarize(original), "isolated_gameplay": summarize(repaired)}
    atomic_json(proof_folder / "protocol.json", protocol)
    atomic_json(proof_folder / "report.json", report)
    atomic_json(proof_folder / "completion.json", {"protocol_sha256": sha256(proof_folder / "protocol.json"),
                                                 "report_sha256": sha256(proof_folder / "report.json")})
    calls = {"training": [], "assessment": [], "before_outcome": "death", "after_floor": 11}

    def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
        module.trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
        calls["assessment"].append((Path(root), label, episodes, validation_start))
        seeds = evaluation_seeds("validation", episodes, 7, start=validation_start)
        result = assessment(manifest, state, label, seeds, validation_start,
                            floor=10 if label == "before_rl" else calls["after_floor"])
        if label == "before_rl" and calls["before_outcome"] != "death":
            result["episodes"][0]["outcome"] = calls["before_outcome"]
            result["gameplay"] = summarize(result["episodes"])
        return result

    def train(root, config, upstream, *, steps):
        calls["training"].append((Path(root), steps))
        manifest = read_json(root / "experiment.json")
        path, state = verified_checkpoint(root, manifest["fingerprint"])
        model = PreservedPolicy.load(path)
        rollout = config.n_envs * config.n_steps
        model.num_timesteps += ((steps + rollout - 1) // rollout) * rollout
        model.policy(torch.ones(1, 1)).sum().backward()
        model.policy.optimizer.step()
        commit_checkpoint(model, root, [SimpleNamespace(next_index=i + config.n_envs) for i in state["next_indices"]],
                          manifest["fingerprint"])
        return {"status": "completed", "total_steps": model.num_timesteps}

    def forbidden(*args, **kwargs):
        pytest.fail("Corrected cycles must not clone, replace PPO optimizer or promote")

    monkeypatch.setattr(module, "assess", assess)
    monkeypatch.setattr(runner, "train", train)
    monkeypatch.setattr(runner, "promote", forbidden)
    monkeypatch.setitem(module.trial["selection"], "fresh_policy", forbidden)
    return SimpleNamespace(source=source, series=series, upstream=upstream, config=config, parent=parent,
                           calls=calls, proof=proof_folder, Isolation=Isolated)


def run(module, setup, cycle=1, **kwargs):
    return module.progress(setup.source, setup.series, setup.upstream, cycle=cycle,
                           new_steps=kwargs.pop("new_steps", 75), episodes=1000, **kwargs)


def create(module, setup, **kwargs):
    return module.create_cycle(setup.source, setup.series / "cycle-0001", setup.upstream, cycle=1,
                               base_source=setup.source, series=setup.series,
                               new_steps=kwargs.pop("new_steps", 80000), episodes=1000, **kwargs)


def test_initial_fork_preserves_policy_value_adam_counter_and_all_original_bytes(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    frozen = bytes_in(setup.source)
    lineage = create(module, setup)
    root = setup.series / "cycle-0001"
    imported = lineage["initial_checkpoint"]
    assert lineage["source_state"] == setup.parent["state"]
    assert imported["num_timesteps"] == 323640 and lineage["target_total_steps"] == 403640
    assert lineage["plan"] == {"new_steps": 80000, "episodes": 1000, "validation_start": 7800}
    assert imported["next_indices"] == lineage["training_seed_indices"] == [50, 55]
    assert imported["fingerprint"] != setup.parent["manifest"]["fingerprint"]
    original = PreservedPolicy.load(setup.parent["checkpoint"])
    migrated = PreservedPolicy.load(root / imported["model"])
    for key, tensor in original.policy.state_dict().items():
        torch.testing.assert_close(migrated.policy.state_dict()[key], tensor, rtol=0, atol=0)
    assert {key.split(".")[0] for key in migrated.policy.state_dict()} == {"action_net", "value_net"}
    assert_optimizer_equal(original, migrated)
    assert bytes_in(setup.source) == frozen and setup.calls["training"] == []
    assert lineage["repair_evidence"]["status"] == "verified"
    assert lineage["parent_gate"]["original_gate"]["eligible"] is False
    assert not (root / "champion.json").exists()


def test_factory_has_one_event_and_one_potion_wrapper_and_restores_exact_original_factory(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    lineage = create(module, setup)
    original_make = runner._make
    with pytest.raises(RuntimeError, match="inside repaired factory"):
        with module.factory(lineage["base_factory"], lineage["original_plan"], lineage["original_source_state"],
                            lineage["plan"], lineage["source_state"]):
            env = runner._make(setup.config, setup.upstream)
            layers, layer = [], env
            while hasattr(layer, "env"):
                layers.append(layer)
                layer = layer.env
            assert sum(isinstance(layer, setup.Isolation) for layer in layers) == 1
            assert sum(isinstance(layer, PotionCapacityMask) for layer in layers) == 1
            assert env.observation_space == PreservedPolicy.observation_space
            assert env.action_space == PreservedPolicy.action_space
            assert env.backend_metadata["rl_cycle_plan"] == lineage["original_plan"]
            assert env.backend_metadata["event_isolated_plan"] == lineage["plan"]
            env.close()
            raise RuntimeError("inside repaired factory")
    assert runner._make is original_make


def test_completed_cycle_repeats_without_replay_training_or_byte_changes(isolated_cycles, isolated_setup):
    first = run(isolated_cycles, isolated_setup)
    root = isolated_setup.series / "cycle-0001"
    frozen = bytes_in(root)
    second = run(isolated_cycles, isolated_setup)
    assert second == first and bytes_in(root) == frozen
    assert first["stage"] == "after_isolated_rl_cycle"
    assert first["source_rl_steps"] == 323640 and first["new_rl_steps"] == 80
    assert first["cumulative_rl_steps"] == 323720 and first["next_cycle_eligible"] is True
    assert first["champion_updated"] is False
    assert isolated_setup.calls["training"] == [(root, 75)]
    assert [call[1:] for call in isolated_setup.calls["assessment"]] == [("before_rl", 1000, 7800), ("after_rl", 1000, 7800)]


def test_second_cycle_uses_flat_original_factory_latest_optimizer_and_new_seed_block(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    first = run(module, setup)
    first_root = setup.series / "cycle-0001"
    first_lineage = read_json(first_root / "event-lineage.json")
    frozen = bytes_in(first_root)
    second = run(module, setup, 2)
    second_root = setup.series / "cycle-0002"
    lineage = read_json(second_root / "event-lineage.json")
    assert lineage["source_state"] == first["checkpoint"]
    assert lineage["base_factory"] == first_lineage["base_factory"]
    assert lineage["original_plan"] == first_lineage["original_plan"]
    assert lineage["original_source_state"] == first_lineage["original_source_state"]
    assert lineage["plan"]["validation_start"] == second["validation_start"] == 8800
    assert second["source_rl_steps"] == 323720 and second["cumulative_rl_steps"] == 323800
    assert_optimizer_equal(PreservedPolicy.load(first_root / first["checkpoint"]["model"]),
                           PreservedPolicy.load(second_root / lineage["initial_checkpoint"]["model"]))
    assert bytes_in(first_root) == frozen


@pytest.mark.parametrize("tamper", ["missing_completion", "raw_mechanics", "model_identity"])
def test_first_fork_requires_complete_checksum_verified_raw_repair_proof(isolated_cycles, isolated_setup, tamper):
    setup = isolated_setup
    if tamper == "missing_completion":
        (setup.proof / "completion.json").unlink()
    else:
        path = setup.proof / ("report.json" if tamper == "raw_mechanics" else "protocol.json")
        document = read_json(path)
        if tamper == "raw_mechanics":
            document["mechanics"]["neow_isolated"]["matches_solo"] = False
            document["mechanics"]["neow_isolated"]["interleaved_actual_reward"] = [{"reward": "wrong-seed"}]
        else:
            document["checkpoint"]["sha256"] = "wrong-policy"
        atomic_json(path, document)
        completion = read_json(setup.proof / "completion.json")
        completion[f"{'report' if tamper == 'raw_mechanics' else 'protocol'}_sha256"] = sha256(path)
        atomic_json(setup.proof / "completion.json", completion)
    frozen = bytes_in(setup.source)
    with pytest.raises(ValueError, match="proof|mechanics|source|check"):
        create(isolated_cycles, setup)
    assert not (setup.series / "cycle-0001").exists() and setup.calls["training"] == []
    assert bytes_in(setup.source) == frozen


def test_initialization_failure_removes_staging_and_preserves_source(isolated_cycles, isolated_setup, monkeypatch):
    setup = isolated_setup
    frozen = bytes_in(setup.source)
    original_factory = runner._make
    class FailedLoad:
        @classmethod
        def load(cls, *args, **kwargs):
            raise RuntimeError("model restoration failed")
    monkeypatch.setattr(isolated_cycles, "MaskablePPO", FailedLoad)
    with pytest.raises(RuntimeError, match="restoration failed"):
        create(isolated_cycles, setup)
    assert not (setup.series / "cycle-0001").exists()
    assert not list(setup.series.glob(".cycle-0001-initializing-*"))
    assert bytes_in(setup.source) == frozen and runner._make is original_factory


def test_partial_checkpoint_resumes_only_remaining_fixed_budget(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    lineage = create(module, setup, new_steps=75)
    root = setup.series / "cycle-0001"
    manifest = read_json(root / "experiment.json")
    model = PreservedPolicy.load(root / lineage["initial_checkpoint"]["model"])
    model.num_timesteps += 32
    commit_checkpoint(model, root, [SimpleNamespace(next_index=52), SimpleNamespace(next_index=57)],
                      manifest["fingerprint"], interrupted=True)
    summary = run(module, setup)
    assert setup.calls["training"] == [(root, 43)]
    assert summary["new_rl_steps"] == 80 and summary["cumulative_rl_steps"] == 323720


@pytest.mark.parametrize("state", ["below_initial", "overshot", "interrupted_target"])
def test_invalid_latest_counter_never_allocates_training(isolated_cycles, isolated_setup, state):
    module, setup = isolated_cycles, isolated_setup
    lineage = create(module, setup, new_steps=75)
    root = setup.series / "cycle-0001"
    manifest = read_json(root / "experiment.json")
    model = PreservedPolicy.load(root / lineage["initial_checkpoint"]["model"])
    model.num_timesteps = (323639 if state == "below_initial" else
                           lineage["target_total_steps"] + 8 if state == "overshot" else
                           lineage["target_total_steps"])
    commit_checkpoint(model, root, [SimpleNamespace(next_index=52), SimpleNamespace(next_index=57)],
                      manifest["fingerprint"], interrupted=state == "interrupted_target")
    with pytest.raises(ValueError, match="budget|Interrupted|counter"):
        run(module, setup)
    assert setup.calls["training"] == []


def test_latest_counter_must_match_embedded_model_before_allocating_remainder(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    create(module, setup, new_steps=75)
    root = setup.series / "cycle-0001"
    latest = read_json(root / "latest.json")
    latest["num_timesteps"] += 32  # The actual checksummed zip still contains 323640.
    atomic_json(root / "latest.json", latest)
    with pytest.raises(ValueError, match="counter|timestep"):
        run(module, setup)
    assert setup.calls["training"] == []


def test_flat_raw_result_blocks_next_cycle_despite_forged_positive_flag(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    setup.calls["after_floor"] = 10
    result = run(module, setup)
    assert result["next_cycle_eligible"] is False
    root = setup.series / "cycle-0001"
    summary_path = root / "event-progress.json"
    summary = read_json(summary_path)
    summary["next_cycle_eligible"] = True
    atomic_json(summary_path, summary)
    lineage_path = root / "event-lineage.json"
    lineage = read_json(lineage_path)
    lineage["event_progress_sha256"] = sha256(summary_path)
    atomic_json(lineage_path, lineage)
    with pytest.raises(ValueError, match="raw|stopped|evidence|summary"):
        run(module, setup, 2)
    assert not (setup.series / "cycle-0002").exists()
    assert len(setup.calls["training"]) == 1


def test_unclean_starting_assessment_and_changed_resume_budget_cannot_train(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    setup.calls["before_outcome"] = "aborted"
    with pytest.raises(ValueError, match="Starting-policy"):
        run(module, setup)
    assert setup.calls["training"] == []
    with pytest.raises(ValueError, match="fixed|original"):
        run(module, setup, new_steps=76)
    assert setup.calls["training"] == []


def test_missing_summary_crash_recovery_preserves_original_training_evidence(isolated_cycles, isolated_setup):
    module, setup = isolated_cycles, isolated_setup
    first = run(module, setup)
    root = setup.series / "cycle-0001"
    evidence = (root / "event-evidence.json").read_bytes()
    lineage_path = root / "event-lineage.json"
    lineage = read_json(lineage_path)
    lineage.pop("event_progress_sha256")
    atomic_json(lineage_path, lineage)
    (root / "event-progress.json").unlink()
    second = run(module, setup)
    assert first == second and (root / "event-evidence.json").read_bytes() == evidence
    assert setup.calls["training"] == [(root, 75)]
