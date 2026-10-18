"""A semantic representation trial compares two continued, fixed-budget learners.

These tests persist separate real Torch action/value parameters and Adam state.
Training, semantic migration and game results are controlled here; real encoder
and short PPO runtime checks are separate. No fixture result measures strength.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from spire2_rl import runner
from spire2_rl.experiment import (
    append_jsonl, atomic_json, commit_checkpoint, read_json, sha256,
    verified_checkpoint,
)
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import evaluation_seeds, summarize

from test_act1_continuation import load_script
from test_act1_trial import bytes_in
from test_isolated_act1_cycles import PreservedPolicy
from test_learning_session import assessed, setup as session_setup


class TrialPolicy(PreservedPolicy):
    """Persist inherited and fresh residual parameters with actual Adam moments."""

    def __init__(self, num_timesteps=0, weight=4):
        super().__init__(num_timesteps, weight)
        self.gamma = 0.999
        self.gae_lambda = 0.95
        self.n_steps = 256
        self.batch_size = 128
        self.n_epochs = 4
        self.ent_coef = 0.01
        self.learning_rate = 0.0003
        self.clip_range = lambda progress: 0.2
        self.clip_range_vf = None
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.normalize_advantage = True
        self.policy_kwargs = {"features_dim": 256}
        self.n_envs = 4
        self._n_updates = 16
        self._current_progress_remaining = 0.0
        self._total_timesteps = num_timesteps
        self._num_timesteps_at_start = num_timesteps

    def add_residual(self):
        self.policy.residual = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.policy.residual.weight.zero_()
        self.policy.optimizer.param_groups[0]["params"].extend(self.policy.residual.parameters())

    def set_logger(self, logger):
        self.logger = logger

    def train(self):
        """Controlled metrics exercise the actual telemetry wrapper's final dump."""
        self._n_updates += 1
        for name, value in {"train/n_updates": self._n_updates,
                            "train/entropy_loss": -0.7, "train/value_loss": 0.1,
                            "train/explained_variance": 0.0, "train/approx_kl": 0.01,
                            "train/clip_fraction": 0.1}.items():
            self.logger.record(name, value)

    @classmethod
    def load(cls, path, **kwargs):
        if "env" in kwargs:
            assert kwargs["env"].observation_space == cls.observation_space
            assert kwargs["env"].action_space == cls.action_space
        saved = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(saved["num_timesteps"])
        if "residual.weight" in saved["weights"]:
            result.add_residual()
        result.policy.load_state_dict(saved["weights"])
        result.policy.optimizer.load_state_dict(saved["optimizer"])
        return result


@pytest.fixture(scope="module")
def choice_trial():
    return load_script("choice_credit_trial.py")


def inherited_state_equal(source, imported):
    for key, tensor in source.policy.state_dict().items():
        torch.testing.assert_close(tensor, imported.policy.state_dict()[key], rtol=0, atol=0)
    before, after = source.policy.optimizer.state_dict(), imported.policy.optimizer.state_dict()
    old_group, new_group = before["param_groups"][0], after["param_groups"][0]
    assert {key: value for key, value in old_group.items() if key != "params"} == {
        key: value for key, value in new_group.items() if key != "params"}
    assert new_group["params"][:len(old_group["params"])] == old_group["params"]
    for key, entry in before["state"].items():
        for name, value in entry.items():
            torch.testing.assert_close(value, after["state"][key][name], rtol=0, atol=0)


@pytest.fixture
def setup(choice_trial, monkeypatch, tmp_path):
    old_module = load_script("run_learning_session.py")
    old = session_setup.__wrapped__(old_module, monkeypatch, tmp_path)
    monkeypatch.setattr(old_module, "MaskablePPO", TrialPolicy)
    old_module.progress(old.source, old.series, old.upstream)
    source = old.series / "session-0001"
    root = old.series / "session-choice-credit-trial"
    module = choice_trial
    monkeypatch.setattr(module, "sessions", old_module.__dict__)
    monkeypatch.setattr(module, "MaskablePPO", TrialPolicy)
    calls = {"training": [], "assessments": [], "migration": [],
             "interrupt_arm": None, "engine_abort_arm": None,
             "control_timeouts": 0, "treatment_timeouts": 0,
             "control_aborts": 0, "treatment_aborts": 0,
             "control_clears": 0, "treatment_clears": 0}

    bank = np.zeros((3, 2, 24), dtype=np.float32)
    bank_metadata = {"schema": "public-choice-card-residual-v1", "bank_shape": list(bank.shape),
                     "bank_sha256": "controlled-public-bank", "mapping_sha256": "controlled-card-map",
                     "card_names_by_integer_id": [None, "STRIKE_IRONCLAD", "SHRUG_IT_OFF"]}

    def public_bank(upstream, observation_space):
        return bank.copy(), deepcopy(bank_metadata)

    def migrate(source_model, vec, config, upstream):
        calls["migration"].append(source_model.num_timesteps)
        candidate = deepcopy(source_model)
        candidate.add_residual()
        inherited_state_equal(source_model, candidate)
        assert candidate.policy.optimizer.state_dict()["param_groups"][0]["params"] == [0, 1, 2, 3, 4]
        assert 4 not in candidate.policy.optimizer.state_dict()["state"]
        return candidate, {"schema": "public-choice-card-residual-v1",
            "old_parameter_tensors_equal": True, "old_optimizer_states_equal": True,
            "initial_feature_residual_zero": True, "num_timesteps": candidate.num_timesteps,
            "new_parameters_have_empty_optimizer_state": True, "rng_preserved": True,
            "source_parameters_copied": 4, "optimizer_states_copied": 4,
            "original_optimizer_groups": 1, "new_parameter_groups": 0,
            "new_parameter_tensors": 1, "semantic_bank_metadata": deepcopy(bank_metadata)}

    def initial_identity(source_model, candidate, observations, masks):
        inherited_state_equal(source_model, candidate)
        residual = getattr(candidate.policy, "residual", None)
        if residual is not None:
            assert torch.count_nonzero(residual.weight) == 0
        return {"distribution_exact": True, "value_exact": True,
                "distribution_max_abs_diff": 0.0, "value_max_abs_diff": 0.0}

    monkeypatch.setattr(module, "encoder_api", lambda: SimpleNamespace(
        build_public_card_bank=public_bank, migrate_model=migrate,
        verify_initial_identity=initial_identity))
    monkeypatch.setattr(module, "initial_observations", lambda vec, model, indices, config:
                        ({"visible": np.zeros((config.n_envs, 1), dtype=np.float32)},
                         np.ones((config.n_envs, 2), dtype=bool)))

    def train(root, config, upstream, *, steps, **kwargs):
        root = Path(root)
        arm = root.name
        manifest = read_json(root / "experiment.json")
        model_path, state = verified_checkpoint(root, manifest["fingerprint"])
        model = runner.MaskablePPO.load(model_path)
        calls["training"].append((arm, steps, model.num_timesteps))
        interrupted = calls["interrupt_arm"] == arm
        rollout = config.n_envs * config.n_steps
        addition = 512 if interrupted else ((steps + rollout - 1) // rollout) * rollout
        model.num_timesteps += addition
        model.policy.optimizer.zero_grad()
        model.policy(torch.ones(1, 1)).sum().backward()
        if hasattr(model.policy, "residual"):
            model.policy.residual(torch.ones(1, 1)).sum().backward()
        model.policy.optimizer.step()
        saved = commit_checkpoint(model, root,
                    [SimpleNamespace(next_index=i + config.n_envs) for i in state["next_indices"]],
                    manifest["fingerprint"], interrupted=interrupted)
        model.train()
        if interrupted:
            calls["interrupt_arm"] = None
            append_jsonl(root / "episodes.jsonl", {"outcome": "aborted", "reason": "KeyboardInterrupt: "})
            append_jsonl(root / "sessions.jsonl", {"status": "interrupted", "checkpoint": saved})
            raise KeyboardInterrupt
        append_jsonl(root / "episodes.jsonl", {"outcome": "death"})
        for rank in range(config.n_envs):
            append_jsonl(root / "episodes.jsonl", {"outcome": "aborted",
                "reason": "training budget ended; unfinished episode discarded"})
        if calls["engine_abort_arm"] == arm:
            append_jsonl(root / "episodes.jsonl", {"outcome": "aborted",
                "reason": "SimulatorError: advertised legal action rejected"})
        append_jsonl(root / "sessions.jsonl", {"status": "completed", "checkpoint": saved})
        return {"status": "completed", "checkpoint": saved, "total_steps": model.num_timesteps,
                "added_steps": addition, "requested_steps": steps}

    def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
        root = Path(root)
        arm = root.name
        old_module.trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
        calls["assessments"].append((arm, label, episodes, validation_start, deepcopy(state)))
        report = assessed(manifest, state, label, episodes, validation_start)
        offset = calls[f"{arm}_timeouts"]
        for row in report["episodes"][:offset]:
            row.update(outcome="timeout", steps=4000, hp=10.0, **{"return": 0.0})
        abort_count = calls[f"{arm}_aborts"]
        for row in report["episodes"][offset:offset + abort_count]:
            row.update(outcome="aborted", error="SimulatorError: rejected legal action")
        offset += abort_count
        for row in report["episodes"][offset:offset + calls[f"{arm}_clears"]]:
            row.update(outcome="win", floor=17, **{"return": 1.0})
        report["gameplay"] = summarize(report["episodes"])
        return report

    monkeypatch.setattr(runner, "train", train)
    monkeypatch.setattr(runner, "MaskablePPO", TrialPolicy)
    monkeypatch.setattr(old_module, "assess", assess)
    # Source has no replay case, so any replays here must be newly recorded.
    old.calls["replays"].clear()
    return SimpleNamespace(module=module, source=source, root=root, upstream=old.upstream,
                           config=old.config, old=old, calls=calls, bank_metadata=bank_metadata)


def execute(setup, *, steps=240000, chunk_steps=80000, episodes=1000):
    return setup.module.progress(setup.source, setup.root, setup.upstream,
                                 steps=steps, chunk_steps=chunk_steps, episodes=episodes)


def branch(setup, arm):
    return setup.root / "branches" / arm


def test_fixed_global_budget_spent_equally_on_both_continued_arms_without_starting_eval(setup):
    frozen_source = bytes_in(setup.source)
    result = execute(setup)
    for arm in ("control", "treatment"):
        calls = [(steps, initial) for which, steps, initial in setup.calls["training"] if which == arm]
        assert calls == [(80000, 645176), (80000, 726072), (78208, 806968)]
        state = verified_checkpoint(branch(setup, arm))[1]
        assert state["num_timesteps"] == 885816
        assert not (branch(setup, arm) / "before_rl.json").exists()
        assert not (branch(setup, arm) / "champion.json").exists()
    assert len(setup.calls["assessments"]) == 2
    assert {call[:4] for call in setup.calls["assessments"]} == {
        ("control", "after_rl", 1000, 9800), ("treatment", "after_rl", 1000, 9800)}
    assert all(call[4]["num_timesteps"] == 885816 for call in setup.calls["assessments"])
    assert setup.calls["migration"] == [645176]
    assert bytes_in(setup.source) == frozen_source
    assert result["status"] == "completed"


def test_completed_repeat_returns_cached_result_and_preserves_all_trial_bytes(setup):
    first = execute(setup, steps=1500, chunk_steps=700, episodes=25)
    frozen = bytes_in(setup.root)
    calls = deepcopy(setup.calls)
    second = execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert second == first
    assert bytes_in(setup.root) == frozen
    assert setup.calls == calls


@pytest.mark.parametrize("change", [{"steps": 240001}, {"chunk_steps": 81000}, {"episodes": 1001}])
def test_repeat_cannot_change_fixed_budget_or_prespecified_paired_block(setup, change):
    execute(setup)
    calls = deepcopy(setup.calls)
    with pytest.raises(ValueError, match="fixed|protocol|original|plan|argument"):
        execute(setup, **change)
    assert setup.calls == calls


def test_interrupt_resumes_only_remaining_arm_allowance_without_remigration(setup):
    setup.calls["interrupt_arm"] = "control"
    with pytest.raises(KeyboardInterrupt):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert verified_checkpoint(branch(setup, "control"))[1]["num_timesteps"] == 645688
    execute(setup, steps=1500, chunk_steps=700, episodes=25)
    requests = [steps for arm, steps, initial in setup.calls["training"] if arm == "control"]
    assert requests == [700, 700]
    assert verified_checkpoint(branch(setup, "control"))[1]["num_timesteps"] == 646712
    assert setup.calls["migration"] == [645176]
    assert len(setup.calls["assessments"]) == 2


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_training_simulator_error_blocks_remaining_budget_and_strength_publication(setup, arm):
    setup.calls["engine_abort_arm"] = arm
    with pytest.raises(ValueError, match="abort|Simulator|simulator|health|infrastructure"):
        execute(setup)
    failures = [request for which, request, _ in setup.calls["training"] if which == arm]
    assert failures == [80000]
    assert not (setup.root / "choice-trial-evidence.json").exists()
    assert not any(call[0] == arm for call in setup.calls["assessments"])


def test_source_bytes_tampered_before_initialization_do_not_start_either_arm(setup):
    path = setup.source / "after_rl.json"
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(ValueError, match="changed|checksum|protocol|evaluation"):
        execute(setup)
    assert setup.calls["training"] == [] and setup.calls["migration"] == []


def create(setup, *, steps=1500, chunk_steps=700, episodes=25):
    return setup.module.create_trial(setup.source, setup.root, setup.upstream,
                                     steps=steps, chunk_steps=chunk_steps, episodes=episodes)


def test_initial_branches_preserve_action_value_adam_spaces_config_and_independent_cursors(setup):
    frozen_source = bytes_in(setup.source)
    lineage = create(setup)
    source_state = verified_checkpoint(setup.source)[1]
    source_model = TrialPolicy.load(setup.source / source_state["model"])
    imported = {}
    for arm in ("control", "treatment"):
        root = branch(setup, arm)
        state = verified_checkpoint(root)[1]
        imported[arm] = state
        model = TrialPolicy.load(root / state["model"])
        inherited_state_equal(source_model, model)
        assert state["num_timesteps"] == 645176
        assert state["next_indices"] == lineage["training_seed_indices"] == [4892, 4893, 4894, 4895]
        assert read_json(root / "experiment.json")["config"] == asdict(setup.config)
        assert model.observation_space == source_model.observation_space
        assert model.action_space == source_model.action_space
        if arm == "control":
            assert not hasattr(model.policy, "residual")
        else:
            assert torch.count_nonzero(model.policy.residual.weight) == 0
            assert len(model.policy.optimizer.param_groups) == 1
            assert 4 not in model.policy.optimizer.state_dict()["state"]
    # Each branch owns its cursor state; a write in one branch cannot advance the other.
    changed = deepcopy(imported["control"])
    changed["next_indices"][0] += 4
    atomic_json(branch(setup, "control") / "latest.json", changed)
    assert read_json(branch(setup, "treatment") / "latest.json") == imported["treatment"]
    assert bytes_in(setup.source) == frozen_source and setup.calls["training"] == []


def test_arm_factory_installs_one_private_event_and_potion_layer_and_restores_on_error(setup):
    lineage = create(setup)
    original = runner._make
    with pytest.raises(RuntimeError, match="cleanup"):
        with setup.module.arm_factory(lineage["descriptor"], lineage["plan"], lineage["source_state"],
                                      "treatment", lineage["public_bank_metadata"]):
            env = runner._make(setup.config, setup.upstream)
            layers, layer = [], env
            while hasattr(layer, "env"):
                layers.append(layer)
                layer = layer.env
            assert sum(isinstance(layer, setup.old.Isolation) for layer in layers) == 1
            assert sum(isinstance(layer, PotionCapacityMask) for layer in layers) == 1
            assert env.backend_metadata["choice_features_trial_arm"] == "treatment"
            env.close()
            raise RuntimeError("exercise cleanup")
    assert runner._make is original


def test_pending_trial_reservation_is_visible_to_unchanged_future_session_scheduler(setup):
    lineage = create(setup)
    protocol = read_json(setup.root / "session-protocol.json")
    assert (protocol["split"], protocol["episodes"], protocol["seed_start"]) == ("validation", 25, 9800)
    parent = setup.old.module.load_source(setup.source, setup.upstream)
    seeds, start, files = setup.old.module.fresh_schedule(parent, setup.source.parent, 25)
    assert start == 9825
    assert str(setup.root / "session-protocol.json") in files
    reserved = set(evaluation_seeds("validation", 25, 7, start=lineage["plan"]["validation_start"]))
    assert reserved.isdisjoint(seeds)
    assert setup.calls["training"] == setup.calls["assessments"] == []


@pytest.mark.parametrize("refresh_hash", [False, True])
def test_fixed_protocol_edit_rejected_even_if_outer_checksum_is_refreshed(setup, refresh_hash):
    create(setup)
    path = setup.root / "session-protocol.json"
    protocol = read_json(path)
    protocol["plan"]["new_steps"] += 1000
    atomic_json(path, protocol)
    if refresh_hash:
        lineage_path = setup.root / "choice-trial-lineage.json"
        lineage = read_json(lineage_path)
        lineage["protocol_sha256"] = sha256(path)
        atomic_json(lineage_path, lineage)
    with pytest.raises(ValueError, match="protocol|fixed|immutable"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert setup.calls["training"] == setup.calls["assessments"] == []


def test_latest_metadata_cannot_claim_unsaved_decisions(setup):
    create(setup)
    path = branch(setup, "control") / "latest.json"
    state = read_json(path)
    state["num_timesteps"] += 4
    atomic_json(path, state)
    with pytest.raises(ValueError, match="counter|checkpoint|timestep"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert setup.calls["training"] == setup.calls["assessments"] == []


@pytest.mark.parametrize("tamper", ["raw_evaluation", "derived_summary", "implementation_pin", "source_extra_file"])
def test_completed_frozen_trial_rejects_changed_evidence_without_more_games(setup, tamper):
    execute(setup, steps=1500, chunk_steps=700, episodes=25)
    if tamper == "raw_evaluation":
        path = branch(setup, "treatment") / "after_rl.json"
        raw = read_json(path)
        raw["episodes"][0].update(outcome="win", floor=17, **{"return": 1.0})
        raw["gameplay"] = summarize(raw["episodes"])
    elif tamper == "derived_summary":
        path = setup.root / "choice-trial-progress.json"
        raw = read_json(path)
        raw["establishes_choice_feature_gain"] = True
    elif tamper == "implementation_pin":
        path = setup.root / "choice-trial-lineage.json"
        raw = read_json(path)
        raw["implementations"][str(Path(setup.module.__file__).resolve())] = "wrong source hash"
    else:
        path = setup.source / "unexpected-experiment-change.json"
        raw = {"changed": True}
    atomic_json(path, raw)
    calls = deepcopy(setup.calls)
    with pytest.raises(ValueError, match="changed|checksum|protocol|summary|source"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert setup.calls == calls


def test_final_timeouts_stay_nonclears_in_paired_endpoint_and_bounded_replays_are_disclosed(setup):
    setup.calls.update(control_timeouts=1, treatment_timeouts=5, treatment_clears=1)
    result = execute(setup, steps=1024, chunk_steps=1024, episodes=25)
    assert result["control"]["outcomes"]["timeout"] == 1
    assert result["treatment"]["outcomes"]["timeout"] == 5
    assert result["treatment"]["episodes"] == 25
    assert result["treatment"]["clear_rate"] == 1 / 25
    assert result["comparison"]["rl_act1_clear_rate_gain"] == 1 / 25
    assert result["final_replays"]["treatment"]["total_failed_cases"] == 5
    assert result["final_replays"]["treatment"]["untraced_cases"] == 1
    assert len(setup.old.calls["replays"]) == 5
    raw = read_json(branch(setup, "treatment") / "after_rl.json")
    assert sum(row["outcome"] == "timeout" for row in raw["episodes"]) == 5


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_final_simulator_abort_blocks_strength_publication(setup, arm):
    setup.calls[f"{arm}_aborts"] = 1
    with pytest.raises(ValueError, match="abort|error|simulator"):
        execute(setup, steps=1024, chunk_steps=1024, episodes=25)
    assert not (setup.root / "choice-trial-evidence.json").exists()
    assert read_json(setup.root / "choice-trial-pause.json")["status"] == "paused"


def test_transient_final_publication_failure_recovers_without_more_training_or_evaluation(setup, monkeypatch):
    original = setup.module.atomic_json
    failed = False
    def fault(path, value):
        nonlocal failed
        if Path(path).name == "choice-trial-progress.json" and not failed:
            failed = True
            raise OSError("Controlled transient disk write failure")
        return original(path, value)
    monkeypatch.setattr(setup.module, "atomic_json", fault)
    with pytest.raises(OSError, match="transient"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    calls = deepcopy(setup.calls)
    restored = execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert restored["status"] == "completed" and setup.calls == calls
    lineage = read_json(setup.root / "choice-trial-lineage.json")
    assert lineage["evidence_sha256"] == sha256(setup.root / "choice-trial-evidence.json")
    assert lineage["progress_sha256"] == sha256(setup.root / "choice-trial-progress.json")
    assert all((setup.root / name).is_file() for name in ("report.txt", "report.json", "diagnostics.zip"))


def test_transient_report_failure_recovers_generated_artifacts_without_more_games(setup, monkeypatch):
    original = setup.module.write_report
    failed = False
    def fault(root, summary):
        nonlocal failed
        if summary["status"] == "completed" and not failed:
            failed = True
            raise OSError("Controlled transient report failure")
        return original(root, summary)
    monkeypatch.setattr(setup.module, "write_report", fault)
    with pytest.raises(OSError, match="report failure"):
        execute(setup, steps=1500, chunk_steps=700, episodes=25)
    calls = deepcopy(setup.calls)
    restored = execute(setup, steps=1500, chunk_steps=700, episodes=25)
    assert restored["status"] == "completed" and setup.calls == calls
    assert read_json(setup.root / "report.json")["status"] == "completed"
    assert "Status: completed" in (setup.root / "report.txt").read_text(encoding="utf-8")


def test_both_update_telemetry_and_raw_arm_evidence_are_bundled_without_model_files(setup):
    execute(setup, steps=1500, chunk_steps=700, episodes=25)
    for arm in ("control", "treatment"):
        logs = list((branch(setup, arm) / "telemetry").glob("chunk-*/progress.json"))
        assert len(logs) == 2
        for path in logs:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 1 and rows[0]["train/n_updates"] == 17
            assert "train/entropy_loss" in rows[0] and "train/explained_variance" in rows[0]
    # These files must never enter an evidence bundle.
    atomic_json(branch(setup, "control") / "private-observations.json", {"hidden": "do not bundle"})
    setup.module.write_report(setup.root, read_json(setup.root / "choice-trial-progress.json"))
    with zipfile.ZipFile(setup.root / "diagnostics.zip") as archive:
        names = set(archive.namelist())
        for arm in ("control", "treatment"):
            assert f"branches/{arm}/after_rl.json" in names
            assert f"branches/{arm}/episodes.jsonl" in names
            assert sum(name.startswith(f"branches/{arm}/telemetry/") and name.endswith("progress.json")
                       for name in names) == 2
        assert "session-protocol.json" in names and "choice-trial-evidence.json" in names
        assert not any("checkpoints/" in name or "observations" in name for name in names)
        assert archive.testzip() is None



