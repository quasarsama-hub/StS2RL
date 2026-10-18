"""One controlled, resumable trial of public card-choice semantic features.

Both arms inherit the same completed PPO state and fixed additional allowance.
The treatment starts with a zero residual, preserving the original policy and
value function, and adds public card descriptions to the feature extractor.
The only strength comparison is treatment versus continued control on one
fresh, reserved final seed block. No promotion or automatic extension occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from spire2_rl import runner
from spire2_rl.experiment import (ExperimentConfig, atomic_json, commit_checkpoint,
                                 experiment_lock, read_json, sha256)
from spire2_rl.study import evaluation_seeds, seed_for

SCRIPT_DIR = Path(__file__).resolve().parent
sessions = runpy.run_path(str(SCRIPT_DIR / "run_learning_session.py"))
reports = sessions["reports"]
ARMS = ("control", "treatment")
DEFAULT_SOURCE = Path("experiments/ironclad-a0-learning-sessions/session-0001")
DEFAULT_ROOT = Path("experiments/ironclad-a0-learning-sessions/session-choice-features-trial")
ENCODER_PATH = SCRIPT_DIR.parent / "src/spire2_rl/choice_encoder.py"
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_FILE_BYTES = 4 * 1024 * 1024


def encoder_api():
    from spire2_rl import choice_encoder
    return choice_encoder


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def snapshot_files(root):
    """Guard every original source file, excluding only its temporary lock."""
    root = Path(root).resolve()
    return {str(path): sha256(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != ".running.lock"}


def check_source(lineage):
    sessions["check_files"](lineage["source_files"])
    if snapshot_files(lineage["source_experiment"]) != lineage["source_snapshot"]:
        raise ValueError("The original source directory changed during the trial")


def implementation_pins():
    return {str(path): sha256(path) for path in (
        Path(__file__), ENCODER_PATH, SCRIPT_DIR / "run_learning_session.py",
        SCRIPT_DIR / "learning_session_report.py", SCRIPT_DIR / "learning_session_trace.py")}


def _tensor_tree_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _tensor_tree_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _tensor_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def control_identity(source, control):
    weights = _tensor_tree_equal(source.policy.state_dict(), control.policy.state_dict())
    optimizer = _tensor_tree_equal(source.policy.optimizer.state_dict(), control.policy.optimizer.state_dict())
    counter = int(source.num_timesteps) == int(control.num_timesteps)
    if not weights or not optimizer or not counter:
        raise ValueError("Control import changed source policy, value, optimizer or counter")
    return {"old_parameter_tensors_equal": weights, "old_optimizer_states_equal": optimizer,
            "num_timesteps": int(control.num_timesteps), "source_policy_and_value_preserved": True}


@contextmanager
def arm_factory(descriptor, plan, state, arm, bank_metadata):
    if arm not in ARMS:
        raise ValueError("Unknown trial arm")
    with sessions["factory"](descriptor, plan, state):
        original = runner._make
        def make(config, upstream):
            env = original(config, upstream)
            env.backend_metadata.update(
                choice_features_trial="public-choice-semantics-controlled-v1",
                choice_features_trial_runner_sha256=sha256(Path(__file__)),
                choice_features_encoder_sha256=sha256(ENCODER_PATH),
                choice_features_public_bank_sha256=bank_metadata["bank_sha256"],
                choice_features_public_mapping_sha256=bank_metadata["mapping_sha256"],
                choice_features_trial_plan=dict(plan), choice_features_trial_arm=arm,
                choice_features_trial_source_checkpoint=dict(state),
            )
            return env
        runner._make = make
        try:
            yield
        finally:
            runner._make = original


def immutable_protocol(lineage):
    return {key: lineage[key] for key in (
        "source_experiment", "source_state", "source_files", "source_snapshot", "descriptor",
        "config", "plan", "training_seed_indices", "public_bank_metadata", "implementations",
        "source_replays", "source_replay_files", "arms")} | {
            "split": "validation", "episodes": lineage["plan"]["episodes"],
            "seed_start": lineage["plan"]["validation_start"]}


def validate_protocol(root, lineage):
    path = Path(root) / "session-protocol.json"
    if (not path.is_file() or sha256(path) != lineage.get("protocol_sha256")
            or read_json(path) != immutable_protocol(lineage)):
        raise ValueError("Immutable choice-feature trial protocol changed")
    if lineage["implementations"] != implementation_pins():
        raise ValueError("Choice-feature trial implementation changed; preserve this trial")
    for name, digest in lineage["source_replay_files"].items():
        replay = (Path(root) / name).resolve()
        if not replay.is_relative_to(Path(root).resolve()) or not replay.is_file() or sha256(replay) != digest:
            raise ValueError("Frozen source replay evidence changed")
    check_source(lineage)


def validate_budget(lineage, state, config, *, final=False):
    start, target = lineage["source_state"]["num_timesteps"], lineage["plan"]["target_total_steps"]
    if (target != start + lineage["plan"]["new_steps"] or state["num_timesteps"] < start
            or state["num_timesteps"] >= target + config.n_envs * config.n_steps):
        raise ValueError("Arm checkpoint lies outside its fixed global allowance")
    if final and (state["interrupted"] or state["num_timesteps"] < target):
        raise ValueError("Finish this arm's fixed allowance before final evaluation")


def fresh_schedule(parent, source, root, episodes):
    seeds, start, files = sessions["fresh_schedule"](parent, Path(source).parent, episodes)
    # Audits can reserve source-evaluation subsets; any new diagnostic blocks
    # elsewhere are also pinned rather than silently reused by this experiment.
    audit_series = Path(source).parent.parent / "ironclad-a0-learning-session-audits"
    used = set()
    for path in sorted(audit_series.glob("session-*/protocol.json")):
        document = read_json(path)
        files[str(path)] = sha256(path)
        used.update(sessions["isolated"]["strength_checker"]["raw_seeds"](document))
        reservation = sessions["_reservation"](document)
        if reservation is not None:
            count, block_start = reservation
            if isinstance(count, int) and isinstance(block_start, int) and count > 0 and block_start >= 0:
                used.update(evaluation_seeds("validation", count, parent["config"].experiment_seed, start=block_start))
                start = max(start, block_start + count)
    seeds = evaluation_seeds("validation", episodes, parent["config"].experiment_seed, start=start)
    if used.intersection(seeds):
        raise ValueError("Final choice-feature validation block overlaps a prior audit")
    return seeds, start, files


def initial_observations(vec, model, indices, config):
    """Public start states; this does not reserve or consume journalled seeds."""
    rows, masks = [], []
    for rank, env in enumerate(vec.envs):
        obs, _ = env.reset(seed=seed_for(indices[rank], "train", config.experiment_seed))
        rows.append(obs)
        masks.append(env.action_masks())
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}, np.stack(masks)


def create_trial(source, root, upstream, *, steps, chunk_steps, episodes):
    source, root, upstream = map(lambda path: Path(path).resolve(), (source, root, upstream))
    if root.exists() or root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Use a new separate directory for this controlled trial")
    with sessions["trial"]["selection"]["workflow_lock"](source.parent), experiment_lock(source):
        parent = sessions["load_source"](source, upstream)
        if parent["kind"] != "session":
            raise ValueError("Choice-feature trial starts from the completed learning session")
        config, state = parent["config"], parent["state"]
        if (config.n_envs, config.n_steps) != (4, 256):
            raise ValueError("This trial preserves the original four-by-256 PPO rollout")
        source_snapshot = snapshot_files(source)
        _, start, schedule_files = fresh_schedule(parent, source, root, episodes)
        parent["source_files"].update(schedule_files)
        plan = {"schema": "choice-features-controlled-trial-v1", "new_steps": steps,
                "chunk_steps": chunk_steps, "episodes": episodes, "validation_start": start,
                "target_total_steps": state["num_timesteps"] + steps,
                "training_threads": 2, "device": "cpu", "replay_cap_per_arm": sessions["REPLAY_CAP"],
                "primary_outcome": "Treatment versus continued control: Act 1 clear within the unchanged cap",
                "treatment": "zero-residual public card-choice semantic features",
                "control": "unchanged continued PPO", "automatic_next_allowance": False,
                "champion_updated": False, "inference": "deterministic", "before_evaluation": False}
        indices = sessions["trial"]["source_indices"](source, state, config)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            parent["replay_root"] = staging
            replays = sessions["source_replays"](source, upstream, parent)
            replay_files = {str(path.relative_to(staging)): sha256(path)
                            for path in (staging / "diagnostics").rglob("*.json")}
            with sessions["factory"](parent["descriptor"], plan, state):
                env = runner._make(config, upstream)
                try:
                    _, bank_metadata = encoder_api().build_public_card_bank(upstream, env.observation_space)
                finally:
                    env.close()
            arms = {}
            for arm in ARMS:
                branch = staging / "branches" / arm
                with arm_factory(parent["descriptor"], plan, state, arm, bank_metadata):
                    manifest = runner.initialize(branch, config, upstream)
                    vec = DummyVecEnv([lambda: runner._make(config, upstream) for _ in range(config.n_envs)])
                    try:
                        original = MaskablePPO.load(parent["checkpoint"], device="cpu")
                        if int(original.num_timesteps) != state["num_timesteps"]:
                            raise ValueError("Source model counter differs from checkpoint metadata")
                        if arm == "control":
                            model = MaskablePPO.load(parent["checkpoint"], env=vec, device="cpu", force_reset=True)
                            migration = control_identity(original, model)
                        else:
                            model, migration = encoder_api().migrate_model(original, vec, config, upstream)
                            for field in ("old_parameter_tensors_equal", "old_optimizer_states_equal",
                                          "initial_feature_residual_zero", "rng_preserved"):
                                if migration.get(field) is not True:
                                    raise ValueError("Treatment migration did not preserve " + field)
                            if migration.get("semantic_bank_metadata") != bank_metadata:
                                raise ValueError("Treatment migration used a different public card bank")
                        if (int(model.num_timesteps) != state["num_timesteps"]
                                or model.observation_space != vec.observation_space or model.action_space != vec.action_space):
                            raise ValueError("Trial import changed counter or public spaces")
                        observations, masks = initial_observations(vec, model, indices, config)
                        identity = encoder_api().verify_initial_identity(original, model, observations, masks)
                        if identity.get("distribution_exact") is not True or identity.get("value_exact") is not True:
                            raise ValueError("Trial initialization changed action distribution or value prediction")
                        vec._seeds = [None] * config.n_envs
                        initial = commit_checkpoint(model, branch,
                            [SimpleNamespace(next_index=i) for i in indices], manifest["fingerprint"])
                        arms[arm] = {"fingerprint": manifest["fingerprint"], "initial_checkpoint": initial,
                                     "migration": migration, "initial_identity": identity}
                    finally:
                        vec.close()
            lineage = {"source_experiment": str(source), "source_state": state,
                       "source_files": parent["source_files"], "source_snapshot": source_snapshot,
                       "descriptor": parent["descriptor"], "config": asdict(config), "plan": plan,
                       "training_seed_indices": indices, "public_bank_metadata": bank_metadata,
                       "implementations": implementation_pins(), "source_replays": replays,
                       "source_replay_files": replay_files, "arms": arms}
            check_source(lineage)
            atomic_json(staging / "session-protocol.json", immutable_protocol(lineage))
            lineage["protocol_sha256"] = sha256(staging / "session-protocol.json")
            atomic_json(staging / "choice-trial-lineage.json", lineage)
            if root.exists():
                raise ValueError("Trial directory appeared during initialization")
            staging.rename(root)
            return lineage
        except BaseException:
            failed = root.parent / f"{root.name}-initialization-paused-{time.time_ns()}"
            staging.rename(failed)
            raise


@contextmanager
def telemetry(root):
    """Capture each PPO update, including the final one, without changing it."""
    real_class, original_train = runner.MaskablePPO, MaskablePPO.train
    token = str(time.time_ns())
    log_dir = Path(root) / "telemetry" / f"chunk-{token}"
    active = {}
    class ObservedLoad:
        @staticmethod
        def load(*args, **kwargs):
            model = real_class.load(*args, **kwargs)
            model.set_logger(configure(str(log_dir), format_strings=["csv", "json"]))
            active[id(model)] = model.logger
            return model
    def observed_train(model, *args, **kwargs):
        result = original_train(model, *args, **kwargs)
        if id(model) in active:
            model.logger.record("time/total_timesteps", int(model.num_timesteps))
            model.logger.dump(step=int(model.num_timesteps))
        return result
    runner.MaskablePPO, MaskablePPO.train = ObservedLoad, observed_train
    try:
        yield log_dir
    finally:
        runner.MaskablePPO, MaskablePPO.train = real_class, original_train
        for logger in active.values():
            logger.close()


def train_chunk(root, config, upstream, *, steps):
    with telemetry(root):
        return runner.train(Path(root), config, Path(upstream), steps=steps)


def cached_assessment(root, upstream, manifest, lineage, arm, state):
    root, plan = Path(root), lineage["plan"]
    path = root / "after_rl.json"
    pins = lineage.setdefault("evaluation_files", {})
    if path.exists():
        if arm in pins and sha256(path) != pins[arm]:
            raise ValueError("Frozen arm evaluation checksum changed")
        report = read_json(path)
    elif arm in pins:
        raise ValueError("Frozen arm evaluation is missing")
    else:
        report = sessions["assess"](root, upstream, manifest, state, label="after_rl",
                                      episodes=plan["episodes"], validation_start=plan["validation_start"])
        atomic_json(path, report)
    sessions["trial"]["act1"]["validate_assessment"](
        report, manifest, state, "after_rl", plan["episodes"], plan["validation_start"])
    reports["checked_rows"](report["episodes"])
    if arm not in pins:
        pins[arm] = sha256(path)
        atomic_json(root.parents[1] / "choice-trial-lineage.json", lineage)
    return report


def final_summary(root, lineage, config, states, evaluations, replays):
    comparison = reports["compare_attempts"](evaluations["control"]["episodes"], evaluations["treatment"]["episodes"])
    return {"stage": "after_choice_feature_trial", "status": "completed", "victory_scope": "act1",
            "primary_outcome": lineage["plan"]["primary_outcome"],
            "control": comparison["before_rl"], "treatment": comparison["after_rl"], "comparison": comparison,
            "establishes_choice_feature_gain": comparison["establishes_positive_act1_clear_gain"],
            "fixed_additional_steps_per_arm": lineage["plan"]["new_steps"],
            "new_rl_steps_per_arm": {arm: states[arm]["num_timesteps"] - lineage["source_state"]["num_timesteps"] for arm in ARMS},
            "source_rl_steps": lineage["source_state"]["num_timesteps"], "checkpoints": states,
            "validation_start": lineage["plan"]["validation_start"], "training_records": {
                arm: sessions["training_health"](root / "branches" / arm, config, states[arm]) for arm in ARMS},
            "source_replays": lineage["source_replays"], "final_replays": replays,
            "champion_updated": False, "automatic_next_allowance": False,
            "note": "One prespecified treatment-versus-continued-control comparison. Both inherited source policy/value/Adam/counter and seed highwaters; only treatment adds a zero-initialized public choice-feature residual. Same PPO, simulator, actions, masks, rewards, gamma and GAE. Timeouts remain failed clears; infrastructure failures pause. Emulator Act 1; native transfer unverified."}


def write_report(root, summary):
    """Bounded evidence bundle, including PPO CSV/JSON and both raw arms."""
    root = Path(root).resolve()
    atomic_json(root / "report.json", summary)
    lines = ["Slay the Spire 2 controlled choice-feature trial", "Status: " + summary["status"]]
    for arm in ARMS:
        row = summary.get(arm)
        if row:
            lines.append(f"{arm}: {row['act1_clears']}/{row['episodes']} Act 1 clears; {row['outcomes']['timeout']} timeouts; {row['outcomes']['aborted']} simulator aborts.")
    comparison = summary.get("comparison", {})
    if comparison:
        lines += [f"Treatment minus continued control: {comparison['rl_act1_clear_rate_gain']:+.2%}; 95% paired interval: {comparison['rl_act1_clear_rate_gain_ci95']}.",
                  f"Exact paired p-value: {comparison['paired_clear_evidence']['exact_two_sided_p']}; supports feature gain: {summary['establishes_choice_feature_gain']}."]
    if summary.get("error"):
        lines.append("Error: " + summary["error"])
    if summary.get("new_rl_steps_per_arm"):
        lines.append("Additional decisions per arm: " + json.dumps(summary["new_rl_steps_per_arm"], sort_keys=True))
    lines += [summary.get("note", "Fixed trial paused; preserve its evidence and resume only after diagnosis."),
              "No promotion or automatic budget extension. Emulator Act 1 only.",
              "Diagnostics: " + str(root / "diagnostics.zip")]
    temp = root / "report.txt.tmp"
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(root / "report.txt")
    allowed = {"session-protocol.json", "choice-trial-lineage.json", "choice-trial-progress.json",
               "choice-trial-evidence.json", "choice-trial-pause.json", "report.json", "report.txt"}
    branch_names = {"experiment.json", "latest.json", "after_rl.json", "episodes.jsonl", "sessions.jsonl"}
    selected = [root / name for name in sorted(allowed)]
    for arm in ARMS:
        branch = root / "branches" / arm
        selected += [branch / name for name in sorted(branch_names)]
        selected += sorted((branch / "telemetry").glob("chunk-*/progress.csv"))
        selected += sorted((branch / "telemetry").glob("chunk-*/progress.json"))
    selected += sorted(path for path in (root / "diagnostics").rglob("*.json")
                       if path.name in {"protocol.json", "completion.json", "report.json", "summary.json"})
    used, omitted = 0, []
    target = root / "diagnostics.zip.tmp"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES or used + size > MAX_BUNDLE_BYTES - 1048576:
                omitted.append({"file": str(path.relative_to(root)), "reason": "size bound"})
                continue
            archive.write(path, path.relative_to(root))
            used += size
        log = root / "run.log"
        if log.is_file() and not log.is_symlink():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 524288))
                archive.writestr("logs/run.log.tail.txt", stream.read(524288))
        archive.writestr("bundle-manifest.json", json.dumps({"included_evidence_bytes": used, "omitted": omitted,
                            "note": "No model/game files or full action/observation catalogs."}, indent=2))
    if target.stat().st_size > MAX_BUNDLE_BYTES:
        target.unlink()
        raise ValueError("Trial diagnostics exceeded its total size bound")
    target.replace(root / "diagnostics.zip")
    return {"text_report": str(root / "report.txt"), "diagnostics_zip": str(root / "diagnostics.zip")}


def _progress(source, root, upstream, *, steps, chunk_steps, episodes, console):
    if root.exists() and not (root / "choice-trial-lineage.json").exists():
        # Initialization never trains. A previous pause bundle is preserved
        # before a user-requested retry; unknown artifacts require inspection.
        known = {"choice-trial-pause.json", "report.json", "report.txt", "diagnostics.zip", "run.log"}
        if not (root / "choice-trial-pause.json").is_file() or any(
                not path.is_file() or path.name not in known for path in root.iterdir()):
            raise ValueError("Trial initialization is incomplete and contains unknown artifacts; preserve it for inspection")
        root.rename(root.parent / f"{root.name}-preinitialization-pause-{time.time_ns()}")
    if not root.exists():
        with sessions["phase"](root, console, "Verifying source and initializing matched trial arms"):
            create_trial(source, root, upstream, steps=steps, chunk_steps=chunk_steps, episodes=episodes)
    with experiment_lock(root):
        lineage = read_json(root / "choice-trial-lineage.json")
        plan = lineage["plan"]
        if (lineage["source_experiment"] != str(source) or plan["new_steps"] != steps
                or plan["chunk_steps"] != chunk_steps or plan["episodes"] != episodes):
            raise ValueError("Trial source, budget, chunks and final sample are fixed; use its original command")
        validate_protocol(root, lineage)
        config = ExperimentConfig(**lineage["config"])
        evaluations, replays, states = {}, {}, {}
        evidence_path, progress_path = root / "choice-trial-evidence.json", root / "choice-trial-progress.json"
        finished = evidence_path.exists() or progress_path.exists() or bool(lineage.get("evidence_sha256") or lineage.get("progress_sha256"))
        if finished and not evidence_path.is_file():
            raise ValueError("Published final evidence is missing; preserve the trial for inspection")
        for arm in ARMS:
            branch = root / "branches" / arm
            with arm_factory(lineage["descriptor"], plan, lineage["source_state"], arm, lineage["public_bank_metadata"]):
                manifest = runner.initialize(branch, config, upstream)
                if manifest["fingerprint"] != lineage["arms"][arm]["fingerprint"]:
                    raise ValueError("Arm manifest differs from the prespecified factory")
                sessions["trial"]["act1"]["checked_state"](branch, lineage["arms"][arm]["initial_checkpoint"], manifest["fingerprint"])
                while True:
                    with experiment_lock(branch):
                        _, state = sessions["actual_checkpoint"](branch, manifest)
                        validate_budget(lineage, state, config)
                        sessions["training_health"](branch, config, state)
                    remaining = max(0, plan["target_total_steps"] - state["num_timesteps"])
                    if not remaining:
                        validate_budget(lineage, state, config, final=True)
                        break
                    if finished:
                        raise ValueError("Published final evidence points at an unfinished arm")
                    with sessions["phase"](branch, console, f"{arm}: training {min(chunk_steps, remaining):,} decisions ({remaining:,} remain)"):
                        train = train_chunk(branch, config, upstream, steps=min(chunk_steps, remaining))
                    if train.get("status") != "completed":
                        raise ValueError("Training chunk did not complete")
                    with experiment_lock(branch):
                        _, following = sessions["actual_checkpoint"](branch, manifest)
                        if following["num_timesteps"] <= state["num_timesteps"] or following["interrupted"]:
                            raise ValueError("Training did not publish a forward completed checkpoint")
                        validate_budget(lineage, following, config)
                        sessions["training_health"](branch, config, following)
                    check_source(lineage)
                states[arm] = state
        # Neither arm's held-out results are inspected until both fixed
        # training allowances have finished.
        for arm in ARMS:
            branch = root / "branches" / arm
            with arm_factory(lineage["descriptor"], plan, lineage["source_state"], arm, lineage["public_bank_metadata"]):
                manifest = runner.initialize(branch, config, upstream)
                with experiment_lock(branch):
                    with sessions["phase"](branch, console, f"{arm}: evaluating {episodes:,} reserved final runs"):
                        evaluations[arm] = cached_assessment(branch, upstream, manifest, lineage, arm, states[arm])
                        replays[arm] = sessions["diagnostic_replays"](branch, upstream, manifest, states[arm],
                            evaluations[arm], label=f"final-{arm}", out_root=root / "diagnostics" / arm)
        check_source(lineage)
        summary = final_summary(root, lineage, config, states, evaluations, replays)
        evidence = {"control": evaluations["control"], "treatment": evaluations["treatment"],
                    "replays": replays, "summary": summary}
        if finished:
            for name, key in (("choice-trial-evidence.json", "evidence_sha256"), ("choice-trial-progress.json", "progress_sha256")):
                if lineage.get(key) is not None and (not (root / name).is_file() or lineage[key] != sha256(root / name)):
                    raise ValueError("Published choice-feature trial evidence changed")
            if read_json(evidence_path) != evidence or (progress_path.exists() and read_json(progress_path) != summary):
                raise ValueError("Published summary differs from frozen raw arm results")
            if not progress_path.exists():
                atomic_json(progress_path, summary)
            if lineage.get("evidence_sha256") is None or lineage.get("progress_sha256") is None:
                lineage["evidence_sha256"], lineage["progress_sha256"] = sha256(evidence_path), sha256(progress_path)
                atomic_json(root / "choice-trial-lineage.json", lineage)
            report_current = False
            try:
                report_current = read_json(root / "report.json") == summary
            except (OSError, ValueError):
                pass
            if not report_current or any(not (root / name).is_file() for name in ("report.txt", "diagnostics.zip")):
                write_report(root, summary)
            return summary
        atomic_json(root / "choice-trial-evidence.json", evidence)
        atomic_json(root / "choice-trial-progress.json", summary)
        lineage["evidence_sha256"], lineage["progress_sha256"] = sha256(root / "choice-trial-evidence.json"), sha256(root / "choice-trial-progress.json")
        atomic_json(root / "choice-trial-lineage.json", lineage)
        write_report(root, summary)
        return summary


def progress(source, root, upstream, *, steps=240000, chunk_steps=80000, episodes=1000):
    for value, label in ((steps, "steps"), (chunk_steps, "chunk_steps"), (episodes, "episodes")):
        _positive(value, label)
    if chunk_steps > 80000:
        raise ValueError("Recovery chunks must not exceed the original 80,000-decision size")
    source, root, upstream = map(lambda path: Path(path).resolve(), (source, root, upstream))
    if source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Keep the original source and controlled trial directories separate")
    if root.parent != source.parent or not root.name.startswith("session-"):
        raise ValueError("Use a session-* trial directory beside the source so future sessions reserve its validation block")
    console = sys.stdout
    torch.set_num_threads(2)
    # Completed repeats do not append logs or rewrite publication artifacts.
    cached = (root / "choice-trial-evidence.json").exists()
    if cached:
        result = _progress(source, root, upstream, steps=steps, chunk_steps=chunk_steps, episodes=episodes, console=console)
        print((root / "report.txt").read_text(encoding="utf-8"), file=console, flush=True)
        return result
    root.parent.mkdir(parents=True, exist_ok=True)
    log_path = root.parent / f"{root.name}.log"
    with log_path.open("a", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            result = _progress(source, root, upstream, steps=steps, chunk_steps=chunk_steps, episodes=episodes, console=console)
        except BaseException as exc:
            traceback.print_exc()
            log.flush()
            root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, root / "run.log")
            summary = {"stage": "choice_feature_trial_paused", "status": "paused", "error": f"{type(exc).__name__}: {exc}",
                       "champion_updated": False, "automatic_next_allowance": False}
            atomic_json(root / "choice-trial-pause.json", summary)
            write_report(root, summary)
            print(f"Trial paused: {exc}\nDiagnostics: {root / 'diagnostics.zip'}", file=console, flush=True)
            raise
        log.flush()
        shutil.copy2(log_path, root / "run.log")
        write_report(root, result)
    print((root / "report.txt").read_text(encoding="utf-8"), file=console, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--steps", type=int, default=240000)
    parser.add_argument("--chunk-steps", type=int, default=80000)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args(argv)
    return progress(args.source, args.experiment, args.upstream, steps=args.steps,
                    chunk_steps=args.chunk_steps, episodes=args.episodes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
