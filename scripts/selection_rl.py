"""Bounded PPO continuation with finite card selection and verified weight transfer.

This additive launcher leaves the original experiment and fingerprinted source
files intact. Use this launcher for the new experiment on subsequent runs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from spire2_rl import runner
from spire2_rl.experiment import (
    ExperimentConfig, atomic_json, commit_checkpoint, experiment_lock,
    read_json, sha256, verified_checkpoint,
)
from spire2_rl.selection import MonotoneSelections
from spire2_rl.study import evaluation_seeds, paired_compare, seed_for, summarize


@contextmanager
def finite_selection_runner():
    """One contained factory override; the normal launcher keeps its old API."""
    original = runner._make

    def make(config, upstream):
        wrapped = MonotoneSelections(original(config, upstream))
        wrapped.backend_metadata["selection_runner_sha256"] = sha256(Path(__file__))
        return wrapped

    runner._make = make
    try:
        yield
    finally:
        runner._make = original


def fresh_policy(source_model, config, vec):
    """Transfer policy/value weights, with a fresh optimizer and rollout state."""
    if source_model.observation_space != vec.observation_space or source_model.action_space != vec.action_space:
        raise ValueError("Weight transfer requires identical observation and action spaces")
    fresh = MaskablePPO(
        source_model.policy_class, vec, learning_rate=config.learning_rate,
        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
        gamma=config.gamma, ent_coef=config.ent_coef,
        policy_kwargs=source_model.policy_kwargs, seed=config.experiment_seed,
        device="cpu", verbose=0,
    )
    fresh.policy.load_state_dict(source_model.policy.state_dict(), strict=True)
    # The journal, not SB3's pending VecEnv seeds, owns training seed assignment.
    vec._seeds = [None] * config.n_envs
    return fresh


def create_warm_experiment(source: Path, root: Path, upstream: Path) -> dict:
    """Build in a private staging directory, then publish the complete experiment."""
    source, root, upstream = source.resolve(), root.resolve(), upstream.resolve()
    if source == root or root.is_relative_to(source):
        raise ValueError("Use a new experiment directory for the changed selection interface")
    if not source.is_dir():
        raise ValueError(f"Source experiment does not exist: {source}")
    if root.exists():
        raise ValueError(f"Destination already exists: {root}")
    with experiment_lock(source):
        source_manifest = read_json(source / "experiment.json")
        config = ExperimentConfig(**source_manifest["config"])
        # Check the source against the original factory, before overriding it.
        runner.initialize(source, config, upstream)
        checkpoint, source_state = verified_checkpoint(source, source_manifest["fingerprint"])
        if source_state.get("interrupted"):
            raise ValueError("Use a completed source checkpoint for this weight transfer")
        indices = list(source_state["next_indices"])
        if len(indices) != config.n_envs:
            raise ValueError("Source checkpoint worker count differs from its config")
        for rank in range(config.n_envs):
            seed_for(indices[rank], "train", config.experiment_seed)
            if indices[rank] % config.n_envs != rank:
                raise ValueError("Source checkpoint seed index has the wrong worker rank")
            cursor = source / "seed-cursors" / f"worker-{rank}.json"
            if cursor.exists():
                persisted_index = read_json(cursor)["next_index"]
                seed_for(persisted_index, "train", config.experiment_seed)
                if persisted_index % config.n_envs != rank:
                    raise ValueError("Source training seed cursor has the wrong worker rank")
                indices[rank] = max(indices[rank], persisted_index)
        source_model = MaskablePPO.load(checkpoint, device="cpu")
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            with finite_selection_runner():
                manifest = runner.initialize(staging, config, upstream)
                vec = DummyVecEnv([lambda: runner._make(config, upstream) for _ in range(config.n_envs)])
                try:
                    model = fresh_policy(source_model, config, vec)
                    initial = commit_checkpoint(
                        model, staging, [SimpleNamespace(next_index=index) for index in indices],
                        manifest["fingerprint"],
                    )
                finally:
                    vec.close()
            if sha256(checkpoint) != source_state["sha256"]:
                raise ValueError("Source checkpoint changed during weight transfer")
            lineage = {
                "transfer": "weights_only", "optimizer": "fresh", "new_num_timesteps": 0,
                "source_experiment": str(source), "source_fingerprint": source_manifest["fingerprint"],
                "source_checkpoint": source_state, "initial_checkpoint": initial,
                "training_seed_indices": indices, "config": asdict(config),
                "note": "Old policy/value weights imported; new optimizer and experiment. Initial evaluation measures the interface change, not additional RL.",
            }
            atomic_json(staging / "warm-start.json", lineage)
            if root.exists():
                raise ValueError("Destination was created by another process during initialization")
            staging.rename(root)
            return lineage
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


def compact(report):
    return {key: report[key] for key in ("candidate", "baseline", "comparison", "report_path")}


def validate_initial_report(report, manifest, initial_state, episodes, validation_start):
    if report["checkpoint"] != initial_state or report["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Initial evaluation does not match the imported policy and current interface")
    if report["seed_start"] != validation_start or report["split"] != "validation":
        raise ValueError("Keep the same --episodes and --validation-start settings for this bounded experiment")
    if report["baseline_identity"] != {"kind": "random"}:
        raise ValueError("Initial reference evaluation must use the legal-random baseline")
    if report["inference"] != {"candidate": "deterministic", "baseline": "seeded legal random"}:
        raise ValueError("Initial evaluation inference settings changed")
    seeds = evaluation_seeds("validation", episodes, ExperimentConfig(**manifest["config"]).experiment_seed, start=validation_start)
    for label in ("candidate", "baseline"):
        rows = report[f"{label}_episodes"]
        if [row["seed"] for row in rows] != seeds:
            raise ValueError("Initial evaluation seed evidence does not match the validation schedule")
        if summarize(rows) != report[label]:
            raise ValueError("Initial evaluation summary does not match raw episode evidence")
    if any(row["policy"] != "random" for row in report["baseline_episodes"]):
        raise ValueError("Initial baseline episodes are not legal-random episodes")
    if any(row["policy"] != Path(initial_state["model"]).name for row in report["candidate_episodes"]):
        raise ValueError("Initial candidate episodes belong to another checkpoint")
    if paired_compare(report["candidate_episodes"], report["baseline_episodes"]) != report["comparison"]:
        raise ValueError("Initial comparison does not match raw episode evidence")


def check_original_timeouts(root: Path, upstream: Path, lineage: dict, config: ExperimentConfig) -> dict:
    """Known failure seeds are regression checks, separate from fresh validation."""
    target = root / "selection-repair-check.json"
    if target.exists():
        return read_json(target)
    source_report = Path(lineage["source_experiment"]) / "last-evaluation.json"
    if not source_report.exists():
        return {"status": "no_source_evaluation", "runs": []}
    report = read_json(source_report)
    if report["checkpoint"]["sha256"] != lineage["source_checkpoint"]["sha256"] or report["fingerprint"] != lineage["source_fingerprint"]:
        return {"status": "source_evaluation_is_for_another_checkpoint", "runs": []}
    seeds = [int(row["seed"]) for row in report["candidate_episodes"] if row["outcome"] == "timeout"]
    if not seeds:
        return {"status": "no_source_timeouts", "runs": []}
    initial_path = root / lineage["initial_checkpoint"]["model"]
    model = MaskablePPO.load(initial_path, device="cpu")
    env = runner._make(config, upstream)
    try:
        rows = runner._rollout(env, seeds, model=model, label="imported-policy-finite-selections")
    finally:
        env.close()
    result = {
        "status": "checked", "checkpoint_sha256": lineage["initial_checkpoint"]["sha256"],
        "selection_interface": "add-only-v1", "runs": rows,
        "passed": all(row["outcome"] in {"win", "death"} for row in rows),
        "note": "Known timeout seeds only; this is an interface regression check, not evidence of additional learning.",
    }
    atomic_json(target, result)
    return result


@contextmanager
def workflow_lock(root: Path):
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / f".{root.name}.selection-running.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Selection workflow is already running: {path}. If its process stopped, remove the stale lock.") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "started": time.time()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def progress(source: Path, root: Path, upstream: Path, *, target_steps: int, episodes: int,
             validation_start: int = 100) -> dict:
    source, root, upstream = source.resolve(), root.resolve(), upstream.resolve()
    with workflow_lock(root):
        return _progress(source, root, upstream, target_steps=target_steps,
                         episodes=episodes, validation_start=validation_start)


def _progress(source: Path, root: Path, upstream: Path, *, target_steps: int, episodes: int,
              validation_start: int) -> dict:
    if target_steps < 1 or episodes < 1:
        raise ValueError("target_steps and episodes must be positive")
    if not root.exists():
        create_warm_experiment(source, root, upstream)
    lineage = read_json(root / "warm-start.json")
    initial_state = lineage["initial_checkpoint"]
    initial_path = (root / initial_state["model"]).resolve()
    if not initial_path.is_relative_to(root) or sha256(initial_path) != initial_state["sha256"]:
        raise ValueError("Initial imported policy checksum mismatch")
    initial_report_path = root / "selection-initial-evaluation.json"
    with finite_selection_runner():
        manifest = read_json(root / "experiment.json")
        config = ExperimentConfig(**manifest["config"])
        runner.initialize(root, config, upstream)
        with experiment_lock(root):
            repair = check_original_timeouts(root, upstream, lineage, config)
        print(json.dumps({"stage": "known_timeout_regression_check", **repair}, indent=2), flush=True)
        if repair.get("passed") is False:
            raise ValueError("Known timeout regression check did not complete cleanly; inspect selection-repair-check.json before further training")
        if initial_report_path.exists():
            expected_hash = lineage.get("initial_evaluation_sha256")
            if expected_hash is not None and sha256(initial_report_path) != expected_hash:
                raise ValueError("Cached initial evaluation checksum changed")
            initial_report = read_json(initial_report_path)
        else:
            _, latest = verified_checkpoint(root, manifest["fingerprint"])
            if latest["sha256"] != initial_state["sha256"]:
                # Recover an evaluation that completed just before interruption.
                matching = [{**read_json(path), "report_path": str(path.resolve())}
                            for path in sorted((root / "evaluations").glob("validation-*.json"))]
                matching = [report for report in matching if report["checkpoint"]["sha256"] == initial_state["sha256"]
                            and report["baseline_identity"] == {"kind": "random"}
                            and report["candidate"]["episodes"] == episodes
                            and report["seed_start"] == validation_start]
                if not matching:
                    raise ValueError("Initial evaluation is missing after training; preserve artifacts and inspect this experiment")
                initial_report = matching[-1]
            else:
                print("Evaluating imported weights with finite selections, before additional RL...", flush=True)
                initial_report = runner.evaluate(root, upstream, episodes=episodes, split="validation", baseline="random", start=validation_start)
            atomic_json(initial_report_path, initial_report)
        validate_initial_report(initial_report, manifest, initial_state, episodes, validation_start)
        if lineage.get("initial_evaluation_sha256") is None:
            # The core evaluation also persists an independent immutable report;
            # require that exact evidence before pinning a recovered cache.
            original_path = Path(initial_report["report_path"]).resolve()
            if not original_path.is_relative_to(root / "evaluations"):
                raise ValueError("Initial evaluation evidence path escapes this experiment")
            original = read_json(original_path)
            if {key: value for key, value in initial_report.items() if key != "report_path"} != original:
                raise ValueError("Cached initial evaluation differs from its persisted raw report")
            lineage["initial_evaluation_sha256"] = sha256(initial_report_path)
            atomic_json(root / "warm-start.json", lineage)
        print(json.dumps({"stage": "before_additional_rl", **compact(initial_report)}, indent=2), flush=True)
        if any(initial_report[label]["outcomes"][outcome] for label in ("candidate", "baseline") for outcome in ("timeout", "aborted")):
            raise ValueError("Initial evaluation contains timeouts or aborts; inspect the report before further training")
        _, latest = verified_checkpoint(root, manifest["fingerprint"])
        remaining = max(0, target_steps - int(latest["num_timesteps"]))
        if remaining:
            print(f"Training up to {target_steps} new decisions, starting from imported weights...", flush=True)
            training = runner.train(root, config, upstream, steps=remaining)
        else:
            training = {"status": "target_already_reached", "total_steps": latest["num_timesteps"]}
        print("Evaluating trained policy against the imported policy on the same validation seeds...", flush=True)
        final = runner.evaluate(root, upstream, episodes=episodes, split="validation", baseline=str(initial_path), start=validation_start)
        # The saved random baseline was executed under the same interface on the
        # same seeds. Reuse those raw episodes; do not invent additional runs.
        random_comparison = paired_compare(final["candidate_episodes"], initial_report["baseline_episodes"])
        summary = {
            "stage": "after_additional_rl", "selection_interface": "add-only-v1",
            "known_timeout_regression_check": repair, "validation_seed_start": validation_start,
            "source_optimization_steps": lineage["source_checkpoint"]["num_timesteps"],
            "new_optimization_steps": training["total_steps"],
            "before_training": initial_report["candidate"], "after_training": final["candidate"],
            "learning_comparison": final["comparison"], "random_comparison": random_comparison,
            "random_baseline": initial_report["baseline"], "training": training,
            "initial_evaluation_report": str(initial_report_path), "final_evaluation_report": final["report_path"],
            "summary_path": str(root / "selection-progress.json"),
            "note": "Only the before/after comparison under the same finite-selection interface measures additional RL. Gameplay uses the Python emulator; native transfer remains unverified.",
        }
        atomic_json(root / "selection-progress.json", summary)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/ironclad-a0"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-add-only"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--target-steps", type=int, default=80000, help="New-experiment total; repeated invocation does not add another full budget")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-start", type=int, default=100, help="Use fresh validation seeds after the original diagnostic set")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    return progress(args.source.resolve(), args.experiment.resolve(), args.upstream.resolve(),
                    target_steps=args.target_steps, episodes=args.episodes, validation_start=args.validation_start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
