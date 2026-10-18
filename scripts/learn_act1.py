"""One bounded combat-imitation / Act 1 PPO experiment.

Existing full-run experiments are read-only sources. Imitation demonstrations
come exclusively from a reserved training-seed block. Evaluation runs the
imported, cloned and RL policies on the same separate validation seeds.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import runpy
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
from spire2_rl.imitation import behavior_clone, collect_demonstrations
from spire2_rl.study import evaluation_seeds, paired_compare, seed_for, summarize

SCRIPT_DIR = Path(__file__).resolve().parent
selection = runpy.run_path(str(SCRIPT_DIR / "selection_rl.py"))
FairTeacher = runpy.run_path(str(SCRIPT_DIR / "fair_teacher.py"))["FairTeacher"]


def default_recipe():
    return {
        "version": "combat-cloning-act1-v1", "demonstration_episodes": 256,
        "bc_epochs": 12, "bc_batch_size": 256, "bc_learning_rate": 0.001,
        "ppo_learning_rate": 0.0001,
        "labels": "ordinary combat decisions only; no pending-choice or macro labels",
        "objective": "Act 1 boss cleared; not a full-run victory",
    }


@contextmanager
def act1_runner(recipe):
    with selection["finite_selection_runner"]():
        original = runner._make

        def make(config, upstream):
            if config.scope != "fullrun" or config.max_acts != 1:
                raise ValueError("This experiment requires the full-run schema with max_acts=1")
            env = original(config, upstream)
            env.backend_metadata.update(
                curriculum="act1-v1", curriculum_runner_sha256=sha256(Path(__file__)),
                imitation_module_sha256=sha256(Path(__file__).resolve().parents[1] / "src/spire2_rl/imitation.py"),
                teacher_sha256=sha256(SCRIPT_DIR / "fair_teacher.py"),
                training_recipe=dict(recipe),
            )
            return env

        runner._make = make
        try:
            yield
        finally:
            runner._make = original


def seed_plan(source, state, config, count):
    """Reserve a new contiguous demo block, then interleaved PPO worker seeds."""
    if count < 5:
        raise ValueError("At least five demonstration episodes are required")
    indices = list(state["next_indices"])
    if len(indices) != config.n_envs:
        raise ValueError("Source worker count changed")
    for rank, index in enumerate(indices):
        seed_for(index, "train", config.experiment_seed)
        if index % config.n_envs != rank:
            raise ValueError("Source checkpoint seed cursor has the wrong worker rank")
        cursor = source / "seed-cursors" / f"worker-{rank}.json"
        if cursor.exists():
            highwater = read_json(cursor)["next_index"]
            seed_for(highwater, "train", config.experiment_seed)
            if highwater % config.n_envs != rank:
                raise ValueError("Source persisted seed cursor has the wrong worker rank")
            indices[rank] = max(index, highwater)
    n = config.n_envs
    start = ((max(indices) + n - 1) // n) * n
    end = start + count
    rl_start = ((end + n - 1) // n) * n
    next_indices = [rl_start + rank for rank in range(n)]
    for index in next_indices:
        seed_for(index, "train", config.experiment_seed)
    return {"source_highwater_indices": indices, "demonstration_index_start": start,
            "demonstration_index_end_exclusive": end,
            "demonstration_seeds": [seed_for(i, "train", config.experiment_seed)
                                    for i in range(start, end)],
            "rl_next_indices": next_indices}


def create_experiment(source, root, upstream, recipe):
    source, root, upstream = source.resolve(), root.resolve(), upstream.resolve()
    if source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source and destination experiments must not overlap")
    if not source.is_dir() or root.exists():
        raise ValueError("Use an existing source and a new destination directory")
    with experiment_lock(source):
        source_manifest = read_json(source / "experiment.json")
        source_config = ExperimentConfig(**source_manifest["config"])
        if source_config.scope != "fullrun" or source_config.max_acts != 3:
            raise ValueError("Source must be the existing finite-selection full-run experiment")
        with selection["finite_selection_runner"]():
            runner.initialize(source, source_config, upstream)
        checkpoint, source_state = verified_checkpoint(source, source_manifest["fingerprint"])
        if source_state.get("interrupted"):
            raise ValueError("Import a completed checkpoint")
        source_model = MaskablePPO.load(checkpoint, device="cpu")
        if int(source_model.num_timesteps) != source_state["num_timesteps"]:
            raise ValueError("Source timestep metadata does not match its model")
        plan = seed_plan(source, source_state, source_config, recipe["demonstration_episodes"])
        config = replace(source_config, max_acts=1, learning_rate=recipe["ppo_learning_rate"])
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            with act1_runner(recipe):
                manifest = runner.initialize(staging, config, upstream)
                vec = DummyVecEnv([lambda: runner._make(config, upstream) for _ in range(config.n_envs)])
                cursors = [SimpleNamespace(next_index=i) for i in plan["rl_next_indices"]]
                try:
                    imported = selection["fresh_policy"](source_model, config, vec)
                    initial_state = commit_checkpoint(imported, staging, cursors, manifest["fingerprint"])
                    print(f"Collecting teacher combat demonstrations on {recipe['demonstration_episodes']} training seeds...", flush=True)
                    env = runner._make(config, upstream)
                    try:
                        demos = collect_demonstrations(env, FairTeacher(), plan["demonstration_seeds"],
                                                       staging / "demonstrations.npz")
                    finally:
                        env.close()
                    print("Teaching combat choices by imitation; PPO steps remain zero...", flush=True)
                    cloning = behavior_clone(imported, staging / "demonstrations.npz",
                                             epochs=recipe["bc_epochs"], batch_size=recipe["bc_batch_size"],
                                             learning_rate=recipe["bc_learning_rate"], seed=config.experiment_seed)
                    # The supervised Adam state must not become the PPO optimizer.
                    cloned = selection["fresh_policy"](imported, config, vec)
                    cloned_state = commit_checkpoint(cloned, staging, cursors, manifest["fingerprint"])
                finally:
                    vec.close()
            if sha256(checkpoint) != source_state["sha256"]:
                raise ValueError("Source checkpoint changed during import")
            lineage = {
                "source_experiment": str(source), "source_fingerprint": source_manifest["fingerprint"],
                "source_checkpoint": source_state, "imported_checkpoint": initial_state,
                "cloned_checkpoint": cloned_state, "seed_plan": plan,
                "recipe": recipe, "config": asdict(config), "demonstrations": demos,
                "cloning": cloning, "created": time.time_ns(),
                "note": "Combat imitation is supervised learning, not RL. Both initial checkpoints have zero new PPO steps. PPO gets a fresh optimizer. Act 1 wins are not full-run wins. A cancelled initialization can deterministically regenerate its training demonstrations; the source is unchanged.",
            }
            atomic_json(staging / "act1-lineage.json", lineage)
            if root.exists():
                raise ValueError("Destination was created during initialization")
            staging.rename(root)
            return lineage
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


def checked_state(root, state, identity):
    if state.get("kind") == "public_rule_teacher":
        if state != {"kind": "public_rule_teacher", "sha256": sha256(SCRIPT_DIR / "fair_teacher.py")}:
            raise ValueError("Teacher source changed during evaluation")
        return None
    path = (root / state["model"]).resolve()
    if (not path.is_relative_to(root.resolve()) or state["fingerprint"] != identity
            or state.get("interrupted") or sha256(path) != state["sha256"]):
        raise ValueError("Learning stage checkpoint is incompatible, interrupted or corrupt")
    return path


def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
    config = ExperimentConfig(**manifest["config"])
    path = checked_state(root, state, manifest["fingerprint"])
    model = MaskablePPO.load(path, device="cpu") if path is not None else None
    if model is not None and int(model.num_timesteps) != state["num_timesteps"]:
        raise ValueError("Evaluation model timestep count differs from its checkpoint")
    seeds = evaluation_seeds("validation", episodes, config.experiment_seed, start=validation_start)
    env = runner._make(config, upstream)
    try:
        if model is None:
            teacher = FairTeacher()
            class TeacherPredictor:
                def predict(self, obs, *, action_masks, deterministic):
                    return teacher.select(env), None
            model = TeacherPredictor()
        rows = runner._rollout(env, seeds, model=model, label=label)
    finally:
        env.close()
    checked_state(root, state, manifest["fingerprint"])
    return {"stage": label, "fingerprint": manifest["fingerprint"], "checkpoint": state,
            "split": "validation", "seed_start": validation_start, "episodes": rows,
            "gameplay": summarize(rows), "inference": "deterministic",
            "victory_scope": "act1", "note": "Wins in this report mean Act 1 clears only."}


def validate_assessment(report, manifest, state, label, episodes, validation_start):
    if (report["fingerprint"] != manifest["fingerprint"] or report["checkpoint"] != state
            or report["stage"] != label or report["split"] != "validation"
            or report["victory_scope"] != "act1" or report["inference"] != "deterministic"
            or report["seed_start"] != validation_start):
        raise ValueError("Cached evaluation belongs to another stage or schedule")
    config = ExperimentConfig(**manifest["config"])
    seeds = evaluation_seeds("validation", episodes, config.experiment_seed, start=validation_start)
    if ([row["seed"] for row in report["episodes"]] != seeds
            or any(row["policy"] != label for row in report["episodes"])
            or summarize(report["episodes"]) != report["gameplay"]):
        raise ValueError("Cached evaluation does not match its raw seed/outcome evidence")


def cached_assessment(root, upstream, manifest, lineage, key, state, episodes, validation_start):
    target = root / f"{key}.json"
    checked_state(root, state, manifest["fingerprint"])
    if target.exists():
        expected = lineage.get(f"{key}_sha256")
        if expected is not None and expected != sha256(target):
            raise ValueError("Cached evaluation checksum changed or was not committed")
        report = read_json(target)
        validate_assessment(report, manifest, state, key, episodes, validation_start)
        if expected is None:
            # Recover a complete, validated evaluation written immediately before
            # interruption between report commit and lineage hash commit.
            lineage[f"{key}_sha256"] = sha256(target)
            atomic_json(root / "act1-lineage.json", lineage)
    else:
        if lineage.get(f"{key}_sha256") is not None:
            raise ValueError("Pinned evaluation is missing")
        print(f"Evaluating {key}: {episodes} fresh matched validation runs...", flush=True)
        report = assess(root, upstream, manifest, state, label=key,
                        episodes=episodes, validation_start=validation_start)
        validate_assessment(report, manifest, state, key, episodes, validation_start)
        atomic_json(target, report)
        lineage[f"{key}_sha256"] = sha256(target)
        atomic_json(root / "act1-lineage.json", lineage)
    validate_assessment(report, manifest, state, key, episodes, validation_start)
    return report


def clean(report):
    return report["gameplay"]["outcomes"]["timeout"] == report["gameplay"]["outcomes"]["aborted"] == 0


def progress(source, root, upstream, *, target_steps=80000, episodes=100, validation_start=400, recipe=None):
    if target_steps < 1 or episodes < 1:
        raise ValueError("Training budget and evaluation count must be positive")
    recipe = default_recipe() if recipe is None else recipe
    source, root, upstream = source.resolve(), root.resolve(), upstream.resolve()
    if source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source and destination must not overlap")
    torch.set_num_threads(2)
    with selection["workflow_lock"](root):
        if not root.exists():
            create_experiment(source, root, upstream, recipe)
        lineage = read_json(root / "act1-lineage.json")
        if lineage["source_experiment"] != str(source) or lineage["recipe"] != recipe:
            raise ValueError("Use the original source and recipe to resume this experiment")
        if sha256(root / "demonstrations.npz") != lineage["demonstrations"]["sha256"]:
            raise ValueError("Teacher demonstration checksum changed")
        config = ExperimentConfig(**lineage["config"])
        with act1_runner(recipe):
            manifest = runner.initialize(root, config, upstream)
            with experiment_lock(root):
                before = cached_assessment(root, upstream, manifest, lineage, "before_imitation",
                                           lineage["imported_checkpoint"], episodes, validation_start)
                cloned = cached_assessment(root, upstream, manifest, lineage, "after_imitation_before_rl",
                                           lineage["cloned_checkpoint"], episodes, validation_start)
                teacher_report = cached_assessment(root, upstream, manifest, lineage, "teacher_act1",
                                                   {"kind": "public_rule_teacher", "sha256": sha256(SCRIPT_DIR / "fair_teacher.py")},
                                                   episodes, validation_start)
            imitation_comparison = paired_compare(cloned["episodes"], before["episodes"])
            print(json.dumps({"stage": "imitation_results", "victory_scope": "act1",
                              "before_imitation": before["gameplay"], "after_imitation": cloned["gameplay"],
                              "teacher": teacher_report["gameplay"],
                              "comparison": imitation_comparison}, indent=2), flush=True)
            status = "completed"
            if not clean(before) or not clean(cloned) or not clean(teacher_report):
                status = "stopped_incomplete_evaluation"
            elif not imitation_comparison["promote"]:
                status = "stopped_imitation_did_not_establish_improvement"
            elif lineage["demonstrations"]["gameplay"]["outcomes"]["win"] == 0:
                status = "stopped_teacher_produced_no_act1_successes"
            if status != "completed":
                result = {"stage": "act1_progress", "status": status, "before_imitation": before["gameplay"],
                          "after_imitation": cloned["gameplay"], "imitation_comparison": imitation_comparison,
                          "teacher": teacher_report["gameplay"],
                          "new_rl_steps": 0, "victory_scope": "act1",
                          "source_rl_steps": lineage["source_checkpoint"]["num_timesteps"],
                          "summary_path": str(root / "act1-progress.json"),
                          "note": "No PPO started. Inspect these measurements before allocating another budget."}
                atomic_json(root / "act1-progress.json", result)
                print(json.dumps(result, indent=2), flush=True)
                return result
            _, latest = verified_checkpoint(root, manifest["fingerprint"])
            remaining = max(0, target_steps - int(latest["num_timesteps"]))
            if remaining:
                print(f"Act 1 PPO: target {target_steps} new decisions; imitation is excluded from this count...", flush=True)
                training = runner.train(root, config, upstream, steps=remaining)
            else:
                training = {"status": "target_already_reached", "total_steps": latest["num_timesteps"]}
            with experiment_lock(root):
                _, latest = verified_checkpoint(root, manifest["fingerprint"])
                final = assess(root, upstream, manifest, latest, label="after_act1_rl",
                               episodes=episodes, validation_start=validation_start)
                output = root / "evaluations" / f"act1-{time.time_ns()}.json"
                atomic_json(output, final)
            incremental = paired_compare(final["episodes"], cloned["episodes"])
            overall = paired_compare(final["episodes"], before["episodes"])
            strict_win_gain = clean(final) and incremental["eligible_pairs"] >= 100 and incremental["win_delta_ci95"][0] > 0
            summary = {
                "stage": "after_act1_rl", "status": "completed", "victory_scope": "act1",
                "before_imitation": before["gameplay"], "after_imitation_before_rl": cloned["gameplay"],
                "after_rl": final["gameplay"], "imitation_comparison": imitation_comparison,
                "teacher": teacher_report["gameplay"],
                "additional_rl_comparison": incremental, "overall_comparison": overall,
                "additional_rl_establishes_more_act1_clears": bool(strict_win_gain),
                "new_rl_steps": training["total_steps"], "training": training,
                "source_rl_steps": lineage["source_checkpoint"]["num_timesteps"],
                "teacher_training_episodes": lineage["demonstrations"]["gameplay"],
                "cloning": lineage["cloning"], "evaluation_report": str(output),
                "summary_path": str(root / "act1-progress.json"),
                "next_action": "Review the separate imitation and PPO gains before allocating another training budget.",
                "note": "Repeated-run PPO is evaluated against the cloned checkpoint, not the old source. Wins mean Act 1 clears in the Python emulator; full-run wins and native transfer remain unverified.",
            }
            atomic_json(root / "act1-progress.json", summary)
            print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/ironclad-a0-add-only"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--target-steps", type=int, default=80000, help="Total new PPO steps; rerunning does not add the full budget again")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-start", type=int, default=400)
    args = parser.parse_args(argv)
    return progress(args.source, args.experiment, args.upstream, target_steps=args.target_steps,
                    episodes=args.episodes, validation_start=args.validation_start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
