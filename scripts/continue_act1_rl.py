"""Continue the saved Act 1 learner under a verified potion-capacity repair.

One fixed additional PPO budget, preserved weights and Adam state, a frozen
starting reference, and fresh matched evaluation. No imitation or promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
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
from spire2_rl.experiment import (ExperimentConfig, atomic_json, commit_checkpoint,
                                 experiment_lock, read_json, sha256, verified_checkpoint)
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.study import paired_compare, summarize

SCRIPT_DIR = Path(__file__).resolve().parent
potion = runpy.run_path(str(SCRIPT_DIR / "check_potion_failures.py"))
checker = potion["checker"]
trial = potion["trial"]


def contained(path, folder):
    path = Path(path).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Repair evidence path escapes its diagnostic directory")
    return path


def trace_evidence(row, folder, files):
    paths = {key: contained(row[key], folder) for key in ("actions_trace", "observations_trace")}
    for path in paths.values():
        files[str(path)] = sha256(path)
    actions = [json.loads(line) for line in paths["actions_trace"].read_text(encoding="utf-8").splitlines() if line.strip()]
    observations = [json.loads(line) for line in paths["observations_trace"].read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = row["steps"] + int(row["outcome"] == "aborted" and row["error_stage"] == "game_step")
    if (not actions or not observations or len(actions) != expected_count
            or any(item["seed"] != row["seed"] or item["step"] != i for i, item in enumerate(actions))):
        raise ValueError("Repair trace does not match its recorded episode")
    observed, visits = {}, Counter()
    for item in observations:
        public = {key: item[key] for key in ("observation", "legal_actions")}
        digest = hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != item["observation_sha256"] or digest in observed:
            raise ValueError("Repair public observation checksum changed")
        observed[digest] = public
    for item in actions:
        digest = item["observation_sha256"]
        visits[digest] += 1
        if (digest not in observed or item["action_slot"] not in observed[digest]["legal_actions"]
                or item["observation_visits"] != visits[digest]):
            raise ValueError("Repair action trace differs from its legal public observation")
    if row["unique_public_observations"] != len(observed) or row["most_repeated_observation_visits"] != max(visits.values()):
        raise ValueError("Repair observation counts differ from raw evidence")
    tail = [{key: value for key, value in item.items()
             if key not in {"seed", "observation_sha256", "observation_visits"}} for item in actions[-6:]]
    count = sum("potion_inventory" in item and not any(item["potion_inventory"]["pick_action_mask"])
                for item in actions)
    if row["last_decisions"] != tail or row["capacity_mask_decisions"] != count:
        raise ValueError("Repair summary differs from its raw action trace")
    return actions


def validated_repair(source, source_lineage, manifest, state, repair_path=None):
    """Use the exact successful diagnostic for these source weights, not its flag alone."""
    if repair_path is None:
        candidates = sorted((source / "diagnostics").glob("potion-*/summary.json"))
        if not candidates:
            raise ValueError("Run check_potion_failures.py successfully before continuing")
        repair_path = candidates[-1]
    repair_path = contained(repair_path, source / "diagnostics")
    folder = repair_path.parent
    paths = [repair_path, folder / "protocol.json", folder / "runs.json"]
    files = {str(path): sha256(path) for path in paths}
    summary, protocol, raw = [read_json(path) for path in paths]
    if (summary["stage"] != "potion_capacity_repair_check" or summary["victory_scope"] != "act1"
            or summary["repair_verified"] is not True or summary["additional_training_steps"] != 0
            or summary["champion_updated"] is not False or raw["protocol"] != protocol
            or protocol["additional_training_steps"] != 0
            or protocol["replay_script_sha256"] != sha256(SCRIPT_DIR / "check_potion_failures.py")
            or protocol["repair_adapter_sha256"] != sha256(Path(sys.modules[PotionCapacityMask.__module__].__file__))):
        raise ValueError("Potion repair evidence is incomplete or belongs to different code")
    strength_path = contained(protocol["source_report"], source / "diagnostics")
    files[str(strength_path)] = sha256(strength_path)
    if files[str(strength_path)] != protocol["source_report_sha256"] or summary["source_report_sha256"] != protocol["source_report_sha256"]:
        raise ValueError("Original strength report checksum changed")
    strength = read_json(strength_path)
    original_protocol = strength["protocol"]
    if (protocol["source_protocol"] != original_protocol
            or original_protocol["fingerprint"] != manifest["fingerprint"]
            or original_protocol["candidate_checkpoint"] != state
            or original_protocol["baseline_checkpoint"] != source_lineage["initial_checkpoint"]
            or original_protocol["checker_sha256"] != sha256(SCRIPT_DIR / "check_act1_trial.py")
            or original_protocol["split"] != "validation" or original_protocol["inference"] != "deterministic"
            or original_protocol["victory_scope"] != "act1"):
        raise ValueError("Repair does not assess the current frozen source policy")
    config = ExperimentConfig(**source_lineage["config"])
    seeds = checker["fresh_seeds"](source, source_lineage, config, original_protocol["episodes"], original_protocol["seed_start"])
    checker["validate_report"](strength, original_protocol, seeds)
    trial["act1"]["checked_state"](source, source_lineage["initial_checkpoint"], manifest["fingerprint"])
    expected = [(key, row) for key in ("candidate", "baseline") for row in strength[f"{key}_episodes"]
                if row["outcome"] in {"aborted", "timeout"}]
    if (not expected or len(raw["runs"]) != len(expected) or len(summary["runs"]) != len(expected)
            or summary["failed_evaluation_runs"] != len(expected)):
        raise ValueError("Repair evidence does not cover every failed evaluation run")
    for (key, failed), row, compact in zip(expected, raw["runs"], summary["runs"]):
        if (row["policy"] != key or row["seed"] != failed["seed"] or row["original_evaluation"] != failed
                or compact["policy"] != key or compact["seed"] != failed["seed"]
                or not potion["verified_full_capacity_failure"](row["original"], failed)):
            raise ValueError("Original full-capacity failure was not exactly reproduced")
        repaired = row["capacity_mask"]
        if (repaired["seed"] != failed["seed"] or repaired["outcome"] not in {"win", "death"}
                or repaired["error"] is not None or repaired["error_stage"] is not None
                or repaired["capacity_mask_decisions"] < 1):
            raise ValueError("Repaired run did not complete normally with a capacity intervention")
        if any(row[name] is not True or compact[name] is not True for name in ("confirmed_full_capacity_failure", "repair_completed_run")):
            raise ValueError("Repair evidence does not verify its original and repaired outcomes")
        for mode in ("original", "capacity_mask"):
            if compact[mode] != potion["brief"](row[mode]):
                raise ValueError("Repair compact summary differs from its raw replay")
        original_actions = trace_evidence(row["original"], folder, files)
        repaired_actions = trace_evidence(repaired, folder, files)
        fields = ("seed", "step", "phase", "floor", "hp", "action_slot", "named_action")
        if len(repaired_actions) < len(original_actions) or any(
                any(a[field] != b[field] for field in fields)
                for a, b in zip(original_actions[:-1], repaired_actions)):
            raise ValueError("Repair changed the trajectory before the original failed decision")
        before, after = original_actions[-1], repaired_actions[len(original_actions) - 1]
        if (any(before[field] != after[field] for field in ("seed", "step", "phase", "floor", "hp"))
                or any(before["potion_inventory"][field] != after["potion_inventory"][field]
                       for field in ("max_slots", "slot_potion_ids", "free_slots", "pick_action_slots"))
                or any(after["potion_inventory"]["pick_action_mask"])
                or after["named_action"]["action"] != "skip_potion"):
            raise ValueError("Repair did not mask the same impossible pickup with explicit skip")
    return {"summary_path": str(repair_path), "files": files, "strength_report": strength, "summary": summary}


@contextmanager
def continuation_runner(source_lineage, plan, source_state, repair_evidence):
    with trial["trial_runner"](source_lineage["recipe"], source_lineage["plan"]):
        original = runner._make
        def make(config, upstream):
            env = PotionCapacityMask(original(config, upstream))
            env.backend_metadata.update(
                continuation="bounded-act1-capacity-ppo-v1", continuation_runner_sha256=sha256(Path(__file__)),
                continuation_plan=dict(plan), continuation_source_checkpoint=dict(source_state),
                continuation_repair_files=dict(repair_evidence["files"]),
            )
            return env
        runner._make = make
        try:
            yield
        finally:
            runner._make = original


def create_continuation(source, root, upstream, plan, repair_path=None):
    source, root, upstream = Path(source).resolve(), Path(root).resolve(), Path(upstream).resolve()
    if source == root or root.is_relative_to(source) or source.is_relative_to(root) or root.exists():
        raise ValueError("Use a new, separate continuation directory")
    with experiment_lock(source):
        source_lineage = read_json(source / "trial-lineage.json")
        config = ExperimentConfig(**source_lineage["config"])
        with trial["trial_runner"](source_lineage["recipe"], source_lineage["plan"]):
            source_manifest = runner.initialize(source, config, upstream)
            checkpoint, state = verified_checkpoint(source, source_manifest["fingerprint"])
            if state["interrupted"] or state["num_timesteps"] < source_lineage["plan"]["target_steps"]:
                raise ValueError("Complete the original source trial before continuing")
            repair = validated_repair(source, source_lineage, source_manifest, state, repair_path)
            seeds = checker["fresh_seeds"](source, source_lineage, config, plan["episodes"], plan["validation_start"])
        if set(seeds) & {row["seed"] for row in repair["strength_report"]["candidate_episodes"]}:
            raise ValueError("Continuation validation seeds overlap the frozen strength check")
        indices = trial["source_indices"](source, state, config)
        source_evidence = {str(source / name): sha256(source / name) for name in ("trial-lineage.json", "experiment.json")}
        source_evidence.update(repair["files"])
        source_evidence[str(checkpoint)] = state["sha256"]
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            with continuation_runner(source_lineage, plan, state, repair):
                manifest = runner.initialize(staging, config, upstream)
                vec = DummyVecEnv([lambda: runner._make(config, upstream) for _ in range(config.n_envs)])
                try:
                    # load restores both policy/value tensors and PPO Adam moments;
                    # force_reset discards old episode observations, not learned state.
                    model = MaskablePPO.load(checkpoint, env=vec, device="cpu", force_reset=True)
                    if int(model.num_timesteps) != state["num_timesteps"]:
                        raise ValueError("Source model step count differs from its checkpoint")
                    if model.observation_space != vec.observation_space or model.action_space != vec.action_space:
                        raise ValueError("Capacity repair changed the policy's observation or action space")
                    vec._seeds = [None] * config.n_envs
                    initial = commit_checkpoint(model, staging, [SimpleNamespace(next_index=i) for i in indices], manifest["fingerprint"])
                finally:
                    vec.close()
            if any(sha256(Path(path)) != digest for path, digest in source_evidence.items()):
                raise ValueError("Source evidence changed during continuation initialization")
            lineage = {"source_experiment": str(source), "source_fingerprint": source_manifest["fingerprint"],
                       "source_state": state, "source_lineage": source_lineage, "source_evidence": source_evidence,
                       "config": asdict(config), "plan": plan, "initial_checkpoint": initial,
                       "target_total_steps": state["num_timesteps"] + plan["new_steps"],
                       "repair_evidence": {"summary_path": repair["summary_path"], "files": repair["files"]},
                       "training_seed_indices": indices, "optimizer": "preserved PPO Adam state",
                       "note": "New interface identity; existing learned policy, value and optimizer retained. Active episodes restart beyond the source seed high-water marks. No cloning or automatic promotion."}
            atomic_json(staging / "capacity-lineage.json", lineage)
            if root.exists():
                raise ValueError("Continuation directory was created during initialization")
            staging.rename(root)
            return lineage
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
    config = ExperimentConfig(**manifest["config"])
    path = trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
    model = MaskablePPO.load(path, device="cpu")
    if int(model.num_timesteps) != state["num_timesteps"]:
        raise ValueError("Evaluation model step count differs from its checkpoint")
    seeds = checker["evaluation_seeds"]("validation", episodes, config.experiment_seed, start=validation_start)
    env = runner._make(config, upstream)
    rows = []
    try:
        for offset in range(0, episodes, 50):
            rows.extend(runner._rollout(env, seeds[offset:offset + 50], model=model, label=label))
            print(f"{label}: checked {len(rows)}/{episodes} Act 1 runs", flush=True)
    finally:
        env.close()
    trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
    return {"stage": label, "fingerprint": manifest["fingerprint"], "checkpoint": state,
            "split": "validation", "seed_start": validation_start, "episodes": rows,
            "gameplay": summarize(rows), "inference": "deterministic", "victory_scope": "act1"}


def cached_assessment(root, upstream, manifest, lineage, key, state):
    plan = lineage["plan"]
    path = root / f"{key}.json"
    trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
    if path.exists():
        if lineage.get(f"{key}_sha256") is not None and sha256(path) != lineage[f"{key}_sha256"]:
            raise ValueError("Cached continuation evaluation checksum changed")
        report = read_json(path)
    else:
        if lineage.get(f"{key}_sha256") is not None:
            raise ValueError("Pinned continuation evaluation is missing")
        report = assess(root, upstream, manifest, state, label=key, episodes=plan["episodes"], validation_start=plan["validation_start"])
        trial["act1"]["validate_assessment"](report, manifest, state, key, plan["episodes"], plan["validation_start"])
        atomic_json(path, report)
    trial["act1"]["validate_assessment"](report, manifest, state, key, plan["episodes"], plan["validation_start"])
    if lineage.get(f"{key}_sha256") is None:
        lineage[f"{key}_sha256"] = sha256(path)
        atomic_json(root / "capacity-lineage.json", lineage)
    return report


def progress(source, root, upstream, *, new_steps=80000, episodes=1000, validation_start=1700, repair_check=None):
    if new_steps < 1 or episodes < 1000 or validation_start < 1700:
        raise ValueError("Use a positive fixed additional budget, at least 1000 pairs and fresh validation indices >=1700")
    source, root, upstream = Path(source).resolve(), Path(root).resolve(), Path(upstream).resolve()
    if source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source and continuation directories must not overlap")
    torch.set_num_threads(2)
    plan = {"new_steps": new_steps, "episodes": episodes, "validation_start": validation_start}
    with trial["selection"]["workflow_lock"](root):
        if not root.exists():
            create_continuation(source, root, upstream, plan, repair_check)
        lineage = read_json(root / "capacity-lineage.json")
        if lineage["source_experiment"] != str(source) or lineage["plan"] != plan:
            raise ValueError("This continuation's source, additional budget and evaluation schedule are fixed; resume with the original command")
        if repair_check is not None and str(Path(repair_check).resolve()) != lineage["repair_evidence"]["summary_path"]:
            raise ValueError("Use this continuation's original repair evidence")
        if any(sha256(Path(path)) != digest for path, digest in lineage["source_evidence"].items()):
            raise ValueError("Frozen source model or repair evidence changed")
        config = ExperimentConfig(**lineage["config"])
        initial = lineage["initial_checkpoint"]
        if initial["num_timesteps"] != lineage["source_state"]["num_timesteps"] or lineage["target_total_steps"] != initial["num_timesteps"] + new_steps:
            raise ValueError("Continuation budget differs from its frozen starting checkpoint")
        with continuation_runner(lineage["source_lineage"], plan, lineage["source_state"], lineage["repair_evidence"]):
            manifest = runner.initialize(root, config, upstream)
            with experiment_lock(root):
                before = cached_assessment(root, upstream, manifest, lineage, "before_rl", initial)
            if not trial["act1"]["clean"](before):
                raise ValueError("Repaired starting-policy evaluation contains errors or timeouts; no PPO started")
            print(json.dumps({"stage": "before_continued_rl", "victory_scope": "act1",
                              "starting_policy": trial["brief_gameplay"](before["gameplay"]),
                              "fixed_additional_steps": new_steps, "source_rl_steps": initial["num_timesteps"],
                              "target_total_steps": lineage["target_total_steps"], "optimizer": lineage["optimizer"]}, indent=2), flush=True)
            _, latest = verified_checkpoint(root, manifest["fingerprint"])
            remaining = max(0, lineage["target_total_steps"] - latest["num_timesteps"])
            if not remaining and latest["interrupted"]:
                raise ValueError("Interrupted checkpoint meets the counter target without completed training; preserve it for inspection")
            if remaining:
                print(f"Continuing PPO: {remaining} decisions remain in the fixed {new_steps}-decision additional budget...", flush=True)
                training = runner.train(root, config, upstream, steps=remaining)
            else:
                training = {"status": "target_already_reached", "total_steps": latest["num_timesteps"]}
            with experiment_lock(root):
                _, latest = verified_checkpoint(root, manifest["fingerprint"])
                if latest["interrupted"]:
                    raise ValueError("Final continuation checkpoint is interrupted")
                after = cached_assessment(root, upstream, manifest, lineage, "after_rl", latest)
            comparison = paired_compare(after["episodes"], before["episodes"])
            clears = checker["exact_win_test"](after["episodes"], before["episodes"])
            clean = comparison["excluded_pairs"] == 0
            floor_ci = comparison["floor_delta_ci95"]
            summary = {"stage": "after_continued_rl", "victory_scope": "act1", "status": "completed" if clean else "incomplete_evaluation",
                       "before_rl": trial["brief_gameplay"](before["gameplay"]), "after_rl": trial["brief_gameplay"](after["gameplay"]),
                       "new_rl_steps": latest["num_timesteps"] - initial["num_timesteps"],
                       "source_rl_steps": initial["num_timesteps"], "cumulative_rl_steps": latest["num_timesteps"],
                       "rl_floor_gain": comparison["floor_delta"], "rl_floor_gain_ci95": floor_ci,
                       "rl_act1_clear_rate_gain": comparison["win_delta"], "rl_act1_clear_rate_gain_ci95": comparison["win_delta_ci95"],
                       "establishes_positive_floor_gain": bool(clean and comparison["eligible_pairs"] >= 1000 and floor_ci is not None and floor_ci[0] > 0),
                       "establishes_positive_act1_clear_gain": bool(clean and comparison["eligible_pairs"] >= 1000 and clears["supports_more_clears"]),
                       "paired_clear_evidence": clears, "checkpoint": latest,
                       "training_record": checker["training_record"](root, config, latest),
                       "summary_path": str(root / "continuation-progress.json"), "champion_updated": False,
                       "note": "Same repaired interface and fresh matched seeds before and after additional RL. PPO weights, value network, optimizer and counter were retained. No new imitation or automatic budget extension. Wins are emulator Act 1 clears only."}
            atomic_json(root / "continuation-evidence.json", {"before": before, "after": after, "comparison": comparison, "training": training})
            atomic_json(root / "continuation-progress.json", summary)
            print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/ironclad-a0-act1-rl-trial"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-rl-continuation"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--new-steps", type=int, default=80000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--validation-start", type=int, default=1700)
    parser.add_argument("--repair-check", type=Path)
    args = parser.parse_args(argv)
    return progress(args.source, args.experiment, args.upstream, new_steps=args.new_steps,
                    episodes=args.episodes, validation_start=args.validation_start, repair_check=args.repair_check)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
