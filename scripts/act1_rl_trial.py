"""A fixed-budget exploratory PPO trial from the saved combat clone.

Training eligibility is distinct from champion promotion. A measured floor gain
can justify this trial while winning ability remains unresolved. This launcher
preserves the source experiment, repeats no imitation, and promotes no model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
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
from spire2_rl.study import evaluation_seeds, paired_compare, seed_for

act1 = runpy.run_path(str(Path(__file__).with_name("learn_act1.py")))
selection = act1["selection"]


def exploration_gate(before, cloned):
    comparison = paired_compare(cloned["episodes"], before["episodes"])
    reasons = []
    if comparison["excluded_pairs"] or comparison["eligible_pairs"] < 100:
        reasons.append("Requires at least 100 clean matched Act 1 episodes")
    if comparison["floor_delta_ci95"] is None or comparison["floor_delta_ci95"][0] <= 0:
        reasons.append("Cloning did not establish a positive floor gain")
    if comparison["win_delta_ci95"] is None or comparison["win_delta_ci95"][1] < 0:
        reasons.append("The evidence establishes an Act 1 clear-rate decrease")
    return {"eligible": not reasons, "reasons": reasons, "comparison": comparison,
            "purpose": "Eligibility for a bounded training trial only",
            "original_promotion_rule_passed": comparison["promote"],
            "clear_rate_noninferiority_claim": "not made",
            "note": "No established decrease is not evidence of safety, equivalence or noninferiority. The original promotion gate remains unchanged; this trial makes no champion update."}


def eligible_source(source, upstream):
    lineage = read_json(source / "act1-lineage.json")
    config = ExperimentConfig(**lineage["config"])
    with act1["act1_runner"](lineage["recipe"]):
        manifest = runner.initialize(source, config, upstream)
    reports = []
    for key, state_key in (("before_imitation", "imported_checkpoint"),
                           ("after_imitation_before_rl", "cloned_checkpoint")):
        path = source / f"{key}.json"
        if lineage.get(f"{key}_sha256") != sha256(path):
            raise ValueError("Source imitation evidence checksum mismatch")
        state = lineage[state_key]
        act1["checked_state"](source, state, manifest["fingerprint"])
        report = read_json(path)
        act1["validate_assessment"](report, manifest, state, key, len(report["episodes"]), report["seed_start"])
        reports.append(report)
    if reports[0]["seed_start"] != reports[1]["seed_start"]:
        raise ValueError("Source imitation stages do not use the same seed schedule")
    gate = exploration_gate(*reports)
    if not gate["eligible"]:
        raise ValueError("Exploratory trial is ineligible: " + "; ".join(gate["reasons"]))
    if lineage["demonstrations"]["gameplay"]["outcomes"]["win"] < 1:
        raise ValueError("Teacher collection supplied no Act 1 success examples")
    _, latest = verified_checkpoint(source, manifest["fingerprint"])
    if latest != lineage["cloned_checkpoint"] or latest["num_timesteps"] != 0:
        raise ValueError("Source must be the saved cloned initializer before any PPO continuation")
    return lineage, manifest, config, gate


@contextmanager
def trial_runner(recipe, plan):
    with act1["act1_runner"](recipe):
        original = runner._make
        def make(config, upstream):
            env = original(config, upstream)
            env.backend_metadata.update(
                trial="bounded-act1-exploratory-ppo-v1",
                trial_runner_sha256=sha256(Path(__file__)), trial_plan=dict(plan),
            )
            return env
        runner._make = make
        try:
            yield
        finally:
            runner._make = original


def source_indices(source, state, config):
    indices = list(state["next_indices"])
    if len(indices) != config.n_envs:
        raise ValueError("Source worker count differs from its seed cursors")
    for rank, index in enumerate(indices):
        cursor = source / "seed-cursors" / f"worker-{rank}.json"
        values = [index] + ([read_json(cursor)["next_index"]] if cursor.exists() else [])
        for value in values:
            seed_for(value, "train", config.experiment_seed)
            if value % config.n_envs != rank:
                raise ValueError("Source training seed cursor has the wrong worker rank")
        indices[rank] = max(values)
    return indices


def create_trial(source, root, upstream, plan):
    source, root, upstream = source.resolve(), root.resolve(), upstream.resolve()
    if source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Trial must use a separate experiment directory")
    if not source.is_dir() or root.exists():
        raise ValueError("Use an existing imitation source and a new trial directory")
    with experiment_lock(source):
        source_lineage, source_manifest, config, gate = eligible_source(source, upstream)
        planned_seeds = set(evaluation_seeds("validation", plan["episodes"], config.experiment_seed,
                                            start=plan["validation_start"]))
        for key in ("before_imitation", "after_imitation_before_rl"):
            used_seeds = {row["seed"] for row in read_json(source / f"{key}.json")["episodes"]}
            if planned_seeds & used_seeds:
                raise ValueError("Trial validation seeds overlap the source imitation evaluations; choose fresh indices")
        states = [source_lineage[key] for key in ("imported_checkpoint", "cloned_checkpoint")]
        paths = [act1["checked_state"](source, state, source_manifest["fingerprint"]) for state in states]
        models = [MaskablePPO.load(path, device="cpu") for path in paths]
        if any(int(model.num_timesteps) != state["num_timesteps"] for model, state in zip(models, states)):
            raise ValueError("Source checkpoint timestep metadata differs from its model")
        indices = source_indices(source, states[1], config)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            with trial_runner(source_lineage["recipe"], plan):
                manifest = runner.initialize(staging, config, upstream)
                vec = DummyVecEnv([lambda: runner._make(config, upstream) for _ in range(config.n_envs)])
                try:
                    imported_states = []
                    for model in models:
                        fresh = selection["fresh_policy"](model, config, vec)
                        imported_states.append(commit_checkpoint(
                            fresh, staging, [SimpleNamespace(next_index=i) for i in indices], manifest["fingerprint"]))
                finally:
                    vec.close()
            for path, state in zip(paths, states):
                if sha256(path) != state["sha256"]:
                    raise ValueError("Source checkpoint changed during trial initialization")
            lineage = {"source_experiment": str(source), "source_fingerprint": source_manifest["fingerprint"],
                       "source_imported_checkpoint": states[0], "source_cloned_checkpoint": states[1],
                       "incumbent_checkpoint": imported_states[0], "initial_checkpoint": imported_states[1],
                       "recipe": source_lineage["recipe"], "config": asdict(config), "plan": plan,
                       "exploration_entry": gate, "source_rl_steps": source_lineage["source_checkpoint"]["num_timesteps"],
                       "training_seed_indices": indices,
                       "note": "Only weights transferred; fresh PPO optimizer. No new demonstrations or cloning. Original incumbent and clone remain frozen in the source. This is exploration, not promotion."}
            atomic_json(staging / "trial-lineage.json", lineage)
            if root.exists():
                raise ValueError("Trial destination was created during initialization")
            staging.rename(root)
            return lineage
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


def cached_reference(root, upstream, manifest, lineage, key, state):
    plan = lineage["plan"]
    target = root / f"{key}.json"
    act1["checked_state"](root, state, manifest["fingerprint"])
    if target.exists():
        expected = lineage.get(f"{key}_sha256")
        if expected is not None and sha256(target) != expected:
            raise ValueError("Trial reference evaluation checksum changed")
        report = read_json(target)
    else:
        if lineage.get(f"{key}_sha256") is not None:
            raise ValueError("Pinned trial reference evaluation is missing")
        print(f"Evaluating {key}: {plan['episodes']} fresh matched Act 1 runs...", flush=True)
        report = act1["assess"](root, upstream, manifest, state, label=key,
                               episodes=plan["episodes"], validation_start=plan["validation_start"])
        atomic_json(target, report)
    act1["validate_assessment"](report, manifest, state, key, plan["episodes"], plan["validation_start"])
    if lineage.get(f"{key}_sha256") is None:
        lineage[f"{key}_sha256"] = sha256(target)
        atomic_json(root / "trial-lineage.json", lineage)
    return report


def brief_gameplay(report):
    return {"mean_floor": report["mean_floor"], "act1_clears": report["outcomes"]["win"],
            "episodes": report["episodes"], "timeouts": report["outcomes"]["timeout"],
            "aborted": report["outcomes"]["aborted"]}


def progress(source, root, upstream, *, target_steps=80000, episodes=100, validation_start=500):
    if target_steps < 1 or episodes < 100 or validation_start < 500:
        raise ValueError("Use a positive bounded budget, at least 100 evaluation episodes, and fresh validation indices >=500")
    source, root, upstream = source.resolve(), root.resolve(), upstream.resolve()
    if source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Source and trial directories must not overlap")
    torch.set_num_threads(2)
    plan = {"target_steps": target_steps, "episodes": episodes, "validation_start": validation_start}
    with selection["workflow_lock"](root):
        if not root.exists():
            create_trial(source, root, upstream, plan)
        lineage = read_json(root / "trial-lineage.json")
        if lineage["source_experiment"] != str(source) or lineage["plan"] != plan:
            raise ValueError("This trial's source, budget and evaluation schedule are fixed; use its original command to resume")
        config = ExperimentConfig(**lineage["config"])
        with trial_runner(lineage["recipe"], plan):
            manifest = runner.initialize(root, config, upstream)
            with experiment_lock(root):
                before = cached_reference(root, upstream, manifest, lineage, "before_rl", lineage["initial_checkpoint"])
                incumbent = cached_reference(root, upstream, manifest, lineage, "incumbent_reference", lineage["incumbent_checkpoint"])
            if not act1["clean"](before) or not act1["clean"](incumbent):
                raise ValueError("Trial reference runs contain timeouts or errors; no further PPO started")
            print(json.dumps({"stage": "before_exploratory_rl", "victory_scope": "act1",
                              "cloned_initializer": brief_gameplay(before["gameplay"]),
                              "incumbent": brief_gameplay(incumbent["gameplay"]),
                              "target_steps": target_steps, "promotion": "not performed"}, indent=2), flush=True)
            _, latest = verified_checkpoint(root, manifest["fingerprint"])
            remaining = max(0, target_steps - int(latest["num_timesteps"]))
            training = (runner.train(root, config, upstream, steps=remaining) if remaining else
                        {"status": "target_already_reached", "total_steps": latest["num_timesteps"]})
            with experiment_lock(root):
                _, latest = verified_checkpoint(root, manifest["fingerprint"])
                final = act1["assess"](root, upstream, manifest, latest, label="after_exploratory_rl",
                                      episodes=episodes, validation_start=validation_start)
                target = root / "evaluations" / f"trial-{time.time_ns()}.json"
                atomic_json(target, final)
            incremental = paired_compare(final["episodes"], before["episodes"])
            overall = paired_compare(final["episodes"], incumbent["episodes"])
            result = {"stage": "after_exploratory_rl", "status": "completed", "victory_scope": "act1",
                      "before_rl": before["gameplay"], "after_rl": final["gameplay"], "incumbent": incumbent["gameplay"],
                      "additional_rl_comparison": incremental, "incumbent_comparison": overall,
                      "new_rl_steps": training["total_steps"], "source_rl_steps": lineage["source_rl_steps"],
                      "training": training, "exploration_entry": lineage["exploration_entry"],
                      "champion_updated": False, "evaluation_report": str(target),
                      "summary_path": str(root / "trial-progress.json"),
                      "note": "PPO is compared against the frozen clone to isolate RL gains and the pre-imitation incumbent for overall performance. No champion update or additional budget is automatic. Wins are emulator Act 1 clears only."}
            atomic_json(root / "trial-progress.json", result)
            print(json.dumps({"stage": result["stage"], "victory_scope": "act1",
                              "before_rl": brief_gameplay(before["gameplay"]), "after_rl": brief_gameplay(final["gameplay"]),
                              "new_rl_steps": result["new_rl_steps"],
                              "rl_floor_gain": incremental["floor_delta"], "rl_floor_gain_ci95": incremental["floor_delta_ci95"],
                              "rl_act1_clear_rate_gain": incremental["win_delta"], "rl_act1_clear_rate_gain_ci95": incremental["win_delta_ci95"],
                              "summary_path": result["summary_path"], "champion_updated": False}, indent=2), flush=True)
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/ironclad-a0-act1"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-rl-trial"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--target-steps", type=int, default=80000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-start", type=int, default=500)
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
