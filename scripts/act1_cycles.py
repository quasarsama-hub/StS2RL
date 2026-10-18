"""Numbered, bounded Act 1 RL trials from the latest evaluated learner.

Repeat a cycle number to resume it. Select the next number only after reviewing
its predecessor. Each cycle preserves PPO state and has fresh matched seeds.
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
from types import SimpleNamespace

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from spire2_rl import runner
from spire2_rl.experiment import (ExperimentConfig, atomic_json, commit_checkpoint,
                                 experiment_lock, read_json, sha256, verified_checkpoint)
from spire2_rl.study import evaluation_seeds, paired_compare, summarize

continued = runpy.run_path(str(Path(__file__).with_name("continue_act1_rl.py")))
trial = continued["trial"]
checker = continued["checker"]


@contextmanager
def base_runner(base_factory):
    with continued["continuation_runner"](
            base_factory["source_lineage"], base_factory["plan"],
            base_factory["source_state"], base_factory["repair_evidence"]):
        yield


@contextmanager
def cycle_runner(base_factory, plan, source_state):
    """Reconstruct the stable repaired factory once, then label this trial."""
    with base_runner(base_factory):
        original = runner._make
        def make(config, upstream):
            env = original(config, upstream)
            env.backend_metadata.update(
                rl_cycle="bounded-act1-ppo-cycle-v1", rl_cycle_runner_sha256=sha256(Path(__file__)),
                rl_cycle_plan=dict(plan), rl_cycle_source_checkpoint=dict(source_state),
            )
            return env
        runner._make = make
        try:
            yield
        finally:
            runner._make = original


def continuation_gate(before, after):
    """A measured gain justifies another trial; it does not promote a champion."""
    comparison = paired_compare(after["episodes"], before["episodes"])
    clears = checker["exact_win_test"](after["episodes"], before["episodes"])
    floor_ci = comparison["floor_delta_ci95"]
    floor_gain = floor_ci is not None and floor_ci[0] > 0
    win_regression = (clears["baseline_only_clears"] > clears["candidate_only_clears"]
                      and clears["exact_two_sided_p"] < 0.05)
    reasons = []
    if comparison["excluded_pairs"] or comparison["eligible_pairs"] < 1000:
        reasons.append("Requires at least 1000 clean matched evaluation pairs")
    if not floor_gain and not clears["supports_more_clears"]:
        reasons.append("The previous trial did not establish further floor or clear-rate progress")
    if win_regression:
        reasons.append("The previous trial established a clear-rate regression")
    return {"eligible": not reasons, "reasons": reasons, "comparison": comparison,
            "paired_clear_evidence": clears, "purpose": "Eligibility for one further bounded trial only",
            "note": "No clear-rate equivalence, noninferiority, automatic promotion or combined confidence claim across repeated trials."}


def load_parent(source, upstream):
    source, upstream = Path(source).resolve(), Path(upstream).resolve()
    if (source / "cycle-lineage.json").exists():
        lineage_path = source / "cycle-lineage.json"
        summary_path = source / "cycle-progress.json"
        lineage = read_json(lineage_path)
        base_factory = lineage["base_factory"]
        context = cycle_runner(base_factory, lineage["plan"], lineage["source_state"])
        inherited = lineage["source_files"]
    elif (source / "capacity-lineage.json").exists():
        lineage_path = source / "capacity-lineage.json"
        summary_path = source / "continuation-progress.json"
        lineage = read_json(lineage_path)
        base_factory = {key: lineage[key] for key in ("source_lineage", "plan", "source_state", "repair_evidence")}
        context = base_runner(base_factory)
        inherited = lineage["source_evidence"]
    else:
        raise ValueError("Source must be the completed repaired continuation or an earlier numbered cycle")
    if any(sha256(Path(path)) != digest for path, digest in inherited.items()):
        raise ValueError("Parent's frozen model or repair evidence changed")
    config = ExperimentConfig(**lineage["config"])
    with context:
        manifest = runner.initialize(source, config, upstream)
        checkpoint, state = verified_checkpoint(source, manifest["fingerprint"])
        initial = lineage["initial_checkpoint"]
        if (state["interrupted"] or state["num_timesteps"] < lineage["target_total_steps"]
                or state["num_timesteps"] >= lineage["target_total_steps"] + config.n_envs * config.n_steps
                or initial["num_timesteps"] != lineage["source_state"]["num_timesteps"]
                or lineage["target_total_steps"] != initial["num_timesteps"] + lineage["plan"]["new_steps"]):
            raise ValueError("Complete the parent's fixed budget before starting the next cycle")
        reports = []
        source_files = dict(inherited)
        for key, report_state in (("before_rl", initial), ("after_rl", state)):
            path = source / f"{key}.json"
            if lineage.get(f"{key}_sha256") != sha256(path):
                raise ValueError("Parent evaluation checksum changed or was not committed")
            report = read_json(path)
            trial["act1"]["checked_state"](source, report_state, manifest["fingerprint"])
            trial["act1"]["validate_assessment"](
                report, manifest, report_state, key, lineage["plan"]["episodes"], lineage["plan"]["validation_start"])
            reports.append(report)
            source_files[str(path)] = sha256(path)
            report_model = trial["act1"]["checked_state"](source, report_state, manifest["fingerprint"])
            source_files[str(report_model)] = report_state["sha256"]
    before, after = reports
    gate = continuation_gate(before, after)
    summary = read_json(summary_path)
    comparison, clears = gate["comparison"], gate["paired_clear_evidence"]
    expected_stage = "after_rl_cycle" if lineage_path.name == "cycle-lineage.json" else "after_continued_rl"
    if (summary["stage"] != expected_stage or summary["status"] != "completed" or summary["checkpoint"] != state
            or summary["champion_updated"] is not False or summary["victory_scope"] != "act1"
            or summary["before_rl"] != trial["brief_gameplay"](before["gameplay"])
            or summary["after_rl"] != trial["brief_gameplay"](after["gameplay"])
            or summary["new_rl_steps"] != state["num_timesteps"] - initial["num_timesteps"]
            or any(summary[key] != comparison[value] for key, value in (
                ("rl_floor_gain", "floor_delta"), ("rl_floor_gain_ci95", "floor_delta_ci95"),
                ("rl_act1_clear_rate_gain", "win_delta"), ("rl_act1_clear_rate_gain_ci95", "win_delta_ci95")))
            or summary["paired_clear_evidence"] != clears):
        raise ValueError("Parent progress summary differs from its frozen raw evaluations")
    if lineage_path.name == "cycle-lineage.json":
        evidence_path = source / "cycle-evidence.json"
        if (summary["cycle"] != lineage["cycle"]
                or lineage.get("cycle_progress_sha256") != sha256(summary_path)
                or lineage.get("cycle_evidence_sha256") != sha256(evidence_path)):
            raise ValueError("Parent cycle final evidence checksum changed or was not committed")
        evidence = read_json(evidence_path)
        if (evidence["before"] != before or evidence["after"] != after
                or evidence["comparison"] != comparison or evidence["next_cycle_gate"] != gate
                or evidence["training"]["total_steps"] != state["num_timesteps"]
                or evidence["training"]["status"] not in {"completed", "target_already_reached"}):
            raise ValueError("Parent cycle final evidence differs from its raw evaluated models")
        source_files[str(evidence_path)] = sha256(evidence_path)
    for path in (lineage_path, source / "experiment.json", summary_path):
        source_files[str(path)] = sha256(path)
    return {"lineage": lineage, "base_factory": base_factory, "manifest": manifest,
            "state": state, "checkpoint": checkpoint, "config": config, "before": before,
            "after": after, "gate": gate, "source_files": source_files}


def create_cycle(source, root, upstream, *, cycle, base_source, series, new_steps, episodes):
    source, root, upstream = Path(source).resolve(), Path(root).resolve(), Path(upstream).resolve()
    if root.exists() or source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Use a new, separate cycle directory")
    with experiment_lock(source):
        parent = load_parent(source, upstream)
        if "cycle" in parent["lineage"] and (
                parent["lineage"]["cycle"] != cycle - 1
                or parent["lineage"]["series"] != str(Path(series).resolve())
                or parent["lineage"]["base_source"] != str(Path(base_source).resolve())):
            raise ValueError("Cycle chain does not match its original source and immediately preceding trial")
        if not parent["gate"]["eligible"]:
            raise ValueError("Next cycle stopped: " + "; ".join(parent["gate"]["reasons"]))
        config, state = parent["config"], parent["state"]
        start = parent["after"]["seed_start"] + len(parent["after"]["episodes"])
        plan = {"new_steps": new_steps, "episodes": episodes, "validation_start": start}
        seeds = evaluation_seeds("validation", episodes, config.experiment_seed, start=start)
        used = {row["seed"] for report in (parent["before"], parent["after"]) for row in report["episodes"]}
        # Check the earlier imitation/trial evidence as well as the immediate parent.
        original_lineage = parent["base_factory"]["source_lineage"]
        checker["fresh_seeds"](source, original_lineage, config, episodes, start)
        for path in parent["source_files"]:
            if Path(path).name == "report.json" and "strength-" in Path(path).parent.name:
                report = read_json(Path(path))
                used.update(row["seed"] for row in report["candidate_episodes"])
            elif Path(path).name in {"before_rl.json", "after_rl.json"}:
                used.update(row["seed"] for row in read_json(Path(path))["episodes"])
        if set(seeds) & used:
            raise ValueError("Cycle validation seeds overlap earlier evaluations")
        indices = trial["source_indices"](source, state, config)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            with cycle_runner(parent["base_factory"], plan, state):
                manifest = runner.initialize(staging, config, upstream)
                vec = DummyVecEnv([lambda: runner._make(config, upstream) for _ in range(config.n_envs)])
                try:
                    model = MaskablePPO.load(parent["checkpoint"], env=vec, device="cpu", force_reset=True)
                    if (int(model.num_timesteps) != state["num_timesteps"]
                            or model.observation_space != vec.observation_space or model.action_space != vec.action_space):
                        raise ValueError("Parent model counter or public spaces changed")
                    vec._seeds = [None] * config.n_envs
                    initial = commit_checkpoint(model, staging, [SimpleNamespace(next_index=i) for i in indices], manifest["fingerprint"])
                finally:
                    vec.close()
            if any(sha256(Path(path)) != digest for path, digest in parent["source_files"].items()):
                raise ValueError("Parent evidence changed during cycle initialization")
            lineage = {"cycle": cycle, "series": str(Path(series).resolve()), "base_source": str(Path(base_source).resolve()),
                       "source_experiment": str(source), "source_state": state, "source_files": parent["source_files"],
                       "base_factory": parent["base_factory"], "config": asdict(config), "plan": plan,
                       "initial_checkpoint": initial, "target_total_steps": state["num_timesteps"] + new_steps,
                       "training_seed_indices": indices, "parent_gate": parent["gate"],
                       "optimizer": "preserved PPO Adam state"}
            atomic_json(staging / "cycle-lineage.json", lineage)
            if root.exists():
                raise ValueError("Cycle directory was created during initialization")
            staging.rename(root)
            return lineage
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise


def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
    # Reuse the unchanged batched evaluator; the active factory defines this cycle.
    return continued["assess"](root, upstream, manifest, state, label=label,
                               episodes=episodes, validation_start=validation_start)


def cached_assessment(root, upstream, manifest, lineage, key, state):
    path, plan = root / f"{key}.json", lineage["plan"]
    trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
    if path.exists():
        if lineage.get(f"{key}_sha256") is not None and sha256(path) != lineage[f"{key}_sha256"]:
            raise ValueError("Cached cycle evaluation checksum changed")
        report = read_json(path)
    else:
        if lineage.get(f"{key}_sha256") is not None:
            raise ValueError("Pinned cycle evaluation is missing")
        report = assess(root, upstream, manifest, state, label=key, episodes=plan["episodes"], validation_start=plan["validation_start"])
        trial["act1"]["validate_assessment"](report, manifest, state, key, plan["episodes"], plan["validation_start"])
        atomic_json(path, report)
    trial["act1"]["validate_assessment"](report, manifest, state, key, plan["episodes"], plan["validation_start"])
    if lineage.get(f"{key}_sha256") is None:
        lineage[f"{key}_sha256"] = sha256(path)
        atomic_json(root / "cycle-lineage.json", lineage)
    return report


def cycle_summary(root, config, lineage, latest, before, after):
    gate = continuation_gate(before, after)
    comparison, clears = gate["comparison"], gate["paired_clear_evidence"]
    clean = comparison["excluded_pairs"] == 0
    floor_ci = comparison["floor_delta_ci95"]
    cycle, initial, plan = lineage["cycle"], lineage["initial_checkpoint"], lineage["plan"]
    summary = {"stage": "after_rl_cycle", "cycle": cycle, "victory_scope": "act1",
               "status": "completed" if clean else "incomplete_evaluation",
               "before_rl": trial["brief_gameplay"](before["gameplay"]), "after_rl": trial["brief_gameplay"](after["gameplay"]),
               "new_rl_steps": latest["num_timesteps"] - initial["num_timesteps"],
               "source_rl_steps": initial["num_timesteps"], "cumulative_rl_steps": latest["num_timesteps"],
               "rl_floor_gain": comparison["floor_delta"], "rl_floor_gain_ci95": floor_ci,
               "rl_act1_clear_rate_gain": comparison["win_delta"], "rl_act1_clear_rate_gain_ci95": comparison["win_delta_ci95"],
               "establishes_positive_floor_gain": bool(clean and comparison["eligible_pairs"] >= 1000 and floor_ci is not None and floor_ci[0] > 0),
               "establishes_positive_act1_clear_gain": bool(clean and comparison["eligible_pairs"] >= 1000 and clears["supports_more_clears"]),
               "paired_clear_evidence": clears, "checkpoint": latest,
               "training_record": checker["training_record"](root, config, latest),
               "next_cycle_eligible": gate["eligible"], "next_cycle_reasons": gate["reasons"],
               "next_cycle": cycle + 1, "validation_start": plan["validation_start"],
               "summary_path": str(root / "cycle-progress.json"), "champion_updated": False,
               "note": "Same repaired interface and matched fresh validation seeds. PPO policy, value, optimizer and counter preserved. Each cycle is a fixed trial; no automatic next budget, imitation or promotion. Wins are emulator Act 1 clears only."}
    return summary, gate


def pin_final(root, lineage):
    changed = False
    for key, name in (("cycle_evidence_sha256", "cycle-evidence.json"), ("cycle_progress_sha256", "cycle-progress.json")):
        digest = sha256(root / name)
        if lineage.get(key) is not None and lineage[key] != digest:
            raise ValueError("Pinned final cycle evidence changed")
        if lineage.get(key) is None:
            lineage[key] = digest
            changed = True
    if changed:
        atomic_json(root / "cycle-lineage.json", lineage)


def completed_result(root, upstream, manifest, lineage, config, latest):
    """A completed cycle is read-only; recover only an unfinished final publication."""
    evidence_path, summary_path = root / "cycle-evidence.json", root / "cycle-progress.json"
    if not evidence_path.exists() and not summary_path.exists():
        if lineage.get("cycle_evidence_sha256") is not None or lineage.get("cycle_progress_sha256") is not None:
            raise ValueError("Pinned final cycle evidence is missing")
        return None
    if not evidence_path.exists():
        raise ValueError("Final cycle evidence is missing")
    for key, path in (("cycle_evidence_sha256", evidence_path), ("cycle_progress_sha256", summary_path)):
        if lineage.get(key) is not None and (not path.exists() or sha256(path) != lineage[key]):
            raise ValueError("Pinned final cycle evidence changed or is missing")
    if latest["interrupted"] or latest["num_timesteps"] < lineage["target_total_steps"]:
        raise ValueError("Final cycle evidence belongs to an unfinished budget")
    before = cached_assessment(root, upstream, manifest, lineage, "before_rl", lineage["initial_checkpoint"])
    after = cached_assessment(root, upstream, manifest, lineage, "after_rl", latest)
    expected, gate = cycle_summary(root, config, lineage, latest, before, after)
    evidence = read_json(evidence_path)
    if (evidence["before"] != before or evidence["after"] != after
            or evidence["comparison"] != gate["comparison"] or evidence["next_cycle_gate"] != gate
            or evidence["training"]["total_steps"] != latest["num_timesteps"]
            or evidence["training"]["status"] not in {"completed", "target_already_reached"}):
        raise ValueError("Final cycle evidence differs from its raw evaluated models")
    if summary_path.exists():
        if read_json(summary_path) != expected:
            raise ValueError("Final cycle summary differs from raw evidence")
    else:
        atomic_json(summary_path, expected)
    pin_final(root, lineage)
    return expected


def progress(source, series, upstream, *, cycle=1, new_steps=80000, episodes=1000):
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (cycle, new_steps, episodes)) or cycle < 1 or new_steps < 1 or episodes < 1000:
        raise ValueError("Use a positive cycle number and fixed budget, with at least 1000 evaluation pairs")
    base_source, series, upstream = Path(source).resolve(), Path(series).resolve(), Path(upstream).resolve()
    if base_source == series or series.is_relative_to(base_source) or base_source.is_relative_to(series):
        raise ValueError("Source and cycle series must not overlap")
    root = series / f"cycle-{cycle:04d}"
    parent = base_source if cycle == 1 else series / f"cycle-{cycle - 1:04d}"
    if not parent.is_dir():
        raise ValueError("Complete the immediately preceding cycle first")
    torch.set_num_threads(2)
    with trial["selection"]["workflow_lock"](series):
        if not root.exists():
            create_cycle(parent, root, upstream, cycle=cycle, base_source=base_source, series=series, new_steps=new_steps, episodes=episodes)
        lineage = read_json(root / "cycle-lineage.json")
        if (lineage["cycle"] != cycle or lineage["series"] != str(series) or lineage["base_source"] != str(base_source)
                or lineage["source_experiment"] != str(parent) or lineage["plan"]["new_steps"] != new_steps or lineage["plan"]["episodes"] != episodes):
            raise ValueError("This cycle's source, budget and evaluation schedule are fixed; resume with its original command")
        if any(sha256(Path(path)) != digest for path, digest in lineage["source_files"].items()):
            raise ValueError("Pinned parent evidence changed")
        config = ExperimentConfig(**lineage["config"])
        initial, plan = lineage["initial_checkpoint"], lineage["plan"]
        if initial["num_timesteps"] != lineage["source_state"]["num_timesteps"] or lineage["target_total_steps"] != initial["num_timesteps"] + new_steps:
            raise ValueError("Cycle counter target differs from its frozen starting model")
        with cycle_runner(lineage["base_factory"], plan, lineage["source_state"]):
            manifest = runner.initialize(root, config, upstream)
            with experiment_lock(root):
                _, latest = verified_checkpoint(root, manifest["fingerprint"])
                if (latest["num_timesteps"] < initial["num_timesteps"]
                        or latest["num_timesteps"] >= lineage["target_total_steps"] + config.n_envs * config.n_steps):
                    raise ValueError("Latest cycle counter falls outside its fixed training budget")
                cached = completed_result(root, upstream, manifest, lineage, config, latest)
                if cached is not None:
                    print(json.dumps(cached, indent=2, allow_nan=False), flush=True)
                    return cached
                before = cached_assessment(root, upstream, manifest, lineage, "before_rl", initial)
            if not trial["act1"]["clean"](before):
                raise ValueError("Starting-policy evaluation contains errors or timeouts; no additional RL started")
            print(json.dumps({"stage": "before_rl_cycle", "cycle": cycle, "victory_scope": "act1",
                              "starting_policy": trial["brief_gameplay"](before["gameplay"]),
                              "fixed_additional_steps": new_steps, "source_rl_steps": initial["num_timesteps"],
                              "validation_start": plan["validation_start"], "optimizer": lineage["optimizer"]}, indent=2), flush=True)
            _, latest = verified_checkpoint(root, manifest["fingerprint"])
            remaining = max(0, lineage["target_total_steps"] - latest["num_timesteps"])
            if not remaining and latest["interrupted"]:
                raise ValueError("Interrupted checkpoint meets the counter target without completed training")
            if remaining:
                print(f"Cycle {cycle}: {remaining} decisions remain in its fixed {new_steps}-decision budget...", flush=True)
                training = runner.train(root, config, upstream, steps=remaining)
            else:
                training = {"status": "target_already_reached", "total_steps": latest["num_timesteps"]}
            with experiment_lock(root):
                _, latest = verified_checkpoint(root, manifest["fingerprint"])
                if (latest["interrupted"] or latest["num_timesteps"] < lineage["target_total_steps"]
                        or latest["num_timesteps"] >= lineage["target_total_steps"] + config.n_envs * config.n_steps):
                    raise ValueError("Final cycle checkpoint is interrupted or outside its fixed budget")
                after = cached_assessment(root, upstream, manifest, lineage, "after_rl", latest)
            summary, gate = cycle_summary(root, config, lineage, latest, before, after)
            atomic_json(root / "cycle-evidence.json", {"before": before, "after": after, "comparison": gate["comparison"], "training": training, "next_cycle_gate": gate})
            atomic_json(root / "cycle-progress.json", summary)
            pin_final(root, lineage)
            print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/ironclad-a0-act1-rl-continuation"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-cycles"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--new-steps", type=int, default=80000)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args(argv)
    return progress(args.source, args.experiment, args.upstream, cycle=args.cycle, new_steps=args.new_steps, episodes=args.episodes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
