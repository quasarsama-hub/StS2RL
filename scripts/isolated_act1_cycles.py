"""Bounded Act 1 PPO cycles with fresh private event state for every run.

The first cycle forks a verified event-repair check into a new experiment.
Later cycles require measured clean progress; repeating a number resumes only
that number's fixed budget. Existing learning evidence is never rewritten.
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
from spire2_rl.event_isolation import IsolatedEvents
from spire2_rl.experiment import (ExperimentConfig, atomic_json, commit_checkpoint,
                                 experiment_lock, read_json, sha256, verified_checkpoint)
from spire2_rl.study import evaluation_seeds

cycles = runpy.run_path(str(Path(__file__).with_name("act1_cycles.py")))
strength_checker = runpy.run_path(str(Path(__file__).with_name("check_act1_cycle.py")))
repair_checker = runpy.run_path(str(Path(__file__).with_name("check_cycle_event_isolation.py")))
trial, checker = cycles["trial"], cycles["checker"]


@contextmanager
def factory(base_factory, original_plan, original_source_state, plan, source_state):
    """Rebuild the original flat factory, add event isolation exactly once."""
    with cycles["cycle_runner"](base_factory, original_plan, original_source_state):
        original = runner._make
        def make(config, upstream):
            raw = original(config, upstream)
            try:
                env = IsolatedEvents(raw)
            except BaseException:
                raw.close()
                raise
            env.backend_metadata.update(
                event_isolated_cycle="bounded-act1-private-events-v1",
                event_isolated_runner_sha256=sha256(Path(__file__)),
                event_isolated_plan=dict(plan),
                event_isolated_source_checkpoint=dict(source_state),
            )
            return env
        runner._make = make
        try:
            yield
        finally:
            runner._make = original


def continuation_gate(before, after):
    return cycles["continuation_gate"](before, after)


def repair_evidence(source, upstream, parent):
    """Validate the completed same-weights proof, not an asserted success flag."""
    source, upstream = Path(source).resolve(), Path(upstream).resolve()
    folder = source / "diagnostics" / f"event-isolation-{parent['state']['sha256'][:16]}"
    paths = {name: folder / f"{name}.json" for name in ("protocol", "report", "completion")}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("Complete the same-weights event-isolation check before starting repaired RL")
    protocol, report, completion = (read_json(paths[key]) for key in ("protocol", "report", "completion"))
    if (completion["protocol_sha256"] != sha256(paths["protocol"])
            or completion["report_sha256"] != sha256(paths["report"])):
        raise ValueError("Event-repair proof checksum changed or was not committed")
    guarded = {**parent["source_files"], **repair_checker["behavior"]["strength_evidence"](source, parent)}
    with cycles["cycle_runner"](parent["base_factory"], parent["lineage"]["plan"], parent["lineage"]["source_state"]):
        env = IsolatedEvents(runner._make(parent["config"], upstream))
        try:
            isolated_identity = env.backend_metadata
        finally:
            env.close()
    expected = {"cycle": parent["lineage"]["cycle"], "fingerprint": parent["manifest"]["fingerprint"],
                "checkpoint": parent["state"], "split": "validation", "episodes": protocol["episodes"],
                "seed_start": protocol["seed_start"], "checker_sha256": sha256(Path(__file__).with_name("check_cycle_event_isolation.py")),
                "repair_sha256": sha256(Path(__file__).parents[1] / "src/spire2_rl/event_isolation.py"),
                "source_strength_evidence": guarded, "inference": "deterministic", "victory_scope": "act1",
                "scope": "same-weights event-instance isolation check", "isolated_backend": isolated_identity,
                "automatic_training": False}
    if (protocol != expected or isinstance(protocol["episodes"], bool) or not isinstance(protocol["episodes"], int)
            or protocol["episodes"] < 100 or isinstance(protocol["seed_start"], bool)
            or not isinstance(protocol["seed_start"], int) or protocol["seed_start"] < 0):
        raise ValueError("Event-repair proof differs from its frozen source, repair or checker")
    seeds, _ = strength_checker["fresh_schedule"](source, parent, protocol["episodes"], protocol["seed_start"], exclude=paths["protocol"])
    repair_checker["validate_report"](report, protocol, seeds)
    view = repair_checker["view"](report, paths["report"])
    if view["status"] != "verified":
        raise ValueError("Resolve event-repair errors or timeouts before additional RL")
    guarded.update({str(path): sha256(path) for path in paths.values()})
    if any(sha256(Path(path)) != digest for path, digest in guarded.items()):
        raise ValueError("Event-repair source evidence changed")
    return {"source_files": guarded, "protocol": protocol, "status": "verified",
            "mechanics": report["mechanics"], "view": view}


def _runtime_files(source, files):
    result = dict(files)
    for name in ("latest.json", "episodes.jsonl", "sessions.jsonl", "champion.json"):
        path = source / name
        if path.exists():
            result[str(path)] = sha256(path)
    for path in source.glob("seed-cursors/worker-*.json"):
        result[str(path)] = sha256(path)
    return result


def load_parent(source, upstream):
    source, upstream = Path(source).resolve(), Path(upstream).resolve()
    lineage_path = source / "event-lineage.json"
    lineage = read_json(lineage_path)
    if any(sha256(Path(path)) != digest for path, digest in lineage["source_files"].items()):
        raise ValueError("Parent's frozen source or event-repair proof changed")
    config = ExperimentConfig(**lineage["config"])
    initial, plan = lineage["initial_checkpoint"], lineage["plan"]
    with factory(lineage["base_factory"], lineage["original_plan"], lineage["original_source_state"], plan, lineage["source_state"]):
        manifest = runner.initialize(source, config, upstream)
        checkpoint, state = verified_checkpoint(source, manifest["fingerprint"])
        if (state["interrupted"] or state["num_timesteps"] < lineage["target_total_steps"]
                or state["num_timesteps"] >= lineage["target_total_steps"] + config.n_envs * config.n_steps
                or initial["num_timesteps"] != lineage["source_state"]["num_timesteps"]
                or lineage["target_total_steps"] != initial["num_timesteps"] + plan["new_steps"]):
            raise ValueError("Complete the parent's fixed repaired budget before starting the next cycle")
        # Unlike progress(), parent loading cannot repair incomplete final publication.
        if any(lineage.get(key) is None for key in ("before_rl_sha256", "after_rl_sha256", "event_progress_sha256", "event_evidence_sha256")):
            raise ValueError("Finish publishing the parent's evaluated evidence before another cycle")
        expected = completed_result(source, upstream, manifest, lineage, config, state)
        if expected is None or expected["status"] != "completed":
            raise ValueError("Parent repaired cycle has no clean completed evaluation")
        before, after = read_json(source / "before_rl.json"), read_json(source / "after_rl.json")
        gate = continuation_gate(before, after)
        files = dict(lineage["source_files"])
        for report_state in (initial, state):
            model = trial["act1"]["checked_state"](source, report_state, manifest["fingerprint"])
            files[str(model)] = report_state["sha256"]
        for name in ("event-lineage.json", "experiment.json", "before_rl.json", "after_rl.json", "event-progress.json", "event-evidence.json"):
            path = source / name
            files[str(path)] = sha256(path)
    return {"lineage": lineage, "base_factory": lineage["base_factory"], "original_plan": lineage["original_plan"],
            "original_source_state": lineage["original_source_state"], "manifest": manifest,
            "state": state, "checkpoint": checkpoint, "config": config, "before": before, "after": after,
            "gate": gate, "repair_evidence": lineage["repair_evidence"], "source_files": _runtime_files(source, files)}


def fresh_schedule(base_source, series, parent, episodes, original_parent=None):
    """Reserve original and repaired series reports and pending protocols."""
    base_source, series = Path(base_source).resolve(), Path(series).resolve()
    if original_parent is None:
        # The initial source's original factory/evidence is already frozen by
        # the inherited source hashes; it is reconstructed only for scheduling.
        original_parent = {"config": parent["config"], "before": parent["before"], "after": parent["after"],
                           "base_factory": parent["base_factory"], "source_files": parent["source_files"]}
        # Parent may be a repaired cycle: scan the original folder's reports
        # and lineage as well, rather than treating its later seed start as
        # evidence that prior diagnostic blocks were reserved.
        original_parent["before"] = read_json(base_source / "before_rl.json")
        original_parent["after"] = read_json(base_source / "after_rl.json")
    _, start = strength_checker["fresh_schedule"](base_source, original_parent, episodes)
    config, used = parent["config"], set()
    paths = {Path(path) for path in parent["source_files"] if Path(path).suffix == ".json"}
    paths.update(base_source.parent.glob("cycle-*/before_rl.json"))
    paths.update(base_source.parent.glob("cycle-*/after_rl.json"))
    paths.update(base_source.parent.glob("cycle-*/cycle-lineage.json"))
    paths.update(base_source.parent.glob("cycle-*/diagnostics/*/protocol.json"))
    paths.update(series.glob("cycle-*/before_rl.json"))
    paths.update(series.glob("cycle-*/after_rl.json"))
    paths.update(series.glob("cycle-*/event-lineage.json"))
    paths.update(series.glob("cycle-*/diagnostics/*/protocol.json"))
    schedule_files = {}
    for path in sorted(paths):
        document = read_json(path)
        schedule_files[str(path)] = sha256(path)
        used.update(strength_checker["raw_seeds"](document))
        if path.name in {"event-lineage.json", "cycle-lineage.json"}:
            block = document["plan"]
            count, block_start = block["episodes"], block["validation_start"]
        elif document.get("split") == "validation":
            count = len(document["episodes"]) if isinstance(document.get("episodes"), list) else document.get("episodes")
            block_start = document.get("seed_start")
        elif isinstance(document.get("protocol"), dict) and document["protocol"].get("split") == "validation":
            count, block_start = document["protocol"]["episodes"], document["protocol"]["seed_start"]
        else:
            continue
        if not isinstance(count, int) or not isinstance(block_start, int):
            raise ValueError("Recorded validation reservation is malformed")
        block_seeds = evaluation_seeds("validation", count, config.experiment_seed, start=block_start)
        used.update(block_seeds)
        start = max(start, block_start + count)
    seeds = evaluation_seeds("validation", episodes, config.experiment_seed, start=start)
    checker["fresh_seeds"](base_source, parent["base_factory"]["source_lineage"], config, episodes, start)
    if set(seeds) & used:
        raise ValueError("Repaired-cycle validation overlaps a prior report or reserved protocol")
    return seeds, start, schedule_files


def create_cycle(source, root, upstream, *, cycle, base_source, series, new_steps, episodes):
    source, root, upstream = Path(source).resolve(), Path(root).resolve(), Path(upstream).resolve()
    if root.exists() or source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Use a new, separate cycle directory")
    with experiment_lock(source):
        if cycle == 1:
            parent = cycles["load_parent"](source, upstream)
            if source != Path(base_source).resolve() or "cycle" not in parent["lineage"]:
                raise ValueError("First repaired cycle must fork its original numbered source")
            proof = repair_evidence(source, upstream, parent)
            parent["source_files"] = _runtime_files(source, proof["source_files"])
            parent["repair_evidence"] = proof
            parent["original_plan"] = parent["lineage"]["plan"]
            parent["original_source_state"] = parent["lineage"]["source_state"]
            parent_gate = {"eligible": True, "reason": "First bounded trial after verified mutable-event repair; new experiment identity", "original_gate": parent["gate"]}
            original_parent = parent
        else:
            parent = load_parent(source, upstream)
            if (parent["lineage"]["cycle"] != cycle - 1
                    or parent["lineage"]["series"] != str(Path(series).resolve())
                    or parent["lineage"]["base_source"] != str(Path(base_source).resolve())):
                raise ValueError("Repaired cycle chain differs from its source and immediately preceding trial")
            if not parent["gate"]["eligible"]:
                raise ValueError("Next repaired cycle stopped: " + "; ".join(parent["gate"]["reasons"]))
            parent_gate, original_parent = parent["gate"], None
        config, state = parent["config"], parent["state"]
        _, start, schedule_files = fresh_schedule(base_source, series, parent, episodes, original_parent)
        parent["source_files"].update(schedule_files)
        plan = {"new_steps": new_steps, "episodes": episodes, "validation_start": start}
        indices = trial["source_indices"](source, state, config)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            with factory(parent["base_factory"], parent["original_plan"], parent["original_source_state"], plan, state):
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
                       "base_factory": parent["base_factory"], "original_plan": parent["original_plan"],
                       "original_source_state": parent["original_source_state"], "repair_evidence": parent["repair_evidence"],
                       "config": asdict(config), "plan": plan,
                       "initial_checkpoint": initial, "target_total_steps": state["num_timesteps"] + new_steps,
                       "training_seed_indices": indices, "parent_gate": parent_gate,
                       "optimizer": "preserved PPO Adam state"}
            atomic_json(staging / "event-lineage.json", lineage)
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
    return cycles["assess"](root, upstream, manifest, state, label=label,
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
        atomic_json(root / "event-lineage.json", lineage)
    return report


def cycle_summary(root, config, lineage, latest, before, after):
    gate = continuation_gate(before, after)
    comparison, clears = gate["comparison"], gate["paired_clear_evidence"]
    clean = comparison["excluded_pairs"] == 0
    floor_ci = comparison["floor_delta_ci95"]
    cycle, initial, plan = lineage["cycle"], lineage["initial_checkpoint"], lineage["plan"]
    summary = {"stage": "after_isolated_rl_cycle", "cycle": cycle, "victory_scope": "act1",
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
               "summary_path": str(root / "event-progress.json"), "champion_updated": False,
               "event_state_isolation": "private-registry-per-run-v1",
               "note": "Private event objects per environment/run, finite selections and repaired potion masks; fresh matched validation seeds. PPO policy, value, optimizer and counter retained. A separate mechanics-corrected experiment, with no automatic next budget, imitation or promotion. Wins are emulator Act 1 clears; native transfer and other globals remain unverified."}
    return summary, gate


def pin_final(root, lineage):
    changed = False
    for key, name in (("event_evidence_sha256", "event-evidence.json"), ("event_progress_sha256", "event-progress.json")):
        digest = sha256(root / name)
        if lineage.get(key) is not None and lineage[key] != digest:
            raise ValueError("Pinned final cycle evidence changed")
        if lineage.get(key) is None:
            lineage[key] = digest
            changed = True
    if changed:
        atomic_json(root / "event-lineage.json", lineage)


def completed_result(root, upstream, manifest, lineage, config, latest):
    """A completed cycle is read-only; recover only an unfinished final publication."""
    evidence_path, summary_path = root / "event-evidence.json", root / "event-progress.json"
    if not evidence_path.exists() and not summary_path.exists():
        if lineage.get("event_evidence_sha256") is not None or lineage.get("event_progress_sha256") is not None:
            raise ValueError("Pinned final cycle evidence is missing")
        return None
    if not evidence_path.exists():
        raise ValueError("Final cycle evidence is missing")
    for key, path in (("event_evidence_sha256", evidence_path), ("event_progress_sha256", summary_path)):
        if lineage.get(key) is not None and (not path.exists() or sha256(path) != lineage[key]):
            raise ValueError("Pinned final cycle evidence changed or is missing")
    if (latest["interrupted"] or latest["num_timesteps"] < lineage["target_total_steps"]
            or latest["num_timesteps"] >= lineage["target_total_steps"] + config.n_envs * config.n_steps):
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
        lineage = read_json(root / "event-lineage.json")
        if (lineage["cycle"] != cycle or lineage["series"] != str(series) or lineage["base_source"] != str(base_source)
                or lineage["source_experiment"] != str(parent) or lineage["plan"]["new_steps"] != new_steps or lineage["plan"]["episodes"] != episodes):
            raise ValueError("This cycle's source, budget and evaluation schedule are fixed; resume with its original command")
        if any(sha256(Path(path)) != digest for path, digest in lineage["source_files"].items()):
            raise ValueError("Pinned parent evidence changed")
        config = ExperimentConfig(**lineage["config"])
        initial, plan = lineage["initial_checkpoint"], lineage["plan"]
        if initial["num_timesteps"] != lineage["source_state"]["num_timesteps"] or lineage["target_total_steps"] != initial["num_timesteps"] + new_steps:
            raise ValueError("Cycle counter target differs from its frozen starting model")
        with factory(lineage["base_factory"], lineage["original_plan"], lineage["original_source_state"], plan, lineage["source_state"]):
            manifest = runner.initialize(root, config, upstream)
            with experiment_lock(root):
                latest_path, latest = verified_checkpoint(root, manifest["fingerprint"])
                loaded = MaskablePPO.load(latest_path, device="cpu")
                if int(loaded.num_timesteps) != latest["num_timesteps"]:
                    raise ValueError("Latest cycle model counter differs from its saved checkpoint metadata")
                del loaded
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
            print(json.dumps({"stage": "before_isolated_rl_cycle", "cycle": cycle, "victory_scope": "act1",
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
            if any(sha256(Path(path)) != digest for path, digest in lineage["source_files"].items()):
                raise ValueError("Pinned source evidence changed during the repaired trial")
            summary, gate = cycle_summary(root, config, lineage, latest, before, after)
            atomic_json(root / "event-evidence.json", {"before": before, "after": after, "comparison": gate["comparison"], "training": training, "next_cycle_gate": gate})
            atomic_json(root / "event-progress.json", summary)
            pin_final(root, lineage)
            print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
            return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("experiments/ironclad-a0-act1-cycles/cycle-0002"))
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-event-isolated-cycles"))
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
