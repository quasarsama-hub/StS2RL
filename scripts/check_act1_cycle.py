"""One fixed fresh comparison of a completed cycle's frozen before/after policies.

Act 1 clear rate is the primary outcome; floor reached is secondary. This
diagnostic neither trains nor changes the numbered-cycle continuation gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import torch
from sb3_contrib import MaskablePPO

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, experiment_lock, read_json, sha256
from spire2_rl.study import evaluation_seeds, paired_compare, summarize

cycles = runpy.run_path(str(Path(__file__).with_name("act1_cycles.py")))
trial, checker = cycles["trial"], cycles["checker"]


def raw_seeds(document):
    seeds = set()
    for key in ("episodes", "candidate_episodes", "baseline_episodes"):
        rows = document.get(key)
        if isinstance(rows, list):
            seeds.update(row["seed"] for row in rows)
    return seeds


def fresh_schedule(root, parent, episodes, start=None, *, exclude=None):
    """Reserve a fresh block after known evaluations, including partial checks."""
    root = Path(root)
    config = parent["config"]
    used = raw_seeds(parent["before"]) | raw_seeds(parent["after"])
    next_index = parent["after"]["seed_start"] + len(parent["after"]["episodes"])
    paths = {Path(path) for path in parent["source_files"] if Path(path).suffix == ".json"}
    paths.update(root.parent.glob("cycle-*/before_rl.json"))
    paths.update(root.parent.glob("cycle-*/after_rl.json"))
    paths.update(root.parent.glob("cycle-*/cycle-lineage.json"))
    paths.update(root.parent.glob("cycle-*/diagnostics/*/protocol.json"))
    for path in sorted(paths):
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        document = read_json(path)
        used.update(raw_seeds(document))
        if isinstance(document.get("episodes"), list) and document.get("split") == "validation":
            next_index = max(next_index, document["seed_start"] + len(document["episodes"]))
        elif document.get("split") == "validation" and isinstance(document.get("episodes"), int):
            block = evaluation_seeds("validation", document["episodes"], config.experiment_seed, start=document["seed_start"])
            used.update(block)
            next_index = max(next_index, document["seed_start"] + document["episodes"])
        if path.name == "cycle-lineage.json":
            plan = document["plan"]
            used.update(evaluation_seeds("validation", plan["episodes"], config.experiment_seed, start=plan["validation_start"]))
            next_index = max(next_index, plan["validation_start"] + plan["episodes"])
        if "protocol" in document and document["protocol"].get("split") == "validation":
            protocol = document["protocol"]
            next_index = max(next_index, protocol["seed_start"] + protocol["episodes"])
    selected = next_index if start is None else start
    original = parent["base_factory"]["source_lineage"]
    seeds = checker["fresh_seeds"](root, original, config, episodes, selected)
    if set(seeds) & used:
        raise ValueError("Frozen-check seeds overlap earlier evaluations or a reserved diagnostic block")
    return seeds, selected


def validate_report(report, protocol, seeds):
    if report["protocol"] != protocol:
        raise ValueError("Frozen diagnostic belongs to another model or fixed schedule")
    for key, label in (("candidate_episodes", "frozen-cycle-candidate"),
                       ("baseline_episodes", "frozen-cycle-baseline")):
        rows = report[key]
        if ([row["seed"] for row in rows] != seeds or any(row["policy"] != label for row in rows)
                or any(row["outcome"] not in {"win", "death", "timeout", "aborted"} for row in rows)):
            raise ValueError("Frozen diagnostic raw rows differ from their models or seed schedule")
    candidate, baseline = report["candidate_episodes"], report["baseline_episodes"]
    if (report["candidate"] != summarize(candidate) or report["baseline"] != summarize(baseline)
            or report["comparison"] != paired_compare(candidate, baseline)
            or report["paired_clear_evidence"] != checker["exact_win_test"](candidate, baseline)):
        raise ValueError("Frozen diagnostic statistics differ from their raw evidence")


def view(report, target):
    comparison, clears = report["comparison"], report["paired_clear_evidence"]
    clean = comparison["excluded_pairs"] == 0
    floor_ci = comparison["floor_delta_ci95"]
    enough = comparison["eligible_pairs"] >= 1000
    return {"stage": "frozen_cycle_strength_check", "cycle": report["protocol"]["cycle"],
            "victory_scope": "act1", "primary_outcome": "Act 1 clear rate", "secondary_outcome": "floor reached",
            "status": "completed" if clean else "incomplete_evaluation",
            "before_rl": trial["brief_gameplay"](report["baseline"]),
            "after_rl": trial["brief_gameplay"](report["candidate"]),
            "rl_floor_gain": comparison["floor_delta"], "rl_floor_gain_ci95": floor_ci,
            "rl_act1_clear_rate_gain": comparison["win_delta"], "rl_act1_clear_rate_gain_ci95": comparison["win_delta_ci95"],
            "establishes_positive_floor_gain": bool(clean and enough and floor_ci is not None and floor_ci[0] > 0),
            "establishes_positive_act1_clear_gain": bool(clean and enough and clears["supports_more_clears"]),
            "paired_clear_evidence": clears, "checkpoint": report["protocol"]["candidate_checkpoint"],
            "validation_start": report["protocol"]["seed_start"], "report_path": str(target),
            "additional_training_steps": 0, "champion_updated": False, "training_gate_unchanged": True,
            "note": "One prespecified fresh check of frozen emulator Act 1 policies. No training, checkpoint replacement or automatic continuation. Do not increase the sample size after seeing results or pool it with the earlier data-dependent comparison. Per-comparison evidence is not a global confidence guarantee for the learning series."}


def evaluate_pair(root, upstream, parent, seeds):
    initial, final = parent["lineage"]["initial_checkpoint"], parent["state"]
    identity = parent["manifest"]["fingerprint"]
    baseline_path = trial["act1"]["checked_state"](root, initial, identity)
    candidate_path = trial["act1"]["checked_state"](root, final, identity)
    candidate = MaskablePPO.load(candidate_path, device="cpu")
    baseline = MaskablePPO.load(baseline_path, device="cpu")
    if int(candidate.num_timesteps) != final["num_timesteps"] or int(baseline.num_timesteps) != initial["num_timesteps"]:
        raise ValueError("Frozen model counter differs from its checkpoint evidence")
    env = runner._make(parent["config"], upstream)
    candidate_rows, baseline_rows = [], []
    print(f"Frozen cycle policies: checking {len(seeds)} fresh pairs ({2 * len(seeds)} simulated runs), with no training...", flush=True)
    try:
        for offset in range(0, len(seeds), 50):
            batch = seeds[offset:offset + 50]
            candidate_rows.extend(runner._rollout(env, batch, model=candidate, label="frozen-cycle-candidate"))
            baseline_rows.extend(runner._rollout(env, batch, model=baseline, label="frozen-cycle-baseline"))
            print(f"Checked {len(candidate_rows)}/{len(seeds)} pairs", flush=True)
    finally:
        env.close()
    trial["act1"]["checked_state"](root, initial, identity)
    trial["act1"]["checked_state"](root, final, identity)
    return candidate_rows, baseline_rows


def check(series, upstream, *, cycle=2, episodes=3000, start=None):
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (cycle, episodes))
            or cycle < 1 or episodes < 1000
            or start is not None and (isinstance(start, bool) or not isinstance(start, int) or start < 0)):
        raise ValueError("Predeclare a positive cycle and at least 1000 fresh matched pairs")
    series, upstream = Path(series).resolve(), Path(upstream).resolve()
    root = series / f"cycle-{cycle:04d}"
    if not root.is_dir():
        raise ValueError("Complete the numbered cycle before checking its frozen policies")
    torch.set_num_threads(2)
    with trial["selection"]["workflow_lock"](series), experiment_lock(root):
        parent = cycles["load_parent"](root, upstream)
        if "cycle" not in parent["lineage"] or parent["lineage"]["cycle"] != cycle:
            raise ValueError("Frozen check must name the completed numbered cycle")
        final, initial = parent["state"], parent["lineage"]["initial_checkpoint"]
        diagnostic = root / "diagnostics" / f"cycle-strength-{final['sha256'][:16]}"
        protocol_path, target = diagnostic / "protocol.json", diagnostic / "report.json"
        completion_path = diagnostic / "completion.json"
        previous = read_json(protocol_path) if protocol_path.exists() else None
        selected = previous["seed_start"] if previous is not None and start is None else start
        seeds, selected = fresh_schedule(root, parent, episodes, selected, exclude=protocol_path)
        protocol = {"cycle": cycle, "fingerprint": parent["manifest"]["fingerprint"],
                    "candidate_checkpoint": final, "baseline_checkpoint": initial,
                    "split": "validation", "episodes": episodes, "seed_start": selected,
                    "inference": "deterministic", "checker_sha256": sha256(Path(__file__)),
                    "victory_scope": "act1", "primary_outcome": "Act 1 clear rate",
                    "primary_test": "Exact paired two-sided McNemar; positive direction and p<0.05",
                    "secondary_outcome": "floor reached", "automatic_training": False}
        if previous is not None and previous != protocol:
            raise ValueError("This frozen diagnostic's models, sample size and schedule are fixed; reuse its original command")
        if previous is None:
            if target.exists() or completion_path.exists():
                raise ValueError("Frozen diagnostic protocol is missing")
            diagnostic.mkdir(parents=True, exist_ok=True)
            atomic_json(protocol_path, protocol)
        guarded = dict(parent["source_files"])
        for name in ("latest.json", "episodes.jsonl", "sessions.jsonl", "champion.json"):
            path = root / name
            if path.exists():
                guarded[str(path)] = sha256(path)
        completion = read_json(completion_path) if completion_path.exists() else None
        if completion is not None and (completion["protocol_sha256"] != sha256(protocol_path)
                or not target.exists() or completion["report_sha256"] != sha256(target)):
            raise ValueError("Pinned frozen diagnostic evidence changed or is missing")
        with cycles["cycle_runner"](parent["base_factory"], parent["lineage"]["plan"], parent["lineage"]["source_state"]):
            if target.exists():
                report = read_json(target)
                validate_report(report, protocol, seeds)
            else:
                candidate_rows, baseline_rows = evaluate_pair(root, upstream, parent, seeds)
                if any(sha256(Path(path)) != digest for path, digest in guarded.items()):
                    raise ValueError("Frozen learning files changed during evaluation")
                report = {"protocol": protocol, "candidate": summarize(candidate_rows), "baseline": summarize(baseline_rows),
                          "candidate_episodes": candidate_rows, "baseline_episodes": baseline_rows,
                          "comparison": paired_compare(candidate_rows, baseline_rows),
                          "paired_clear_evidence": checker["exact_win_test"](candidate_rows, baseline_rows),
                          "created": time.time_ns()}
                validate_report(report, protocol, seeds)
                atomic_json(target, report)
        if any(sha256(Path(path)) != digest for path, digest in guarded.items()):
            raise ValueError("Frozen learning files changed during the diagnostic")
        if completion is None:
            atomic_json(completion_path, {"protocol_sha256": sha256(protocol_path), "report_sha256": sha256(target)})
        summary = view(report, target)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-cycles"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--cycle", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--validation-start", type=int)
    args = parser.parse_args(argv)
    return check(args.experiment, args.upstream, cycle=args.cycle, episodes=args.episodes, start=args.validation_start)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"Frozen cycle check stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
