"""Freeze the trial models and check RL gains on one larger fresh seed set.

This writes diagnostic evidence only. It does not train, collect demonstrations,
change the trial budget, replace a checkpoint or update a champion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import runpy
import sys
import time

import torch
from sb3_contrib import MaskablePPO

from spire2_rl import runner
from spire2_rl.experiment import (ExperimentConfig, atomic_json, experiment_lock,
                                 read_json, sha256, verified_checkpoint)
from spire2_rl.study import evaluation_seeds, paired_compare, summarize

trial = runpy.run_path(str(Path(__file__).with_name("act1_rl_trial.py")))
act1 = trial["act1"]


def training_record(root, config, latest):
    """Signal accounting for changing policies, not a strength estimate."""
    path = root / "episodes.jsonl"
    sessions_path = root / "sessions.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
        sessions = [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines() if line.strip()] if sessions_path.exists() else []
        if any(row["outcome"] not in {"win", "death", "timeout", "aborted"} for row in rows):
            raise ValueError("Training journal contains an unknown outcome")
        if any(row["status"] not in {"completed", "interrupted"} for row in sessions):
            raise ValueError("Training journal contains an unknown session status")
        discarded = [row for row in rows if row["outcome"] == "aborted" and row.get("reason") == "training budget ended; unfinished episode discarded"]
        other_aborts = [row for row in rows if row["outcome"] == "aborted" and row not in discarded]
        counts = Counter(row["outcome"] for row in rows if row["outcome"] != "aborted")
        session_counts = Counter(row["status"] for row in sessions)
        interrupted_states = [row["checkpoint"]["num_timesteps"] for row in sessions
                              if row["status"] == "interrupted" and row.get("checkpoint") is not None]
        return {"status": "read", "training_outcomes": {key: counts[key] for key in ("win", "death", "timeout")},
                "budget_boundary_discards": len(discarded), "other_aborted_episodes": len(other_aborts),
                "other_abort_reasons": dict(Counter(row.get("reason", "unknown") for row in other_aborts)),
                "session_statuses": dict(session_counts), "interrupted_checkpoint_steps": interrupted_states,
                "latest_decisions": latest["num_timesteps"], "rollout_size": config.n_envs * config.n_steps,
                "latest_remainder_within_rollout": latest["num_timesteps"] % (config.n_envs * config.n_steps),
                "note": "Training policies change. Counts locate positive reward examples, not held-out strength. Collected decisions can include a discarded partial rollout after interruption."}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"}


def fresh_seeds(root, lineage, config, episodes, start):
    seeds = evaluation_seeds("validation", episodes, config.experiment_seed, start=start)
    requested = set(seeds)
    plan = lineage["plan"]
    previous = set(evaluation_seeds("validation", plan["episodes"], config.experiment_seed,
                                    start=plan["validation_start"]))
    source = Path(lineage["source_experiment"])
    for key in ("before_imitation", "after_imitation_before_rl"):
        path = source / f"{key}.json"
        if not path.exists():
            raise ValueError(f"Source evaluation evidence is missing: {path}")
        previous.update(row["seed"] for row in read_json(path)["episodes"])
    if requested & previous:
        raise ValueError("Strength-check seeds overlap the previous evaluations; choose fresh indices")
    return seeds


def validate_report(report, protocol, seeds):
    if report["protocol"] != protocol:
        raise ValueError("Cached diagnostic belongs to another model or schedule")
    for key, label in (("candidate_episodes", "frozen-rl-candidate"), ("baseline_episodes", "frozen-cloned-initializer")):
        if ([row["seed"] for row in report[key]] != seeds
                or any(row["policy"] != label for row in report[key])):
            raise ValueError("Diagnostic raw rows differ from their frozen models or seed schedule")
    if (summarize(report["candidate_episodes"]) != report["candidate"]
            or summarize(report["baseline_episodes"]) != report["baseline"]
            or paired_compare(report["candidate_episodes"], report["baseline_episodes"]) != report["comparison"]):
        raise ValueError("Diagnostic statistics differ from their raw evidence")


def exact_win_test(candidate_rows, baseline_rows):
    """Two-sided exact McNemar test for matched, rare clear outcomes."""
    candidate = {row["seed"]: row for row in candidate_rows}
    baseline = {row["seed"]: row for row in baseline_rows}
    if candidate.keys() != baseline.keys():
        raise ValueError("Exact clear comparison requires matched seed sets")
    gains = losses = 0
    for seed, row in candidate.items():
        previous = baseline[seed]
        if row["outcome"] not in {"win", "death"} or previous["outcome"] not in {"win", "death"}:
            continue
        gains += int(row["outcome"] == "win" and previous["outcome"] == "death")
        losses += int(row["outcome"] == "death" and previous["outcome"] == "win")
    discordant = gains + losses
    probability = (min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / (1 << discordant))
                   if discordant else 1.0)
    return {"candidate_only_clears": gains, "baseline_only_clears": losses,
            "discordant_pairs": discordant, "exact_two_sided_p": probability,
            "supports_more_clears": gains > losses and probability < 0.05,
            "method": "Exact paired McNemar test; two-sided threshold 0.05",
            "note": "No clear-rate equivalence or noninferiority claim is made."}


def view(report, target):
    comparison = report["comparison"]
    clean = comparison["excluded_pairs"] == 0
    floor_gain = clean and comparison["eligible_pairs"] >= 100 and comparison["floor_delta_ci95"][0] > 0
    win_evidence = exact_win_test(report["candidate_episodes"], report["baseline_episodes"])
    win_gain = clean and comparison["eligible_pairs"] >= 100 and win_evidence["supports_more_clears"]
    return {"stage": "frozen_trial_strength_check", "victory_scope": "act1",
            "before_rl": trial["brief_gameplay"](report["baseline"]),
            "after_rl": trial["brief_gameplay"](report["candidate"]),
            "rl_floor_gain": comparison["floor_delta"], "rl_floor_gain_ci95": comparison["floor_delta_ci95"],
            "rl_act1_clear_rate_gain": comparison["win_delta"], "rl_act1_clear_rate_gain_ci95": comparison["win_delta_ci95"],
            "establishes_positive_floor_gain": bool(floor_gain), "establishes_positive_act1_clear_gain": bool(win_gain),
            "paired_clear_evidence": win_evidence,
            "training_record": report["training_record"], "checkpoint": report["protocol"]["candidate_checkpoint"],
            "report_path": str(target), "additional_training_steps": 0, "champion_updated": False,
            "note": "Paired intervals assess these frozen policies. Wins are emulator Act 1 clears only. A zero-width win-difference interval when neither model wins does not prove equal winning ability."}


def check(root, upstream, *, episodes=1000, start=700):
    if episodes < 1000 or start < 700:
        raise ValueError("Predeclare at least 1000 paired runs and fresh validation indices >=700")
    root, upstream = root.resolve(), upstream.resolve()
    torch.set_num_threads(2)
    with trial["selection"]["workflow_lock"](root):
        with experiment_lock(root):
            lineage = read_json(root / "trial-lineage.json")
            config = ExperimentConfig(**lineage["config"])
            with trial["trial_runner"](lineage["recipe"], lineage["plan"]):
                manifest = runner.initialize(root, config, upstream)
                candidate_path, candidate_state = verified_checkpoint(root, manifest["fingerprint"])
                if candidate_state["interrupted"] or candidate_state["num_timesteps"] < lineage["plan"]["target_steps"]:
                    raise ValueError("Complete the fixed trial budget before checking its final model")
                baseline_state = lineage["initial_checkpoint"]
                baseline_path = act1["checked_state"](root, baseline_state, manifest["fingerprint"])
                seeds = fresh_seeds(root, lineage, config, episodes, start)
                diagnostic = root / "diagnostics" / f"strength-{candidate_state['sha256'][:16]}"
                diagnostic.mkdir(parents=True, exist_ok=True)
                protocol_path = diagnostic / "protocol.json"
                target = diagnostic / "report.json"
                protocol = {"fingerprint": manifest["fingerprint"], "candidate_checkpoint": candidate_state,
                            "baseline_checkpoint": baseline_state, "split": "validation",
                            "episodes": episodes, "seed_start": start, "inference": "deterministic",
                            "checker_sha256": sha256(Path(__file__)), "victory_scope": "act1"}
                if protocol_path.exists():
                    if read_json(protocol_path) != protocol:
                        raise ValueError("This model's diagnostic sample size and schedule are already frozen; reuse the original command")
                else:
                    atomic_json(protocol_path, protocol)
                if target.exists():
                    report = read_json(target)
                    validate_report(report, protocol, seeds)
                else:
                    candidate = MaskablePPO.load(candidate_path, device="cpu")
                    baseline = MaskablePPO.load(baseline_path, device="cpu")
                    if int(candidate.num_timesteps) != candidate_state["num_timesteps"] or int(baseline.num_timesteps) != baseline_state["num_timesteps"]:
                        raise ValueError("Model timestep count differs from its checkpoint evidence")
                    env = runner._make(config, upstream)
                    candidate_rows, baseline_rows = [], []
                    print(f"Frozen models: checking {episodes} fresh pairs ({episodes * 2} simulated runs), without training...", flush=True)
                    try:
                        for offset in range(0, episodes, 50):
                            batch = seeds[offset:offset + 50]
                            candidate_rows.extend(runner._rollout(env, batch, model=candidate, label="frozen-rl-candidate"))
                            baseline_rows.extend(runner._rollout(env, batch, model=baseline, label="frozen-cloned-initializer"))
                            print(f"Checked {len(candidate_rows)}/{episodes} pairs", flush=True)
                    finally:
                        env.close()
                    if sha256(candidate_path) != candidate_state["sha256"] or sha256(baseline_path) != baseline_state["sha256"]:
                        raise ValueError("Frozen model checksum changed during evaluation")
                    report = {"protocol": protocol, "candidate": summarize(candidate_rows), "baseline": summarize(baseline_rows),
                              "comparison": paired_compare(candidate_rows, baseline_rows),
                              "candidate_episodes": candidate_rows, "baseline_episodes": baseline_rows,
                              "training_record": training_record(root, config, candidate_state), "created": time.time_ns()}
                    validate_report(report, protocol, seeds)
                    atomic_json(target, report)
                summary = view(report, target)
                print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
                return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-rl-trial"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--validation-start", type=int, default=700)
    args = parser.parse_args(argv)
    return check(args.experiment, args.upstream, episodes=args.episodes, start=args.validation_start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
