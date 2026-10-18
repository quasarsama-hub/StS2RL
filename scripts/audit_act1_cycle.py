"""Bounded behavior audit after a frozen numbered-cycle strength check.

Play five fixed controllers on the same fresh seeds. Diagnose combat and other
decisions separately; do not train, change a budget gate or promote a policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, experiment_lock, read_json, sha256
from spire2_rl.study import evaluation_seeds

strength_checker = runpy.run_path(str(Path(__file__).with_name("check_act1_cycle.py")))
cycles = strength_checker["cycles"]
auditor = runpy.run_path(str(Path(__file__).with_name("audit_combat.py")))
observer_path = Path(__file__).with_name("cycle_audit_observer.py")
CycleAuditObserver = runpy.run_path(str(observer_path))["CycleAuditObserver"]
teacher_path = Path(__file__).with_name("fair_teacher.py")
FairTeacher = runpy.run_path(str(teacher_path))["FairTeacher"]
LABELS = ("current", "previous", "fair_teacher", "teacher_combat", "teacher_noncombat")


class HybridController:
    """Teacher in one decision category, frozen current policy in the other."""
    def __init__(self, model, teacher, mode):
        if mode not in {"teacher_combat", "teacher_noncombat"}:
            raise ValueError("Unknown hybrid decision category")
        self.model, self.teacher, self.mode = model, teacher, mode

    def select(self, env):
        mgr = env.raw_env._mgr
        combat = mgr.get_combat_state() if mgr.phase == mgr.PHASE_COMBAT else None
        ordinary = combat is not None and combat.pending_choice is None
        use_teacher = ordinary if self.mode == "teacher_combat" else not ordinary
        if use_teacher:
            return int(self.teacher.select(env))
        action, _ = self.model.predict(env.current_observation, action_masks=env.action_masks(), deterministic=True)
        return int(action)


def strength_evidence(root, parent):
    folder = root / "diagnostics" / f"cycle-strength-{parent['state']['sha256'][:16]}"
    paths = {name: folder / f"{name}.json" for name in ("protocol", "report", "completion")}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("Complete the fixed frozen-cycle strength check before auditing behavior")
    protocol, report, completion = (read_json(paths[key]) for key in ("protocol", "report", "completion"))
    if (completion["protocol_sha256"] != sha256(paths["protocol"])
            or completion["report_sha256"] != sha256(paths["report"])
            or protocol["cycle"] != parent["lineage"]["cycle"]
            or protocol["fingerprint"] != parent["manifest"]["fingerprint"]
            or protocol["candidate_checkpoint"] != parent["state"]
            or protocol["baseline_checkpoint"] != parent["lineage"]["initial_checkpoint"]
            or protocol["episodes"] < 1000 or protocol["split"] != "validation"
            or protocol["inference"] != "deterministic" or protocol["victory_scope"] != "act1"
            or protocol["checker_sha256"] != sha256(Path(__file__).with_name("check_act1_cycle.py"))):
        raise ValueError("Frozen strength evidence differs from the completed cycle or its checker")
    seeds = evaluation_seeds("validation", protocol["episodes"], parent["config"].experiment_seed, start=protocol["seed_start"])
    strength_checker["validate_report"](report, protocol, seeds)
    if report["comparison"]["excluded_pairs"]:
        raise ValueError("Resolve errors or timeouts in the frozen strength check before behavior auditing")
    return {str(path): sha256(path) for path in paths.values()}


def summarize_rows(rows):
    summary = auditor["describe"](rows)
    summary["floor_distribution"] = {str(key): value for key, value in summary["floor_distribution"].items()}
    rooms, choices, paths, cards, rests = Counter(), Counter(), Counter(), Counter(), Counter()
    combat_entries, elites, bosses = [], [], []
    observer_error_runs = 0
    for row in rows:
        observed = row["audit"]
        observer_error_runs += bool(observed["observer_errors"])
        if row["outcome"] == "death":
            context = observed["death_context"] or {}
            rooms[context.get("room_type") or "UNKNOWN"] += 1
        for destination, field in ((choices, "macro_action_counts"), (paths, "map_room_choices"),
                                   (cards, "picked_card_counts"), (rests, "rest_option_counts")):
            destination.update(observed[field])
        entries = observed["combat_entries"]
        combat_entries.extend(entries)
        elites.extend(entry for entry in entries if "ELITE" in str(entry.get("room_type")))
        bosses.extend(entry for entry in entries if "BOSS" in str(entry.get("room_type")))
    def mean_hp(entries):
        values = [entry["hp"] for entry in entries if entry.get("hp") is not None]
        return sum(values) / len(values) if values else None
    summary.update(death_room_counts=dict(rooms), macro_action_counts=dict(choices), map_room_choices=dict(paths),
                   picked_card_counts=dict(cards), rest_option_counts=dict(rests),
                   combat_arrivals=len(combat_entries), elite_arrivals=len(elites), boss_arrivals=len(bosses),
                   mean_hp_entering_combat=mean_hp(combat_entries), mean_hp_entering_elites=mean_hp(elites),
                   mean_hp_entering_bosses=mean_hp(bosses), observer_error_runs=observer_error_runs)
    return summary


def validate_report(report, protocol, seeds):
    if report["protocol"] != protocol or set(report["episodes"]) != set(LABELS):
        raise ValueError("Behavior audit differs from its fixed models or controllers")
    for label in LABELS:
        rows = report["episodes"][label]
        if ([row["seed"] for row in rows] != seeds or any(row["policy"] != label for row in rows)
                or any(row["act1_cleared"] != (row["outcome"] == "win") for row in rows)
                or report["controllers"][label] != summarize_rows(rows)):
            raise ValueError("Behavior audit statistics or rows differ from their fixed evidence")


def view(report, target):
    controllers = {}
    for label in LABELS:
        summary = report["controllers"][label]
        controllers[label] = {"gameplay": cycles["trial"]["brief_gameplay"](summary["gameplay"]),
                              **{key: summary[key] for key in (
                                  "early_end_turn_fraction", "mean_unused_energy_per_end_turn",
                                  "deaths_with_potions_left", "deaths_with_legal_potion_before_last_action",
                                  "death_room_counts", "elite_arrivals", "boss_arrivals",
                                  "mean_hp_entering_elites", "mean_hp_entering_bosses", "observer_error_runs",
                                  "map_room_choices", "rest_option_counts", "picked_card_counts")}}
    examples = [{"seed": row["seed"], "floor": row["floor"], "death_context": row["audit"]["death_context"],
                 "flagged_decisions": row["audit"]["flagged_decisions"]}
                for row in report["episodes"]["current"] if row["outcome"] == "death"][:3]
    incomplete = any(summary["gameplay"]["timeouts"] or summary["gameplay"]["aborted"] or summary["observer_error_runs"]
                     for summary in controllers.values())
    return {"stage": "act1_cycle_behavior_audit", "cycle": report["protocol"]["cycle"], "victory_scope": "act1",
            "status": "diagnostic_errors" if incomplete else "completed", "controllers": controllers,
            "current_loss_examples": examples, "validation_start": report["protocol"]["seed_start"],
            "episodes_per_controller": report["protocol"]["episodes"], "report_path": str(target),
            "additional_training_steps": 0, "champion_updated": False, "training_gate_unchanged": True,
            "note": "Descriptive behavior audit, not a new strength or promotion test. Hybrids change only ordinary combat or all other decisions. Paths and encounters may change as a consequence. Unused cards, energy or potions can be intentional; flags locate choices to inspect. Observer failures are disclosed separately. No hidden future state is fed to controllers. Emulator Act 1 only; native transfer unverified."}


def play_controller(env, seeds, *, model, label, controller=None):
    rows = []
    for seed in seeds:
        row = auditor["run_policy"](env, [seed], model=model, label=label, controller=controller)[0]
        row["act1_cleared"] = row["outcome"] == "win"
        row["audit"] = deepcopy(env.episode_audit)
        rows.append(row)
        if len(rows) % 25 == 0 or len(rows) == len(seeds):
            print(f"{label}: audited {len(rows)}/{len(seeds)} runs", flush=True)
    return rows


def audit(series, upstream, *, cycle=2, episodes=100, start=None):
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (cycle, episodes))
            or cycle < 1 or episodes < 1
            or start is not None and (isinstance(start, bool) or not isinstance(start, int) or start < 0)):
        raise ValueError("Use a positive cycle, fixed diagnostic sample, and fresh seed indices")
    series, upstream = Path(series).resolve(), Path(upstream).resolve()
    root = series / f"cycle-{cycle:04d}"
    if not root.is_dir():
        raise ValueError("Complete the numbered cycle before auditing it")
    torch.set_num_threads(2)
    with cycles["trial"]["selection"]["workflow_lock"](series), experiment_lock(root):
        parent = cycles["load_parent"](root, upstream)
        if parent["config"].max_acts != 1 or parent["config"].scope != "fullrun":
            raise ValueError("This diagnostic requires an Act 1 run policy")
        guarded = {**parent["source_files"], **strength_evidence(root, parent)}
        initial, final = parent["lineage"]["initial_checkpoint"], parent["state"]
        diagnostic = root / "diagnostics" / f"cycle-audit-{final['sha256'][:16]}"
        protocol_path, target = diagnostic / "protocol.json", diagnostic / "report.json"
        summary_path, completion_path = diagnostic / "summary.json", diagnostic / "completion.json"
        previous = read_json(protocol_path) if protocol_path.exists() else None
        selected = previous["seed_start"] if previous is not None and start is None else start
        seeds, selected = strength_checker["fresh_schedule"](root, parent, episodes, selected, exclude=protocol_path)
        protocol = {"cycle": cycle, "fingerprint": parent["manifest"]["fingerprint"],
                    "candidate_checkpoint": final, "baseline_checkpoint": initial,
                    "split": "validation", "episodes": episodes, "seed_start": selected,
                    "controllers": list(LABELS), "victory_scope": "act1", "inference": "deterministic",
                    "audit_sha256": sha256(Path(__file__)), "observer_sha256": sha256(observer_path),
                    "metric_runner_sha256": sha256(Path(__file__).with_name("audit_combat.py")),
                    "teacher_sha256": sha256(teacher_path), "strength_evidence": guarded.copy(),
                    "use_lookahead": False, "automatic_training": False}
        if previous is not None and previous != protocol:
            raise ValueError("This behavior audit's sample, models and schedule are fixed; reuse its original command")
        if previous is None:
            if target.exists() or completion_path.exists() or summary_path.exists():
                raise ValueError("Behavior audit protocol is missing")
            diagnostic.mkdir(parents=True, exist_ok=True)
            atomic_json(protocol_path, protocol)
        for name in ("latest.json", "episodes.jsonl", "sessions.jsonl", "champion.json"):
            path = root / name
            if path.exists():
                guarded[str(path)] = sha256(path)
        completion = read_json(completion_path) if completion_path.exists() else None
        if completion is not None and any(not path.exists() or completion[key] != sha256(path) for key, path in (
                ("protocol_sha256", protocol_path), ("report_sha256", target), ("summary_sha256", summary_path))):
            raise ValueError("Pinned behavior audit evidence changed or is missing")
        with cycles["cycle_runner"](parent["base_factory"], parent["lineage"]["plan"], parent["lineage"]["source_state"]):
            if target.exists():
                report = read_json(target)
                validate_report(report, protocol, seeds)
            else:
                identity = parent["manifest"]["fingerprint"]
                paths = {"current": cycles["trial"]["act1"]["checked_state"](root, final, identity),
                         "previous": cycles["trial"]["act1"]["checked_state"](root, initial, identity)}
                models = {key: MaskablePPO.load(path, device="cpu") for key, path in paths.items()}
                if (int(models["current"].num_timesteps) != final["num_timesteps"]
                        or int(models["previous"].num_timesteps) != initial["num_timesteps"]):
                    raise ValueError("Behavior audit model counter differs from frozen evidence")
                env = CycleAuditObserver(runner._make(parent["config"], upstream))
                all_rows = {}
                try:
                    for label in LABELS:
                        teacher = FairTeacher() if label not in {"current", "previous"} else None
                        if teacher is not None and teacher.bot.use_lookahead:
                            raise ValueError("Behavior audit teacher must use no lookahead")
                        controller = (teacher if label == "fair_teacher" else
                                      HybridController(models["current"], teacher, label) if teacher is not None else None)
                        print(f"Auditing {label}: {episodes} fixed matched runs...", flush=True)
                        all_rows[label] = play_controller(env, seeds, model=models.get(label), label=label, controller=controller)
                finally:
                    env.close()
                if any(sha256(Path(path)) != digest for path, digest in guarded.items()):
                    raise ValueError("Frozen learning or strength files changed during the behavior audit")
                report = {"protocol": protocol, "episodes": all_rows,
                          "controllers": {label: summarize_rows(all_rows[label]) for label in LABELS}, "created": time.time_ns()}
                validate_report(report, protocol, seeds)
                atomic_json(target, report)
        if any(sha256(Path(path)) != digest for path, digest in guarded.items()):
            raise ValueError("Frozen files changed during the behavior audit")
        summary = view(report, target)
        if summary_path.exists():
            if read_json(summary_path) != summary:
                raise ValueError("Behavior audit summary differs from raw evidence")
        else:
            atomic_json(summary_path, summary)
        if completion is None:
            atomic_json(completion_path, {"protocol_sha256": sha256(protocol_path), "report_sha256": sha256(target),
                                         "summary_sha256": sha256(summary_path)})
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-cycles"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--cycle", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-start", type=int)
    args = parser.parse_args(argv)
    return audit(args.experiment, args.upstream, cycle=args.cycle, episodes=args.episodes, start=args.validation_start)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"Cycle behavior audit stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
