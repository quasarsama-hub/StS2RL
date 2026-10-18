"""Replay the frozen trial's engine failures with a narrow potion-capacity mask.

The original strength report and policies stay intact. This diagnoses an
interface repair; it does not train, replace checkpoints or promote a model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import time

import torch
from sb3_contrib import MaskablePPO

from spire2_rl import runner
from spire2_rl.experiment import (ExperimentConfig, atomic_json, experiment_lock,
                                 read_json, sha256, verified_checkpoint)
from spire2_rl.potion_capacity import PotionCapacityMask

SCRIPT_DIR = Path(__file__).resolve().parent
checker = runpy.run_path(str(SCRIPT_DIR / "check_act1_trial.py"))
trial = checker["trial"]
spec = importlib.util.spec_from_file_location("potion_failure_trace", SCRIPT_DIR / "trace_timeouts.py")
trace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trace)
_describe = trace.describe_decision


def describe_decision(env, action):
    """Add the inventory visible before the action, before a failed pick mutates it."""
    result = _describe(env, action)
    raw, mgr = env.raw_env, env.raw_env._mgr
    if mgr.phase == mgr.PHASE_COMBAT:
        return result
    slots = raw.public_slots()
    potion_slots = [i for i, item in enumerate(slots) if item["action"] == "pick_potion"]
    if not potion_slots:
        return result
    player = mgr.run_state.player
    capacity, potions = int(player.max_potion_slots), player.potions
    free = [i for i in range(capacity) if i >= len(potions) or potions[i] is None]
    mask = env.action_masks()
    result["potion_inventory"] = {
        "max_slots": capacity,
        "slot_potion_ids": [None if item is None else item.potion_id for item in potions],
        "free_slots": free,
        "pick_action_slots": potion_slots,
        "pick_action_mask": [bool(mask[i]) for i in potion_slots],
    }
    return result


def replay(env, model, seed, output):
    original = trace.describe_decision
    trace.describe_decision = describe_decision
    try:
        result = trace.replay(env, model, seed, output)
    finally:
        trace.describe_decision = original
    actions = [json.loads(line) for line in Path(result["actions_trace"]).read_text(encoding="utf-8").splitlines()]
    result["capacity_mask_decisions"] = sum(
        "potion_inventory" in row and not any(row["potion_inventory"]["pick_action_mask"])
        for row in actions
    )
    return result


def verified_full_capacity_failure(original, expected):
    """Require an exact replay plus evidence of the specific broken contract."""
    if any(original.get(key) != expected.get(key) for key in ("seed", "outcome", "steps", "floor", "hp", "error")):
        return False
    if (original.get("outcome") != "aborted" or original.get("error_stage") != "game_step"
            or "Engine rejected advertised legal pick_potion" not in str(original.get("error"))):
        return False
    tail = original.get("last_decisions", [])
    if not tail or tail[-1].get("named_action", {}).get("action") != "pick_potion":
        return False
    inventory = tail[-1].get("potion_inventory", {})
    capacity = inventory.get("max_slots")
    slots, masks = inventory.get("pick_action_slots"), inventory.get("pick_action_mask")
    held = inventory.get("slot_potion_ids")
    return (isinstance(capacity, int) and not isinstance(capacity, bool) and capacity >= 0
            and isinstance(held, list) and len(held) >= capacity
            and all(item is not None for item in held[:capacity])
            and inventory.get("free_slots") == []
            and isinstance(slots, list) and bool(slots)
            and isinstance(masks, list) and len(masks) == len(slots)
            and all(item is True for item in masks))


def brief(row):
    return {key: row[key] for key in ("seed", "outcome", "steps", "floor", "hp", "error",
                                     "error_stage", "last_decisions", "capacity_mask_decisions")}


@contextmanager
def repair_runner():
    """Separate factory for diagnosis only; never initialize the old experiment with it."""
    original = runner._make
    runner._make = lambda config, upstream: PotionCapacityMask(original(config, upstream))
    try:
        yield
    finally:
        runner._make = original


def check(root, upstream, report_path=None):
    root, upstream = Path(root).resolve(), Path(upstream).resolve()
    torch.set_num_threads(2)
    with trial["selection"]["workflow_lock"](root):
        with experiment_lock(root):
            lineage = read_json(root / "trial-lineage.json")
            config = ExperimentConfig(**lineage["config"])
            with trial["trial_runner"](lineage["recipe"], lineage["plan"]):
                manifest = runner.initialize(root, config, upstream)
                if report_path is None:
                    _, latest = verified_checkpoint(root, manifest["fingerprint"])
                    report_path = root / "diagnostics" / f"strength-{latest['sha256'][:16]}" / "report.json"
                report_path = Path(report_path).resolve()
                report_hash = sha256(report_path)
                report = read_json(report_path)
                protocol = report["protocol"]
                if (protocol["fingerprint"] != manifest["fingerprint"]
                        or protocol["checker_sha256"] != sha256(SCRIPT_DIR / "check_act1_trial.py")
                        or protocol["baseline_checkpoint"] != lineage["initial_checkpoint"]
                        or protocol["split"] != "validation" or protocol["inference"] != "deterministic"
                        or protocol["victory_scope"] != "act1"):
                    raise ValueError("Strength report differs from the saved trial models or protocol")
                seeds = checker["fresh_seeds"](root, lineage, config, protocol["episodes"], protocol["seed_start"])
                checker["validate_report"](report, protocol, seeds)
                states = {"candidate": protocol["candidate_checkpoint"], "baseline": protocol["baseline_checkpoint"]}
                paths = {key: trial["act1"]["checked_state"](root, state, manifest["fingerprint"])
                         for key, state in states.items()}
                if any(state.get("interrupted") for state in states.values()):
                    raise ValueError("Replay requires completed frozen policies")
                failed = [(key, row) for key in states for row in report[f"{key}_episodes"]
                          if row["outcome"] in {"aborted", "timeout"}]
                output = root / "diagnostics" / f"potion-{time.time_ns()}"
                output.mkdir(parents=True, exist_ok=False)
                protocol_out = {"source_report": str(report_path), "source_report_sha256": report_hash,
                                "source_protocol": protocol, "replay_script_sha256": sha256(Path(__file__)),
                                "repair_adapter_sha256": sha256(Path(sys.modules[PotionCapacityMask.__module__].__file__)),
                                "additional_training_steps": 0}
                atomic_json(output / "protocol.json", protocol_out)
                models = {key: MaskablePPO.load(paths[key], device="cpu") for key in {key for key, _ in failed}}
                if any(int(model.num_timesteps) != states[key]["num_timesteps"] for key, model in models.items()):
                    raise ValueError("Frozen model step count differs from its checkpoint")
                runs = []
                for key, expected in failed:
                    item = {"policy": key, "seed": expected["seed"], "original_evaluation": expected}
                    for mode in ("original", "capacity_mask"):
                        folder = output / f"{key}-{expected['seed']}" / mode
                        folder.mkdir(parents=True, exist_ok=False)
                        if mode == "capacity_mask":
                            with repair_runner():
                                env = runner._make(config, upstream)
                        else:
                            env = runner._make(config, upstream)
                        try:
                            result = replay(env, models[key], expected["seed"], folder)
                        finally:
                            env.close()
                        item[mode] = result
                        print(f"{key}, seed {expected['seed']}, {mode}: {result['outcome']}, {result['steps']} decisions", flush=True)
                    item["confirmed_full_capacity_failure"] = verified_full_capacity_failure(item["original"], expected)
                    item["repair_completed_run"] = (item["capacity_mask"]["outcome"] in {"win", "death"}
                                                     and item["capacity_mask"]["error"] is None
                                                     and item["capacity_mask"]["capacity_mask_decisions"] > 0)
                    runs.append(item)
                if sha256(report_path) != report_hash or any(sha256(paths[key]) != state["sha256"] for key, state in states.items()):
                    raise ValueError("Original report or model checksum changed during replay")
                summary = {"stage": "potion_capacity_repair_check", "victory_scope": "act1",
                           "failed_evaluation_runs": len(failed), "runs": [
                               {**{key: item[key] for key in ("policy", "seed", "confirmed_full_capacity_failure", "repair_completed_run")},
                                "original": brief(item["original"]), "capacity_mask": brief(item["capacity_mask"])} for item in runs],
                           "repair_verified": bool(runs) and all(item["confirmed_full_capacity_failure"] and item["repair_completed_run"] for item in runs),
                           "additional_training_steps": 0, "champion_updated": False,
                           "source_report_sha256": report_hash, "summary_path": str(output / "summary.json"),
                           "note": "Same frozen weights on known failed seeds. A repaired run verifies this interface case, not additional learning or a new held-out strength estimate. The original strength report is preserved. A training continuation needs a new experiment identity."}
                atomic_json(output / "runs.json", {"protocol": protocol_out, "runs": runs})
                atomic_json(output / "summary.json", summary)
                print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
                return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-act1-rl-trial"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--report", type=Path, help="Default: latest checkpoint's frozen strength report")
    args = parser.parse_args(argv)
    return check(args.experiment, args.upstream, args.report)


if __name__ == "__main__":
    try:
        result = main()
        raise SystemExit(0 if result["repair_verified"] else 2)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
