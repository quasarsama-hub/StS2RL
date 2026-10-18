"""Verify private event state, then replay the same frozen policy on both backends.

No learning, checkpoint replacement, budget extension or promotion occurs.
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
from spire2_rl.emulator import make_env
from spire2_rl.event_isolation import IsolatedEvents
from spire2_rl.experiment import atomic_json, experiment_lock, read_json, sha256
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.selection import MonotoneSelections
from spire2_rl.study import summarize

behavior = runpy.run_path(str(Path(__file__).with_name("audit_act1_cycle.py")))
strength_checker, cycles = behavior["strength_checker"], behavior["cycles"]
FairTeacher = behavior["FairTeacher"]


def mechanics_probe(upstream):
    def make(isolated):
        env = PotionCapacityMask(MonotoneSelections(make_env(scope="fullrun", max_acts=1, max_steps=4000, upstream_path=upstream)))
        return IsolatedEvents(env) if isolated else env
    def positive(env):
        return next(i for i, slot in enumerate(env.raw_env.public_slots())
                    if slot.get("action") == "event_choice" and slot.get("option_id") == "positive_1")
    def visible_rewards(env):
        return json.loads(json.dumps(env.raw_env.public_slots(), sort_keys=True))
    def neow(isolated):
        solo = make(isolated)
        try:
            solo.reset(seed=42)
            solo.step(positive(solo))
            expected = visible_rewards(solo)
        finally:
            solo.close()
        first, second = make(isolated), make(isolated)
        try:
            first.reset(seed=42)
            shown = [option.label for option in first.raw_env._mgr._event_options]
            action = positive(first)
            second.reset(seed=43)
            shared = first.raw_env._mgr._event_model is second.raw_env._mgr._event_model
            first.step(action)
            actual = visible_rewards(first)
            return {"shared_event_model": shared, "seed42_offered_labels": shown,
                    "solo_expected_reward": expected, "interleaved_actual_reward": actual,
                    "matches_solo": expected == actual}
        finally:
            first.close()
            second.close()
    def tablet(isolated):
        first, second = make(isolated), make(isolated)
        teacher = FairTeacher()
        try:
            first.reset(seed=44)
            second.reset(seed=44)
            for step in range(4000):
                ma, mb = first.raw_env._mgr, second.raw_env._mgr
                ea, eb = ma._event_model, mb._event_model
                before = [ma.run_state.player.max_hp, mb.run_state.player.max_hp]
                action = teacher.select(first)
                result_a, result_b = first.step(action), second.step(action)
                after = [ma.run_state.player.max_hp, mb.run_state.player.max_hp]
                if (ea is not None and ea.event_id == "TabletOfTruth" and before[0] == before[1]
                        and after[0] < before[0] and after[1] < before[1]):
                    return {"decision": step + 1, "shared_event_model": ea is eb,
                            "max_hp_before": before, "max_hp_after": after, "matches": after[0] == after[1]}
                if any(result[2] or result[3] for result in (result_a, result_b)):
                    break
            return {"target_not_reached": True, "matches": False}
        finally:
            first.close()
            second.close()
    bootstrap = make(False)
    bootstrap.close()
    import sts2_env.run.events as events
    original_registry = events._EVENT_REGISTRY
    report = {"neow_original": neow(False), "neow_isolated": neow(True),
              "tablet_original": tablet(False), "tablet_isolated": tablet(True)}
    report["global_registry_restored"] = events._EVENT_REGISTRY is original_registry
    report["passed"] = (report["neow_original"]["shared_event_model"] and not report["neow_original"]["matches_solo"]
                        and not report["neow_isolated"]["shared_event_model"] and report["neow_isolated"]["matches_solo"]
                        and report["tablet_original"].get("shared_event_model") and not report["tablet_original"]["matches"]
                        and not report["tablet_isolated"].get("shared_event_model") and report["tablet_isolated"]["matches"]
                        and report["global_registry_restored"])
    return report


def mechanics_passed(proof):
    for key in ("neow_original", "neow_isolated"):
        row = proof[key]
        if row["matches_solo"] != (row["solo_expected_reward"] == row["interleaved_actual_reward"]):
            raise ValueError("Event repair Neow proof differs from its recorded rewards")
    for key in ("tablet_original", "tablet_isolated"):
        row = proof[key]
        if not row.get("target_not_reached") and row["matches"] != (row["max_hp_after"][0] == row["max_hp_after"][1]):
            raise ValueError("Event repair Tablet proof differs from its recorded HP")
    return bool(proof["neow_original"]["shared_event_model"] and not proof["neow_original"]["matches_solo"]
                and not proof["neow_isolated"]["shared_event_model"] and proof["neow_isolated"]["matches_solo"]
                and proof["tablet_original"].get("shared_event_model") and not proof["tablet_original"]["matches"]
                and not proof["tablet_isolated"].get("shared_event_model") and proof["tablet_isolated"]["matches"]
                and proof["global_registry_restored"])


def validate_report(report, protocol, seeds):
    if report["protocol"] != protocol:
        raise ValueError("Event repair check differs from its fixed source and schedule")
    if report["mechanics"]["passed"] != mechanics_passed(report["mechanics"]):
        raise ValueError("Event repair success flag differs from its recorded mechanics")
    for key in ("original", "isolated"):
        rows = report[f"{key}_episodes"]
        if ([row["seed"] for row in rows] != seeds or any(row["policy"] != f"same-policy-{key}" for row in rows)
                or report[f"{key}_gameplay"] != summarize(rows)):
            raise ValueError("Event repair rows or statistics differ from their frozen evidence")


def view(report, target):
    clean = not report["isolated_gameplay"]["outcomes"]["timeout"] and not report["isolated_gameplay"]["outcomes"]["aborted"]
    return {"stage": "cycle_event_isolation_check", "cycle": report["protocol"]["cycle"],
            "status": "verified" if report["mechanics"]["passed"] and clean else "inspection_required",
            "same_frozen_checkpoint": report["protocol"]["checkpoint"], "mechanics": report["mechanics"],
            "original": cycles["trial"]["brief_gameplay"](report["original_gameplay"]),
            "isolated": cycles["trial"]["brief_gameplay"](report["isolated_gameplay"]),
            "matching_episode_outcomes": sum(a["outcome"] == b["outcome"] for a, b in zip(report["original_episodes"], report["isolated_episodes"])),
            "matching_episode_floors": sum(a["floor"] == b["floor"] for a, b in zip(report["original_episodes"], report["isolated_episodes"])),
            "validation_start": report["protocol"]["seed_start"], "report_path": str(target),
            "additional_training_steps": 0, "champion_updated": False, "training_gate_unchanged": True,
            "note": "Event-instance isolation regression and same-weights replay, not additional learning or improved strength evidence. The original four-env trainer shares mutable event objects; this repair supplies fresh event objects per env/run. Other simulator globals and native transfer remain unverified. Original learning files and continuation gate are unchanged."}


def check(series, upstream, *, cycle=2, episodes=100, start=None):
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (cycle, episodes))
            or cycle < 1 or episodes < 1
            or start is not None and (isinstance(start, bool) or not isinstance(start, int) or start < 0)):
        raise ValueError("Use a positive cycle and fixed fresh replay sample")
    series, upstream = Path(series).resolve(), Path(upstream).resolve()
    root = series / f"cycle-{cycle:04d}"
    if not root.is_dir():
        raise ValueError("Complete the numbered cycle before checking its event repair")
    torch.set_num_threads(2)
    with cycles["trial"]["selection"]["workflow_lock"](series), experiment_lock(root):
        parent = cycles["load_parent"](root, upstream)
        guarded = {**parent["source_files"], **behavior["strength_evidence"](root, parent)}
        state = parent["state"]
        diagnostic = root / "diagnostics" / f"event-isolation-{state['sha256'][:16]}"
        protocol_path, target, completion_path = (diagnostic / name for name in ("protocol.json", "report.json", "completion.json"))
        previous = read_json(protocol_path) if protocol_path.exists() else None
        selected = previous["seed_start"] if previous is not None and start is None else start
        seeds, selected = strength_checker["fresh_schedule"](root, parent, episodes, selected, exclude=protocol_path)
        with cycles["cycle_runner"](parent["base_factory"], parent["lineage"]["plan"], parent["lineage"]["source_state"]):
            identity_env = IsolatedEvents(runner._make(parent["config"], upstream))
            try:
                isolated_identity = identity_env.backend_metadata
            finally:
                identity_env.close()
        protocol = {"cycle": cycle, "fingerprint": parent["manifest"]["fingerprint"], "checkpoint": state,
                    "split": "validation", "episodes": episodes, "seed_start": selected,
                    "checker_sha256": sha256(Path(__file__)), "repair_sha256": sha256(Path(__file__).parents[1] / "src/spire2_rl/event_isolation.py"),
                    "source_strength_evidence": guarded.copy(), "inference": "deterministic", "victory_scope": "act1",
                    "scope": "same-weights event-instance isolation check", "isolated_backend": isolated_identity,
                    "automatic_training": False}
        if previous is not None and previous != protocol:
            raise ValueError("This event repair check's model, sample and schedule are fixed; reuse its original command")
        if previous is None:
            if target.exists() or completion_path.exists():
                raise ValueError("Event repair protocol is missing")
            diagnostic.mkdir(parents=True, exist_ok=True)
            atomic_json(protocol_path, protocol)
        for name in ("latest.json", "episodes.jsonl", "sessions.jsonl", "champion.json"):
            path = root / name
            if path.exists():
                guarded[str(path)] = sha256(path)
        completion = read_json(completion_path) if completion_path.exists() else None
        if completion is not None and (completion["protocol_sha256"] != sha256(protocol_path)
                or not target.exists() or completion["report_sha256"] != sha256(target)):
            raise ValueError("Pinned event repair evidence changed or is missing")
        with cycles["cycle_runner"](parent["base_factory"], parent["lineage"]["plan"], parent["lineage"]["source_state"]):
            if target.exists():
                report = read_json(target)
                validate_report(report, protocol, seeds)
            else:
                model_path = cycles["trial"]["act1"]["checked_state"](root, state, parent["manifest"]["fingerprint"])
                model = MaskablePPO.load(model_path, device="cpu")
                if int(model.num_timesteps) != state["num_timesteps"]:
                    raise ValueError("Event replay model counter differs from frozen evidence")
                print("Checking independent Neow and Tablet event state with interleaved games...", flush=True)
                proof = mechanics_probe(upstream)
                all_rows = {}
                for label in ("original", "isolated"):
                    env = runner._make(parent["config"], upstream)
                    if label == "isolated":
                        env = IsolatedEvents(env)
                    try:
                        if model.observation_space != env.observation_space or model.action_space != env.action_space:
                            raise ValueError("Event repair changed the policy's observation/action spaces")
                        rows = []
                        for offset in range(0, episodes, 25):
                            rows.extend(runner._rollout(env, seeds[offset:offset + 25], model=model, label=f"same-policy-{label}"))
                            print(f"{label}: replayed {len(rows)}/{episodes} frozen-policy runs", flush=True)
                        all_rows[label] = rows
                    finally:
                        env.close()
                if any(sha256(Path(path)) != digest for path, digest in guarded.items()):
                    raise ValueError("Frozen source or learning files changed during the event check")
                report = {"protocol": protocol, "mechanics": proof,
                          "original_episodes": all_rows["original"], "isolated_episodes": all_rows["isolated"],
                          "original_gameplay": summarize(all_rows["original"]), "isolated_gameplay": summarize(all_rows["isolated"]),
                          "created": time.time_ns()}
                validate_report(report, protocol, seeds)
                atomic_json(target, report)
        if any(sha256(Path(path)) != digest for path, digest in guarded.items()):
            raise ValueError("Frozen learning files changed during the event repair check")
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
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-start", type=int)
    args = parser.parse_args(argv)
    return check(args.experiment, args.upstream, cycle=args.cycle, episodes=args.episodes, start=args.validation_start)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"Event isolation check stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
