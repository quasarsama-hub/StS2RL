"""Replay evaluation timeouts with the exact evaluated policy, without training.

Run with the project's virtual-environment Python. This separate diagnostic
does not change the adapter, experiment fingerprint, checkpoints or seed cursors.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from spire2_rl.experiment import (
    ExperimentConfig, atomic_json, experiment_lock, read_json, sha256,
)
from spire2_rl.runner import _make, initialize


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def describe_decision(env, action: int) -> dict:
    """Describe only the current public decision, including choice toggles."""
    raw = env.raw_env
    mgr = raw._mgr
    choice = mgr.run_state.pending_choice
    if mgr.phase == mgr.PHASE_COMBAT:
        combat = mgr.get_combat_state()
        choice = combat.pending_choice
        module = importlib.import_module("sts2_env.gym_env.run_env")
        layout = module._LAYOUT
        if layout.player_select_start <= action < layout.player_select_start + layout.player_select_size:
            players = [item for item in mgr.get_available_actions() if item["action"] == "select_player"]
            named = dict(players[action - layout.player_select_start])
        elif choice is not None:
            named = {"action": "confirm_choice"} if action == 0 else {
                "action": "choose", "index": action - 1,
                "card_id": choice.options[action - 1].card.card_id.name,
                "selected_before": action - 1 in choice.selected_indices,
            }
        elif module.is_potion_action(action):
            slot, target = module.action_to_potion_and_target(action)
            named = {"action": "use_potion", "slot_index": slot, "target_index": target}
        else:
            hand, target = module.action_to_card_and_target(action)
            named = {"action": "end_turn"} if hand is None else {
                "action": "play_card", "hand_index": hand, "target_index": target,
            }
            if hand is not None and hand < len(combat.hand):
                named["card_id"] = combat.hand[hand].card_id.name
    else:
        named = dict(raw.public_slots()[action])
        if choice is None and mgr._event_model is not None:
            choice = mgr._event_model.pending_choice
    result = {"named_action": named}
    if choice is not None:
        result["pending_choice"] = {
            "prompt": choice.prompt, "min_choices": choice.min_choices,
            "max_choices": choice.max_choices, "allow_skip": choice.allow_skip,
            "selected_indices": sorted(choice.selected_indices),
            "can_confirm": choice.can_confirm(),
        }
        if named["action"] == "choose":
            named["selected_before"] = named["index"] in choice.selected_indices
    return serializable(result)


def replay(env, model, seed: int, output: Path) -> dict:
    trace_path = output / f"seed-{seed}-actions.jsonl"
    states_path = output / f"seed-{seed}-observations.jsonl"
    seen = Counter()
    tail = deque(maxlen=6)
    info, length, outcome, error, error_stage = {}, 0, "aborted", None, None
    stage = "reset"
    with trace_path.open("w", encoding="utf-8") as trace, states_path.open("w", encoding="utf-8") as states:
        try:
            obs, info = env.reset(seed=seed)
            while True:
                stage = "policy_inference"
                masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                action = int(action)
                stage = "diagnostic_logging"
                public = serializable({"observation": obs, "legal_actions": np.flatnonzero(masks)})
                encoded = json.dumps(public, sort_keys=True, separators=(",", ":"), allow_nan=False)
                digest = hashlib.sha256(encoded.encode()).hexdigest()
                if not seen[digest]:
                    states.write(json.dumps({"observation_sha256": digest, **public}, allow_nan=False) + "\n")
                seen[digest] += 1
                row = {
                    "seed": seed, "step": length, "phase": info.get("phase"),
                    "floor": info.get("floor"), "hp": info.get("hp"),
                    "action_slot": action, "observation_sha256": digest,
                    "observation_visits": seen[digest], **describe_decision(env, action),
                }
                trace.write(json.dumps(row, allow_nan=False) + "\n")
                tail.append({key: value for key, value in row.items()
                             if key not in {"seed", "observation_sha256", "observation_visits"}})
                stage = "game_step"
                obs, _, terminated, truncated, info = env.step(action)
                length += 1
                if terminated or truncated:
                    outcome = info["outcome"]
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_stage = stage
    return {
        "seed": seed, "outcome": outcome, "steps": length,
        "floor": info.get("floor"), "hp": info.get("hp"),
        "final_phase": info.get("phase"), "error": error, "error_stage": error_stage,
        "unique_public_observations": len(seen),
        "most_repeated_observation_visits": max(seen.values(), default=0),
        "last_decisions": list(tail), "actions_trace": str(trace_path),
        "observations_trace": str(states_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--report", type=Path, help="Defaults to experiment/last-evaluation.json")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override timeout seeds for diagnostic testing")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    root, upstream = args.experiment.resolve(), args.upstream.resolve()
    with experiment_lock(root):
        manifest = read_json(root / "experiment.json")
        config = ExperimentConfig(**manifest["config"])
        if config.scope != "fullrun":
            raise ValueError("This diagnostic currently supports fullrun experiments only")
        initialize(root, config, upstream)
        report_path = (args.report or root / "last-evaluation.json").resolve()
        report = read_json(report_path)
        state = report["checkpoint"]
        if report["fingerprint"] != manifest["fingerprint"] or state["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Evaluation belongs to a different experiment fingerprint")
        checkpoint = (root / state["model"]).resolve()
        if not checkpoint.is_relative_to(root) or sha256(checkpoint) != state["sha256"]:
            raise ValueError("Evaluated checkpoint path or checksum mismatch")
        seeds = args.seeds if args.seeds is not None else [
            int(row["seed"]) for row in report["candidate_episodes"] if row["outcome"] == "timeout"
        ]
        seeds = list(dict.fromkeys(seeds))
        if not seeds:
            print(json.dumps({"status": "no_timeout_seeds", "report": str(report_path)}, indent=2))
            return 0
        output = (args.out or root / "diagnostics" / f"timeouts-{time.time_ns()}").resolve()
        # New output directory only: never overwrite existing diagnostic evidence.
        output.mkdir(parents=True, exist_ok=False)
        model = MaskablePPO.load(checkpoint, device="cpu")
        env = _make(config, upstream)
        results = []
        expected = {int(row["seed"]): row for row in report["candidate_episodes"]}
        try:
            for seed in seeds:
                result = replay(env, model, seed, output)
                prior = expected.get(seed)
                if prior is not None:
                    result["original_result"] = {key: prior[key] for key in ("outcome", "steps", "floor", "hp")}
                    result["matches_original_result"] = all(result[key] == prior[key] for key in ("outcome", "steps", "floor", "hp"))
                results.append(result)
                print(f"Replayed seed {seed}: {result['outcome']}, {result['steps']} decisions", file=sys.stderr, flush=True)
        finally:
            env.close()
        summary = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": state["sha256"],
            "evaluation_report": str(report_path), "runs": results,
            "note": "Repeated public observations help locate stalls; they alone do not prove an engine cycle. No training performed.",
            "summary_path": str(output / "summary.json"),
        }
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 1 if any(row["outcome"] == "aborted" for row in results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
