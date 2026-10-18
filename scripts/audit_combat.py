"""Read-only combat audit of two checkpoints and two public-action baselines.

Uses the existing finite-selection launcher without changing its identity.
Writes new diagnostic reports only; never trains or updates checkpoint pointers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from spire2_rl.experiment import (
    ExperimentConfig, atomic_json, experiment_lock, read_json, sha256, verified_checkpoint,
)
from spire2_rl.study import evaluation_seeds, paired_compare, summarize


def checkpoint_at_step(root: Path, manifest: dict, step: int):
    matching = []
    for path in sorted((root / "checkpoints").glob("step-*.json")):
        state = read_json(path)
        if state.get("num_timesteps") != step or state.get("fingerprint") != manifest["fingerprint"] or state.get("interrupted"):
            continue
        model = (root / state["model"]).resolve()
        if not model.is_relative_to(root.resolve()) or sha256(model) != state["sha256"]:
            raise ValueError(f"Invalid checkpoint artifact: {path}")
        matching.append((model, state))
    if not matching:
        raise ValueError(f"No completed checksum-verified checkpoint at step {step}")
    return matching[-1]


def card_actions(legal, module):
    # End-turn is 0. Exclude potions and the separate player-selection region.
    return [int(action) for action in legal
            if 0 < action < module._LAYOUT.combat_size and not module.is_potion_action(int(action))]


def choice_for(env, combat):
    mgr = env.raw_env._mgr
    if combat is not None:
        return combat.pending_choice
    choice = mgr.run_state.pending_choice
    if choice is None and mgr.phase == mgr.PHASE_EVENT and mgr._event_model is not None:
        choice = mgr._event_model.pending_choice
    return choice


def run_policy(env, seeds, *, model, label, controller=None):
    module = importlib.import_module("sts2_env.gym_env.run_env")
    rows = []
    for seed in seeds:
        metrics = {key: 0 for key in (
            "normal_combat_decisions", "card_plays", "end_turns",
            "end_turns_with_playable_cards", "end_turns_with_energy", "unused_energy_sum",
            "combat_wins", "combat_losses", "hp_lost_on_end_turn",
            "hp_lost_other_combat_actions", "potion_actions", "multi_choice_confirms",
            "empty_multi_choice_confirms",
        )}
        metrics.update(potions_left_at_death=None, legal_potion_slots_before_death=None)
        phases = Counter()
        info, length, reward_total = {}, 0, 0.0
        max_act = 0
        outcome, error, error_stage = "aborted", None, None
        stage = "reset"
        try:
            obs, info = env.reset(seed=seed)
            max_act = int(info.get("act", 0))
            rng = np.random.default_rng(seed)
            while True:
                stage = "public_diagnostic_state"
                mgr = env.raw_env._mgr
                combat = mgr.get_combat_state() if mgr.phase == mgr.PHASE_COMBAT else None
                choice = choice_for(env, combat)
                regular = combat is not None and choice is None
                masks = env.action_masks()
                legal = np.flatnonzero(masks)
                cards = card_actions(legal, module) if regular else []
                before_hp = float(combat.player.current_hp) if combat is not None else None
                before_energy = int(combat.energy) if regular else None
                was_over = bool(combat.is_over) if combat is not None else False
                legal_potion_slots = {module.action_to_potion_and_target(int(action))[0]
                                      for action in legal if regular and module.is_potion_action(int(action))}
                stage = "policy_inference"
                if controller is not None:
                    action = int(controller.select(env))
                elif model is None:
                    # Public action legality and phase only; no hidden RNG/deck
                    # information, card scoring, lookahead or automatic selections.
                    action = int(rng.choice(cards if cards else legal))
                else:
                    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                    action = int(action)
                stage = "diagnostic_logging"
                phases[str(info.get("phase"))] += 1
                if regular:
                    metrics["normal_combat_decisions"] += 1
                    if action in cards:
                        metrics["card_plays"] += 1
                    elif module.is_potion_action(action):
                        metrics["potion_actions"] += 1
                    elif action == 0:
                        metrics["end_turns"] += 1
                        metrics["end_turns_with_playable_cards"] += int(bool(cards))
                        metrics["end_turns_with_energy"] += int(before_energy > 0)
                        metrics["unused_energy_sum"] += before_energy
                if choice is not None and choice.is_multi:
                    confirming = action == 0 if combat is not None else env.raw_env.public_slots()[action]["action"] == "confirm_choice"
                    if confirming:
                        metrics["multi_choice_confirms"] += 1
                        metrics["empty_multi_choice_confirms"] += int(not choice.selected_indices)
                stage = "game_step"
                obs, reward, terminated, truncated, info = env.step(action)
                max_act = max(max_act, int(info.get("act", 0)))
                length += 1
                reward_total += float(reward)
                stage = "post_step_diagnostic_state"
                if combat is not None:
                    loss = max(0.0, before_hp - float(combat.player.current_hp))
                    metrics["hp_lost_on_end_turn" if regular and action == 0 else "hp_lost_other_combat_actions"] += loss
                    if not was_over and combat.is_over:
                        metrics["combat_wins" if combat.player_won else "combat_losses"] += 1
                if terminated or truncated:
                    outcome = info["outcome"]
                    if outcome == "death":
                        # Upstream's death branch does not sync combat potions
                        # back to run state. Retain this combat reference instead.
                        inventory = combat.potions if combat is not None else mgr.run_state.player.potions
                        metrics["potions_left_at_death"] = sum(potion is not None for potion in inventory)
                        metrics["legal_potion_slots_before_death"] = len(legal_potion_slots) if regular else None
                    break
        except Exception as exc:
            outcome = "aborted"
            error, error_stage = f"{type(exc).__name__}: {exc}", stage
        rows.append({
            "seed": seed, "policy": label, "outcome": outcome,
            "floor": int(info.get("floor", 0)), "hp": float(info.get("hp", 0)),
            "steps": length, "return": reward_total, "error": error, "error_stage": error_stage,
            "metrics": metrics, "phase_decisions": dict(phases),
            "max_act_index": max_act, "act1_cleared": max_act >= 1,
        })
    return rows


def describe(rows):
    gameplay = [row for row in rows if row["outcome"] in {"win", "death"}]
    totals = {key: sum(row["metrics"][key] for row in gameplay)
              for key in gameplay[0]["metrics"] if not key.endswith("at_death") and key != "legal_potion_slots_before_death"} if gameplay else {}
    turns = totals.get("end_turns", 0)
    deaths = [row for row in gameplay if row["outcome"] == "death"]
    return {
        "gameplay": summarize(rows), "metrics_episodes": len(gameplay), "totals": totals,
        "act1_clears": sum(bool(row.get("act1_cleared", False)) for row in gameplay),
        "early_end_turn_fraction": totals.get("end_turns_with_playable_cards", 0) / turns if turns else None,
        "mean_unused_energy_per_end_turn": totals.get("unused_energy_sum", 0) / turns if turns else None,
        "mean_combat_wins_per_completed_run": totals.get("combat_wins", 0) / len(gameplay) if gameplay else None,
        "deaths_with_potions_left": sum(row["metrics"]["potions_left_at_death"] > 0 for row in deaths),
        "deaths_with_legal_potion_before_last_action": sum((row["metrics"]["legal_potion_slots_before_death"] or 0) > 0 for row in deaths),
        "floor_distribution": dict(sorted(Counter(row["floor"] for row in gameplay).items())),
        "note": "Available cards, energy or potions can sometimes be intentionally left unused. These flags locate behavior to inspect; they do not label every flagged action a mistake. HP-loss counters measure net HP decreases after resolution and healing, not raw incoming damage.",
    }


def training_record(root: Path):
    path = root / "episodes.jsonl"
    if not path.exists():
        return {"status": "journal_missing"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    discarded = [row for row in rows if row.get("outcome") == "aborted"
                 and str(row.get("reason") or "").startswith("training budget ended;")]
    assessed = [row for row in rows if row not in discarded]
    return {
        "status": "read", "recorded_episodes": summarize(assessed),
        "discarded_session_boundary_episodes": len(discarded),
        "note": "Training episodes are sampled by changing policies; this is reward-signal accounting, not a held-out strength estimate.",
    }


def audit(root: Path, upstream: Path, *, previous_step: int, episodes: int, start: int):
    if previous_step < 0 or episodes < 1 or start < 0:
        raise ValueError("Invalid checkpoint step, episode count or seed index")
    support = runpy.run_path(str(Path(__file__).with_name("selection_rl.py")))
    with support["workflow_lock"](root), support["finite_selection_runner"](), experiment_lock(root):
        runner = support["runner"]
        manifest = read_json(root / "experiment.json")
        config = ExperimentConfig(**manifest["config"])
        runner.initialize(root, config, upstream)
        current_path, current_state = verified_checkpoint(root, manifest["fingerprint"])
        previous_path, previous_state = checkpoint_at_step(root, manifest, previous_step)
        if current_state.get("interrupted") or current_state["num_timesteps"] <= previous_step:
            raise ValueError("Current checkpoint must be completed and newer than the requested previous checkpoint")
        seeds = evaluation_seeds("validation", episodes, config.experiment_seed, start=start)
        output = root / "diagnostics" / f"combat-audit-{time.time_ns()}"
        output.mkdir(parents=True, exist_ok=False)
        policies = (
            ("current", current_path, current_state),
            ("previous", previous_path, previous_state),
            ("play_available_cards", None, None),
            ("fair_teacher", None, None),
        )
        all_rows, summaries = {}, {}
        env = runner._make(config, upstream)
        teacher_path = Path(__file__).with_name("fair_teacher.py")
        teacher_class = runpy.run_path(str(teacher_path))["FairTeacher"]
        try:
            for label, path, state in policies:
                print(f"Auditing {label}: {episodes} matched validation runs...", file=sys.stderr, flush=True)
                model = MaskablePPO.load(path, device="cpu") if path is not None else None
                if model is not None and int(model.num_timesteps) != state["num_timesteps"]:
                    raise ValueError(f"Embedded model timestep count differs from checkpoint state: {path}")
                controller = teacher_class() if label == "fair_teacher" else None
                rows = run_policy(env, seeds, model=model, label=label, controller=controller)
                all_rows[label], summaries[label] = rows, describe(rows)
                atomic_json(output / f"{label}-episodes.json", {"episodes": rows})
        finally:
            env.close()
        for path, state in ((current_path, current_state), (previous_path, previous_state)):
            if sha256(path) != state["sha256"]:
                raise ValueError("Checkpoint changed during audit")
        report = {
            "stage": "combat_audit", "fingerprint": manifest["fingerprint"],
            "audit_source_sha256": sha256(Path(__file__)), "seed_start": start, "episodes": episodes,
            "teacher_source_sha256": sha256(teacher_path),
            "current_checkpoint": current_state, "previous_checkpoint": previous_state,
            "training_journal": training_record(root),
            "controllers": summaries,
            "incremental_learning_comparison": paired_compare(all_rows["current"], all_rows["previous"]),
            "simple_controller_comparison": paired_compare(all_rows["current"], all_rows["play_available_cards"]),
            "fair_teacher_comparison": paired_compare(all_rows["current"], all_rows["fair_teacher"]),
            "summary_path": str(output / "summary.json"),
            "note": "Fresh matched validation seeds. No training performed; simulator mechanics and native transfer remain unverified.",
        }
        atomic_json(output / "summary.json", report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 1 if any(row["outcome"] == "aborted" for rows in all_rows.values() for row in rows) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-add-only"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--previous-step", type=int, default=80896)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start", type=int, default=200)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    return audit(args.experiment.resolve(), args.upstream.resolve(), previous_step=args.previous_step,
                 episodes=args.episodes, start=args.start)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
