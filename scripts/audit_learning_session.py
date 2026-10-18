"""One bounded, read-only behavior audit of a completed learning session.

Five frozen controllers replay the same prespecified source evaluation seeds. This diagnostic locates
combat, deck-building and routing weaknesses; it never trains or promotes a
model. Cached per-controller episodes make interruptions resumable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import zipfile

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from spire2_rl import runner
from spire2_rl.experiment import atomic_json, experiment_lock, read_json, sha256
from spire2_rl.study import evaluation_seeds

SCRIPT_DIR = Path(__file__).resolve().parent
sessions = runpy.run_path(str(SCRIPT_DIR / "run_learning_session.py"))
metrics = runpy.run_path(str(SCRIPT_DIR / "audit_combat.py"))
reports = runpy.run_path(str(SCRIPT_DIR / "learning_session_report.py"))
CycleAuditObserver = runpy.run_path(str(SCRIPT_DIR / "cycle_audit_observer.py"))["CycleAuditObserver"]
FairTeacher = runpy.run_path(str(SCRIPT_DIR / "fair_teacher.py"))["FairTeacher"]
LABELS = ("current", "previous", "fair_teacher", "teacher_combat", "teacher_noncombat")
DEFAULT_SOURCE = Path("experiments/ironclad-a0-learning-sessions/session-0001")
PARTITION = "teacher_combat: all COMBAT decisions including pending choices; teacher_noncombat: all other phases"
TRACE_LIMIT = 6
PROBE_STRIDE = 16
PROBE_LIMIT = 16
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
GAME_FIELDS = ("seed", "outcome", "floor", "hp", "steps", "return", "error")


class HybridController:
    """Partition by actual phase, including combat card selections."""
    def __init__(self, model, teacher, mode):
        if mode not in {"teacher_combat", "teacher_noncombat"}:
            raise ValueError("Unknown hybrid controller")
        self.model, self.teacher, self.mode = model, teacher, mode

    def select(self, env):
        mgr = env.raw_env._mgr
        combat = mgr.phase == mgr.PHASE_COMBAT
        if combat == (self.mode == "teacher_combat"):
            return int(self.teacher.select(env))
        action, _ = self.model.predict(env.current_observation, action_masks=env.action_masks(), deterministic=True)
        return int(action)


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite public diagnostic {label}")
    return value


class LearningAuditObserver(CycleAuditObserver):
    """Extend existing observations with public deck and encounter context."""
    def __init__(self, env, *, model=None, trace_limit=TRACE_LIMIT):
        self.probe_model = model
        super().__init__(env, trace_limit=trace_limit)

    def _clear(self):
        super()._clear()
        self.episode_audit.update(public_choices=[], policy_probes=[])

    @staticmethod
    def _powers(creature):
        powers = list(creature.powers.items())
        if len(powers) > 64:
            raise ValueError("Public power list exceeds diagnostic bound")
        return [{"power_id": str(getattr(key, "name", key)),
                 "amount": _finite(value.amount, "power amount")} for key, value in powers]

    def _snapshot(self, mgr, combat=None):
        snapshot = super()._snapshot(mgr, combat)
        player = combat.player if combat is not None else mgr.run_state.player
        deck = Counter((str(card.card_id.name), bool(card.upgraded)) for card in mgr.run_state.player.deck)
        if len(deck) > 128 or len(mgr.run_state.player.relics) > 128:
            raise ValueError("Public permanent inventory exceeds diagnostic bound")
        snapshot.update(
            permanent_deck=[{"card_id": name, "upgraded": upgraded, "count": count}
                            for (name, upgraded), count in sorted(deck.items())],
            relics=[str(name) for name in mgr.run_state.player.relics],
        )
        if combat is not None:
            snapshot["player_powers"] = self._powers(player)
            if len(combat.enemies) > 32:
                raise ValueError("Public enemy list exceeds diagnostic bound")
            snapshot["enemy_state"] = [
                {"enemy_id": str(enemy.monster_id), "hp": _finite(enemy.current_hp, "enemy hp"),
                 "max_hp": _finite(enemy.max_hp, "enemy max hp"), "block": _finite(enemy.block, "enemy block"),
                 "powers": self._powers(enemy)} for enemy in combat.enemies]
        return snapshot

    def _choice(self, action):
        mgr = self.raw_env._mgr
        combat = mgr.get_combat_state() if mgr.phase == mgr.PHASE_COMBAT else None
        pending = metrics["choice_for"](self, combat)
        if combat is None:
            slots = self.raw_env.public_slots()
            named = dict(slots[int(action)])
            if named["action"] in {"pick_card", "pick_card_bundle", "skip", "buy_card", "event_choice", "rest_option"}:
                offered = [{key: item[key] for key in ("action", "card_id", "card_ids", "option_id", "index") if key in item}
                           for item in slots if item["action"] in {"pick_card", "pick_card_bundle", "buy_card", "rest_option"}]
                self._record_choice({"step": self._steps, "floor": mgr.run_state.total_floor,
                                     "phase": mgr.phase, "chosen": named, "offered": offered[:128]})
        if pending is not None:
            selected = {"action": "confirm_choice"} if combat is not None and int(action) == 0 else (
                {"action": "choose", "index": int(action) - 1} if combat is not None
                else dict(self.raw_env.public_slots()[int(action)]))
            options = pending.options
            if len(options) > 128:
                raise ValueError("Public pending choices exceed diagnostic bound")
            self._record_choice({"step": self._steps, "floor": mgr.run_state.total_floor, "phase": mgr.phase,
                                 "prompt": str(pending.prompt)[:1024], "chosen": selected,
                                 "offered_cards": [str(option.card.card_id.name) for option in options],
                                 "selected_before": sorted(pending.selected_indices)})

    def _record_choice(self, row):
        choices = self.episode_audit["public_choices"]
        if len(choices) < 96:
            choices.append(row)
        else:
            self.episode_audit["public_choices_omitted"] = self.episode_audit.get("public_choices_omitted", 0) + 1

    def _probe(self):
        probes = self.episode_audit["policy_probes"]
        if self.probe_model is None or self._steps % PROBE_STRIDE or len(probes) >= PROBE_LIMIT:
            return
        mask = self.action_masks()
        legal_count = int(np.count_nonzero(mask))
        if not legal_count:
            raise ValueError("Probe observes no legal actions")
        with torch.no_grad():
            observation = self.probe_model.policy.obs_to_tensor(self.current_observation)[0]
            distribution = self.probe_model.policy.get_distribution(observation, action_masks=mask)
            entropy = _finite(distribution.entropy().item(), "masked policy entropy")
            value = _finite(self.probe_model.policy.predict_values(observation).item(), "critic value")
        probes.append({"step": self._steps, "phase": self.raw_env._mgr.phase, "legal_actions": legal_count,
                       "masked_entropy": entropy, "normalized_masked_entropy": entropy / math.log(legal_count)
                       if legal_count > 1 else 0.0, "critic_value": value})

    def step(self, action):
        self._safe("public_choice_logging", lambda: self._choice(action))
        self._safe("frozen_policy_probe", self._probe)
        return super().step(action)


def play_controller(env, seeds, *, model, label, controller=None, on_row=None):
    rows = []
    for seed in seeds:
        row = metrics["run_policy"](env, [seed], model=model, label=label, controller=controller)[0]
        row["act1_cleared"] = row["outcome"] == "win"
        row["audit"] = deepcopy(env.episode_audit)
        rows.append(row)
        if on_row is not None:
            on_row(row)
        if _failures([row]):
            break
    return rows


def summarize_rows(rows):
    gameplay = reports["summarize_attempts"](rows)
    counts = {name: Counter() for name in (
        "death_rooms", "death_enemies", "macro_action_counts", "map_room_choices", "picked_card_counts", "rest_option_counts")}
    entries, bosses, elites, probes = [], [], [], []
    totals = Counter()
    observer_error_runs = deaths_with_potions = deaths_with_legal_potion = 0
    boss_deaths = 0
    for row in rows:
        observed = row["audit"]
        observer_error_runs += bool(observed["observer_errors"])
        for field in ("macro_action_counts", "map_room_choices", "picked_card_counts", "rest_option_counts"):
            counts[field].update(observed[field])
        if row["outcome"] == "death":
            death = observed["death_context"] or {}
            counts["death_rooms"][death.get("room_type", "UNKNOWN")] += 1
            counts["death_enemies"].update(death.get("enemies", []))
            boss_deaths += "BOSS" in str(death.get("room_type", ""))
            deaths_with_potions += (row["metrics"].get("potions_left_at_death") or 0) > 0
            deaths_with_legal_potion += (row["metrics"].get("legal_potion_slots_before_death") or 0) > 0
        # Include capped games in action accounting; game errors remain visible.
        for key, value in row["metrics"].items():
            if isinstance(value, (int, float)) and value is not None:
                totals[key] += value
        for entry in observed["combat_entries"]:
            entries.append(entry)
            if "BOSS" in str(entry["room_type"]):
                bosses.append(entry)
            if "ELITE" in str(entry["room_type"]):
                elites.append(entry)
        probes.extend(observed.get("policy_probes", []))
    def mean(values):
        return sum(values) / len(values) if values else None
    turns = totals.get("end_turns", 0)
    boss_run_count = sum(any("BOSS" in str(entry["room_type"]) for entry in row["audit"]["combat_entries"]) for row in rows)
    return {
        "gameplay": gameplay, **{key: dict(value) for key, value in counts.items()}, "totals": dict(totals),
        "floor_distribution": dict(sorted(Counter(str(row["floor"]) for row in rows).items())),
        "combat_arrivals": len(entries), "elite_arrivals": len(elites), "boss_arrivals": len(bosses),
        "runs_entering_boss": boss_run_count, "deaths_in_boss": boss_deaths,
        "boss_clear_fraction_conditional_on_arrival": gameplay["act1_clears"] / boss_run_count if boss_run_count else None,
        "mean_hp_entering_combat": mean([entry["hp"] for entry in entries]),
        "mean_hp_entering_elites": mean([entry["hp"] for entry in elites]),
        "mean_hp_entering_bosses": mean([entry["hp"] for entry in bosses]),
        "mean_hp_fraction_entering_bosses": mean([entry["hp"] / entry["max_hp"] for entry in bosses if entry["max_hp"] > 0]),
        "mean_deck_size_entering_bosses": mean([entry["deck_size"] for entry in bosses]),
        "early_end_turn_fraction": totals.get("end_turns_with_playable_cards", 0) / turns if turns else None,
        "mean_unused_energy_per_end_turn": totals.get("unused_energy_sum", 0) / turns if turns else None,
        "deaths_with_potions_left": deaths_with_potions,
        "deaths_with_legal_potion_before_last_action": deaths_with_legal_potion,
        "observer_error_runs": observer_error_runs,
        "sampled_policy_probes": {"samples": len(probes),
                                 "mean_masked_entropy": mean([p["masked_entropy"] for p in probes]),
                                 "mean_normalized_masked_entropy": mean([p["normalized_masked_entropy"] for p in probes]),
                                 "mean_critic_value": mean([p["critic_value"] for p in probes]),
                                 "note": "Frozen RL model on sampled visible states; in hybrids this is not the teacher distribution. Values are not calibrated against deterministic returns. No collapse or critic-calibration claim."},
        "note": "Descriptive all-attempt audit. Boss conversion conditions on arrival; unused cards, energy and potions can be intentional. Public-rule teacher sees richer public state than the neural encoder. Hybrids can change paths and encounters.",
    }


def _paired_descriptive(baseline, candidate):
    if [row["seed"] for row in baseline] != [row["seed"] for row in candidate]:
        raise ValueError("Diagnostic controllers must share the same ordered seeds")
    gained = [a["seed"] for b, a in zip(baseline, candidate) if a["outcome"] == "win" and b["outcome"] != "win"]
    lost = [a["seed"] for b, a in zip(baseline, candidate) if b["outcome"] == "win" and a["outcome"] != "win"]
    return {"pairs": len(baseline), "candidate_only_clears": len(gained), "baseline_only_clears": len(lost),
            "clear_rate_difference": (len(gained) - len(lost)) / len(baseline),
            "candidate_only_clear_seeds": gained, "baseline_only_clear_seeds": lost,
            "scope": "Descriptive matched diagnostic comparison; not promotion or a confirmed causal effect."}


def _checkpoint_models(source, parent):
    states = {"current": parent["state"], "previous": parent["lineage"]["initial_checkpoint"]}
    models = {}
    for label, state in states.items():
        path = sessions["trial"]["act1"]["checked_state"](source, state, parent["manifest"]["fingerprint"])
        model = MaskablePPO.load(path, device="cpu")
        if int(model.num_timesteps) != state["num_timesteps"]:
            raise ValueError(f"Frozen {label} model counter differs from metadata")
        if getattr(model, "policy", None) is not None:
            model.policy.set_training_mode(False)
        models[label] = model
    return models


def select_cohort(parent, episodes, start=None):
    """Replay an outcome-independent ordered subset of the already used block.

    This avoids reserving fresh seeds unknown to the released training runner.
    Future sessions already exclude the entire completed source block.
    """
    before, after = parent["before"], parent["after"]
    rows_before = reports["checked_rows"](before["episodes"])
    rows_after = reports["checked_rows"](after["episodes"])
    block = after["seed_start"]
    if before["seed_start"] != block or [row["seed"] for row in rows_before] != [row["seed"] for row in rows_after]:
        raise ValueError("Source before/after evaluations do not share one ordered block")
    expected = evaluation_seeds("validation", len(rows_after), parent["config"].experiment_seed, start=block)
    if [row["seed"] for row in rows_after] != expected:
        raise ValueError("Source raw rows differ from their recorded seed block")
    selected = block if start is None else start
    if (isinstance(selected, bool) or not isinstance(selected, int) or selected < block
            or selected + episodes > block + len(rows_after)):
        raise ValueError(f"Choose a contiguous diagnostic subset inside the recorded block {block}..{block + len(rows_after) - 1}")
    offset = selected - block
    seeds = expected[offset:offset + episodes]
    return seeds, selected, {"current": rows_after[offset:offset + episodes],
                             "previous": rows_before[offset:offset + episodes]}


def _protocol(source, parent, episodes, start, files):
    return {
        "schema": "learning-session-behavior-audit-v1", "source_experiment": str(source),
        "fingerprint": parent["manifest"]["fingerprint"], "config": asdict(parent["config"]),
        "current_checkpoint": parent["state"], "previous_checkpoint": parent["lineage"]["initial_checkpoint"],
        "candidate_checkpoint": parent["state"], "baseline_checkpoint": parent["lineage"]["initial_checkpoint"],
        "source_files": files, "controllers": list(LABELS), "phase_partition": PARTITION,
        "split": "validation", "episodes": episodes, "seed_start": start, "inference": "deterministic",
        "selection_rule": "First N ordered seeds of the source's completed matched block by default; an explicit contiguous subset must remain inside that block. No outcome-based selection.",
        "cohort_scope": "Diagnostic replay of already used evaluation seeds, not a fresh strength check",
        "frozen_replay_match_fields": list(GAME_FIELDS),
        "factory_descriptor": parent["descriptor"], "factory_plan": parent["lineage"]["plan"],
        "factory_source_state": parent["lineage"]["source_state"], "use_lookahead": False,
        "audit_sha256": sha256(Path(__file__)), "session_loader_sha256": sha256(SCRIPT_DIR / "run_learning_session.py"),
        "teacher_sha256": sha256(SCRIPT_DIR / "fair_teacher.py"),
        "observer_sha256": sha256(SCRIPT_DIR / "cycle_audit_observer.py"),
        "metric_helper_sha256": sha256(SCRIPT_DIR / "audit_combat.py"),
        "summary_helper_sha256": sha256(SCRIPT_DIR / "learning_session_report.py"),
        "trace_limit": TRACE_LIMIT, "probe_stride": PROBE_STRIDE, "probe_limit": PROBE_LIMIT,
        "additional_training_steps": 0, "champion_updated": False,
        "scope": "Fixed descriptive behavior diagnostic; no checkpoint selection, training or promotion.",
    }


def _payload_digest(payload):
    content = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_payload(payload, label, seeds, protocol_hash, *, originals=None, cap=4000):
    if (not isinstance(payload, dict) or payload.get("protocol_sha256") != protocol_hash
            or payload.get("label") != label or not isinstance(payload.get("complete"), bool)):
        raise ValueError(f"Controller cache identity changed: {label}")
    if payload.get("evidence_sha256") != _payload_digest(payload):
        raise ValueError(f"Controller cache evidence checksum changed: {label}")
    rows = reports["checked_rows"](payload.get("episodes"))
    if len(rows) > len(seeds) or [row["seed"] for row in rows] != seeds[:len(rows)]:
        raise ValueError(f"Controller cache seeds changed: {label}")
    if payload["complete"] != (len(rows) == len(seeds)):
        raise ValueError(f"Controller completion differs from its rows: {label}")
    for index, row in enumerate(rows):
        if (row.get("policy") != label or row.get("act1_cleared") != (row["outcome"] == "win")
                or not isinstance(row.get("metrics"), dict) or not isinstance(row.get("audit"), dict)):
            raise ValueError(f"Controller raw evidence is malformed: {label}")
        if (isinstance(row.get("steps"), bool) or not isinstance(row.get("steps"), int)
                or not 0 <= row["steps"] <= cap):
            raise ValueError(f"Controller decision count is outside the unchanged cap: {label}")
        if row["outcome"] == "timeout" and row["steps"] != cap:
            raise ValueError(f"Controller timeout ended before its unchanged cap: {label}")
        for key, value in row["metrics"].items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"Controller metric is malformed: {label}/{key}")
        audit = row["audit"]
        for name in ("combat_entries", "flagged_decisions", "observer_errors", "public_choices", "policy_probes"):
            if not isinstance(audit.get(name), list):
                raise ValueError(f"Controller observation evidence is missing: {label}/{name}")
        if originals is not None:
            original = originals[index]
            expected_match = all(row.get(key) == original[key] for key in GAME_FIELDS)
            if (row.get("matches_source_result") is not expected_match
                    or row.get("original_result") != {key: original[key] for key in GAME_FIELDS}):
                raise ValueError(f"Frozen replay classification differs from its source: {label}")
    for name in ("teacher_decisions", "teacher_phases"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"Controller teacher accounting is malformed: {label}")
    return rows


def _failures(rows):
    return [row for row in rows if row["outcome"] == "aborted" or row["error"] is not None
            or row["audit"]["observer_errors"] or row.get("matches_source_result") is False]


def _collect_controller(output, env, seeds, *, model, label, protocol_hash, controller=None, teacher=None,
                        originals=None, cap=4000):
    path = output / "controllers" / f"{label}.json"
    payload = read_json(path) if path.exists() else {
        "protocol_sha256": protocol_hash, "label": label, "episodes": [], "complete": False,
        "teacher_decisions": {}, "teacher_phases": {}}
    if not path.exists():
        payload["evidence_sha256"] = _payload_digest(payload)
    rows = _validate_payload(payload, label, seeds, protocol_hash, originals=originals, cap=cap)
    if _failures(rows):
        raise RuntimeError(f"Recorded game or observer failure in {label}; inspect {path}. No failed run is replayed automatically.")
    if payload["complete"]:
        return payload
    if teacher is not None:
        if hasattr(teacher, "decisions"):
            teacher.decisions.update(payload["teacher_decisions"])
        if hasattr(teacher, "phases"):
            teacher.phases.update(payload["teacher_phases"])
    def save(row):
        if originals is not None:
            original = originals[len(payload["episodes"])]
            row["matches_source_result"] = all(row.get(key) == original[key] for key in GAME_FIELDS)
            row["original_result"] = {key: original[key] for key in GAME_FIELDS}
        payload["episodes"].append(row)
        payload["complete"] = len(payload["episodes"]) == len(seeds)
        if teacher is not None:
            payload["teacher_decisions"] = dict(getattr(teacher, "decisions", {}))
            payload["teacher_phases"] = dict(getattr(teacher, "phases", {}))
        payload["evidence_sha256"] = _payload_digest(payload)
        _validate_payload(payload, label, seeds, protocol_hash, originals=originals, cap=cap)
        atomic_json(path, payload)
        count = len(payload["episodes"])
        if count % 25 == 0 or count == len(seeds):
            print(f"{label}: audited {count}/{len(seeds)} frozen runs", flush=True)
        if _failures([row]):
            raise RuntimeError(f"Game, observer or exact frozen replay failure in {label}; inspect {path}.")
    play_controller(env, seeds[len(rows):], model=model, label=label, controller=controller, on_row=save)
    _validate_payload(payload, label, seeds, protocol_hash, originals=originals, cap=cap)
    if _failures(payload["episodes"]):
        raise RuntimeError(f"Game or observer failure in {label}; raw evidence saved to {path}.")
    if not payload["complete"]:
        raise RuntimeError(f"Controller {label} did not complete its fixed diagnostic sample")
    return payload


def _human_report(summary):
    lines = ["Slay the Spire 2 frozen behavior audit", f"Status: {summary['status']}",
             "No additional training or checkpoint promotion."]
    for label, values in summary.get("controllers", {}).items():
        game = values["gameplay"]
        lines.append(f"{label}: {game['act1_clears']}/{game['attempts']} clears; "
                     f"{game['outcomes']['timeout']} timeouts; {game['outcomes']['aborted']} simulator aborts; "
                     f"{values['observer_error_runs']} observer-error runs.")
        lines.append(f"  Boss arrivals: {values['runs_entering_boss']}; deaths in boss: {values['deaths_in_boss']}; "
                     f"mean HP entering boss: {values['mean_hp_entering_bosses']}; "
                     f"conditional clear fraction: {values['boss_clear_fraction_conditional_on_arrival']}.")
        lines.append(f"  Death rooms: {json.dumps(values['death_rooms'], sort_keys=True)}; "
                     f"deaths with potions: {values['deaths_with_potions_left']}.")
    if summary.get("pause_reason"):
        lines.append("Pause reason: " + summary["pause_reason"])
    lines += [PARTITION, "Teacher uses public rules without lookahead, and sees richer public state than the neural encoder.",
              "Matched hybrid outcomes are descriptive. Changed decisions can change paths and encounters.",
              "Unused energy/cards/potions need not be mistakes; sampled values do not prove critic calibration.",
              "Emulator Act 1 only; transfer to the installed game remains unverified."]
    return "\n".join(lines) + "\n"


def _write_outputs(output, summary):
    atomic_json(output / "report.json", summary)
    target = output / "report.txt"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(_human_report(summary), encoding="utf-8")
    temporary.replace(target)
    _write_bundle(output)


def _write_bundle(output):
    selected = [output / name for name in ("protocol.json", "report.json", "report.txt", "pause.json")]
    selected += sorted((output / "controllers").glob("*.json"))
    used, omitted, included = 0, [], []
    target = output / "diagnostics.zip"
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                continue
            size = path.stat().st_size
            if used + size > MAX_BUNDLE_BYTES - 65536:
                omitted.append({"file": str(path.relative_to(output)), "bytes": size, "reason": "bundle bound"})
                continue
            archive.write(path, str(path.relative_to(output)))
            included.append(str(path.relative_to(output)))
            used += size
        archive.writestr("bundle-manifest.json", json.dumps({"included": included, "omitted": omitted,
                         "uncompressed_bytes": used, "max_uncompressed_bytes": MAX_BUNDLE_BYTES,
                         "note": "No models, game files, hidden state or training changes included."}, indent=2))
    if temporary.stat().st_size > MAX_BUNDLE_BYTES:
        temporary.unlink()
        raise ValueError("Diagnostic ZIP exceeds its final size bound")
    temporary.replace(target)


def _summary(output, protocol, payloads, *, status="completed", reason=None):
    controllers = {label: summarize_rows(payload["episodes"]) for label, payload in payloads.items() if payload["episodes"]}
    complete = set(payloads) == set(LABELS) and all(payload["complete"] for payload in payloads.values())
    errors = any(_failures(payload["episodes"]) for payload in payloads.values())
    comparisons = {}
    if status == "completed" and complete and not errors:
        for label in LABELS[1:]:
            comparisons[label + "_versus_current"] = _paired_descriptive(payloads["current"]["episodes"], payloads[label]["episodes"])
    return {"stage": "frozen_learning_session_behavior_audit", "status": status,
            "source_experiment": protocol["source_experiment"], "episodes_per_controller": protocol["episodes"],
            "validation_start": protocol["seed_start"], "controllers": controllers,
            "paired_descriptive": comparisons, "observer_or_game_errors": errors,
            "diagnostic_complete": status == "completed" and complete and not errors, "playing_strength_claim": False,
            "additional_training_steps": 0, "champion_updated": False, "training_gate_unchanged": True,
            "pause_reason": reason, "report_path": str(output / "report.txt"),
            "diagnostics_zip": str(output / "diagnostics.zip"),
            "protocol": protocol,
            "note": "One fixed ordered diagnostic subset of the completed source evaluation block, not fresh strength evidence. Frozen models, unchanged gameplay/rewards, no promotion, no additional RL. All attempted outcomes retained; errors and frozen replay mismatches pause. Descriptive comparisons do not establish an intervention's causal effect or a global strength claim."}


def _validate_completion(output, protocol_hash, seeds, *, originals, cap):
    completion = read_json(output / "completion.json")
    if completion.get("protocol_sha256") != protocol_hash:
        raise ValueError("Completed audit protocol changed")
    expected_names = {"report.json", "report.txt", "diagnostics.zip", *(f"controllers/{label}.json" for label in LABELS)}
    if set(completion.get("files", {})) != expected_names:
        raise ValueError("Completed audit evidence pins are malformed")
    for name, digest in completion["files"].items():
        path = output / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Completed audit evidence changed: {name}")
    payloads = {label: read_json(output / "controllers" / f"{label}.json") for label in LABELS}
    for label, payload in payloads.items():
        _validate_payload(payload, label, seeds, protocol_hash, originals=originals.get(label), cap=cap)
        if not payload["complete"] or _failures(payload["episodes"]):
            raise ValueError("Completed audit contains incomplete or failed controller evidence")
    report = read_json(output / "report.json")
    if report != _summary(output, read_json(output / "protocol.json"), payloads):
        raise ValueError("Completed audit summary differs from its raw controller evidence")
    return report


def audit(source, upstream, *, episodes=100, start=None, output=None):
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("Use a positive fixed diagnostic sample")
    source, upstream = Path(source).resolve(), Path(upstream).resolve()
    output = (Path(output) if output is not None else
              source.parent.parent / "ironclad-a0-learning-session-audits" / source.name).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Audit output must be separate from the source learning experiment")
    torch.set_num_threads(2)
    with sessions["trial"]["selection"]["workflow_lock"](source.parent), experiment_lock(source), experiment_lock(output):
        parent = sessions["load_source"](source, upstream)
        if parent["kind"] != "session":
            raise ValueError("This diagnostic requires a completed learning session")
        models = _checkpoint_models(source, parent)
        protocol_path = output / "protocol.json"
        prior = read_json(protocol_path) if protocol_path.exists() else None
        selected = prior["seed_start"] if prior is not None and start is None else start
        seeds, selected, originals = select_cohort(parent, episodes, selected)
        files = dict(parent["source_files"])
        protocol = _protocol(source, parent, episodes, selected, files)
        if prior is not None and prior != protocol:
            raise ValueError("Audit protocol is immutable; reuse the original source, sample and command")
        if prior is None:
            if any((output / name).exists() for name in ("report.json", "completion.json", "controllers")):
                raise ValueError("Audit protocol is missing beside cached evidence")
            atomic_json(protocol_path, protocol)
        protocol_hash = sha256(protocol_path)
        sessions["check_files"](files)
        for label in LABELS:
            path = output / "controllers" / f"{label}.json"
            if path.exists():
                _validate_payload(read_json(path), label, seeds, protocol_hash,
                                  originals=originals.get(label), cap=parent["config"].max_steps)
        if (output / "completion.json").exists():
            result = _validate_completion(output, protocol_hash, seeds, originals=originals, cap=parent["config"].max_steps)
            sessions["check_files"](files)
            print(_human_report(result), end="", flush=True)
            return result
        payloads = {}
        try:
            with sessions["factory"](parent["descriptor"], parent["lineage"]["plan"], parent["lineage"]["source_state"]):
                for label in LABELS:
                    model = models["previous"] if label == "previous" else models["current"]
                    if label == "fair_teacher":
                        model = None
                    teacher = FairTeacher() if label in {"fair_teacher", "teacher_combat", "teacher_noncombat"} else None
                    if teacher is not None and teacher.bot.use_lookahead:
                        raise ValueError("Diagnostic teacher must have lookahead disabled")
                    controller = teacher if label == "fair_teacher" else (
                        HybridController(model, teacher, label) if teacher is not None else None)
                    env = LearningAuditObserver(runner._make(parent["config"], upstream), model=model)
                    try:
                        payloads[label] = _collect_controller(output, env, seeds,
                                                              model=model if label in {"current", "previous"} else None, label=label,
                                                              protocol_hash=protocol_hash, controller=controller, teacher=teacher,
                                                              originals=originals.get(label), cap=parent["config"].max_steps)
                    finally:
                        env.close()
                    sessions["check_files"](files)
            result = _summary(output, protocol, payloads)
            _write_outputs(output, result)
            sessions["check_files"](files)
            names = ["report.json", "report.txt", "diagnostics.zip", *(f"controllers/{label}.json" for label in LABELS)]
            atomic_json(output / "completion.json", {"protocol_sha256": protocol_hash,
                        "files": {name: sha256(output / name) for name in names}})
        except KeyboardInterrupt:
            sessions["check_files"](files)
            # Completed prefixes are already persisted; the same command resumes.
            payloads = {label: read_json(output / "controllers" / f"{label}.json") for label in LABELS
                        if (output / "controllers" / f"{label}.json").exists()}
            result = _summary(output, protocol, payloads, status="interrupted", reason="User interruption; saved episode prefixes resume with the same command.")
            _write_outputs(output, result)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            try:
                sessions["check_files"](files)
            except Exception as guard_error:
                reason += f"; source guard: {type(guard_error).__name__}: {guard_error}"
            payloads = {label: read_json(output / "controllers" / f"{label}.json") for label in LABELS
                        if (output / "controllers" / f"{label}.json").exists()}
            result = _summary(output, protocol, payloads, status="paused", reason=reason)
            atomic_json(output / "pause.json", {"reason": reason, "additional_training_steps": 0})
            _write_outputs(output, result)
        print(_human_report(result), end="", flush=True)
        print(f"Report: {output / 'report.txt'}\nDiagnostics: {output / 'diagnostics.zip'}", flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--experiment", dest="experiment", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-start", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = audit(args.experiment, args.upstream, episodes=args.episodes, start=args.validation_start, output=args.output)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Audit could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
