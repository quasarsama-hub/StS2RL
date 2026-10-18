"""One resumable, fixed-budget PPO session with automatic failure diagnosis.

The learner and game interface are inherited unchanged. A session evaluates its
frozen starting model once and its final model once on one reserved fresh block.
Intermediate checkpoints serve recovery and infrastructure checks, never model
selection. An explicit next session is a new allowance, not an automatic loop.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from spire2_rl import runner
from spire2_rl.experiment import (ExperimentConfig, atomic_json, commit_checkpoint,
                                 experiment_lock, read_json, sha256, verified_checkpoint)
from spire2_rl.study import evaluation_seeds

SCRIPT_DIR = Path(__file__).resolve().parent
isolated = runpy.run_path(str(SCRIPT_DIR / "isolated_act1_cycles.py"))
cycles, trial, checker = isolated["cycles"], isolated["trial"], isolated["checker"]
reports = runpy.run_path(str(SCRIPT_DIR / "learning_session_report.py"))
tracer = runpy.run_path(str(SCRIPT_DIR / "learning_session_trace.py"))
DEFAULT_SOURCE = Path("experiments/ironclad-a0-act1-event-isolated-cycles/cycle-0001")
GAME_FIELDS = ("seed", "outcome", "floor", "hp", "steps", "return", "error")
REPLAY_CAP = 4


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def check_files(files):
    if not isinstance(files, dict):
        raise ValueError("Frozen file pins are malformed")
    for name, digest in files.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Frozen source evidence changed or is missing: {path}")


@contextmanager
def factory(descriptor, plan, source_state):
    """Reconstruct the flat original factory and isolate event objects once."""
    with isolated["factory"](descriptor["base_factory"], descriptor["original_plan"],
                             descriptor["original_source_state"], plan, source_state):
        original = runner._make
        def make(config, upstream):
            env = original(config, upstream)
            env.backend_metadata.update(
                learning_session="fixed-budget-single-comparison-v1",
                learning_session_runner_sha256=sha256(Path(__file__)),
                learning_session_report_sha256=sha256(SCRIPT_DIR / "learning_session_report.py"),
                learning_session_trace_sha256=sha256(SCRIPT_DIR / "learning_session_trace.py"),
                learning_session_plan=dict(plan),
                learning_session_source_checkpoint=dict(source_state),
            )
            return env
        runner._make = make
        try:
            yield
        finally:
            runner._make = original


def actual_checkpoint(root, manifest):
    path, state = verified_checkpoint(Path(root), manifest["fingerprint"])
    model = MaskablePPO.load(path, device="cpu")
    if int(model.num_timesteps) != state["num_timesteps"]:
        raise ValueError("Model counter differs from its saved checkpoint metadata")
    del model
    return path, state


def validate_budget(lineage, state, config, final=False):
    initial, source, target = lineage["initial_checkpoint"], lineage["source_state"], lineage["target_total_steps"]
    if (initial["num_timesteps"] != source["num_timesteps"]
            or target != initial["num_timesteps"] + lineage["plan"]["new_steps"]
            or state["num_timesteps"] < initial["num_timesteps"]
            or state["num_timesteps"] >= target + config.n_envs * config.n_steps):
        raise ValueError("Checkpoint counter falls outside this session's fixed budget")
    if final and (state["interrupted"] or state["num_timesteps"] < target):
        raise ValueError("Complete this session's fixed training allowance before final evaluation")


def training_health(root, config, state):
    """Only planned episode discards and user interruptions are recoverable."""
    record = checker["training_record"](Path(root), config, state)
    if record["status"] != "read":
        raise ValueError("Training journal is unreadable: " + record.get("error", "unknown error"))
    reasons = record.get("other_abort_reasons", {})
    failures = {reason: count for reason, count in reasons.items()
                if not str(reason).startswith("KeyboardInterrupt:")}
    if failures:
        raise ValueError("Training infrastructure failures require diagnosis: " + json.dumps(failures, sort_keys=True))
    sessions_path = Path(root) / "sessions.jsonl"
    if sessions_path.exists():
        rows = [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row.get("checkpoint_error"):
                raise ValueError("A training interruption failed to save its checkpoint")
            if row["status"] == "interrupted" and row.get("error") and not str(row["error"]).startswith("KeyboardInterrupt"):
                raise ValueError("An interrupted training session contains an infrastructure error: " + str(row["error"]))
    return record


def immutable_protocol(lineage, manifest):
    return {key: lineage[key] for key in (
        "session", "series", "base_source", "source_experiment", "source_state", "source_files",
        "descriptor", "config", "plan", "initial_checkpoint", "target_total_steps", "training_seed_indices",
        "source_replays", "source_replay_files"
    )} | {"schema": "learning-session-v1", "fingerprint": manifest["fingerprint"],
          "split": "validation", "episodes": lineage["plan"]["episodes"],
          "seed_start": lineage["plan"]["validation_start"], "inference": "deterministic",
          "primary_outcome": "Act 1 clear within the unchanged decision cap, all attempted seeds",
          "runner_sha256": sha256(Path(__file__)),
          "report_sha256": sha256(SCRIPT_DIR / "learning_session_report.py"),
          "trace_sha256": sha256(SCRIPT_DIR / "learning_session_trace.py"),
          "automatic_next_session": False, "champion_updated": False}


def validate_protocol(root, lineage, manifest):
    path = Path(root) / "session-protocol.json"
    if (not path.is_file() or lineage.get("protocol_sha256") != sha256(path)
            or read_json(path) != immutable_protocol(lineage, manifest)):
        raise ValueError("This session's immutable protocol differs from its lineage, source or implementation")
    for name, digest in lineage["source_replay_files"].items():
        diagnostic = (Path(root) / name).resolve()
        if (not diagnostic.is_relative_to(Path(root).resolve()) or not diagnostic.is_file()
                or sha256(diagnostic) != digest):
            raise ValueError("Frozen source replay evidence changed or is missing")


def load_source(source, upstream):
    """Verify an evaluated source; legal policy timeouts do not rewrite old gates."""
    source, upstream = Path(source).resolve(), Path(upstream).resolve()
    if (source / "session-lineage.json").is_file():
        lineage = read_json(source / "session-lineage.json")
        check_files(lineage["source_files"])
        config = ExperimentConfig(**lineage["config"])
        descriptor = lineage["descriptor"]
        with factory(descriptor, lineage["plan"], lineage["source_state"]):
            manifest = runner.initialize(source, config, upstream)
            validate_protocol(source, lineage, manifest)
            checkpoint, state = actual_checkpoint(source, manifest)
            validate_budget(lineage, state, config, final=True)
            result = final_result(source, upstream, manifest, lineage, config, state)
            if result is None or result["status"] != "completed":
                raise ValueError("Finish the immediately preceding learning session first")
            before, after = read_json(source / "before_rl.json"), read_json(source / "after_rl.json")
        names = ("session-lineage.json", "session-protocol.json", "session-progress.json", "session-evidence.json")
        kind = "session"
    else:
        lineage = read_json(source / "event-lineage.json")
        check_files(lineage["source_files"])
        config = ExperimentConfig(**lineage["config"])
        descriptor = {key: lineage[key] for key in ("base_factory", "original_plan", "original_source_state")}
        descriptor.update(original_base_source=lineage["base_source"], isolated_series=lineage["series"])
        # Require publication pins first: the reused validator must be read-only.
        if any(lineage.get(key) is None for key in (
                "before_rl_sha256", "after_rl_sha256", "event_progress_sha256", "event_evidence_sha256")):
            raise ValueError("Finish publishing the corrected source's raw evaluation evidence first")
        with isolated["factory"](descriptor["base_factory"], descriptor["original_plan"],
                                 descriptor["original_source_state"], lineage["plan"], lineage["source_state"]):
            manifest = runner.initialize(source, config, upstream)
            checkpoint, state = actual_checkpoint(source, manifest)
            validate_budget(lineage, state, config, final=True)
            result = isolated["completed_result"](source, upstream, manifest, lineage, config, state)
            if result is None or result["status"] not in {"completed", "incomplete_evaluation"}:
                raise ValueError("Complete the corrected source's fixed budget and evaluations first")
            before, after = read_json(source / "before_rl.json"), read_json(source / "after_rl.json")
        names = ("event-lineage.json", "event-progress.json", "event-evidence.json")
        kind = "isolated"
    config.validate()
    if config.scope != "fullrun" or config.max_acts != 1:
        raise ValueError("Learning sessions currently inherit the existing Act 1 specialization only")
    for report in (before, after):
        summary = reports["summarize_attempts"](report["episodes"])
        if summary["outcomes"]["aborted"]:
            raise ValueError("Source evaluations contain simulator aborts; resolve them before training")
    training_health(source, config, state)
    files = dict(lineage["source_files"])
    for report_state in (lineage["initial_checkpoint"], state):
        path = trial["act1"]["checked_state"](source, report_state, manifest["fingerprint"])
        files[str(path)] = report_state["sha256"]
    for name in (*names, "experiment.json", "before_rl.json", "after_rl.json"):
        files[str(source / name)] = sha256(source / name)
    files = isolated["_runtime_files"](source, files)
    check_files(files)
    return {"kind": kind, "source": str(source), "lineage": lineage, "descriptor": descriptor,
            "base_factory": descriptor["base_factory"], "config": config, "manifest": manifest,
            "checkpoint": checkpoint, "state": state, "before": before, "after": after,
            "source_files": files}


def _reservation(document):
    if document.get("split") == "validation":
        count = len(document["episodes"]) if isinstance(document.get("episodes"), list) else document.get("episodes")
        return count, document.get("seed_start")
    if isinstance(document.get("plan"), dict) and "validation_start" in document["plan"]:
        return document["plan"]["episodes"], document["plan"]["validation_start"]
    if isinstance(document.get("protocol"), dict):
        return _reservation(document["protocol"])
    return None


def fresh_schedule(parent, series, episodes):
    """Reserve ancestor reports, corrected cycles, and pending session protocols."""
    descriptor = parent["descriptor"]
    _, start, files = isolated["fresh_schedule"](
        Path(descriptor["original_base_source"]), Path(descriptor["isolated_series"]), parent, episodes)
    used = set()
    paths = {Path(name) for name in parent["source_files"] if Path(name).suffix == ".json"}
    paths.update(Path(series).glob("session-*/session-protocol.json"))
    paths.update(Path(series).glob("session-*/session-lineage.json"))
    paths.update(Path(series).glob("session-*/before_rl.json"))
    paths.update(Path(series).glob("session-*/after_rl.json"))
    for path in sorted(paths):
        document = read_json(path)
        files[str(path)] = sha256(path)
        used.update(isolated["strength_checker"]["raw_seeds"](document))
        reservation = _reservation(document)
        if reservation is None:
            continue
        count, block_start = reservation
        _positive(count, "Recorded evaluation size")
        if isinstance(block_start, bool) or not isinstance(block_start, int) or block_start < 0:
            raise ValueError("Recorded validation reservation is malformed")
        used.update(evaluation_seeds("validation", count, parent["config"].experiment_seed, start=block_start))
        start = max(start, block_start + count)
    seeds = evaluation_seeds("validation", episodes, parent["config"].experiment_seed, start=start)
    checker["fresh_seeds"](Path(descriptor["original_base_source"]), descriptor["base_factory"]["source_lineage"],
                           parent["config"], episodes, start)
    if set(seeds) & used:
        raise ValueError("Session validation overlaps an earlier report or reserved protocol")
    return seeds, start, files


def _replay_valid(result, row, cap):
    original = {key: row[key] for key in GAME_FIELDS}
    if result.get("original_result") != original:
        raise ValueError("Diagnostic replay differs from its frozen original gameplay row")
    replayed = result.get("replayed_result")
    if not isinstance(replayed, dict) or set(replayed) != set(GAME_FIELDS):
        raise ValueError("Diagnostic replay is missing its raw gameplay fields")
    expected_match = replayed == original
    if result.get("original_decision_cap") != cap:
        raise ValueError("Diagnostic replay used a different decision cap")
    requires_pause = (not expected_match or replayed["outcome"] == "aborted"
                      or replayed["error"] is not None or bool(result.get("logging_errors"))
                      or bool(result.get("logging_error_count"))
                      or result.get("error_stage") is not None
                      or (replayed["outcome"] == "timeout" and replayed["steps"] != cap))
    if result.get("matches_original_result") is not expected_match or result.get("requires_pause") is not requires_pause:
        raise ValueError("Diagnostic replay classification differs from its raw result")


def diagnostic_replays(root, upstream, manifest, state, report, *, label, cap=REPLAY_CAP, out_root=None):
    """Exactly replay at most cap recorded failures; disclose the rest."""
    root, upstream = Path(root).resolve(), Path(upstream).resolve()
    rows = reports["checked_rows"](report["episodes"])
    failed = [row for row in rows if row["outcome"] in {"timeout", "aborted"} or row["error"] is not None]
    if any(row["outcome"] == "aborted" or row["error"] is not None for row in rows):
        raise ValueError(f"{label} contains simulator aborts or errors")
    cap = max(0, min(REPLAY_CAP, cap))
    selected = failed[:cap]
    diagnostic = Path(out_root) if out_root is not None else root / "diagnostics"
    folder = diagnostic / f"{label}-{state['sha256'][:16]}"
    protocol = {"schema": "learning-session-replays-v1", "label": label,
                "fingerprint": manifest["fingerprint"], "checkpoint": state,
                "config": manifest["config"], "original_rows": selected,
                "total_failed_cases": len(failed), "cap": cap,
                "inference": "deterministic", "trace_sha256": sha256(SCRIPT_DIR / "learning_session_trace.py")}
    protocol_path, target, completion_path = folder / "protocol.json", folder / "report.json", folder / "completion.json"
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("Diagnostic replay protocol changed")
    elif target.exists() or completion_path.exists():
        raise ValueError("Diagnostic replay protocol is missing")
    else:
        atomic_json(protocol_path, protocol)
    if completion_path.exists():
        completion = read_json(completion_path)
        if (completion["protocol_sha256"] != sha256(protocol_path)
                or not target.exists() or completion["report_sha256"] != sha256(target)):
            raise ValueError("Pinned diagnostic replay evidence changed")
    if target.exists():
        result = read_json(target)
        if result.get("protocol") != protocol or len(result.get("runs", [])) != len(selected):
            raise ValueError("Diagnostic replay cache differs from its frozen cases")
        runs = result["runs"]
    else:
        config = ExperimentConfig(**manifest["config"])
        runs = []
        for row in selected:
            runs.append(tracer["replay"](root, config, upstream, state, row, folder / f"seed-{row['seed']}"))
        result = {"protocol": protocol, "runs": runs, **reports["classify_replays"](runs),
                  "total_failed_cases": len(failed), "untraced_cases": len(failed) - len(runs)}
        atomic_json(target, result)
    for row, replayed in zip(selected, runs):
        _replay_valid(replayed, row, manifest["config"]["max_steps"])
        if replayed.get("checkpoint_sha256") != state["sha256"]:
            raise ValueError("Diagnostic replay differs from its frozen checkpoint")
    expected = {"protocol": protocol, "runs": runs, **reports["classify_replays"](runs),
                "total_failed_cases": len(failed), "untraced_cases": len(failed) - len(runs)}
    if result != expected:
        raise ValueError("Diagnostic summary differs from its replay rows")
    if not completion_path.exists():
        atomic_json(completion_path, {"protocol_sha256": sha256(protocol_path), "report_sha256": sha256(target)})
    if result["requires_pause"]:
        raise ValueError(f"{label} replay found a mismatch, simulator error or logging failure; inspect {target}")
    return result


def source_replays(source, upstream, parent):
    root = Path(parent.get("replay_root", Path(source).parent / f"learning-source-diagnostics-{parent['state']['sha256'][:16]}"))
    if parent["kind"] == "session":
        context = factory(parent["descriptor"], parent["lineage"]["plan"], parent["lineage"]["source_state"])
    else:
        d = parent["descriptor"]
        context = isolated["factory"](d["base_factory"], d["original_plan"], d["original_source_state"],
                                     parent["lineage"]["plan"], parent["lineage"]["source_state"])
    results, runs, remaining = [], [], REPLAY_CAP
    with context:
        for key, state in (("before", parent["lineage"]["initial_checkpoint"]), ("after", parent["state"])):
            result = diagnostic_replays(source, upstream, parent["manifest"], state, parent[key],
                                        label=f"source-{key}", cap=remaining, out_root=root / "diagnostics")
            remaining -= len(result["runs"])
            runs.extend(result["runs"])
            results.append(result)
    return {**reports["classify_replays"](runs), "runs": runs,
            "total_failed_cases": sum(result["total_failed_cases"] for result in results),
            "untraced_cases": sum(result["untraced_cases"] for result in results)}


def create_session(source, root, upstream, *, session, series, steps, chunk_steps, episodes, base_source=None):
    source, root, upstream, series = map(lambda value: Path(value).resolve(), (source, root, upstream, series))
    if root.exists() or source == root or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("Use a new, separate learning-session directory")
    with experiment_lock(source):
        parent = load_source(source, upstream)
        base_source = Path(base_source or (source if session == 1 else parent["lineage"]["base_source"])).resolve()
        if session > 1 and (parent["kind"] != "session" or parent["lineage"]["session"] != session - 1
                            or parent["lineage"]["series"] != str(series)
                            or parent["lineage"]["base_source"] != str(base_source)):
            raise ValueError("The next session must inherit this series' immediately preceding completed model")
        if session == 1 and parent["kind"] != "isolated":
            raise ValueError("First learning session must fork the evaluated event-isolated source")
        config, state = parent["config"], parent["state"]
        _, start, schedule_files = fresh_schedule(parent, series, episodes)
        parent["source_files"].update(schedule_files)
        plan = {"new_steps": steps, "chunk_steps": chunk_steps, "episodes": episodes,
                "validation_start": start, "replay_cap_per_stage": REPLAY_CAP}
        indices = trial["source_indices"](source, state, config)
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-initializing-", dir=root.parent))
        try:
            parent["replay_root"] = staging
            atomic_json(staging / "diagnostics" / "source-context" / "report.json", {
                "source_experiment": str(source), "manifest": parent["manifest"], "lineage": parent["lineage"],
                "checkpoint": state, "before": parent["before"], "after": parent["after"],
                "note": "Frozen source evidence copied for diagnosis; original files remain unchanged."})
            replays = source_replays(source, upstream, parent)
            replay_files = {str(path.relative_to(staging)): sha256(path)
                            for path in (staging / "diagnostics").rglob("*.json")}
            with factory(parent["descriptor"], plan, state):
                manifest = runner.initialize(staging, config, upstream)
                vec = DummyVecEnv([lambda: runner._make(config, upstream) for _ in range(config.n_envs)])
                try:
                    model = MaskablePPO.load(parent["checkpoint"], env=vec, device="cpu", force_reset=True)
                    if (int(model.num_timesteps) != state["num_timesteps"]
                            or model.observation_space != vec.observation_space or model.action_space != vec.action_space):
                        raise ValueError("Imported model counter or public policy spaces changed")
                    vec._seeds = [None] * config.n_envs
                    initial = commit_checkpoint(model, staging, [SimpleNamespace(next_index=i) for i in indices], manifest["fingerprint"])
                finally:
                    vec.close()
            check_files(parent["source_files"])
            lineage = {"session": session, "series": str(series), "base_source": str(base_source),
                       "source_experiment": str(source), "source_state": state, "source_files": parent["source_files"],
                       "descriptor": parent["descriptor"], "config": asdict(config), "plan": plan,
                       "initial_checkpoint": initial, "target_total_steps": state["num_timesteps"] + steps,
                       "training_seed_indices": indices, "source_replays": replays, "source_replay_files": replay_files,
                       "optimizer": "preserved PPO Adam state", "old_training_gate_unchanged": True}
            atomic_json(staging / "session-protocol.json", immutable_protocol(lineage, manifest))
            lineage["protocol_sha256"] = sha256(staging / "session-protocol.json")
            atomic_json(staging / "session-lineage.json", lineage)
            if root.exists():
                raise ValueError("Session directory was created during initialization")
            staging.rename(root)
            return lineage
        except BaseException:
            # Preserve diagnostic evidence for the pause bundle without claiming initialization completed.
            failed = series / "reports" / f"session-{session:04d}" / "initialization"
            failed.parent.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                if failed.exists():
                    shutil.rmtree(failed)
                staging.rename(failed)
            raise


def assess(root, upstream, manifest, state, *, label, episodes, validation_start):
    return cycles["assess"](root, upstream, manifest, state, label=label,
                            episodes=episodes, validation_start=validation_start)


def cached_assessment(root, upstream, manifest, lineage, key, state):
    path, plan = Path(root) / f"{key}.json", lineage["plan"]
    trial["act1"]["checked_state"](Path(root), state, manifest["fingerprint"])
    if path.exists():
        if lineage.get(f"{key}_sha256") is not None and sha256(path) != lineage[f"{key}_sha256"]:
            raise ValueError("Cached session evaluation checksum changed")
        report = read_json(path)
    else:
        if lineage.get(f"{key}_sha256") is not None:
            raise ValueError("Pinned session evaluation is missing")
        report = assess(root, upstream, manifest, state, label=key,
                        episodes=plan["episodes"], validation_start=plan["validation_start"])
        trial["act1"]["validate_assessment"](report, manifest, state, key, plan["episodes"], plan["validation_start"])
        reports["checked_rows"](report["episodes"])
        atomic_json(path, report)
    trial["act1"]["validate_assessment"](report, manifest, state, key, plan["episodes"], plan["validation_start"])
    reports["checked_rows"](report["episodes"])
    if lineage.get(f"{key}_sha256") is None:
        lineage[f"{key}_sha256"] = sha256(path)
        atomic_json(Path(root) / "session-lineage.json", lineage)
    return report


def summary_for(root, config, lineage, latest, before, after, before_replays, after_replays):
    comparison = reports["compare_attempts"](before["episodes"], after["episodes"])
    return {"stage": "after_learning_session", "status": "completed", "session": lineage["session"],
            "victory_scope": "act1", "primary_outcome": "Act 1 clear within the unchanged decision cap, all attempted seeds",
            **comparison, "fixed_additional_steps": lineage["plan"]["new_steps"],
            "new_rl_steps": latest["num_timesteps"] - lineage["initial_checkpoint"]["num_timesteps"],
            "source_rl_steps": lineage["initial_checkpoint"]["num_timesteps"],
            "cumulative_rl_steps": latest["num_timesteps"], "checkpoint": latest,
            "validation_start": lineage["plan"]["validation_start"],
            "training_record": training_health(root, config, latest),
            "source_replays": lineage["source_replays"], "before_replays": before_replays, "after_replays": after_replays,
            "optimizer": lineage["optimizer"], "champion_updated": False, "automatic_next_session": False,
            "report_path": str(Path(root) / "report.txt"), "diagnostics_zip": str(Path(root) / "diagnostics.zip"),
            "log_path": str(Path(root).parent / f"session-{lineage['session']:04d}.log"),
            "note": "One fixed additional PPO allowance, unchanged learner/gameplay settings and one prespecified starting/final comparison. Timeouts remain failed clears; simulator aborts or replay mismatches pause. No intermediate selection, automatic budget extension or promotion. Emulator Act 1 only; real-game transfer remains unverified."}


def final_result(root, upstream, manifest, lineage, config, latest, *, allow_repair=False):
    root = Path(root)
    evidence_path, progress_path = root / "session-evidence.json", root / "session-progress.json"
    if not evidence_path.exists() and not progress_path.exists():
        if lineage.get("session_evidence_sha256") or lineage.get("session_progress_sha256"):
            raise ValueError("Pinned final session evidence is missing")
        return None
    if not evidence_path.exists():
        raise ValueError("Final session publication is incomplete; preserve it for inspection")
    for key, path in (("session_evidence_sha256", evidence_path), ("session_progress_sha256", progress_path)):
        if lineage.get(key) is not None and (not path.exists() or lineage[key] != sha256(path)):
            raise ValueError("Pinned final session evidence changed")
    if (not progress_path.exists() or any(lineage.get(key) is None for key in (
            "session_evidence_sha256", "session_progress_sha256"))) and not allow_repair:
        raise ValueError("Finish publishing the previous session with its original command before starting another")
    if any(lineage.get(f"{key}_sha256") is None for key in ("before_rl", "after_rl")):
        raise ValueError("Final session is missing committed frozen evaluations")
    validate_budget(lineage, latest, config, final=True)
    before = cached_assessment(root, upstream, manifest, lineage, "before_rl", lineage["initial_checkpoint"])
    after = cached_assessment(root, upstream, manifest, lineage, "after_rl", latest)
    evidence = read_json(evidence_path)
    before_replays = diagnostic_replays(root, upstream, manifest, lineage["initial_checkpoint"], before, label="before_rl")
    after_replays = diagnostic_replays(root, upstream, manifest, latest, after, label="after_rl")
    expected = summary_for(root, config, lineage, latest, before, after, before_replays, after_replays)
    if (evidence != {"before": before, "after": after, "before_replays": before_replays,
                     "after_replays": after_replays, "summary": expected}
            or (progress_path.exists() and read_json(progress_path) != expected)):
        raise ValueError("Final session summary differs from frozen raw evaluations or training evidence")
    if not progress_path.exists():
        atomic_json(progress_path, expected)
    if any(lineage.get(key) is None for key in ("session_evidence_sha256", "session_progress_sha256")):
        lineage["session_evidence_sha256"] = sha256(evidence_path)
        lineage["session_progress_sha256"] = sha256(progress_path)
        atomic_json(root / "session-lineage.json", lineage)
    if allow_repair and any(not (root / name).exists() for name in ("report.txt", "report.json", "diagnostics.zip")):
        reports["write_report"](root, expected)
    return expected


class Heartbeat:
    """Read-only status reporting; never performs inference or changes files."""
    def __init__(self, console, root, phase, interval=15):
        self.console, self.root, self.phase, self.interval = console, Path(root), phase, interval
        self.stopped = threading.Event()
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stopped.wait(self.interval):
            elapsed = time.monotonic() - self.start
            counter = ""
            try:
                state = read_json(self.root / "latest.json")
                counter = f"; saved decisions {state['num_timesteps']:,}"
            except (OSError, KeyError, ValueError):
                pass
            print(f"{self.phase}: {elapsed:.0f}s elapsed{counter}", file=self.console, flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()


@contextmanager
def phase(root, console, label):
    print(label + "...", file=console, flush=True)
    with Heartbeat(console, root, label):
        yield


def _progress(source, series, upstream, *, session, steps, chunk_steps, episodes, console):
    root = series / f"session-{session:04d}"
    parent = source if session == 1 else series / f"session-{session - 1:04d}"
    if not parent.is_dir():
        raise ValueError("The saved source or immediately preceding session is missing")
    if not root.exists():
        with phase(root, console, "Verifying source and recorded timeout replays"):
            create_session(parent, root, upstream, session=session, series=series,
                           steps=steps, chunk_steps=chunk_steps, episodes=episodes, base_source=source)
    lineage = read_json(root / "session-lineage.json")
    if (lineage["session"] != session or lineage["series"] != str(series)
            or lineage["base_source"] != str(source) or lineage["source_experiment"] != str(parent)
            or lineage["plan"]["new_steps"] != steps or lineage["plan"]["chunk_steps"] != chunk_steps
            or lineage["plan"]["episodes"] != episodes or lineage["plan"]["replay_cap_per_stage"] != REPLAY_CAP):
        raise ValueError("This session's source, budget and sample size are fixed; resume with its original command")
    check_files(lineage["source_files"])
    config = ExperimentConfig(**lineage["config"])
    with factory(lineage["descriptor"], lineage["plan"], lineage["source_state"]):
        manifest = runner.initialize(root, config, upstream)
        validate_protocol(root, lineage, manifest)
        with experiment_lock(root):
            _, latest = actual_checkpoint(root, manifest)
            validate_budget(lineage, latest, config)
            training_health(root, config, latest)
            cached = final_result(root, upstream, manifest, lineage, config, latest, allow_repair=True)
            if cached is not None:
                return cached
            with phase(root, console, f"Evaluating starting model ({episodes:,} reserved runs)"):
                before = cached_assessment(root, upstream, manifest, lineage, "before_rl", lineage["initial_checkpoint"])
                before_replays = diagnostic_replays(root, upstream, manifest, lineage["initial_checkpoint"], before, label="before_rl")
        while True:
            with experiment_lock(root):
                _, latest = actual_checkpoint(root, manifest)
                validate_budget(lineage, latest, config)
                training_health(root, config, latest)
            remaining = max(0, lineage["target_total_steps"] - latest["num_timesteps"])
            if not remaining:
                validate_budget(lineage, latest, config, final=True)
                break
            requested = min(chunk_steps, remaining)
            with phase(root, console, f"Training {requested:,} decisions ({remaining:,} remain in this session)"):
                training = runner.train(root, config, upstream, steps=requested)
                if training.get("status") != "completed":
                    raise ValueError("Training chunk did not complete normally")
            with experiment_lock(root):
                _, next_state = actual_checkpoint(root, manifest)
                if next_state["num_timesteps"] <= latest["num_timesteps"] or next_state["interrupted"]:
                    raise ValueError("Training did not publish a forward, completed checkpoint")
                validate_budget(lineage, next_state, config)
                training_health(root, config, next_state)
        with experiment_lock(root):
            _, latest = actual_checkpoint(root, manifest)
            validate_budget(lineage, latest, config, final=True)
            with phase(root, console, f"Evaluating final model ({episodes:,} same reserved runs)"):
                after = cached_assessment(root, upstream, manifest, lineage, "after_rl", latest)
                after_replays = diagnostic_replays(root, upstream, manifest, latest, after, label="after_rl")
            check_files(lineage["source_files"])
            summary = summary_for(root, config, lineage, latest, before, after, before_replays, after_replays)
            atomic_json(root / "session-evidence.json", {"before": before, "after": after,
                        "before_replays": before_replays, "after_replays": after_replays, "summary": summary})
            atomic_json(root / "session-progress.json", summary)
            lineage["session_evidence_sha256"] = sha256(root / "session-evidence.json")
            lineage["session_progress_sha256"] = sha256(root / "session-progress.json")
            atomic_json(root / "session-lineage.json", lineage)
            reports["write_report"](root, summary)
            return summary


def _pause_bundle(series, session, exc, root):
    folder = series / "reports" / f"session-{session:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("experiment.json", "latest.json", "session-lineage.json", "session-protocol.json", "before_rl.json",
                 "after_rl.json", "session-evidence.json", "session-progress.json", "episodes.jsonl", "sessions.jsonl"):
        path = root / name
        if path.is_file():
            shutil.copy2(path, folder / name)
    if (root / "diagnostics").is_dir():
        shutil.copytree(root / "diagnostics", folder / "diagnostics", dirs_exist_ok=True)
    # Failed staged initialization is already inside the new reports directory.
    initialization = folder / "initialization"
    if (initialization / "diagnostics").is_dir():
        shutil.copytree(initialization / "diagnostics", folder / "diagnostics", dirs_exist_ok=True)
    summary = {"stage": "learning_session_paused", "status": "paused", "session": session,
               "error": f"{type(exc).__name__}: {exc}", "champion_updated": False,
               "automatic_next_session": False,
               "log_path": str(series / f"session-{session:04d}.log"),
               "note": "No extra allowance was started. Preserve the bundle; simulator errors or replay mismatches require diagnosis. Reuse the original command after resolving the recorded cause."}
    atomic_json(series / f"session-{session:04d}-status.json", summary)
    atomic_json(folder / "pause.json", summary)
    reports["write_report"](folder, summary)
    return folder


def progress(source, series, upstream, *, session=1, steps=240000, chunk_steps=80000, episodes=1000):
    for name, value in (("session", session), ("steps", steps), ("chunk_steps", chunk_steps), ("episodes", episodes)):
        _positive(value, name)
    series, upstream = Path(series).resolve(), Path(upstream).resolve()
    root = series / f"session-{session:04d}"
    if source is None:
        if (root / "session-lineage.json").exists():
            source = read_json(root / "session-lineage.json")["base_source"]
        elif session > 1 and (series / f"session-{session - 1:04d}" / "session-lineage.json").exists():
            source = read_json(series / f"session-{session - 1:04d}" / "session-lineage.json")["base_source"]
        else:
            source = DEFAULT_SOURCE
    source = Path(source).resolve()
    if source == series or source.is_relative_to(series) or series.is_relative_to(source):
        raise ValueError("Source and learning-session series must not overlap")
    torch.set_num_threads(2)
    console = sys.stdout
    with trial["selection"]["workflow_lock"](series):
        series.mkdir(parents=True, exist_ok=True)
        log_path = series / f"session-{session:04d}.log"
        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                with redirect_stdout(log), redirect_stderr(log):
                    result = _progress(source, series, upstream, session=session, steps=steps,
                                       chunk_steps=chunk_steps, episodes=episodes, console=console)
        except BaseException as exc:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exception(type(exc), exc, exc.__traceback__, file=log)
            folder = _pause_bundle(series, session, exc, root)
            print(f"Session paused: {type(exc).__name__}: {exc}\nReport: {folder / 'report.txt'}\nDiagnostics: {folder / 'diagnostics.zip'}", file=console, flush=True)
            raise
    print(f"Session {session} completed: {result['new_rl_steps']:,} additional decisions.\nReport: {result['report_path']}\nDiagnostics: {result['diagnostics_zip']}", file=console, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0-learning-sessions"))
    parser.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--steps", type=int, default=240000)
    parser.add_argument("--chunk-steps", type=int, default=80000)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args(argv)
    return progress(args.source, args.experiment, args.upstream, session=args.session,
                    steps=args.steps, chunk_steps=args.chunk_steps, episodes=args.episodes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Learning session stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
