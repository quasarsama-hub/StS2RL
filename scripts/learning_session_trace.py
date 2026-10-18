"""Bounded exact frozen-policy replays for the combined learning session.

The caller installs the same runner._make factory used by the evaluated policy.
Logging reads public observations/actions only and cannot force a game action.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import importlib.util
import json
from pathlib import Path


GAME_FIELDS = ("seed", "outcome", "floor", "hp", "steps", "return", "error")


def _description_function():
    path = Path(__file__).with_name("trace_timeouts.py")
    spec = importlib.util.spec_from_file_location("learning_session_public_description", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.describe_decision, module.serializable


def _run(env, model, seed, *, cap, tail=40, describe=None, serialize=None):
    """Exactly runner._rollout's game path; observational failures are separate."""
    import numpy as np
    observations, actions = Counter(), Counter()
    last = deque(maxlen=tail)
    info, steps, total, outcome, error, error_stage = {}, 0, 0.0, "aborted", None, None
    failures, failure_count = [], 0
    stage = "reset"

    def log_error(where, exc):
        nonlocal failure_count
        failure_count += 1
        if len(failures) < 40:
            failures.append({"stage": where, "step": steps, "error": f"{type(exc).__name__}: {exc}"})

    if describe is None or serialize is None:
        try:
            describe, serialize = _description_function()
        except Exception as exc:
            log_error("logger_initialization", exc)
            describe = lambda env, action: {"action_slot": action}
            serialize = lambda value: value
    try:
        obs, info = env.reset(seed=seed)
        while True:
            stage = "policy_inference"
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)
            try:
                public = serialize({"observation": obs, "legal_actions": np.flatnonzero(masks)})
                encoded = json.dumps(public, sort_keys=True, separators=(",", ":"), allow_nan=False)
                digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                observations[digest] += 1
                actions[(digest, action)] += 1
                named = describe(env, action)
                decision = {"step": steps, "phase": info.get("phase"), "floor": info.get("floor"),
                            "hp": info.get("hp"), "action_slot": action, "observation_sha256": digest,
                            "observation_visits": observations[digest],
                            "public_observation_action_visits": actions[(digest, action)], **named}
                encoded_decision = json.dumps(decision, allow_nan=False)
                if len(encoded_decision.encode("utf-8")) > 8192:
                    decision = {key: decision[key] for key in (
                        "step", "phase", "floor", "hp", "action_slot", "observation_sha256",
                        "observation_visits", "public_observation_action_visits")}
                    decision["description_omitted"] = "Decision description exceeded the 8 KiB bound"
                last.append(decision)
            except Exception as exc:
                log_error("public_state_logging", exc)
            stage = "game_step"
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += float(reward)
            if terminated or truncated:
                outcome = info["outcome"]
                break
    except Exception as exc:
        error, error_stage = f"{type(exc).__name__}: {exc}", stage
    return {
        "seed": seed, "outcome": outcome, "floor": int(info.get("floor", 0)),
        "hp": float(info.get("hp", 0)), "steps": steps, "return": total, "error": error,
        "error_stage": error_stage, "final_phase": info.get("phase"),
        "original_decision_cap": cap, "logging_errors": failures,
        "logging_error_count": failure_count, "last_decisions": list(last),
        "unique_public_observations": len(observations),
        "most_repeated_observation_visits": max(observations.values(), default=0),
        "most_repeated_public_observation_action_visits": max(actions.values(), default=0),
        "note": "Repeated public states/actions locate behavior to inspect; they alone do not establish an engine cycle. No actions or rewards were changed.",
    }


def replay(root, config, upstream, state, row, out_dir, *, tail=40):
    """Replay one recorded seed under caller-owned, validated factory context."""
    if isinstance(tail, bool) or not isinstance(tail, int) or not 0 <= tail <= 100:
        raise ValueError("Replay tail must be an integer from 0 to 100")
    if getattr(config, "scope", None) != "fullrun" or getattr(config, "max_acts", None) != 1:
        raise ValueError("Session replay requires the same fullrun Act 1 configuration")
    cap = config.max_steps
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("Replay requires the unchanged positive decision cap")
    if any(key not in row for key in GAME_FIELDS):
        raise ValueError("Replay requires all seven original gameplay fields")
    seed = row["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("Replay seed must be a 31-bit nonnegative integer")
    from sb3_contrib import MaskablePPO
    from spire2_rl import runner
    from spire2_rl.experiment import atomic_json, sha256

    root, out_dir = Path(root).resolve(), Path(out_dir).resolve()
    model_path = (root / state["model"]).resolve()
    if not model_path.is_relative_to(root) or state.get("interrupted") or sha256(model_path) != state["sha256"]:
        raise ValueError("Replay checkpoint is interrupted, corrupt, or outside the experiment")
    model = MaskablePPO.load(model_path, device="cpu")
    if int(model.num_timesteps) != state["num_timesteps"]:
        raise ValueError("Replay model counter differs from its checkpoint")
    out_dir.mkdir(parents=True, exist_ok=False)
    env = runner._make(config, Path(upstream).resolve())
    try:
        result = _run(env, model, seed, cap=cap, tail=tail)
    finally:
        env.close()
    original = {key: row[key] for key in GAME_FIELDS}
    played = {key: result[key] for key in GAME_FIELDS}
    matches = original == played
    result.update(original_result=original, replayed_result=played, matches_original_result=matches,
                  checkpoint_sha256=state["sha256"], additional_training_steps=0,
                  summary_path="summary.json")
    result["requires_pause"] = bool(not matches or played["outcome"] == "aborted" or played["error"] is not None
                                     or result["logging_error_count"]
                                     or (played["outcome"] == "timeout" and played["steps"] != cap))
    if sha256(model_path) != state["sha256"]:
        raise ValueError("Frozen replay checkpoint changed during diagnosis")
    try:
        atomic_json(out_dir / "summary.json", result)
    except Exception as exc:
        result["logging_error_count"] += 1
        result["logging_errors"].append({"stage": "summary_logging", "step": result["steps"],
                                          "error": f"{type(exc).__name__}: {exc}"})
        result["requires_pause"] = True
    return result
