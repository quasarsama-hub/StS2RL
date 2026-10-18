"""All-attempt Act 1 measurements and bounded, shareable session reports.

This additive helper does not alter the simulator, PPO, or old promotion rules.
The primary endpoint is clearing Act 1 within the experiment's unchanged cap.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import zipfile


OUTCOMES = ("win", "death", "timeout", "aborted")
GAME_FIELDS = ("seed", "outcome", "floor", "hp", "steps", "return", "error")
ROOT_EVIDENCE = {
    "experiment.json", "latest.json", "session-lineage.json", "session-protocol.json",
    "session-progress.json", "session-evidence.json", "session-summary.json",
    "lineage.json", "protocol.json", "progress.json", "assessment.json",
    "before_rl.json", "after_rl.json", "before.json", "after.json", "report.json",
    "source-replays.json", "before-replays.json", "after-replays.json",
    "episodes.jsonl", "sessions.jsonl", "report.txt", "pause.json",
}
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
MAX_BUNDLE_FILES = 256
MAX_OMITTED_RECORDS = 128
BUNDLE_OVERHEAD_RESERVE = 512 * 1024


def checked_rows(rows):
    """Validate public episode evidence without mutating its original rows."""
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Episode evidence must be a sequence")
    seeds, result = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each episode must be an object")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("Episode seed must be an integer in [0, 2**31)")
        if seed in seeds:
            raise ValueError("Episode seeds must be unique")
        seeds.add(seed)
        if row.get("outcome") not in OUTCOMES:
            raise ValueError("Unknown episode outcome")
        floor = row.get("floor")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor < 0:
            raise ValueError("Episode floor must be a finite nonnegative number")
        for key in ("hp", "return"):
            if key in row and (isinstance(row[key], bool) or not isinstance(row[key], (int, float))
                               or not math.isfinite(row[key])):
                raise ValueError(f"Episode {key} must be finite")
        if "steps" in row and (isinstance(row["steps"], bool) or not isinstance(row["steps"], int) or row["steps"] < 0):
            raise ValueError("Episode steps must be a nonnegative integer")
        if "error" in row and row["error"] is not None and not isinstance(row["error"], str):
            raise ValueError("Episode error must be text or null")
        if row["outcome"] != "aborted" and row.get("error") is not None:
            raise ValueError("A completed or capped game cannot have an infrastructure error")
        result.append(dict(row))
    return result


def _wilson(wins, attempts):
    z = 1.959963984540054
    p, q = wins / attempts, z * z / attempts
    center = (p + q / 2) / (1 + q)
    radius = z * math.sqrt(p * (1 - p) / attempts + z * z / (4 * attempts**2)) / (1 + q)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize_attempts(rows):
    rows = checked_rows(rows)
    if not rows:
        raise ValueError("At least one attempted episode is required")
    counts = Counter(row["outcome"] for row in rows)
    completed = [row["floor"] for row in rows if row["outcome"] in {"win", "death"}]
    observed = [row["floor"] for row in rows if row["outcome"] != "aborted"]
    n, wins = len(rows), counts["win"]
    return {
        "episodes": n, "attempts": n, "outcomes": {kind: counts[kind] for kind in OUTCOMES},
        "act1_clears": wins, "clear_rate": wins / n, "win_rate": wins / n,
        "clear_rate_ci95": _wilson(wins, n), "timeout_rate": counts["timeout"] / n,
        "aborted_rate": counts["aborted"] / n, "infrastructure_failure_rate": counts["aborted"] / n,
        "mean_floor_completed": _mean(completed), "completed_floor_samples": len(completed),
        "mean_floor_observed": _mean(observed), "observed_floor_samples": len(observed),
        "mean_floor": _mean(completed),
        "primary_outcome": "Act 1 clear within the unchanged decision cap, across all attempted seeds",
        "note": "Timeouts remain explicit non-clears. Simulator aborts block a playing-strength claim. Floor is diagnostic.",
    }


def _mcnemar(candidate_only, baseline_only):
    n = candidate_only + baseline_only
    if not n:
        return 1.0
    numerator = 2 * sum(math.comb(n, k) for k in range(min(candidate_only, baseline_only) + 1))
    return min(1.0, numerator / (1 << n))


def compare_attempts(before_rows, after_rows, *, bootstrap_samples=10000, bootstrap_seed=7):
    """Paired empirical bootstrap, computed from -1/0/+1 outcome counts.

    Multinomial count sampling has the same distribution as resampling the
    paired clear differences. It uses a different random stream from old code.
    """
    before, after = checked_rows(before_rows), checked_rows(after_rows)
    if not before or [row["seed"] for row in before] != [row["seed"] for row in after]:
        raise ValueError("Comparison requires the same nonempty ordered seed list")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    import numpy as np
    n = len(before)
    positive = sum(a["outcome"] == "win" and b["outcome"] != "win" for b, a in zip(before, after))
    negative = sum(b["outcome"] == "win" and a["outcome"] != "win" for b, a in zip(before, after))
    counts = np.random.default_rng(bootstrap_seed).multinomial(
        n, [positive / n, negative / n, (n - positive - negative) / n], size=bootstrap_samples)
    interval = np.quantile((counts[:, 0] - counts[:, 1]) / n, [0.025, 0.975]).tolist()
    aborts = sum(row["outcome"] == "aborted" for row in before + after)
    p = _mcnemar(positive, negative)
    supports = not aborts and positive > negative and p < 0.05
    clean = [(b, a) for b, a in zip(before, after) if b["outcome"] in {"win", "death"} and a["outcome"] in {"win", "death"}]
    return {
        "pairs": n, "before_rl": summarize_attempts(before), "after_rl": summarize_attempts(after),
        "rl_act1_clear_rate_gain": (positive - negative) / n,
        "rl_act1_clear_rate_gain_ci95": interval,
        "establishes_positive_act1_clear_gain": bool(not aborts and interval[0] > 0 and supports),
        "paired_clear_evidence": {
            "candidate_only_clears": positive, "baseline_only_clears": negative,
            "discordant_pairs": positive + negative, "exact_two_sided_p": p,
            "supports_more_clears": bool(supports),
            "method": "Exact paired McNemar test; two-sided threshold 0.05",
            "note": "No clear-rate equivalence or noninferiority claim is made.",
        },
        "infrastructure_failure_count": aborts,
        "clean_floor_pairs": len(clean), "excluded_floor_pairs": n - len(clean),
        "rl_floor_gain": _mean([a["floor"] - b["floor"] for b, a in clean]),
        "floor_scope": "Descriptive difference on paired completed games only; no floor strength claim",
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
        "bootstrap_method": "Multinomial resampling of empirical paired clear differences (-1, 0, +1)",
        "note": "A zero-width interval from no discordant clears does not establish equal ability. Per-comparison evidence is not a global guarantee over repeated sessions.",
    }


def classify_replays(rows):
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Replay evidence must be a sequence")
    counts = {key: 0 for key in ("matched_timeouts", "mismatched_cases", "aborted_cases", "logging_failure_cases", "cap_mismatch_cases")}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Replay evidence must be an object")
        played = row.get("replayed_result", row)
        match = False
        original = row.get("original_result")
        if isinstance(original, dict) and isinstance(played, dict):
            match = (row.get("matches_original_result") is True
                     and all(key in original and key in played for key in GAME_FIELDS)
                     and all(played[key] == original[key] for key in GAME_FIELDS))
            try:
                checked_rows([original])
                checked_rows([played])
            except ValueError:
                match = False
        if not isinstance(played, dict):
            raise ValueError("Replayed gameplay result must be an object")
        counts["mismatched_cases"] += not match
        counts["aborted_cases"] += played.get("outcome") == "aborted" or played.get("error") is not None
        counts["logging_failure_cases"] += bool(row.get("logging_errors") or row.get("logging_error_count"))
        cap_failure = (played.get("outcome") == "timeout" and "original_decision_cap" in row
                       and played.get("steps") != row["original_decision_cap"])
        counts["cap_mismatch_cases"] += bool(cap_failure)
        counts["matched_timeouts"] += bool(match and played.get("outcome") == "timeout"
                                          and not row.get("logging_errors") and not row.get("logging_error_count")
                                          and not cap_failure)
    return {"replayed_cases": len(rows), **counts,
            "requires_pause": bool(counts["mismatched_cases"] or counts["aborted_cases"] or counts["logging_failure_cases"]
                                   or counts["cap_mismatch_cases"] or any(row.get("requires_pause") is True for row in rows)),
            "note": "A matched legal timeout is an explicit non-clear; recurrence alone does not prove an engine loop."}


def _atomic_text(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def _pretty_summary(summary):
    lines = ["Slay the Spire 2 learning session", f"Status: {summary.get('status', 'unknown')}"]
    for label, key in (("Before", "before_rl"), ("After", "after_rl")):
        row = summary.get(key)
        if isinstance(row, dict):
            counts, n = row.get("outcomes", {}), row.get("episodes", row.get("attempts", 0))
            wins = row.get("act1_clears", counts.get("win", 0))
            lines.append(f"{label}: {wins}/{n} Act 1 clears; {counts.get('timeout', 0)} timeouts; {counts.get('aborted', 0)} simulator aborts.")
    if "rl_act1_clear_rate_gain" in summary:
        lines.append(f"Clear-rate change: {summary['rl_act1_clear_rate_gain']:+.2%}; 95% paired interval: {summary.get('rl_act1_clear_rate_gain_ci95')}.")
    paired = summary.get("paired_clear_evidence", {})
    if paired:
        lines.append(f"Exact paired p-value: {paired.get('exact_two_sided_p')}; supports more clears: {paired.get('supports_more_clears')}.")
    for key in ("new_rl_steps", "cumulative_rl_steps", "target_total_steps", "pause_reason", "error", "note"):
        if key in summary:
            lines.append(f"{key.replace('_', ' ').capitalize()}: {summary[key]}")
    lines += ["Timeouts count as non-clears; simulator errors block a strength claim.",
              "These are emulator Act 1 results. Transfer to the installed game remains unverified."]
    return "\n".join(lines) + "\n"


def write_report(root, summary):
    """Write concise text/JSON plus a size-bounded allowlisted support ZIP."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_text(root / "report.json", payload)
    _atomic_text(root / "report.txt", _pretty_summary(summary))
    selected = [root / name for name in sorted(ROOT_EVIDENCE)]
    diagnostics = root / "diagnostics"
    if diagnostics.is_dir() and not diagnostics.is_symlink():
        selected += sorted(path for path in diagnostics.rglob("*.json") if path.name in {
            "summary.json", "report.json", "protocol.json", "completion.json", "replays.json", "runs.json"})
    omitted, omitted_count, used, included_files = [], 0, 0, 0
    budget = MAX_BUNDLE_BYTES - BUNDLE_OVERHEAD_RESERVE
    if budget < 1:
        raise ValueError("Bundle limit must leave room for metadata")
    temp = root / "diagnostics.zip.tmp"

    def omit(path, reason, **extra):
        nonlocal omitted_count
        omitted_count += 1
        if len(omitted) < MAX_OMITTED_RECORDS:
            name = str(path)
            if len(name.encode("utf-8")) > 256:
                name = name.encode("utf-8")[:253].decode("utf-8", errors="replace") + "..."
            omitted.append({"path": name, "reason": reason, **extra})

    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in selected:
            if not path.is_file():
                continue
            name = str(path.relative_to(root)).replace("\\", "/")
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                omit(name, "symlink or outside report root")
                continue
            if len(name.encode("utf-8")) > 256 or included_files >= MAX_BUNDLE_FILES:
                omit(name, "bundle entry or path bound")
                continue
            size = path.stat().st_size
            if size > MAX_FILE_BYTES or used + size > budget:
                omit(name, "bundle size bound", bytes=size)
                continue
            bundle.write(path, name)
            used += size
            included_files += 1
        log_paths = [(root / name, "logs/" + name + ".tail.txt") for name in (
            "session.log", "run.log", "learning-session.log")]
        explicit = summary.get("log_path")
        if explicit:
            log_path = Path(explicit).absolute()
            allowed_parents = {root, root.parent, root.parent.parent}
            if (re.fullmatch(r"session-[0-9]+\.log", log_path.name)
                    and log_path.parent.resolve() in allowed_parents
                    and not log_path.is_symlink()
                    and log_path.resolve().parent in allowed_parents):
                log_paths.append((log_path, "log-tail.txt"))
            else:
                omit(log_path, "external log path outside allowed session layout")
        for path, archive_name in log_paths:
            if not path.is_file() or path.is_symlink():
                continue
            if archive_name != "log-tail.txt" and not path.resolve().is_relative_to(root):
                continue
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - MAX_LOG_BYTES))
                tail = stream.read(MAX_LOG_BYTES)
            if used + len(tail) <= budget and included_files < MAX_BUNDLE_FILES:
                bundle.writestr(archive_name, tail)
                used += len(tail)
                included_files += 1
            else:
                omit(path, "bundle size bound")
        bundle.writestr("bundle-manifest.json", json.dumps({
            "included_bytes": used, "included_evidence_files": included_files,
            "omitted": omitted, "omitted_files": omitted_count,
            "omitted_records_truncated": omitted_count - len(omitted),
            "note": "Allowlisted evidence and bounded log tails only; no model, game, action or observation catalogs."}, indent=2))
    if temp.stat().st_size > MAX_BUNDLE_BYTES:
        temp.unlink()
        raise ValueError("Support ZIP exceeded its total size bound")
    temp.replace(root / "diagnostics.zip")
    return {"text_report": str(root / "report.txt"), "json_report": str(root / "report.json"),
            "diagnostics_zip": str(root / "diagnostics.zip")}
