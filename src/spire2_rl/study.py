"""Reproducible seeds and conservative evaluation of repeated-run policies.

This module intentionally has no ML or game dependencies.  Infrastructure
failures are accounted for separately from actual gameplay losses.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Mapping
from typing import Any


SEEDS_PER_SPLIT = 1 << 29
MAX_SEED_INDEX = SEEDS_PER_SPLIT - 1
_SPLIT_OFFSETS = {
    "train": 0,
    "validation": SEEDS_PER_SPLIT,
    "test": 2 * SEEDS_PER_SPLIT,
}
_OUTCOMES = ("win", "death", "timeout", "aborted")
_GAME_OUTCOMES = frozenset(("win", "death"))
_Z_95 = 1.959963984540054


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _split_offset(split: str) -> int:
    if not isinstance(split, str) or split not in _SPLIT_OFFSETS:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    return _SPLIT_OFFSETS[split]


def seed_for(index: int, split: str, experiment_seed: int = 7) -> int:
    """Map an episode counter to a unique positive 31-bit seed in its split.

    Each split owns a fixed, disjoint window of 2**29 seeds.  A keyed affine
    permutation randomizes the order within that window; its odd multiplier
    makes it bijective modulo 2**29.  Bounds are checked before permutation,
    so a counter never silently wraps back onto earlier episodes.

    Persist the next training index in the experiment journal.  Different
    experiment seeds change ordering, but never move a seed between splits.
    """
    offset = _split_offset(split)
    _integer(index, "index", 0)
    _integer(experiment_seed, "experiment_seed")
    if index > MAX_SEED_INDEX:
        raise ValueError(f"index must be at most {MAX_SEED_INDEX}")
    key = hashlib.blake2b(
        f"spire2-rl-seeds-v1:{experiment_seed}".encode("ascii"),
        digest_size=16,
    ).digest()
    multiplier = int.from_bytes(key[:8], "big") | 1
    shift = int.from_bytes(key[8:], "big")
    return offset + ((multiplier * index + shift) % SEEDS_PER_SPLIT) + 1


def evaluation_seeds(
    split: str,
    count: int,
    experiment_seed: int = 7,
    start: int = 0,
) -> list[int]:
    """Return a stable seed slice; use the identical slice for paired policies."""
    _split_offset(split)
    _integer(count, "count", 0)
    _integer(start, "start", 0)
    _integer(experiment_seed, "experiment_seed")
    if start > SEEDS_PER_SPLIT or start + count > SEEDS_PER_SPLIT:
        raise ValueError("requested seed slice exceeds its split")
    return [seed_for(index, split, experiment_seed) for index in range(start, start + count)]


def wilson_interval(wins: int, games: int) -> list[float] | None:
    """Two-sided 95% Wilson interval for wins / actual completed games."""
    _integer(wins, "wins", 0)
    _integer(games, "games", 0)
    if wins > games:
        raise ValueError("wins cannot exceed games")
    if games == 0:
        return None
    proportion = wins / games
    denominator = 1.0 + _Z_95**2 / games
    center = (proportion + _Z_95**2 / (2.0 * games)) / denominator
    radius = (
        _Z_95
        * math.sqrt(proportion * (1.0 - proportion) / games + _Z_95**2 / (4.0 * games**2))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be {'a nonnegative' if nonnegative else 'a'} finite number")
    return result


def _episodes(episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"episode {index} must be a mapping")
        row = dict(episode)
        outcome = row.get("outcome")
        if not isinstance(outcome, str) or outcome not in _OUTCOMES:
            raise ValueError(f"episode {index} has a missing or unknown outcome")
        for field in ("floor", "hp"):
            if row.get(field) is not None:
                row[field] = _number(row[field], field, nonnegative=field == "floor")
        result.append(row)
    return result


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def summarize(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize runs without treating timeouts or adapter aborts as deaths.

    Win rate, floor and HP describe only win/death episodes.  Infrastructure
    outcomes remain visible in counts and rates.  Missing optional floor/HP
    values are disclosed by sample counts rather than filled with zero.
    """
    rows = _episodes(episodes)
    counts = {outcome: 0 for outcome in _OUTCOMES}
    floors: list[float] = []
    hps: list[float] = []
    for row in rows:
        counts[row["outcome"]] += 1
        if row["outcome"] in _GAME_OUTCOMES:
            if row.get("floor") is not None:
                floors.append(row["floor"])
            if row.get("hp") is not None:
                hps.append(row["hp"])
    total = len(rows)
    games = counts["win"] + counts["death"]
    return {
        "episodes": total,
        "outcomes": counts,
        "valid_game_runs": games,
        "win_rate": counts["win"] / games if games else None,
        "win_rate_ci95": wilson_interval(counts["win"], games),
        "timeout_rate": counts["timeout"] / total if total else None,
        "aborted_rate": counts["aborted"] / total if total else None,
        "infrastructure_failure_rate": counts["aborted"] / total if total else None,
        "incomplete_run_rate": (counts["timeout"] + counts["aborted"]) / total if total else None,
        "mean_floor": _mean(floors),
        "floor_samples": len(floors),
        "mean_hp": _mean(hps),
        "hp_samples": len(hps),
    }


def _by_seed(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        seed = row.get("seed")
        _integer(seed, f"{label} episode {index} seed", 0)
        if seed >= 1 << 31:
            raise ValueError(f"{label} seed must fit in 31 bits")
        if seed in result:
            raise ValueError(f"duplicate seed {seed} in {label}")
        if row["outcome"] in _GAME_OUTCOMES and row.get("floor") is None:
            raise ValueError(f"{label} seed {seed} is missing floor")
        result[seed] = row
    return result


def _percentile(ordered: list[float], probability: float) -> float:
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    win_deltas: list[float],
    floor_deltas: list[float],
    samples: int,
    bootstrap_seed: int,
) -> tuple[list[float] | None, list[float] | None]:
    if not win_deltas:
        return None, None
    rng = random.Random(bootstrap_seed)
    pairs = len(win_deltas)
    win_means: list[float] = []
    floor_means: list[float] = []
    for _ in range(samples):
        win_sum = 0.0
        floor_sum = 0.0
        for _ in range(pairs):
            index = rng.randrange(pairs)
            win_sum += win_deltas[index]
            floor_sum += floor_deltas[index]
        win_means.append(win_sum / pairs)
        floor_means.append(floor_sum / pairs)
    win_means.sort()
    floor_means.sort()
    return (
        [_percentile(win_means, 0.025), _percentile(win_means, 0.975)],
        [_percentile(floor_means, 0.025), _percentile(floor_means, 0.975)],
    )


def paired_compare(
    candidate: Iterable[Mapping[str, Any]],
    baseline: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 7,
    min_pairs: int = 100,
) -> dict[str, Any]:
    """Compare identical seed sets with a deterministic paired bootstrap.

    Duplicate/missing seeds, unknown outcomes, and missing gameplay floors
    raise ValueError.  Genuine win/death differences are the comparison signal.
    Any pair containing a timeout/abort is excluded from gameplay deltas and
    prevents promotion; its outcome is still reported in the side summaries.

    Promotion requires at least ``min_pairs`` clean pairs and either a positive
    lower 95% win-delta bound, or a positive lower 95% floor-delta bound with a
    nonnegative lower win-delta bound.  Intervals are percentile-bootstrap
    estimates, not guarantees, and test seeds must remain held out of training
    and checkpoint selection.  The caller must freeze the game/difficulty and
    evaluation budget; matching seeds alone does not enforce those conditions.
    """
    _integer(bootstrap_samples, "bootstrap_samples", 100)
    _integer(bootstrap_seed, "bootstrap_seed")
    _integer(min_pairs, "min_pairs", 1)
    candidate_rows = _episodes(candidate)
    baseline_rows = _episodes(baseline)
    candidate_by_seed = _by_seed(candidate_rows, "candidate")
    baseline_by_seed = _by_seed(baseline_rows, "baseline")
    if candidate_by_seed.keys() != baseline_by_seed.keys():
        missing_candidate = len(baseline_by_seed.keys() - candidate_by_seed.keys())
        missing_baseline = len(candidate_by_seed.keys() - baseline_by_seed.keys())
        raise ValueError(
            "seed sets must match exactly: "
            f"{missing_candidate} missing candidate, {missing_baseline} missing baseline"
        )

    win_deltas: list[float] = []
    floor_deltas: list[float] = []
    excluded_seeds: list[int] = []
    # Sorting ensures shuffled journal rows do not change bootstrap results.
    for seed in sorted(candidate_by_seed):
        current = candidate_by_seed[seed]
        previous = baseline_by_seed[seed]
        if current["outcome"] not in _GAME_OUTCOMES or previous["outcome"] not in _GAME_OUTCOMES:
            excluded_seeds.append(seed)
            continue
        win_deltas.append(float(current["outcome"] == "win") - float(previous["outcome"] == "win"))
        floor_deltas.append(current["floor"] - previous["floor"])

    win_interval, floor_interval = _paired_bootstrap(
        win_deltas, floor_deltas, bootstrap_samples, bootstrap_seed
    )
    reasons: list[str] = []
    if excluded_seeds:
        reasons.append(f"{len(excluded_seeds)} pairs contain timeout or aborted outcomes")
    if len(win_deltas) < min_pairs:
        reasons.append(f"requires at least {min_pairs} clean pairs; observed {len(win_deltas)}")
    improves_wins = win_interval is not None and win_interval[0] > 0.0
    improves_floors_without_win_regression = (
        floor_interval is not None
        and floor_interval[0] > 0.0
        and win_interval is not None
        and win_interval[0] >= 0.0
    )
    if not (improves_wins or improves_floors_without_win_regression):
        reasons.append("95% intervals do not establish the required improvement")
    return {
        "pairs": len(candidate_by_seed),
        "eligible_pairs": len(win_deltas),
        "excluded_pairs": len(excluded_seeds),
        "excluded_seeds": excluded_seeds,
        "candidate": summarize(candidate_rows),
        "baseline": summarize(baseline_rows),
        "win_delta": _mean(win_deltas),
        "win_delta_ci95": win_interval,
        "floor_delta": _mean(floor_deltas),
        "floor_delta_ci95": floor_interval,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "min_pairs": min_pairs,
        "promote": not reasons,
        "reasons": reasons,
        "promotion_rule": (
            "No timeouts/aborts, sufficient clean pairs, and lower 95% win delta > 0; "
            "or lower 95% floor delta > 0 with lower 95% win delta >= 0."
        ),
    }
