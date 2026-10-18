"""Paired statistics for comparing two agents on the same seeds.

Every comparison here is PAIRED: both agents play identical seeds, and evaluation is
deterministic per (model, seed, config), so seed-to-seed luck cancels and a much smaller
difference becomes detectable than with two independent samples.
"""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _binom_cdf(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2), exactly."""
    if k < 0:
        return 0.0
    return sum(math.comb(n, i) for i in range(0, min(k, n) + 1)) / 2**n


def mcnemar_exact(a_only: int, b_only: int) -> dict:
    """Exact McNemar test on discordant pairs.

    `a_only`: seeds where only A succeeded; `b_only`: only B. Returns the two-sided
    p-value and the one-sided p-value for "B succeeds more often than A".
    """
    n = a_only + b_only
    if n == 0:
        return {"discordant": 0, "p_two_sided": 1.0, "p_b_better": 1.0}
    two = min(1.0, 2 * _binom_cdf(min(a_only, b_only), n))
    b_better = 1.0 - _binom_cdf(b_only - 1, n)  # P(X >= b_only)
    return {"discordant": n, "p_two_sided": two, "p_b_better": b_better}


def sign_flip_pvalue(diffs: np.ndarray, resamples: int = 20000, seed: int = 0) -> float:
    """One-sided paired permutation test of mean(diffs) > 0 (random sign flips)."""
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    observed = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(resamples, len(d)))
    null = (signs * np.abs(d)).mean(axis=1)
    # +1 smoothing: the observed labelling is one of the permutations.
    return float((np.sum(null >= observed - 1e-12) + 1) / (resamples + 1))


def bootstrap_ci(diffs: np.ndarray, resamples: int = 10000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def compare_paired(a: dict, b: dict) -> dict:
    """Compare per-seed results of agent A (baseline) and agent B (candidate).

    `a` and `b` map seed -> {"score", "floor", "act1", "won"}. Only seeds both played
    count; a seed missing from either (an engine error) is reported, not imputed.
    """
    seeds = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
    if not seeds:
        return {"pairs": 0, "missing": missing}
    sa = np.array([a[s]["score"] for s in seeds])
    sb = np.array([b[s]["score"] for s in seeds])
    fa = np.array([a[s]["floor"] for s in seeds], dtype=float)
    fb = np.array([b[s]["floor"] for s in seeds], dtype=float)
    ca = np.array([bool(a[s]["act1"]) for s in seeds])
    cb = np.array([bool(b[s]["act1"]) for s in seeds])
    wa = np.array([bool(a[s]["won"]) for s in seeds])
    wb = np.array([bool(b[s]["won"]) for s in seeds])
    d_score = sb - sa
    return {
        "pairs": len(seeds),
        "missing": missing,
        "score_a": float(sa.mean()),
        "score_b": float(sb.mean()),
        "score_diff": float(d_score.mean()),
        "score_diff_ci95": bootstrap_ci(d_score),
        "score_p_b_better": sign_flip_pvalue(d_score),
        "floor_diff": float((fb - fa).mean()),
        "floor_diff_ci95": bootstrap_ci(fb - fa),
        "act1_a": int(ca.sum()),
        "act1_b": int(cb.sum()),
        "act1_mcnemar": mcnemar_exact(int((ca & ~cb).sum()), int((cb & ~ca).sum())),
        "wins_a": int(wa.sum()),
        "wins_b": int(wb.sum()),
        "wins_mcnemar": mcnemar_exact(int((wa & ~wb).sum()), int((wb & ~wa).sum())),
    }


def promotion_decision(cmp: dict, alpha: float = 0.05, min_pairs: int = 32) -> tuple[bool, str]:
    """Should the candidate (B) replace the champion (A)?

    Primary endpoint: run score (progress, and a win counts fully), one-sided paired
    permutation test at `alpha`. Guard: no statistically significant drop in Act 1 clears
    or in wins (two-sided exact McNemar at `alpha`). A candidate that is merely not worse
    is not promoted -- that is how noise used to become "the best checkpoint".
    """
    if cmp.get("pairs", 0) < min_pairs:
        return False, f"only {cmp.get('pairs', 0)} complete pairs (need {min_pairs})"
    if cmp.get("missing"):
        return False, f"{len(cmp['missing'])} seeds missing from one side (engine errors)"
    for key, name in (("act1_mcnemar", "Act 1 clears"), ("wins_mcnemar", "wins")):
        m = cmp[key]
        a, b = (cmp["act1_a"], cmp["act1_b"]) if key == "act1_mcnemar" else (cmp["wins_a"], cmp["wins_b"])
        if b < a and m["p_two_sided"] < alpha:
            return False, f"significantly fewer {name} ({b} vs {a}, p={m['p_two_sided']:.3f})"
    if cmp["score_diff"] <= 0:
        return False, f"score not higher ({cmp['score_diff']:+.4f})"
    p = cmp["score_p_b_better"]
    if p >= alpha:
        return False, f"score gain {cmp['score_diff']:+.4f} not significant (p={p:.3f})"
    return True, f"score {cmp['score_diff']:+.4f} (p={p:.4f}), Act 1 clears {cmp['act1_a']}->{cmp['act1_b']}"
