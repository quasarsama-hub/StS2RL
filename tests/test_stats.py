"""The paired statistics behind checkpoint promotion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sts2rl.stats import compare_paired, mcnemar_exact, promotion_decision, sign_flip_pvalue, wilson


def test_mcnemar_exact_matches_binomial():
    # 0 vs 6 discordant pairs: two-sided p = 2 * (1/2)^6
    m = mcnemar_exact(0, 6)
    assert m["p_two_sided"] == pytest.approx(2 / 64)
    assert m["p_b_better"] == pytest.approx(1 / 64)
    # symmetric case
    assert mcnemar_exact(5, 5)["p_two_sided"] == pytest.approx(1.0)
    # 2 vs 8: P(X<=2 | n=10) = (1 + 10 + 45) / 1024
    assert mcnemar_exact(2, 8)["p_two_sided"] == pytest.approx(2 * 56 / 1024)
    assert mcnemar_exact(0, 0)["p_two_sided"] == 1.0


def test_sign_flip_detects_real_gain_and_ignores_noise():
    rng = np.random.default_rng(1)
    gain = rng.normal(0.03, 0.02, size=64)
    noise = rng.normal(0.0, 0.02, size=64)
    assert sign_flip_pvalue(gain) < 0.001
    assert sign_flip_pvalue(noise) > 0.05
    assert sign_flip_pvalue(np.zeros(10)) == 1.0


def test_wilson_interval():
    lo, hi = wilson(56, 128)
    assert lo < 56 / 128 < hi and 0.35 < lo < 0.36 and 0.52 < hi < 0.53
    assert wilson(0, 0) == (0.0, 0.0)


def rows(scores, act1=None, won=None):
    return {f"S{i}": {"score": float(s), "floor": 10 + 10 * s, "act1": bool(act1[i]) if act1 is not None else s > 0.2,
                      "won": bool(won[i]) if won is not None else False} for i, s in enumerate(scores)}


def test_gate_promotes_only_significant_gains():
    rng = np.random.default_rng(2)
    base = rng.uniform(0.1, 0.25, size=64)
    champion = rows(base)
    better = rows(base + rng.normal(0.02, 0.01, size=64))
    same = rows(base + rng.normal(0.0, 0.01, size=64))
    ok, why = promotion_decision(compare_paired(champion, better))
    assert ok, why
    ok, why = promotion_decision(compare_paired(champion, same))
    assert not ok


def test_gate_blocks_significant_act1_regression_even_with_higher_score():
    n = 64
    base = np.full(n, 0.17)
    act1_a = np.ones(n, dtype=bool)
    act1_b = np.ones(n, dtype=bool)
    act1_b[:12] = False  # B loses 12 Act 1 clears...
    cmp = compare_paired(rows(base, act1_a), rows(base + 0.01, act1_b))  # ...while scoring higher
    ok, why = promotion_decision(cmp)
    assert not ok and "Act 1" in why


def test_gate_handles_engine_errors():
    # The champion errored on a seed: its cached results are fixed, so that seed is left
    # out rather than blocking every later candidate.
    champion = rows(np.full(40, 0.1))
    del champion["S3"]
    ok, why = promotion_decision(compare_paired(champion, rows(np.full(40, 0.3))))
    assert ok, why
    # A candidate erroring on one seed in 40 is tolerated; on several it is refused.
    cand = rows(np.full(40, 0.3))
    del cand["S3"]
    assert promotion_decision(compare_paired(rows(np.full(40, 0.1)), cand))[0]
    for s in ("S4", "S5"):
        del cand[s]
    ok, why = promotion_decision(compare_paired(rows(np.full(40, 0.1)), cand))
    assert not ok and "engine errors" in why
    ok, why = promotion_decision(compare_paired(rows(np.full(8, 0.1)), rows(np.full(8, 0.3))))
    assert not ok and "pairs" in why


def test_compare_paired_counts():
    cmp = compare_paired(rows([0.1, 0.3, 0.3], won=[0, 0, 1]), rows([0.3, 0.1, 1.0], won=[0, 0, 1]))
    assert cmp["pairs"] == 3 and cmp["act1_a"] == 2 and cmp["act1_b"] == 2
    assert cmp["act1_mcnemar"]["discordant"] == 2
    assert math.isclose(cmp["score_diff"], (0.2 - 0.2 + 0.7) / 3)
