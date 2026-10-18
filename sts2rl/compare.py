"""Paired comparison of two agents on identical seeds.

    python -m sts2rl.compare --a heuristic --b models/best.pt --runs 128 --split test
    python -m sts2rl.compare --a models/best.pt --b runs/main/latest.pt --runs 200

Both agents play the same seeds with the same settings; evaluation is deterministic, so
the difference between them is not drowned in seed-to-seed luck. Reports the paired
score difference (bootstrap CI, one-sided permutation p-value), Act 1 clears and wins
(exact McNemar), and whether B would pass the training loop's promotion gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .agent import AgentConfig
from .evaluate import SPLITS, format_summary, per_seed, run_eval, split_offset
from .selfplay import Pool
from .stats import compare_paired, promotion_decision


def _model(spec: str):
    return None if spec == "heuristic" else Path(spec)


def report(cmp: dict, a: str, b: str) -> str:
    lo, hi = cmp["score_diff_ci95"]
    flo, fhi = cmp["floor_diff_ci95"]
    m1, mw = cmp["act1_mcnemar"], cmp["wins_mcnemar"]
    return "\n".join([
        f"{cmp['pairs']} paired seeds   A = {a}   B = {b}",
        f"  run score     A {cmp['score_a']:.3f}   B {cmp['score_b']:.3f}   B-A {cmp['score_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]   p(B>A) = {cmp['score_p_b_better']:.4f}",
        f"  floor reached B-A {cmp['floor_diff']:+.2f} [{flo:+.2f}, {fhi:+.2f}]",
        f"  Act 1 clears  A {cmp['act1_a']}   B {cmp['act1_b']}   ({m1['discordant']} seeds differ; exact McNemar two-sided p = {m1['p_two_sided']:.4f})",
        f"  wins          A {cmp['wins_a']}   B {cmp['wins_b']}   (two-sided p = {mw['p_two_sided']:.4f})",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline: a checkpoint path, or 'heuristic'")
    ap.add_argument("--b", required=True, help="candidate: a checkpoint path, or 'heuristic'")
    ap.add_argument("--runs", type=int, default=128)
    ap.add_argument("--split", choices=sorted(SPLITS), default="validation")
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-act", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", help="write both evaluations and the comparison here")
    args = ap.parse_args()

    cfg = AgentConfig(combat_sims=args.sims, samples=args.samples, max_act=args.max_act)
    offset = split_offset(args.split, args.runs)
    t = time.perf_counter()
    with Pool(args.workers) as pool:
        sa, ra = run_eval(_model(args.a), args.runs, cfg, args.workers, offset, pool=pool)
        sb, rb = run_eval(_model(args.b), args.runs, cfg, args.workers, offset, pool=pool)
    cmp = compare_paired(per_seed(ra), per_seed(rb))
    promote, why = promotion_decision(cmp, alpha=args.alpha)
    print(f"A: {format_summary(sa)}")
    print(f"B: {format_summary(sb)}")
    print(report(cmp, args.a, args.b))
    print(f"  promotion gate (alpha {args.alpha}): {'B WOULD BE PROMOTED' if promote else 'B would not be promoted'} -- {why}")
    print(f"wall time {time.perf_counter() - t:.0f}s on the {args.split} split")
    if args.json:
        Path(args.json).write_text(json.dumps({"split": args.split, "a": args.a, "b": args.b, "summary_a": sa,
                                               "summary_b": sb, "comparison": cmp, "promote": promote, "reason": why}, indent=1))


if __name__ == "__main__":
    main()
