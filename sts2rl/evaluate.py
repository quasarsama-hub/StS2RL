"""Evaluate an agent on a fixed set of seeds.

    python -m sts2rl.evaluate --agent heuristic --runs 64
    python -m sts2rl.evaluate --model runs/main/best.pt --runs 200 --sims 128 --split test

Seeds come in two disjoint splits:

  validation  EVAL000000-EVAL000999  what training uses to pick checkpoints
  test        EVAL001000 onwards     final assessment only; nothing is ever selected on it

Same seeds every time and evaluation is deterministic, so two checkpoints are compared
on identical runs (see sts2rl.compare for the paired statistics).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .agent import AgentConfig, RunRecord
from .selfplay import Pool
from .stats import wilson

SPLITS = {"validation": 0, "test": 1000}
VALIDATION_SIZE = SPLITS["test"] - SPLITS["validation"]


def eval_seeds(n: int, offset: int = 0) -> list[str]:
    return [f"EVAL{i:06d}" for i in range(offset, offset + n)]


def split_offset(split: str, n: int) -> int:
    offset = SPLITS[split]
    if split == "validation" and n > VALIDATION_SIZE:
        raise ValueError(f"validation has {VALIDATION_SIZE} seeds; {n} would run into the test split")
    return offset


def per_seed(records: list[RunRecord]) -> dict:
    """seed -> the outcome fields paired comparisons use. Errored runs are left out."""
    return {
        r.seed: {"score": r.score, "floor": r.final_floor, "act1": r.acts_cleared >= 1, "won": bool(r.won)}
        for r in records
        if not r.death_encounter.startswith("ERROR")
    }


def summarize(records: list[RunRecord], max_act: int) -> dict:
    ok = [r for r in records if not r.death_encounter.startswith("ERROR")]
    n = len(ok)
    wins = sum(r.won for r in ok)
    out = {
        "runs": n,
        "errors": len(records) - n,
        "score": statistics.mean(r.score for r in ok) if ok else 0.0,
        "win_rate": wins / n if n else 0.0,
        "win_ci95": wilson(wins, n),
        "mean_floor": statistics.mean(r.final_floor for r in ok) if ok else 0.0,
        "seconds_per_run": statistics.mean(r.seconds for r in ok) if ok else 0.0,
    }
    for a in range(1, max_act + 1):
        k = sum(r.acts_cleared >= a for r in ok)
        out[f"act{a}_clear"] = k / n if n else 0.0
    out["deaths"] = dict(Counter(r.death_encounter for r in ok if not r.won).most_common(8))
    errors = [r.death_encounter for r in records if r.death_encounter.startswith("ERROR")]
    if errors:
        out["error_samples"] = errors[:3]
    return out


def run_eval(model_path, n_runs: int, cfg: AgentConfig, workers: int, offset: int = 0, pool: Pool | None = None) -> tuple[dict, list[RunRecord]]:
    cfg = AgentConfig(**{**cfg.__dict__, "epsilon": 0.0, "record": False})
    seeds = eval_seeds(n_runs, offset)
    if pool is not None:
        records = list(pool.play(seeds, cfg, model_path, base_seed=7_000_000))
    else:
        with Pool(workers) as own:
            records = list(own.play(seeds, cfg, model_path, base_seed=7_000_000))
    records.sort(key=lambda r: r.seed)
    return summarize(records, cfg.max_act), records


def format_summary(s: dict) -> str:
    lo, hi = s["win_ci95"]
    acts = "  ".join(f"act{a}:{s[f'act{a}_clear']:.0%}" for a in range(1, 4) if f"act{a}_clear" in s)
    return (
        f"score {s['score']:.3f}  win {s['win_rate']:.1%} [{lo:.1%},{hi:.1%}]  {acts}  "
        f"floor {s['mean_floor']:.1f}  ({s['runs']} runs, {s['seconds_per_run']:.1f}s/run)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="value-net checkpoint (.pt)")
    ap.add_argument("--agent", choices=["model", "heuristic"], default="model")
    ap.add_argument("--runs", type=int, default=64)
    ap.add_argument("--split", choices=sorted(SPLITS), default="validation")
    ap.add_argument("--offset", type=int, help="first eval seed index (overrides --split)")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-act", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 0))
    ap.add_argument("--json", help="write the summary and per-run results here")
    args = ap.parse_args()

    model_path = None
    if args.agent == "model":
        if not args.model:
            ap.error("--model is required unless --agent heuristic")
        model_path = Path(args.model)
    cfg = AgentConfig(combat_sims=args.sims, samples=args.samples, max_act=args.max_act)
    t = time.perf_counter()
    offset = args.offset if args.offset is not None else split_offset(args.split, args.runs)
    summary, records = run_eval(model_path, args.runs, cfg, args.workers, offset)
    print(format_summary(summary))
    print("most common deaths:", summary["deaths"])
    if summary.get("errors"):
        print("ERRORS:", summary["errors"], summary.get("error_samples"))
    print(f"wall time {time.perf_counter() - t:.0f}s")
    if args.json:
        rows = [{"seed": r.seed, "won": r.won, "floor": r.final_floor, "score": r.score, "death": r.death_encounter, "deck": r.deck, "relics": r.relics} for r in records]
        Path(args.json).write_text(json.dumps({"summary": summary, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
