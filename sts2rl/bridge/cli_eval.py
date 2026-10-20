"""Evaluate the agent on the REAL game engine, headless, through sts2-cli.

    python -m sts2rl.bridge.cli_eval --cli ~/sts2-cli --model models/best.pt --runs 32 --split test
    python -m sts2rl.bridge.cli_eval --cli ~/sts2-cli --agent heuristic --runs 100 --sims 16 --workers 8

Every run is played on the game's own code (your installed sts2.dll) with the simulator
as a lockstep shadow (see sts2rl.bridge.cli). Seeds and agent settings are those of
`sts2rl.evaluate`, so each run is paired with its simulator twin: a run that stays in
sync to the end finishes exactly as the simulator predicted, and a run that desyncs
reports the first floor and the exact reason the simulator stopped matching the game.

The report separates the two questions:

  fidelity     how many runs stay in sync, how far the rest get, and why they diverge
  performance  outcomes of the runs that stayed in sync (real-engine results)

Outputs go to --out: one JSONL trace per seed (every engine state, command, shadow
auto-step and the desync, if any) plus summary.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shlex
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .. import paths
from ..agent import AgentConfig
from ..evaluate import SPLITS, eval_seeds, split_offset
from ..stats import wilson


def _play(task: dict) -> dict:
    from ..agent import Agent
    from ..selfplay import _load_model
    from . import cli

    out = Path(task["out"])
    seed = task["seed"]
    (out / f"{seed}.started").touch()  # so a run that kills its worker can be found
    model = _load_model(task["model"])
    agent = Agent(model, AgentConfig(**task["cfg"]), seed=task["agent_seed"])
    try:
        with cli.CliGame(task["command"], cwd=task["cwd"], timeout=task["timeout"], stderr_path=out / f"{seed}.stderr.log") as game:
            result = cli.CliSession(game, agent, cli.Names(task["loc"]), seed, ascension=task["ascension"], trace_path=out / f"{seed}.jsonl").play()
        return result.to_dict()
    except Exception as exc:  # sts2-cli would not start, or died outright
        return cli.CliRunResult(seed=seed, in_sync=False, engine_error=f"{type(exc).__name__}: {exc}").to_dict()


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def reason_kind(reason: str) -> str:
    """Group desyncs by what disagreed: 'hp: game=3 sim=5; gold: game=1 sim=2' -> 'gold, hp'.
    Any other message is grouped with its numbers blanked ('map choice # at (#, #) ...')."""
    parts = [p.strip() for p in reason.split(";") if p.strip()]
    if parts and all(re.match(r"^[\w ()-]+: game=", p) for p in parts):
        return ", ".join(sorted({p.split(":")[0].strip() for p in parts}))
    return re.sub(r"\d+", "#", reason)[:90]


def summarize(rows: list[dict], max_act: int) -> dict:
    """Fidelity, then outcomes -- each outcome counted only over the runs whose shadow was
    still in sync when it was decided. (Counting only runs in sync to the very end would
    favour short runs: a run that dies early has less road on which to desync.)"""
    n = len(rows)
    engine_errors = [r for r in rows if r["engine_error"]]
    bench_errors = [r for r in rows if r.get("bench_error")]
    desyncs = [r for r in rows if not r["in_sync"] and not r["engine_error"] and not r.get("bench_error")]
    finished = [r for r in rows if r["in_sync"] and r["finished"]]
    acts = {}
    for a in range(1, max_act + 1):
        # Decided while in sync: the run finished in sync, or it had already cleared act a
        # when it stopped (a desync or an error ends a run, it does not undo its past).
        known = [r for r in rows if r in finished or r["acts_cleared"] >= a]
        cleared = sum(r["acts_cleared"] >= a for r in known)
        acts[a] = {"cleared": cleared, "known": len(known), "rate": cleared / len(known) if known else 0.0,
                   "ci95": wilson(cleared, len(known))}
    wins = sum(r["won"] for r in finished)
    return {
        "runs": n,
        "in_sync": len(finished),
        "desyncs": len(desyncs),
        "engine_errors": len(engine_errors),
        "bench_errors": len(bench_errors),
        "desync_floors": sorted(r["desync_floor"] for r in desyncs),
        "desync_reasons": dict(Counter(reason_kind(r["desync_reason"]) for r in desyncs).most_common()),
        "error_samples": [r["engine_error"] for r in engine_errors[:3]] + [r["bench_error"] for r in bench_errors[:3]],
        "acts": acts,
        "wins": wins,
        "win_rate": wins / len(finished) if finished else 0.0,
        "win_ci95": wilson(wins, len(finished)),
        "synced_mean_floor": statistics.mean(r["final_floor"] for r in finished) if finished else 0.0,
        "forfeits": sum(r.get("forfeits", 0) for r in rows),
        "restricted": sum(r.get("restricted", 0) for r in rows),
        "restricted_runs": sum(1 for r in rows if r.get("restricted", 0)),
        "max_act": max_act,
        "seconds_per_run": statistics.mean(r["seconds"] for r in rows) if rows else 0.0,
    }


def format_report(s: dict) -> str:
    lines = [
        f"{s['runs']} runs:  {s['in_sync']} in sync to the end, {s['desyncs']} desynced, "
        f"{s['engine_errors']} engine errors, {s['bench_errors']} bench errors   ({s['seconds_per_run']:.0f}s/run)",
    ]
    if s["desyncs"]:
        floors = s["desync_floors"]
        lines.append(f"  first desync floor: median {statistics.median(floors):.0f}, range {floors[0]}-{floors[-1]}")
        for kind, count in s["desync_reasons"].items():
            lines.append(f"    {count:3d}  {kind}")
    for sample in s["error_samples"][:2]:
        lines.append(f"  error, e.g.: {sample}")
    lines.append("  outcomes, each over the runs still in sync when it was decided:")
    for a, act in s["acts"].items():
        lo, hi = act["ci95"]
        lines.append(f"    Act {a} cleared  {act['cleared']}/{act['known']} = {act['rate']:.0%} [{lo:.0%}, {hi:.0%}]")
    lo, hi = s["win_ci95"]
    lines.append(f"    won            {s['wins']}/{s['in_sync']} = {s['win_rate']:.0%} [{lo:.0%}, {hi:.0%}]")
    if s["forfeits"]:
        lines.append(
            f"  {s['forfeits']} extra card rewards / rest options were discarded by sts2-cli (and on the shadow); "
            f"runs with one can end differently from their simulator twin"
        )
    if s["restricted"]:
        lines.append(
            f"  {s['restricted']} times (in {s['restricted_runs']} runs) the agent's first choice was a move the "
            f"game does not have; it took its best real one instead"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--cli", help="the sts2-cli checkout (set up by scripts/setup_sts2cli.sh)")
    where.add_argument("--command", help="launch command for an sts2-cli-compatible engine (testing)")
    ap.add_argument("--model", help="value-net checkpoint (.pt)")
    ap.add_argument("--agent", choices=["model", "heuristic"], default="model")
    ap.add_argument("--runs", type=int, default=32)
    ap.add_argument("--split", choices=sorted(SPLITS), default="test")
    ap.add_argument("--offset", type=int, help="first eval seed index (overrides --split)")
    # The same defaults as sts2rl.evaluate: a run is the twin of the simulator run with the
    # same seed, --sims and --samples.
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-act", type=int, default=3)
    ap.add_argument("--ascension", type=int, default=8, help="the simulator models A8 only")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for the engine's answer")
    ap.add_argument("--out", help="output directory (default runs/cli/<timestamp>)")
    args = ap.parse_args()

    if args.agent == "model" and not args.model:
        ap.error("--model is required unless --agent heuristic")
    if args.ascension != 8:
        print("warning: the simulator models Ascension 8; other levels desync at once.")
    from . import cli

    if args.cli:
        cli_dir = Path(args.cli).expanduser().resolve()
        command, cwd, loc = cli.cli_command(cli_dir), str(cli_dir), str(cli_dir / "localization_eng")
        if not (cli_dir / "lib" / "sts2.dll").exists():
            sys.exit(f"{cli_dir}/lib/sts2.dll not found -- run scripts/setup_sts2cli.sh first")
    else:
        command, cwd, loc = shlex.split(args.command), None, None
    out = Path(args.out) if args.out else paths.RUNS_DIR / "cli" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    cfg = AgentConfig(combat_sims=args.sims, samples=args.samples, max_act=args.max_act, epsilon=0.0, record=False)
    offset = args.offset if args.offset is not None else split_offset(args.split, args.runs)
    model = str(Path(args.model).resolve()) if args.agent == "model" else None
    tasks = [
        # Agent seeds as in sts2rl.evaluate, so each run is the twin of its simulator run.
        {"seed": seed, "agent_seed": 7_000_000 + i, "cfg": cfg.__dict__, "model": model, "command": command,
         "cwd": cwd, "loc": loc, "timeout": args.timeout, "out": str(out), "ascension": args.ascension}
        for i, seed in enumerate(eval_seeds(args.runs, offset))
    ]
    t = time.perf_counter()
    rows = run_all(tasks, args.workers)
    summary = summarize(rows, args.max_act)
    print()
    print(format_report(summary))
    print(f"wall time {time.perf_counter() - t:.0f}s; traces in {out}")
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "summary": summary, "runs": rows}, indent=1))


def run_all(tasks: list[dict], workers: int) -> list[dict]:
    """Play every task, surviving a worker process that dies (a native crash in the
    simulator or the engine). After a crash the runs that had started are re-run one at a
    time, which pins the one that kills its worker -- recorded as an error -- and the rest
    carry on as normal."""
    rows: dict[int, dict] = {}

    def started(i: int) -> bool:
        return (Path(tasks[i]["out"]) / f"{tasks[i]['seed']}.started").exists()

    while len(rows) < len(tasks):
        pending = [i for i in range(len(tasks)) if i not in rows]
        if not _run_pool(tasks, pending, workers, rows):
            continue
        suspects = [i for i in pending if i not in rows and started(i)] or [i for i in pending if i not in rows]
        for i in suspects:
            if _run_pool(tasks, [i], 1, rows):
                from .cli import CliRunResult

                rows[i] = CliRunResult(seed=tasks[i]["seed"], in_sync=False, engine_error="the run killed its worker process").to_dict()
                _print_row(rows[i])
    return [rows[i] for i in range(len(tasks))]


def _run_pool(tasks: list[dict], indices: list[int], workers: int, rows: dict) -> bool:
    """Run these tasks in one pool, filling `rows`. True if the pool broke."""
    # spawn, not fork: a forked child inherits the parent's loaded native runtime.
    with ProcessPoolExecutor(min(workers, len(indices)), mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        futures = {ex.submit(_play, tasks[i]): i for i in indices}
        try:
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    rows[i] = fut.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    from .cli import CliRunResult

                    rows[i] = CliRunResult(seed=tasks[i]["seed"], in_sync=False, bench_error=f"{type(exc).__name__}: {exc}").to_dict()
                _print_row(rows[i])
        except BrokenProcessPool:
            return True
    return False


def _print_row(row: dict) -> None:
    if row["in_sync"]:
        status = "in sync"
    elif row["engine_error"] or row.get("bench_error"):
        status = f"ERROR {row['engine_error'] or row['bench_error']}"
    else:
        status = f"DESYNC floor {row['desync_floor']}: {row['desync_reason']}"
    print(f"{row['seed']}  floor {row['final_floor']:2d}  {'WON' if row['won'] else '   '}  {status[:160]}", flush=True)


if __name__ == "__main__":
    main()
