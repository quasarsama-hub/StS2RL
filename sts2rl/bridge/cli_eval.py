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
import shlex
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import paths
from ..agent import AgentConfig
from ..evaluate import SPLITS, eval_seeds, split_offset
from ..stats import wilson


def _play(task: dict) -> dict:
    from ..agent import Agent
    from ..selfplay import _load_model
    from . import cli

    model = _load_model(task["model"])
    agent = Agent(model, AgentConfig(**task["cfg"]), seed=task["agent_seed"])
    out = Path(task["out"])
    seed = task["seed"]
    try:
        with cli.CliGame(task["command"], cwd=task["cwd"], timeout=task["timeout"], stderr_path=out / f"{seed}.stderr.log") as game:
            result = cli.CliSession(game, agent, cli.Names(task["loc"]), seed, trace_path=out / f"{seed}.jsonl").play()
        return result.to_dict()
    except Exception as exc:  # sts2-cli would not start, or died outright
        return cli.CliRunResult(seed=seed, in_sync=False, engine_error=f"{type(exc).__name__}: {exc}").to_dict()


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def reason_kind(reason: str) -> str:
    """'hp: game=3 sim=5; gold: ...' -> 'hp, gold'."""
    return ", ".join(sorted({part.split(":")[0].strip() for part in reason.split(";") if part.strip()}))


def summarize(rows: list[dict], max_act: int) -> dict:
    n = len(rows)
    errors = [r for r in rows if r["engine_error"]]
    desyncs = [r for r in rows if not r["in_sync"] and not r["engine_error"]]
    synced = [r for r in rows if r["in_sync"] and r["finished"]]
    k = len(synced)
    wins = sum(r["won"] for r in synced)
    act1 = sum(r["acts_cleared"] >= 1 for r in synced)
    return {
        "runs": n,
        "in_sync": k,
        "desyncs": len(desyncs),
        "engine_errors": len(errors),
        "desync_floors": sorted(r["desync_floor"] for r in desyncs),
        "desync_reasons": dict(Counter(reason_kind(r["desync_reason"]) for r in desyncs).most_common()),
        "error_samples": [r["engine_error"] for r in errors[:3]],
        "synced_win_rate": wins / k if k else 0.0,
        "synced_win_ci95": wilson(wins, k),
        "synced_act1_clear": act1 / k if k else 0.0,
        "synced_act1_ci95": wilson(act1, k),
        "synced_mean_floor": statistics.mean(r["final_floor"] for r in synced) if synced else 0.0,
        "synced_score": statistics.mean(r["score"] for r in synced) if synced else 0.0,
        "max_act": max_act,
        "seconds_per_run": statistics.mean(r["seconds"] for r in rows) if rows else 0.0,
    }


def format_report(s: dict) -> str:
    lines = [
        f"{s['runs']} runs on the real engine:  {s['in_sync']} in sync to the end, "
        f"{s['desyncs']} desynced, {s['engine_errors']} engine errors   ({s['seconds_per_run']:.0f}s/run)",
    ]
    if s["desyncs"]:
        floors = s["desync_floors"]
        lines.append(f"  first desync floor: median {statistics.median(floors):.0f}, range {floors[0]}-{floors[-1]}")
        for kind, count in s["desync_reasons"].items():
            lines.append(f"    {count:3d}  {kind}")
    if s["error_samples"]:
        lines.append(f"  engine errors, e.g.: {s['error_samples'][0]}")
    if s["in_sync"]:
        lo, hi = s["synced_act1_ci95"]
        wlo, whi = s["synced_win_ci95"]
        lines.append(
            f"  in-sync runs: Act 1 cleared {s['synced_act1_clear']:.0%} [{lo:.0%}, {hi:.0%}]   "
            f"won {s['synced_win_rate']:.0%} [{wlo:.0%}, {whi:.0%}]   mean floor {s['synced_mean_floor']:.1f}"
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
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--samples", type=int, default=2)
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
         "cwd": cwd, "loc": loc, "timeout": args.timeout, "out": str(out)}
        for i, seed in enumerate(eval_seeds(args.runs, offset))
    ]
    t = time.perf_counter()
    rows = []
    # spawn, not fork: a forked child inherits the parent's loaded native runtime.
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as ex:
        for row in ex.map(_play, tasks):
            rows.append(row)
            status = "in sync" if row["in_sync"] else (f"DESYNC floor {row['desync_floor']}: {row['desync_reason']}" if not row["engine_error"] else f"ERROR {row['engine_error']}")
            print(f"{row['seed']}  floor {row['final_floor']:2d}  {'WON' if row['won'] else '   '}  {status[:160]}", flush=True)
    summary = summarize(rows, args.max_act)
    print()
    print(format_report(summary))
    print(f"wall time {time.perf_counter() - t:.0f}s; traces in {out}")
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "summary": summary, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
