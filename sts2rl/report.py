"""Learning curve for a training run.

    python -m sts2rl.report runs/main
    python -m sts2rl.report runs/main --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(run_dir: Path) -> list[dict]:
    out = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "eval" in row:
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    data = rows(args.run_dir)
    # Count self-play runs from the data shards: an iteration interrupted after its games
    # were saved has no metrics row but its runs still trained the network.
    shard_runs = {}
    states = 0
    for shard in sorted((args.run_dir / "shards").glob("iter_*.npz")):
        with np.load(shard) as z:
            shard_runs[int(shard.stem.split("_")[1])] = len(np.unique(z["run_id"]))
            states += len(z["score"])
    header = ["iter", "self-play runs", "eval score", "act 1 clear", "act 2 clear", "full win", "mean floor"]
    lines = []
    for row in data:
        ev = row["eval"]
        it = row["iter"]
        runs = "—" if it < 0 else str(sum(n for i, n in shard_runs.items() if i <= it))
        name = "heuristic (no training)" if it < 0 else str(it)
        lo, hi = ev["win_ci95"]
        lines.append([
            name, runs, f"{ev['score']:.3f}", f"{ev.get('act1_clear', 0):.0%}", f"{ev.get('act2_clear', 0):.0%}",
            f"{ev['win_rate']:.0%} [{lo:.0%}–{hi:.0%}]", f"{ev['mean_floor']:.1f}",
        ])
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for line in lines:
            print("| " + " | ".join(line) + " |")
    else:
        widths = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(header)] if lines else [len(h) for h in header]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for line in lines:
            print("  ".join(c.ljust(w) for c, w in zip(line, widths)))
    if data:
        best = max(data, key=lambda r: r["eval"]["score"])
        print(f"\nbest: iteration {best['iter']} (score {best['eval']['score']:.3f}); {states} training states recorded")


if __name__ == "__main__":
    main()
