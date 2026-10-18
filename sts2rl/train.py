"""Self-improvement loop: play, learn, evaluate, repeat.

    python -m sts2rl.train --name main --iterations 50 --workers 8

Each iteration:
  1. self-play `--runs-per-iter` runs with the current value net (plus exploration)
  2. append them to the replay buffer (one compressed shard per iteration)
  3. train the value net on the last `--replay-iters` shards, TD(lambda) targets
  4. evaluate greedily on fixed seeds; keep the best checkpoint by score

Everything lands in runs/<name>/: shards, latest.pt, best.pt, metrics.jsonl.
Re-running the same command resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from . import paths
from .agent import AgentConfig
from .dataset import Replay, append_jsonl, save_shard
from .evaluate import format_summary, run_eval, summarize
from .model import ValueNet
from .selfplay import Pool


def train_value(model: ValueNet, replay: Replay, *, epochs: int, batch: int, lr: float, lam: float, seed: int) -> dict:
    torch.manual_seed(seed)
    x = replay.x
    n = len(replay)
    # Current predictions for every stored state, for the bootstrapped part of the targets.
    values = np.concatenate([model.predict(x.take(slice(i, i + 4096))) for i in range(0, n, 4096)])
    target = replay.td_targets(values, lam).astype(np.float32)
    won = replay.data["won"]
    floor = replay.data["floor"]

    # Hold out ~5% of RUNS (not states) so validation measures generalisation.
    runs = np.unique(replay.data["run_id"])
    rng = np.random.default_rng(seed)
    held = set(rng.choice(runs, size=max(1, len(runs) // 20), replace=False).tolist())
    is_val = np.array([r in held for r in replay.data["run_id"]])
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_idx) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch + 1, pct_start=0.1)

    def loss_on(idx):
        xb = model.tensors(x.take(idx))
        out = model(*xb)
        t_score = torch.as_tensor(target[idx])
        t_won = torch.as_tensor(won[idx])
        t_floor = torch.as_tensor(floor[idx])
        l_score = Fn.binary_cross_entropy_with_logits(out[:, 0], t_score)
        l_win = Fn.binary_cross_entropy_with_logits(out[:, 1], t_won)
        l_floor = Fn.binary_cross_entropy_with_logits(out[:, 2], t_floor)
        return l_score + 0.5 * l_win + 0.5 * l_floor, l_score

    model.train()
    for _ in range(epochs):
        perm = rng.permutation(train_idx)
        for i in range(steps_per_epoch):
            idx = perm[i * batch : (i + 1) * batch]
            loss, _ = loss_on(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        _, val_score_loss = loss_on(val_idx)
        # Baseline: predicting the mean target. The model must beat this to be learning.
        mean_t = float(target[train_idx].mean())
        p = torch.full((len(val_idx),), mean_t)
        base = Fn.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), torch.as_tensor(target[val_idx]))
    return {"states": int(n), "val_loss": float(val_score_loss), "val_baseline": float(base)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--runs-per-iter", type=int, default=200)
    ap.add_argument("--eval-runs", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=32, help="combat simulations per move in self-play")
    ap.add_argument("--eval-sims", type=int, default=64)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--max-act", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--replay-iters", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    out = paths.RUNS_DIR / args.name
    shards = out / "shards"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    latest, best = out / "latest.pt", out / "best.pt"
    metrics_path = out / "metrics.jsonl"

    done = sorted(shards.glob("iter_*.npz"))
    start = len(done)
    best_score = -1.0
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if "eval" in row:
                best_score = max(best_score, row["eval"]["score"])
    model = ValueNet.load(latest) if latest.exists() else ValueNet(hidden=args.hidden)
    print(f"[{args.name}] starting at iteration {start}; best eval score so far {best_score:.3f}")

    sp_cfg = AgentConfig(combat_sims=args.sims, samples=args.samples, epsilon=args.epsilon, max_act=args.max_act)
    ev_cfg = AgentConfig(combat_sims=args.eval_sims, samples=args.samples, max_act=args.max_act)
    rng = random.Random(1234 + start)

    with Pool(args.workers) as pool:
        if start == 0 and not best.exists():
            t = time.perf_counter()
            summary, _ = run_eval(None, args.eval_runs, ev_cfg, args.workers, pool=pool)
            print(f"[baseline, untrained heuristic] {format_summary(summary)}  ({time.perf_counter() - t:.0f}s)")
            append_jsonl(metrics_path, {"iter": -1, "eval": summary, "agent": "heuristic"})

        for it in range(start, start + args.iterations):
            t0 = time.perf_counter()
            model_path = latest if latest.exists() else None
            seeds = ["".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(10)) for _ in range(args.runs_per_iter)]
            records = list(pool.play(seeds, sp_cfg, model_path, base_seed=it * 100_000))
            sp = summarize(records, args.max_act)
            n_states = save_shard(shards / f"iter_{it:04d}.npz", records)
            t_play = time.perf_counter() - t0

            t1 = time.perf_counter()
            window = sorted(shards.glob("iter_*.npz"))[-args.replay_iters :]
            replay = Replay(window)
            tr = train_value(model, replay, epochs=args.epochs, batch=args.batch, lr=args.lr, lam=args.lam, seed=it)
            model.save(latest, iteration=it, args=vars(args))
            t_train = time.perf_counter() - t1

            row = {"iter": it, "selfplay": sp, "train": tr, "new_states": n_states, "t_play": t_play, "t_train": t_train}
            line = (
                f"[iter {it:3d}] self-play: score {sp['score']:.3f} win {sp['win_rate']:.1%} floor {sp['mean_floor']:.1f} "
                f"| train: {tr['states']} states, val {tr['val_loss']:.4f} vs baseline {tr['val_baseline']:.4f} "
                f"| {t_play:.0f}s play, {t_train:.0f}s train"
            )
            if (it + 1) % args.eval_every == 0:
                t2 = time.perf_counter()
                ev, _ = run_eval(latest, args.eval_runs, ev_cfg, args.workers, pool=pool)
                row["eval"] = ev
                row["t_eval"] = time.perf_counter() - t2
                improved = ev["score"] > best_score
                if improved:
                    best_score = ev["score"]
                    shutil.copy(latest, best)
                line += f"\n           eval: {format_summary(ev)}{'  << new best' if improved else ''}"
            append_jsonl(metrics_path, row)
            print(line, flush=True)


if __name__ == "__main__":
    main()
