"""Self-improvement loop: play, learn, evaluate, repeat.

    python -m sts2rl.train --name main --iterations 50 --workers 8

Each iteration:
  1. self-play `--runs-per-iter` runs with the current value net (plus exploration)
  2. append them to the replay buffer (one compressed shard per iteration)
  3. train the value net on the last `--replay-iters` shards, TD(lambda) targets
  4. evaluate greedily on the validation seeds and run a paired test against the current
     champion on those same seeds; the candidate becomes champion (best.pt) only if its
     score gain is statistically significant and it has no significant drop in Act 1
     clears or wins

The first champion is the checkpoint the run starts from (an existing best.pt, or a
latest.pt put there to continue from), else the untrained heuristic. Everything lands in
runs/<name>/: shards, latest.pt, best.pt (the champion), champion.json (its per-seed
validation results, valid only for the same evaluation settings, simulator build, agent
code and champion file), metrics.jsonl. Re-running the same command resumes where it
stopped, finishing a promotion test an interruption cut short. The test split is never
touched here.

Each test is at --gate-alpha on its own; over many iterations an occasional false
promotion is expected, so check a final champion on the test split (sts2rl.compare).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from . import paths, sim
from .agent import AgentConfig
from .dataset import Replay, append_jsonl, save_shard
from .evaluate import format_summary, per_seed, run_eval, split_offset, summarize
from .stats import compare_paired, promotion_decision
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
    ap.add_argument("--gate-alpha", type=float, default=0.05, help="significance level a new champion must reach")
    ap.add_argument("--selfplay-model", choices=["latest", "champion"], default="latest",
                    help="which agent plays self-play games: the newest network, or the current champion")
    args = ap.parse_args()

    out = paths.RUNS_DIR / args.name
    shards = out / "shards"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    latest, best = out / "latest.pt", out / "best.pt"
    metrics_path = out / "metrics.jsonl"

    champion_path = out / "champion.json"
    split_offset("validation", args.eval_runs)  # refuses a validation set that would overlap the test split

    done = sorted(shards.glob("iter_*.npz"))
    start = len(done)
    model = ValueNet.load(latest) if latest.exists() else ValueNet(hidden=args.hidden)
    print(f"[{args.name}] starting at iteration {start}")

    sp_cfg = AgentConfig(combat_sims=args.sims, samples=args.samples, epsilon=args.epsilon, max_act=args.max_act)
    ev_cfg = AgentConfig(combat_sims=args.eval_sims, samples=args.samples, max_act=args.max_act)
    eval_key = {"runs": args.eval_runs, "sims": args.eval_sims, "samples": args.samples, "max_act": args.max_act,
                "code": code_fingerprint()}
    rng = random.Random(1234 + start)

    with Pool(args.workers) as pool:
        champion = load_champion(champion_path, eval_key, best)
        if champion is None:
            # The checkpoint the run starts from is the first champion: an existing best.pt,
            # or a latest.pt placed in a fresh run to continue from (models/README.md).
            # Without one, the untrained heuristic. Its per-seed validation results are
            # what every candidate is paired against.
            if not best.exists() and latest.exists() and start == 0:
                shutil.copy(latest, best)
            who = best if best.exists() else None
            t = time.perf_counter()
            summary, recs = run_eval(who, args.eval_runs, ev_cfg, args.workers, pool=pool)
            iteration = int(ValueNet.load(best).meta.get("iteration", -1)) if who else -1
            name = f"best.pt (iteration {iteration})" if who else "heuristic"
            print(f"[champion: {name}] {format_summary(summary)}  ({time.perf_counter() - t:.0f}s)")
            champion = {"model": "best.pt" if who else "heuristic", "model_sha": file_sha(best) if who else None,
                        "iteration": iteration, "eval_key": eval_key, "summary": summary, "rows": per_seed(recs)}
            save_champion(champion_path, champion)
            append_jsonl(metrics_path, {"iter": -1, "eval": summary, "agent": name, "start_champion": True})
        print(f"[{args.name}] champion: {champion['model']} (iteration {champion['iteration']}), "
              f"validation score {champion['summary']['score']:.3f}")

        def gate(it: int, row: dict) -> str:
            """Evaluate latest.pt on the validation seeds and promote it if it beats the champion."""
            nonlocal champion
            t2 = time.perf_counter()
            ev, recs = run_eval(latest, args.eval_runs, ev_cfg, args.workers, pool=pool)
            rows = per_seed(recs)
            cmp = compare_paired(champion["rows"], rows)
            promote, why = promotion_decision(cmp, alpha=args.gate_alpha)
            row["eval"] = ev
            row["gate"] = {"vs_champion_iteration": champion["iteration"], "vs_champion": champion["model"],
                           "promoted": promote, "reason": why, **cmp}
            row["t_eval"] = time.perf_counter() - t2
            if promote:
                # Champion record first, then the file: an interruption in between leaves a
                # record whose model_sha no longer matches best.pt, which is re-evaluated.
                tmp = best.with_suffix(".pt.tmp")
                shutil.copy(latest, tmp)
                champion = {"model": "best.pt", "model_sha": file_sha(tmp), "iteration": it, "eval_key": eval_key,
                            "summary": ev, "rows": rows}
                save_champion(champion_path, champion)
                tmp.replace(best)
            verdict = "NEW CHAMPION" if promote else f"champion stays (iter {row['gate']['vs_champion_iteration']})"
            return f"\n           eval: {format_summary(ev)}\n           vs champion: {verdict} -- {why}"

        # An interruption during the last iteration's promotion test: its candidate is
        # still latest.pt, so finish the test before playing on.
        prev = start - 1
        logged = {json.loads(line)["iter"] for line in metrics_path.read_text().splitlines()} if metrics_path.exists() else set()
        if (prev >= 0 and prev not in logged and (prev + 1) % args.eval_every == 0 and latest.exists()
                and ValueNet.load(latest).meta.get("iteration") == prev):
            row = {"iter": prev, "resumed_gate": True}
            line = f"[iter {prev:3d}] promotion test, resumed" + gate(prev, row)
            append_jsonl(metrics_path, row)
            print(line, flush=True)

        for it in range(start, start + args.iterations):
            t0 = time.perf_counter()
            if args.selfplay_model == "champion":
                model_path = best if champion["model"] != "heuristic" else None
            else:
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
                line += gate(it, row)
            append_jsonl(metrics_path, row)
            print(line, flush=True)


def code_fingerprint() -> str:
    """What evaluation results depend on besides the settings: the native simulator build
    and the agent's code. Cached champion results are reused only while it is unchanged."""
    h = hashlib.sha256(Path(sim._lib._name).read_bytes())
    for name in ("agent", "combat", "features", "sim", "model", "gamedata", "evaluate", "selfplay"):
        h.update((Path(__file__).parent / f"{name}.py").read_bytes())
    return h.hexdigest()[:16]


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_champion(path, eval_key: dict, best) -> dict | None:
    """The champion's cached per-seed validation results, if they are still comparable.

    Evaluation is deterministic per (model, seed, settings, simulator and agent code), so
    the cache stays valid while all of those do -- including the champion file itself;
    otherwise the champion is simply re-evaluated.
    """
    if not path.exists():
        return None
    champion = json.loads(path.read_text())
    if champion.get("eval_key") != eval_key:
        return None
    if champion.get("model") != "heuristic" and (not best.exists() or champion.get("model_sha") != file_sha(best)):
        return None
    return champion


def save_champion(path, champion: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(champion, indent=1))
    tmp.replace(path)


if __name__ == "__main__":
    main()
