"""Command line entry point; starts no training or downloads on import."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Ironclad A0 learning experiments")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train new episodes or resume weights and optimizer")
    train.add_argument("--experiment", type=Path, default=Path("experiments/ironclad-a0"))
    train.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    train.add_argument("--steps", type=int, default=20000, help="Additional steps; rounded up to a PPO rollout")
    train.add_argument("--scope", choices=["fullrun", "combat"], default="fullrun")
    train.add_argument("--max-acts", type=int, choices=[1, 2, 3], default=3)
    train.add_argument("--max-steps", type=int, default=4000)
    train.add_argument("--n-envs", type=int, default=4)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--reward-shaping", action="store_true", help="Experimental upstream shaping; new experiment required")
    train.add_argument("--device", default="cpu")
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--checkpoint-every", type=int, default=10000)
    evaluation = commands.add_parser("evaluate", help="Paired evaluation on separate validation/test seeds")
    evaluation.add_argument("--experiment", type=Path, required=True)
    evaluation.add_argument("--upstream", type=Path, default=Path(".external/sts2-rl-agent"))
    evaluation.add_argument("--episodes", type=int, default=100)
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--baseline", default="random", help="random or path to current champion ZIP")
    evaluation.add_argument("--start", type=int, default=0, help="Index within held-out seed partition")
    evaluation.add_argument("--threads", type=int, default=2)
    promotion = commands.add_parser("promote", help="Set champion only if the paired validation gate passed")
    promotion.add_argument("--experiment", type=Path, required=True)
    promotion.add_argument("--report", type=Path, required=True)
    status = commands.add_parser("status", help="Read recorded progress without starting the game")
    status.add_argument("--experiment", type=Path, required=True)
    probe = commands.add_parser("native-probe", help="Start the native runner and capture its first decision")
    probe.add_argument("--config", type=Path, default=Path("artifacts/windows-install.json"))
    probe.add_argument("--seed", type=int, default=7)
    probe.add_argument("--timeout", type=float, default=30)
    probe.add_argument("--out", type=Path, default=Path("artifacts/native-probe.json"))
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "native-probe":
            from .native import NativeAbort, NativeRunner
            from .experiment import atomic_json
            install = json.loads(args.config.read_text(encoding="utf-8-sig"))
            if not install.get("native_command"):
                raise ValueError("Native backend has not been built. Run setup_windows.ps1 -BuildNative first.")
            try:
                with NativeRunner(install["native_command"], cwd=install["native_cwd"],
                                  timeout_s=args.timeout, startup_timeout_s=args.timeout) as native:
                    decision = native.reset(args.seed, character="Ironclad", ascension=0)
                    report = {"status": "first_decision_received", "seed": args.seed,
                              "decision": decision, "game_hashes": install.get("game_hashes"),
                              "protocol_version": native.protocol_version, "revision": native.revision,
                              "parity": "unverified", "note": "Startup probe only; native RL adapter is not implemented."}
            except NativeAbort as exc:
                atomic_json(args.out, {"status": "aborted", "seed": args.seed, "abort": exc.as_dict()})
                raise
            atomic_json(args.out, report)
            print(json.dumps({"status": report["status"], "decision": decision["decision"],
                              "report": str(args.out)}, indent=2))
            return 0
        if args.command == "status":
            from .experiment import read_json, verified_checkpoint
            from .study import summarize
            manifest = read_json(args.experiment / "experiment.json")
            _, checkpoint = verified_checkpoint(args.experiment, manifest["fingerprint"])
            journal = args.experiment / "episodes.jsonl"
            rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
            boundaries = [row for row in rows if row.get("outcome") == "aborted"
                          and row.get("reason", "").startswith("training budget ended;")]
            assessed = [row for row in rows if row not in boundaries]
            report = {"config": manifest["config"], "backend": manifest["backend"],
                      "checkpoint": checkpoint, "training_episodes": summarize(assessed),
                      "discarded_session_boundary_episodes": len(boundaries)}
        elif args.command == "train":
            from .experiment import ExperimentConfig, read_json
            from .runner import train
            path = args.experiment / "experiment.json"
            if path.exists():
                config = ExperimentConfig(**read_json(path)["config"])
                # The manifest owns environment/hyperparameters on resume.
                print("Resuming using saved experiment configuration.")
            else:
                config = ExperimentConfig(scope=args.scope, max_acts=args.max_acts,
                                          max_steps=args.max_steps, n_envs=args.n_envs,
                                          experiment_seed=args.seed, reward_shaping=args.reward_shaping)
            report = train(args.experiment.resolve(), config, args.upstream.resolve(),
                           steps=args.steps, device=args.device, threads=args.threads,
                           checkpoint_every=args.checkpoint_every)
        elif args.command == "evaluate":
            from .runner import evaluate
            report = evaluate(args.experiment.resolve(), args.upstream.resolve(), episodes=args.episodes,
                              split=args.split, baseline=args.baseline, start=args.start, threads=args.threads)
            report = {key: value for key, value in report.items()
                      if key not in {"candidate_episodes", "baseline_episodes"}}
        elif args.command == "promote":
            from .runner import promote
            report = promote(args.experiment.resolve(), args.report.resolve())
        else:
            raise AssertionError(args.command)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
