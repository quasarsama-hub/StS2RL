"""Masked PPO and held-out evaluation. No model-driven source edits."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .emulator import make_env
from .experiment import (EpisodeJournal, ExperimentConfig, append_jsonl, atomic_json,
                         commit_checkpoint, experiment_lock, fingerprint, provenance,
                         read_json, sha256, verified_checkpoint)
from .study import evaluation_seeds, paired_compare, summarize
from .models import PublicFeaturesExtractor


def _make(config: ExperimentConfig, upstream: Path):
    return make_env(scope=config.scope, upstream_path=upstream,
                    max_steps=config.max_steps, max_acts=config.max_acts,
                    ascension=config.ascension, reward_shaping=config.reward_shaping)


def initialize(root: Path, config: ExperimentConfig, upstream: Path) -> dict:
    config.validate()
    backend = provenance(upstream)
    env = _make(config, upstream)
    try:
        observation = env.backend_metadata
    finally:
        env.close()
    identity = fingerprint(config, backend, observation)
    manifest = {"config": asdict(config), "backend": backend,
                "observation": observation, "fingerprint": identity}
    path = root / "experiment.json"
    if path.exists():
        existing = read_json(path)
        if existing["fingerprint"] != identity:
            raise ValueError("Experiment configuration or dependency versions changed; create a new directory")
        return existing
    if (root / "latest.json").exists():
        raise ValueError("Checkpoint exists without its experiment manifest")
    atomic_json(path, manifest)
    return manifest


class SaveProgress(BaseCallback):
    def __init__(self, root, wrappers, identity, checkpoint_every):
        super().__init__()
        self.root, self.wrappers, self.identity = root, wrappers, identity
        self.interval = checkpoint_every
        self.last_saved = 0

    def _on_training_start(self):
        self.last_saved = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_saved >= self.interval:
            commit_checkpoint(self.model, self.root, self.wrappers, self.identity)
            self.last_saved = self.num_timesteps
        return True


def train(root: Path, config: ExperimentConfig, upstream: Path, *, steps: int,
          device: str = "cpu", threads: int = 2, checkpoint_every: int = 10000) -> dict:
    if steps < 1 or threads < 1 or checkpoint_every < 1:
        raise ValueError("steps, threads and checkpoint_every must be positive")
    torch.set_num_threads(threads)
    with experiment_lock(root):
        manifest = initialize(root, config, upstream)
        state = read_json(root / "latest.json") if (root / "latest.json").exists() else None
        next_indices = state["next_indices"] if state else list(range(config.n_envs))
        if len(next_indices) != config.n_envs:
            raise ValueError("Worker count changed")
        session = str(time.time_ns())
        wrappers = []
        def factory(rank):
            def create():
                wrapper = EpisodeJournal(_make(config, upstream), config=config, rank=rank,
                                         next_index=next_indices[rank], journal=root / "episodes.jsonl",
                                         session=session)
                wrappers.append(wrapper)
                return wrapper
            return create
        vec = DummyVecEnv([factory(i) for i in range(config.n_envs)])
        try:
            if state:
                checkpoint, state = verified_checkpoint(root, manifest["fingerprint"])
                model = MaskablePPO.load(checkpoint, env=vec, device=device)
                # Resume is episode-boundary, not bitwise mid-rollout restoration.
                session_seed = config.experiment_seed + int(model.num_timesteps)
                random.seed(session_seed)
                np.random.seed(session_seed % 2**32)
                torch.manual_seed(session_seed)
            else:
                policy = "MultiInputPolicy" if isinstance(vec.observation_space, spaces.Dict) else "MlpPolicy"
                policy_kwargs = {"net_arch": [128, 64]}
                if policy == "MultiInputPolicy":
                    policy_kwargs.update(features_extractor_class=PublicFeaturesExtractor,
                                         features_extractor_kwargs={"features_dim": 256})
                model = MaskablePPO(policy, vec, learning_rate=config.learning_rate,
                                    n_steps=config.n_steps, batch_size=config.batch_size,
                                    n_epochs=config.n_epochs, gamma=config.gamma,
                                    ent_coef=config.ent_coef, policy_kwargs=policy_kwargs,
                                    seed=config.experiment_seed, device=device, verbose=1)
                # SB3 defers its initial seed to reset; our wrapper owns game seeds.
                vec._seeds = [None] * config.n_envs
                commit_checkpoint(model, root, wrappers, manifest["fingerprint"])
            vec._seeds = [None] * config.n_envs
            before = int(model.num_timesteps)
            initial = torch.cat([p.detach().cpu().reshape(-1) for p in model.policy.parameters()])
            start = time.monotonic()
            callback = SaveProgress(root, wrappers, manifest["fingerprint"], checkpoint_every)
            try:
                model.learn(total_timesteps=steps, reset_num_timesteps=False, callback=callback)
            except BaseException as exc:
                for wrapper in wrappers:
                    wrapper.abort(f"{type(exc).__name__}: {exc}")
                # Keep a diagnostic snapshot; failed transitions were never added to the buffer.
                checkpoint_error = None
                try:
                    saved = commit_checkpoint(model, root, wrappers, manifest["fingerprint"], interrupted=True)
                except Exception as checkpoint_exc:
                    saved, checkpoint_error = None, str(checkpoint_exc)
                append_jsonl(root / "sessions.jsonl", {
                    "session": session, "status": "interrupted", "error": str(exc),
                    "checkpoint": saved, "elapsed_seconds": time.monotonic() - start,
                    "checkpoint_error": checkpoint_error,
                })
                raise
            final = torch.cat([p.detach().cpu().reshape(-1) for p in model.policy.parameters()])
            # The last vector reset may have opened unfinished episodes. Classify them explicitly.
            for wrapper in wrappers:
                wrapper.abort("training budget ended; unfinished episode discarded")
            saved = commit_checkpoint(model, root, wrappers, manifest["fingerprint"])
            report = {"session": session, "status": "completed", "requested_steps": steps,
                      "added_steps": int(model.num_timesteps) - before,
                      "total_steps": int(model.num_timesteps),
                      "elapsed_seconds": time.monotonic() - start,
                      "policy_parameter_l2_change": float(torch.linalg.vector_norm(final - initial)),
                      "checkpoint": saved, "backend": manifest["backend"],
                      "note": "Parameter change demonstrates optimization, not improved playing strength."}
            append_jsonl(root / "sessions.jsonl", report)
            return report
        finally:
            vec.close()


def _rollout(env, seeds: list[int], *, model=None, label: str, trace: Path | None = None) -> list[dict]:
    results = []
    for seed in seeds:
        info, length, episode_return = {}, 0, 0.0
        outcome, error = "aborted", None
        try:
            obs, info = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            while True:
                masks = env.action_masks()
                if model is None:
                    action = int(rng.choice(np.flatnonzero(masks)))
                else:
                    action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                    action = int(action)
                if trace:
                    append_jsonl(trace, {"seed": seed, "step": length, "action": action,
                                        "legal_actions": np.flatnonzero(masks).tolist()})
                obs, reward, terminated, truncated, info = env.step(action)
                length += 1
                episode_return += float(reward)
                if terminated or truncated:
                    outcome = info["outcome"]
                    break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        results.append({"seed": seed, "policy": label, "outcome": outcome,
                        "floor": int(info.get("floor", 0)), "hp": float(info.get("hp", 0)),
                        "steps": length, "return": episode_return, "error": error})
    return results


def evaluate(root: Path, upstream: Path, *, episodes: int = 100, split: str = "validation",
             baseline: str = "random", start: int = 0, threads: int = 2) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation uses validation or test seeds, never training seeds")
    if episodes < 1:
        raise ValueError("Evaluation needs at least one episode")
    torch.set_num_threads(threads)
    with experiment_lock(root):
        manifest = read_json(root / "experiment.json")
        config = ExperimentConfig(**manifest["config"])
        initialize(root, config, upstream)  # checks current dependency/backend identity
        checkpoint, state = verified_checkpoint(root, manifest["fingerprint"])
        model = MaskablePPO.load(checkpoint, device="cpu")
        baseline_path = None if baseline == "random" else Path(baseline).resolve()
        baseline_identity = ({"kind": "random"} if baseline_path is None else
                             {"kind": "checkpoint", "path": str(baseline_path), "sha256": sha256(baseline_path)})
        baseline_model = None if baseline_path is None else MaskablePPO.load(baseline_path, device="cpu")
        baseline_label = "random" if baseline_path is None else str(baseline_path)
        seeds = evaluation_seeds(split, episodes, config.experiment_seed, start=start)
        env = _make(config, upstream)
        try:
            candidates = _rollout(env, seeds, model=model, label=str(checkpoint.name))
            baselines = _rollout(env, seeds, model=baseline_model, label=baseline_label)
        finally:
            env.close()
        if baseline_path is not None and sha256(baseline_path) != baseline_identity["sha256"]:
            raise ValueError("Baseline checkpoint changed during evaluation")
        report = {"fingerprint": manifest["fingerprint"], "checkpoint": state,
                  "split": split, "seed_start": start, "backend": manifest["backend"],
                  "baseline_identity": baseline_identity,
                  "inference": {"candidate": "deterministic", "baseline":
                                "seeded legal random" if baseline_path is None else "deterministic"},
                  "candidate": summarize(candidates), "baseline": summarize(baselines),
                  "comparison": paired_compare(candidates, baselines),
                  "candidate_episodes": candidates, "baseline_episodes": baselines}
        target = root / "evaluations" / f"{split}-{time.time_ns()}.json"
        atomic_json(target, report)
        report["report_path"] = str(target)
        atomic_json(root / "last-evaluation.json", report)
        return report


def promote(root: Path, report_path: Path) -> dict:
    """Publish a champion pointer only after an explicit validation comparison."""
    with experiment_lock(root):
        report = read_json(report_path)
        manifest = read_json(root / "experiment.json")
        if report["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Evaluation belongs to a different experiment")
        checked_comparison = paired_compare(report["candidate_episodes"], report["baseline_episodes"])
        if report["split"] != "validation" or not checked_comparison["promote"]:
            raise ValueError("Promotion requires passing the validation gate")
        # Verify the assessed checkpoint, not whatever is latest at promotion time.
        state = report["checkpoint"]
        if state["fingerprint"] != manifest["fingerprint"] or state.get("interrupted", False):
            raise ValueError("Evaluated checkpoint is incompatible or interrupted")
        checkpoint = (root / state["model"]).resolve()
        if not checkpoint.is_relative_to(root.resolve()) or sha256(checkpoint) != state["sha256"]:
            raise ValueError("Evaluated checkpoint no longer matches")
        previous = read_json(root / "champion.json") if (root / "champion.json").exists() else None
        expected_baseline = str((root / previous["checkpoint"]["model"]).resolve()) if previous else "random"
        labels = {row["policy"] for row in report["baseline_episodes"]}
        if labels != {expected_baseline}:
            raise ValueError(f"Promotion must compare against current champion: {expected_baseline}")
        expected_identity = ({"kind": "checkpoint", "path": expected_baseline,
                              "sha256": previous["checkpoint"]["sha256"]}
                             if previous else {"kind": "random"})
        if report.get("baseline_identity") != expected_identity:
            raise ValueError("Baseline identity/checksum differs from the current champion")
        if previous and sha256(Path(expected_baseline)) != expected_identity["sha256"]:
            raise ValueError("Champion checkpoint checksum changed")
        champion = {"checkpoint": state, "evaluation": str(report_path.resolve()),
                    "backend": manifest["backend"], "previous": previous,
                    "comparison": checked_comparison}
        atomic_json(root / "champion.json", champion)
        return champion
