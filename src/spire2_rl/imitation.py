"""Bounded ordinary-combat imitation from public teacher demonstrations.

Demonstrations include clean losing runs. Supervised agreement with a teacher
does not measure gameplay strength, and no value targets are fitted here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
import time

import gymnasium as gym
import numpy as np
import torch

from .experiment import sha256
from .study import SEEDS_PER_SPLIT, summarize


def _train_seeds(seeds) -> list[int]:
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Seeds must be a nonempty list of training seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, Integral)
           or not 1 <= seed <= SEEDS_PER_SPLIT for seed in seeds):
        raise ValueError("Seeds must lie inside the positive training seed window")
    result = [int(seed) for seed in seeds]
    if len(set(result)) != len(result):
        raise ValueError("Seeds must not contain duplicates")
    return result


def _schema(space) -> dict:
    if not isinstance(space, gym.spaces.Dict):
        raise ValueError("Imitation requires the public Dict observation space")
    result = {}
    for key, item in space.spaces.items():
        if not isinstance(item, gym.spaces.Box):
            raise ValueError("Imitation observation components must be numeric Box spaces")
        result[key] = {
            "shape": list(item.shape), "dtype": str(item.dtype),
            "low_sha256": hashlib.sha256(item.low.tobytes()).hexdigest(),
            "high_sha256": hashlib.sha256(item.high.tobytes()).hexdigest(),
        }
    return result


def _public_copy(observation, space) -> dict:
    if not isinstance(observation, dict) or observation.keys() != space.spaces.keys():
        raise ValueError("Public observation keys differ from the declared schema")
    result = {}
    for key, item in space.spaces.items():
        value = np.asarray(observation[key])
        if (value.shape != item.shape or value.dtype != item.dtype
                or not np.isfinite(value).all()
                or not (value >= item.low).all() or not (value <= item.high).all()):
            raise ValueError(f"Invalid public observation component: {key}")
        result[key] = value.copy()
    return result


def _atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".npz", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"Demonstrations already exist: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_demonstrations(env, teacher, seeds: list[int], output: Path) -> dict:
    """Play complete teacher episodes, retaining ordinary-combat public labels.

    The supplied seeds must be in the training window. The environment owns its
    finite episode decision limit. Timeouts, errors, and illegal teacher actions
    reject the whole collection; a partly collected dataset is never published.
    """
    seeds = _train_seeds(seeds)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Demonstrations already exist: {output}")
    schema = _schema(env.observation_space)
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("Imitation requires discrete legal action slots")
    metadata = dict(env.backend_metadata)
    max_steps = metadata.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 1:
        raise ValueError("Demonstration environment needs a finite max_steps limit")
    if getattr(getattr(teacher, "bot", None), "use_lookahead", False):
        raise ValueError("Fair demonstrations must disable hidden-state lookahead")
    observations = {key: [] for key in schema}
    actions, masks, episode_seeds, rows = [], [], [], []
    started = time.monotonic()
    for episode_number, seed in enumerate(seeds, 1):
        info, length, episode_return, labels = {}, 0, 0.0, 0
        try:
            obs, info = env.reset(seed=seed)
            while True:
                if length >= max_steps:
                    raise RuntimeError("Teacher episode exceeded the finite decision limit")
                mask = np.asarray(env.action_masks(), dtype=bool)
                if mask.shape != (env.action_space.n,) or not mask.any():
                    raise RuntimeError("Teacher encountered an invalid legal action mask")
                mgr = env.raw_env._mgr
                combat = mgr.get_combat_state() if mgr.phase == mgr.PHASE_COMBAT else None
                ordinary = combat is not None and combat.pending_choice is None
                copied = _public_copy(obs, env.observation_space) if ordinary else None
                action = teacher.select(env)
                if (isinstance(action, bool) or not isinstance(action, Integral)
                        or not env.action_space.contains(action) or not mask[int(action)]):
                    raise RuntimeError(f"Teacher chose illegal action {action!r}")
                action = int(action)
                if ordinary:
                    for key, value in copied.items():
                        observations[key].append(value)
                    actions.append(action)
                    masks.append(mask.copy())
                    episode_seeds.append(seed)
                    labels += 1
                obs, reward, terminated, truncated, info = env.step(action)
                length += 1
                episode_return += float(reward)
                if terminated or truncated:
                    outcome = info.get("outcome")
                    if truncated or not terminated or outcome not in {"win", "death"}:
                        raise RuntimeError(f"Teacher demonstrations contain {outcome or 'incomplete'} episode")
                    break
        except Exception as exc:
            raise RuntimeError(f"Demonstration collection failed at seed {seed}: {exc}") from exc
        rows.append({
            "seed": seed, "policy": "fair-teacher", "outcome": outcome,
            "floor": int(info.get("floor", 0)), "hp": float(info.get("hp", 0)),
            "steps": length, "return": episode_return, "error": None, "samples": labels,
        })
        if episode_number % 10 == 0 or episode_number == len(seeds):
            print(f"Collected {episode_number}/{len(seeds)} teacher episodes, {len(actions)} ordinary-combat labels", flush=True)
    if not actions:
        raise ValueError("Teacher collection produced no ordinary-combat demonstrations")
    payload = {f"obs__{key}": np.stack(values) for key, values in observations.items()}
    payload.update(
        action=np.asarray(actions, dtype=np.int64),
        action_masks=np.stack(masks), episode_seed=np.asarray(episode_seeds, dtype=np.int64),
    )
    archive_metadata = {
        "version": 1, "kind": "ordinary-combat-v1", "split": "train", "seeds": seeds,
        "episodes": rows, "observation_schema": schema, "action_count": int(env.action_space.n),
        "backend": metadata, "teacher_class": type(teacher).__name__,
        "imitation_sha256": sha256(Path(__file__)),
        "note": "Teacher plays all phases; only public ordinary-combat labels retained. Clean losses included.",
    }
    payload["metadata_json"] = np.array(json.dumps(archive_metadata, sort_keys=True, allow_nan=False))
    _atomic_npz(output, payload)
    gameplay = summarize(rows)
    return {
        "dataset": output.name, "sha256": sha256(output), "split": "train", "seeds": seeds,
        "samples": len(actions), "informative_samples": int((payload["action_masks"].sum(axis=1) > 1).sum()),
        "episodes": rows, "gameplay": gameplay, "summary": gameplay,
        "backend": metadata, "elapsed_seconds": time.monotonic() - started,
        "note": archive_metadata["note"],
    }


def _load_demonstrations(model, dataset: Path):
    with np.load(dataset, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"action", "action_masks", "episode_seed", "metadata_json"}
    if not required <= data.keys():
        raise ValueError("Demonstration archive lacks required evidence fields")
    try:
        metadata = json.loads(data["metadata_json"].item())
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid demonstration metadata") from exc
    if metadata.get("version") != 1 or metadata.get("kind") != "ordinary-combat-v1" or metadata.get("split") != "train":
        raise ValueError("Behavior cloning requires ordinary-combat training demonstrations")
    seeds = _train_seeds(metadata.get("seeds"))
    rows = metadata.get("episodes", [])
    if ([row.get("seed") for row in rows] != seeds
            or any(row.get("outcome") not in {"win", "death"} for row in rows)):
        raise ValueError("Demonstrations must contain only complete clean teacher episodes")
    schema = _schema(model.observation_space)
    if metadata.get("observation_schema") != schema or metadata.get("action_count") != model.action_space.n:
        raise ValueError("Demonstration and model observation/action schemas differ")
    expected_keys = {f"obs__{key}" for key in schema}
    if {key for key in data if key.startswith("obs__")} != expected_keys:
        raise ValueError("Demonstration observation keys differ from the model")
    actions, masks, sample_seeds = data["action"], data["action_masks"], data["episode_seed"]
    count = len(actions) if actions.ndim == 1 else 0
    if (not count or not np.issubdtype(actions.dtype, np.integer)
            or (actions < 0).any() or (actions >= model.action_space.n).any()
            or masks.shape != (count, model.action_space.n) or masks.dtype != np.dtype(bool)
            or not masks.any(axis=1).all() or not masks[np.arange(count), actions].all()):
        raise ValueError("Demonstration actions/masks are invalid or contain an illegal teacher label")
    if (sample_seeds.shape != (count,) or not np.issubdtype(sample_seeds.dtype, np.integer)
            or (sample_seeds < 1).any() or (sample_seeds > SEEDS_PER_SPLIT).any()
            or not np.isin(sample_seeds, seeds).all()):
        raise ValueError("Demonstration samples must belong to the supplied training episode seeds")
    counts = Counter(int(seed) for seed in sample_seeds)
    if any(row.get("samples") != counts[row["seed"]] for row in rows):
        raise ValueError("Demonstration episode sample counts differ from their evidence")
    observations = {}
    for key, item in model.observation_space.spaces.items():
        value = data[f"obs__{key}"]
        if (value.shape != (count, *item.shape) or value.dtype != item.dtype
                or not np.isfinite(value).all()
                or not (value >= item.low).all() or not (value <= item.high).all()):
            raise ValueError(f"Invalid demonstration observation component: {key}")
        observations[key] = value
    if len(seeds) < 2:
        raise ValueError("Behavior cloning needs at least two independent episode seeds")
    holdout_count = max(1, math.ceil(len(seeds) * 0.2))
    train_seeds, holdout_seeds = seeds[:-holdout_count], seeds[-holdout_count:]
    training = np.flatnonzero(np.isin(sample_seeds, train_seeds))
    holdout = np.flatnonzero(np.isin(sample_seeds, holdout_seeds))
    optimization = training[masks[training].sum(axis=1) > 1]
    if not len(training) or not len(holdout) or not len(optimization):
        raise ValueError("Demonstrations need training/holdout labels and informative training decisions")
    return observations, actions, masks, training, holdout, optimization, train_seeds, holdout_seeds


def behavior_clone(model, dataset: Path, *, epochs: int = 12, batch_size: int = 256,
                   learning_rate: float = 0.001, seed: int = 7) -> dict:
    """Fit masked teacher likelihood with a separate Adam; PPO counts unchanged.

    The last 20% of supplied episode seeds are supervised holdout, all within the
    training seed window. Ordinary decisions with only one legal action stay in
    evidence/metrics but are omitted from optimization because their NLL is zero.
    Shared features may change value predictions and noncombat behavior even
    though no critic targets are fitted. Evaluate actual play before further RL.
    """
    if (isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs < 1
            or isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1
            or not np.isfinite(learning_rate) or learning_rate <= 0
            or isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("Invalid behavior-cloning optimization settings")
    dataset = Path(dataset)
    dataset_hash = sha256(dataset)
    (observations, actions, masks, training, holdout, optimization,
     train_seeds, holdout_seeds) = _load_demonstrations(model, dataset)
    policy = model.policy
    previous_mode = policy.training
    original_timesteps = int(model.num_timesteps)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(learning_rate))
    started = time.monotonic()

    def batch(indices):
        tensor_obs, _ = policy.obs_to_tensor({key: value[indices] for key, value in observations.items()})
        labels = torch.as_tensor(actions[indices], dtype=torch.long, device=policy.device)
        legal = torch.as_tensor(masks[indices], dtype=torch.bool, device=policy.device)
        return tensor_obs, labels, legal

    def metrics(indices):
        policy.set_training_mode(False)
        total_nll = total_correct = informative_nll = informative_correct = informative_count = 0
        with torch.no_grad():
            for offset in range(0, len(indices), batch_size):
                selected = indices[offset:offset + batch_size]
                obs, labels, legal = batch(selected)
                _, log_prob, _ = policy.evaluate_actions(obs, labels, action_masks=legal)
                predictions = policy.get_distribution(obs, action_masks=legal).distribution.probs.argmax(dim=1)
                productive = legal.sum(dim=1) > 1
                total_nll += float((-log_prob).sum().item())
                total_correct += int((predictions == labels).sum().item())
                informative_nll += float((-log_prob[productive]).sum().item())
                informative_correct += int(((predictions == labels) & productive).sum().item())
                informative_count += int(productive.sum().item())
        return {
            "samples": len(indices), "nll": total_nll / len(indices),
            "accuracy": total_correct / len(indices), "informative_samples": informative_count,
            "informative_nll": informative_nll / informative_count if informative_count else None,
            "informative_accuracy": informative_correct / informative_count if informative_count else None,
        }

    try:
        initial = {"train": metrics(training), "holdout": metrics(holdout)}
        updates = 0
        history = []
        for epoch in range(1, int(epochs) + 1):
            policy.set_training_mode(True)
            order = rng.permutation(optimization)
            total_loss = 0.0
            for offset in range(0, len(order), batch_size):
                selected = order[offset:offset + batch_size]
                obs, labels, legal = batch(selected)
                _, log_prob, _ = policy.evaluate_actions(obs, labels, action_masks=legal)
                loss = -log_prob.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite behavior-cloning loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.item()) * len(selected)
                updates += 1
            history.append({"epoch": epoch, "optimization_nll": total_loss / len(optimization)})
            if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
                print(f"Behavior cloning epoch {epoch}/{epochs}: masked NLL {history[-1]['optimization_nll']:.4f}", flush=True)
        final = {"train": metrics(training), "holdout": metrics(holdout)}
        if int(model.num_timesteps) != original_timesteps or sha256(dataset) != dataset_hash:
            raise ValueError("PPO counter or demonstration file changed during cloning")
        if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
            raise ValueError("Non-finite behavior-cloned policy parameters")
        return {
            "dataset": dataset.name, "dataset_sha256": dataset_hash, "epochs": int(epochs),
            "batch_size": int(batch_size), "learning_rate": float(learning_rate), "seed": int(seed),
            "gradient_updates": updates, "train_samples": len(training), "holdout_samples": len(holdout),
            "optimization_samples": len(optimization), "train_episode_seeds": train_seeds,
            "holdout_episode_seeds": holdout_seeds, "initial": initial, "final": final,
            "history": history, "num_timesteps": original_timesteps,
            "elapsed_seconds": time.monotonic() - started,
            "note": "Supervised teacher agreement only, not gameplay success. Separate Adam; value is not supervised. Use a fresh PPO optimizer for subsequent RL.",
        }
    finally:
        policy.set_training_mode(previous_mode)
