"""Persistent experiment configuration, episode accounting and checkpoint commits."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym

from .study import seed_for


@dataclass(frozen=True)
class ExperimentConfig:
    schema: int = 1
    scope: str = "fullrun"
    character: str = "Ironclad"
    ascension: int = 0
    max_acts: int = 3
    max_steps: int = 4000
    experiment_seed: int = 7
    n_envs: int = 4
    reward_shaping: bool = False
    gamma: float = 0.999
    learning_rate: float = 0.0003
    n_steps: int = 256
    batch_size: int = 128
    n_epochs: int = 4
    ent_coef: float = 0.01

    def validate(self) -> None:
        if self.scope not in {"fullrun", "combat"} or self.max_acts not in {1, 2, 3}:
            raise ValueError("Unsupported scope/act count")
        if self.character != "Ironclad" or self.ascension != 0:
            raise ValueError("This experiment specializes in Ironclad A0")
        if self.max_steps < 1 or not 1 <= self.n_envs <= 32:
            raise ValueError("max_steps must be positive; n_envs must be 1..32")
        if self.n_steps * self.n_envs < self.batch_size or self.batch_size < 2:
            raise ValueError("PPO rollout must contain at least one batch of two samples")
        if not 0 < self.gamma <= 1 or self.learning_rate <= 0:
            raise ValueError("Invalid PPO hyperparameters")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(upstream: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    # Tracked patches invalidate the promised pin even when HEAD is unchanged.
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    expected = "04a62c35b0ee264f5e4a3b5c6587f7438f1a7e0c"
    if commit != expected or dirty:
        raise ValueError("Emulator must be the clean pinned revision in backends.lock.json")
    return {
        "backend": "python-emulator",
        "commit": commit,
        "game_reference": "v0.106.1",
        "real_game_transfer": "unverified",
        "adapter_sha256": sha256(Path(__file__).with_name("emulator.py")),
        "seed_scheme_sha256": sha256(Path(__file__).with_name("study.py")),
        "policy_encoder_sha256": sha256(Path(__file__).with_name("models.py")),
        "training_runner_sha256": sha256(Path(__file__).with_name("runner.py")),
        "experiment_manager_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in (
            "gymnasium", "numpy", "torch", "stable-baselines3", "sb3-contrib"
        )},
    }


def fingerprint(config: ExperimentConfig, backend: dict, observation: dict) -> str:
    raw = json.dumps({"config": asdict(config), "backend": backend, "observation": observation},
                     sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


@contextmanager
def experiment_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".running.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Experiment is locked: {lock}. If its process stopped, remove the stale lock.") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid(), "started": time.time()}, stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


class EpisodeJournal(gym.Wrapper):
    """Assign disjoint training seeds and persist completed/aborted episodes."""

    def __init__(self, env, *, config: ExperimentConfig, rank: int,
                 next_index: int, journal: Path, session: str):
        super().__init__(env)
        self.config, self.rank = config, rank
        self.next_index = next_index
        self.journal, self.session = journal, session
        self.episode_seed: int | None = None
        self.episode_steps = 0
        self.episode_return = 0.0
        self.latest_info: dict = {}
        self.active = False
        self.cursor_path = journal.parent / "seed-cursors" / f"worker-{rank}.json"
        if self.cursor_path.exists():
            cursor = int(read_json(self.cursor_path)["next_index"])
            if cursor % config.n_envs != rank:
                raise ValueError("Persisted seed cursor belongs to a different worker layout")
            self.next_index = max(self.next_index, cursor)

    def action_masks(self):
        return self.env.action_masks()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("Training seed assignment is owned by EpisodeJournal")
        if self.active:
            self.abort("episode reset before terminal")
        index = self.next_index
        self.next_index += self.config.n_envs
        # Reserve before touching the engine. A hard kill must not replay seeds
        # consumed after the last model checkpoint.
        atomic_json(self.cursor_path, {"next_index": self.next_index})
        self.episode_seed = seed_for(index, "train", self.config.experiment_seed)
        self.episode_steps, self.episode_return, self.active = 0, 0.0, True
        self.latest_info = {}
        obs, info = self.env.reset(seed=self.episode_seed, options=options)
        self.latest_info = dict(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.episode_return += float(reward)
        self.latest_info = dict(info)
        if terminated or truncated:
            outcome = info.get("outcome")
            if outcome not in {"win", "death", "timeout"}:
                raise RuntimeError(f"Missing or invalid terminal outcome: {outcome!r}")
            self._write(outcome)
            self.active = False
        return obs, reward, terminated, truncated, info

    def _write(self, outcome: str, reason: str | None = None):
        append_jsonl(self.journal, {
            "session": self.session, "worker": self.rank, "seed": self.episode_seed,
            "outcome": outcome, "steps": self.episode_steps,
            "return": self.episode_return, "floor": int(self.latest_info.get("floor", 0)),
            "hp": float(self.latest_info.get("hp", 0)),
            "act": int(self.latest_info.get("act", 0)), "reason": reason,
        })

    def abort(self, reason: str):
        if self.active:
            self._write("aborted", reason)
            self.active = False


def commit_checkpoint(model, root: Path, wrappers: list[EpisodeJournal],
                      experiment_fingerprint: str, *, interrupted: bool = False) -> dict:
    """Commit ZIP then state, publishing a pointer after both files are written."""
    if hasattr(model, "policy"):
        import torch
        tensors = list(model.policy.parameters())
        optimizer = getattr(model.policy, "optimizer", None)
        if optimizer is not None:
            tensors.extend(value for state in optimizer.state.values()
                           for value in state.values() if torch.is_tensor(value))
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("Refusing to publish non-finite policy/optimizer checkpoint")
    name = f"step-{model.num_timesteps:012d}-{time.time_ns()}"
    folder = root / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (name + ".tmp.zip")
    target = folder / (name + ".zip")
    model.save(str(temporary))
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    state = {
        "model": str(target.relative_to(root)), "sha256": sha256(target),
        "num_timesteps": int(model.num_timesteps),
        "next_indices": [wrapper.next_index for wrapper in wrappers],
        "fingerprint": experiment_fingerprint, "interrupted": interrupted,
        "resume_semantics": "Weights and optimizer restored; active episodes restart on new seeds.",
    }
    atomic_json(folder / (name + ".json"), state)
    atomic_json(root / "latest.json", state)
    return state


def verified_checkpoint(root: Path, expected_fingerprint: str | None = None) -> tuple[Path, dict]:
    state = read_json(root / "latest.json")
    if expected_fingerprint is not None and state["fingerprint"] != expected_fingerprint:
        raise ValueError("Checkpoint configuration/backend fingerprint changed; use a new experiment")
    model = (root / state["model"]).resolve()
    if not model.is_relative_to(root.resolve()):
        raise ValueError("Checkpoint path escapes experiment directory")
    if sha256(model) != state["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    return model, state
