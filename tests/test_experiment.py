"""Check durable episode ownership and corruption protection without a game."""
import json

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from spire2_rl.experiment import (EpisodeJournal, ExperimentConfig, commit_checkpoint,
                                 experiment_lock, verified_checkpoint)


class ShortGame(gym.Env):
    observation_space = spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        self.seed_seen = seed
        return np.zeros(1, dtype=np.float32), {"floor": 1, "hp": 80}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), -1.0, True, False, {"outcome": "death", "floor": 3, "hp": 0}

    def action_masks(self):
        return np.ones(1, dtype=bool)


def test_hard_kill_seed_cursor_advances_past_checkpoint(tmp_path):
    config = ExperimentConfig(n_envs=2)
    kwargs = dict(config=config, rank=0, journal=tmp_path / "episodes.jsonl", session="old")
    old = EpisodeJournal(ShortGame(), next_index=0, **kwargs)
    old.reset()
    consumed = old.episode_seed
    # Simulate restart from an older model checkpoint while the durable reservation remains.
    restored = EpisodeJournal(ShortGame(), next_index=0, **kwargs)
    restored.reset()
    assert restored.episode_seed != consumed
    assert restored.next_index == 4
    worker1 = EpisodeJournal(ShortGame(), config=config, rank=1, next_index=1,
                            journal=tmp_path / "episodes.jsonl", session="parallel")
    worker1.reset()
    assert len({consumed, restored.episode_seed, worker1.episode_seed}) == 3


def test_game_death_and_interruption_are_distinct_and_not_double_counted(tmp_path):
    journal = tmp_path / "episodes.jsonl"
    env = EpisodeJournal(ShortGame(), config=ExperimentConfig(n_envs=1), rank=0,
                         next_index=0, journal=journal, session="one")
    env.reset()
    env.step(0)
    env.abort("shutdown")
    env.reset()
    env.abort("engine exception")
    env.abort("second cleanup")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["outcome"] for row in rows] == ["death", "aborted"]
    assert rows[0]["hp"] == 0 and rows[0]["floor"] == 3
    assert rows[1]["steps"] == 0 and rows[1]["reason"] == "engine exception"


class FakeCheckpointModel:
    num_timesteps = 123

    def save(self, path):
        from pathlib import Path
        Path(path).write_bytes(b"model weights and optimizer")


def test_checkpoint_commit_rejects_corruption_and_changed_environment(tmp_path):
    state = commit_checkpoint(FakeCheckpointModel(), tmp_path, [], "environment-a")
    model, loaded = verified_checkpoint(tmp_path, "environment-a")
    assert loaded == state
    with pytest.raises(ValueError, match="fingerprint"):
        verified_checkpoint(tmp_path, "environment-b")
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verified_checkpoint(tmp_path, "environment-a")


def test_two_trainers_cannot_write_same_experiment(tmp_path):
    with experiment_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with experiment_lock(tmp_path):
                pass
    with experiment_lock(tmp_path):
        pass


def test_nonfinite_optimizer_does_not_replace_last_good_checkpoint(tmp_path):
    import torch
    model = FakeCheckpointModel()
    first = commit_checkpoint(model, tmp_path, [], "environment-a")
    model.policy = torch.nn.Linear(1, 1)
    model.policy.optimizer = torch.optim.Adam(model.policy.parameters())
    model.policy.optimizer.state[model.policy.weight] = {"exp_avg": torch.tensor([float("nan")])}
    with pytest.raises(ValueError, match="non-finite"):
        commit_checkpoint(model, tmp_path, [], "environment-a", interrupted=True)
    _, retained = verified_checkpoint(tmp_path)
    assert retained == first
