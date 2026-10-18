"""Imitation fits legal public combat labels with an episode-held-out check."""

from copy import deepcopy
import json
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
from sb3_contrib import MaskablePPO
import torch

from spire2_rl.imitation import collect_demonstrations, behavior_clone
from spire2_rl.models import PublicFeaturesExtractor
from spire2_rl.study import SEEDS_PER_SPLIT


class PublicTrajectoryEnv(gym.Env):
    """Four-step transport: event, ordinary combat, choice, ordinary combat."""

    def __init__(self, ending="death"):
        self.ending = ending
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({
            "hand_card_ids": spaces.Box(0, 3, shape=(1,), dtype=np.int64),
            "run_scalar": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        })
        self.backend_metadata = {"max_steps": 4, "observation_schema": "test-public"}
        self.raw_env = self
        self._mgr = SimpleNamespace(PHASE_COMBAT="COMBAT", phase="EVENT",
                                    get_combat_state=lambda: self.combat)
        self.combat = SimpleNamespace(pending_choice=None)
        self.reset_seeds = []
        self.executed = []

    def action_masks(self):
        return np.array([True, True, False])

    def observe(self):
        label = (self.seed + int(self.position == 3)) % 2
        self.observation = {
            "hand_card_ids": np.array([label + 1], dtype=np.int64),
            "run_scalar": np.array([0.1 if label == 0 else 0.9], dtype=np.float32),
        }
        return self.observation

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        self.reset_seeds.append(seed)
        self.position = 0
        self._mgr.phase = "EVENT"
        self.combat.pending_choice = None
        return self.observe(), {"outcome": "running", "floor": 1, "hp": 5}

    def step(self, action):
        self.executed.append((self.seed, self.position, int(action)))
        # Deliberately mutate the returned arrays: collection must already have
        # copied the pre-action public observation.
        for value in self.observation.values():
            value[:] = 0
        self.position += 1
        self._mgr.phase = "COMBAT"
        self.combat.pending_choice = object() if self.position == 2 else None
        done = self.position >= 4
        truncated = done and self.ending == "timeout"
        terminated = done and not truncated
        return self.observe(), -1.0 if done else 0.0, terminated, truncated, {
            "outcome": self.ending if done else "running", "floor": 3 if done else 1,
            "hp": 0 if done else 5, "act": 0,
        }


class VisibleTeacher:
    def select(self, env):
        return (env.seed + int(env.position == 3)) % 2


def _dataset(tmp_path, seeds=None):
    path = tmp_path / "demonstrations.npz"
    report = collect_demonstrations(PublicTrajectoryEnv(), VisibleTeacher(),
                                    seeds or list(range(1, 17)), path)
    return path, report


def _rewrite(path, mutate):
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key].copy() for key in loaded.files}
    mutate(data)
    np.savez_compressed(path, **data)


def _model():
    torch.set_num_threads(2)
    env = PublicTrajectoryEnv()
    model = MaskablePPO(
        "MultiInputPolicy", env, n_steps=4, batch_size=4, seed=7,
        policy_kwargs={"net_arch": [8], "features_extractor_class": PublicFeaturesExtractor,
                       "features_extractor_kwargs": {"features_dim": 8, "embedding_dim": 3}},
        device="cpu", verbose=0,
    )
    return model


def test_collection_uses_only_public_pre_action_ordinary_combat_and_keeps_losses(tmp_path):
    env = PublicTrajectoryEnv()
    path = tmp_path / "data.npz"
    report = collect_demonstrations(env, VisibleTeacher(), [1, 2], path)
    assert report["split"] == "train" and report["samples"] == 4
    assert report["gameplay"]["outcomes"]["death"] == 2
    assert report["dataset"] == path.name
    assert len(report["sha256"]) == 64
    assert env.reset_seeds == [1, 2] and len(env.executed) == 8
    with np.load(path, allow_pickle=False) as data:
        assert data["episode_seed"].tolist() == [1, 1, 2, 2]
        assert data["action"].tolist() == [1, 0, 0, 1]
        assert data["obs__hand_card_ids"].reshape(-1).tolist() == [2, 1, 1, 2]
        assert (data["obs__run_scalar"] > 0).all()
        assert data["action_masks"].shape == (4, 3)
        assert data["action_masks"][np.arange(4), data["action"]].all()
        metadata = json.loads(data["metadata_json"].item())
        assert metadata["split"] == "train"
        assert metadata["seeds"] == [1, 2]


@pytest.mark.parametrize("seeds", [[0], [SEEDS_PER_SPLIT + 1], [1, 1], [True], []])
def test_collection_rejects_nontraining_or_duplicate_seeds_before_play(tmp_path, seeds):
    env = PublicTrajectoryEnv()
    with pytest.raises(ValueError, match="[Ss]eed"):
        collect_demonstrations(env, VisibleTeacher(), seeds, tmp_path / "bad.npz")
    assert env.reset_seeds == []
    assert not (tmp_path / "bad.npz").exists()


def test_failed_or_incomplete_demonstrations_never_publish_dataset(tmp_path):
    for ending in ["timeout", "aborted"]:
        path = tmp_path / f"{ending}.npz"
        with pytest.raises(RuntimeError, match=ending):
            collect_demonstrations(PublicTrajectoryEnv(ending), VisibleTeacher(), [1, 2], path)
        assert not path.exists()

    class IllegalTeacher:
        def select(self, env):
            return 2

    path = tmp_path / "illegal.npz"
    with pytest.raises(RuntimeError, match="illegal"):
        collect_demonstrations(PublicTrajectoryEnv(), IllegalTeacher(), [1, 2], path)
    assert not path.exists()


def test_collection_preserves_existing_evidence(tmp_path):
    path, _ = _dataset(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        collect_demonstrations(PublicTrajectoryEnv(), VisibleTeacher(), [1, 2], path)
    assert path.read_bytes() == original


def test_collection_enforces_environment_decision_budget(tmp_path):
    class NeverFinishes(PublicTrajectoryEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            info["outcome"] = "running"
            return obs, reward, False, False, info

    env = NeverFinishes()
    path = tmp_path / "over-budget.npz"
    with pytest.raises(RuntimeError, match="finite decision limit"):
        collect_demonstrations(env, VisibleTeacher(), [1, 2], path)
    assert len(env.executed) == 4
    assert not path.exists()


def test_zero_information_labels_are_not_optimization_samples(tmp_path):
    path, _ = _dataset(tmp_path)

    def forced_only(data):
        mask = np.zeros_like(data["action_masks"])
        mask[np.arange(len(data["action"])), data["action"]] = True
        data["action_masks"] = mask

    _rewrite(path, forced_only)
    model = _model()
    original = deepcopy(model.policy.state_dict())
    with pytest.raises(ValueError, match="informative"):
        behavior_clone(model, path, epochs=2)
    for key, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["illegal_action", "validation_seed", "nonfinite", "shape", "missing_key", "split"])
def test_clone_refuses_bad_demonstrations_before_optimization(tmp_path, corruption):
    path, _ = _dataset(tmp_path)

    def corrupt(data):
        if corruption == "illegal_action":
            data["action"][0] = 2
        elif corruption == "validation_seed":
            data["episode_seed"][0] = SEEDS_PER_SPLIT + 1
        elif corruption == "nonfinite":
            data["obs__run_scalar"][0, 0] = np.nan
        elif corruption == "shape":
            data["obs__run_scalar"] = data["obs__run_scalar"][:, :0]
        elif corruption == "missing_key":
            del data["obs__hand_card_ids"]
        else:
            meta = json.loads(data["metadata_json"].item())
            meta["split"] = "validation"
            data["metadata_json"] = np.array(json.dumps(meta))

    _rewrite(path, corrupt)
    model = _model()
    original = deepcopy(model.policy.state_dict())
    with pytest.raises(ValueError):
        behavior_clone(model, path, epochs=2, batch_size=4)
    for key, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_clone_reduces_masked_nll_and_keeps_ppo_state_and_value_weights(tmp_path):
    path, _ = _dataset(tmp_path)
    model = _model()
    model.num_timesteps = 123
    model.policy.optimizer.zero_grad()
    model.policy.action_net.weight.sum().backward()
    model.policy.optimizer.step()
    optimizer = deepcopy(model.policy.optimizer.state_dict())
    value_weights = deepcopy(model.policy.value_net.state_dict())
    critic_weights = deepcopy(model.policy.mlp_extractor.value_net.state_dict())
    before = deepcopy(model.policy.state_dict())
    report = behavior_clone(model, path, epochs=16, batch_size=8, learning_rate=0.01)
    assert model.num_timesteps == 123
    assert report["gradient_updates"] > 0
    assert report["final"]["train"]["nll"] < report["initial"]["train"]["nll"]
    assert report["final"]["holdout"]["accuracy"] >= 0.9
    assert not set(report["train_episode_seeds"]) & set(report["holdout_episode_seeds"])
    assert report["holdout_episode_seeds"] == [13, 14, 15, 16]
    assert any(not torch.equal(value, before[key]) for key, value in model.policy.state_dict().items())
    after_optimizer = model.policy.optimizer.state_dict()
    assert after_optimizer["param_groups"] == optimizer["param_groups"]
    assert after_optimizer["state"].keys() == optimizer["state"].keys()
    for key, state in optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(after_optimizer["state"][key][name], value, rtol=0, atol=0)
    for key, value in model.policy.value_net.state_dict().items():
        torch.testing.assert_close(value, value_weights[key], rtol=0, atol=0)
    for key, value in model.policy.mlp_extractor.value_net.state_dict().items():
        torch.testing.assert_close(value, critic_weights[key], rtol=0, atol=0)


def test_holdout_labels_never_enter_gradient_updates(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    path, _ = _dataset(first_dir)
    changed, _ = _dataset(second_dir)

    def change_holdout(data):
        selected = np.isin(data["episode_seed"], [13, 14, 15, 16])
        data["action"][selected] = 1 - data["action"][selected]

    _rewrite(changed, change_holdout)
    first, second = _model(), _model()
    report_first = behavior_clone(first, path, epochs=3, batch_size=8, learning_rate=0.01, seed=7)
    report_second = behavior_clone(second, changed, epochs=3, batch_size=8, learning_rate=0.01, seed=7)
    assert report_first["gradient_updates"] == report_second["gradient_updates"]
    for key, value in first.policy.state_dict().items():
        torch.testing.assert_close(value, second.policy.state_dict()[key], rtol=0, atol=0)
