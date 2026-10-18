"""Verify additive card semantics, behavior identity and Adam continuity."""

import copy
from pathlib import Path
import os
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from spire2_rl.choice_encoder import (
    CHOICE_ENCODER_SCHEMA, SEMANTIC_FIELDS, ChoiceFeaturesExtractor,
    build_public_card_bank, migrate_model, verify_initial_identity,
)
from spire2_rl.models import PublicFeaturesExtractor


@pytest.fixture(scope="module")
def public_bank():
    upstream = Path(os.environ.get("SPIRE2_UPSTREAM_PATH", "/workspace/upstream/sts2-rl-agent"))
    if not (upstream / "sts2_env/gym_env/run_env.py").is_file():
        pytest.skip("Pinned emulator checkout required for public-definition migration tests")
    from spire2_rl.emulator import _load_upstream, PINNED_UPSTREAM_COMMIT
    _load_upstream(upstream.resolve(), PINNED_UPSTREAM_COMMIT)
    from sts2_env.gym_env.run_env import _CARD_ID_TO_INT_IDX
    vocabulary = len(_CARD_ID_TO_INT_IDX) + 1
    space = gym.spaces.Dict({
        "hand_card_ids": gym.spaces.Box(0, vocabulary - 1, shape=(3,), dtype=np.int64),
        "hand_card_features": gym.spaces.Box(-1, 10, shape=(3, 2), dtype=np.float32),
        "choice_features": gym.spaces.Box(-1, 10, shape=(5, 28), dtype=np.float32),
        "run_scalar": gym.spaces.Box(-1, 10, shape=(2,), dtype=np.float32),
    })
    bank, metadata = build_public_card_bank(upstream, space)
    return upstream, space, bank, metadata


def _observations(space, metadata, *, batch=2):
    observations = {name: torch.zeros((batch, *component.shape)) for name, component in space.spaces.items()}
    names = metadata["card_names_by_integer_id"]
    card_id = names.index("POMMEL_STRIKE")
    observations["hand_card_ids"][:, 0] = names.index("STRIKE_IRONCLAD")
    observations["choice_features"][:, 1, 0] = 1
    observations["choice_features"][:, 1, 2] = card_id / len(names)
    observations["choice_features"][1:, 1, 5] = 1
    return observations


def _extractor(public_bank):
    _, space, bank, metadata = public_bank
    return ChoiceFeaturesExtractor(space, semantic_bank=bank, semantic_metadata=metadata)


def test_card_definition_bank_is_static_public_upgraded_and_padding_safe(public_bank):
    _, space, bank, metadata = public_bank
    fields = list(SEMANTIC_FIELDS)
    names = metadata["card_names_by_integer_id"]
    assert bank.shape == (int(space["hand_card_ids"].high.max()) + 1, 2, len(fields))
    assert not np.any(bank[0])
    assert metadata["schema"] == CHOICE_ENCODER_SCHEMA
    assert metadata["unavailable_definitions"] == ["GENERIC"]
    strike = bank[names.index("STRIKE_IRONCLAD")]
    assert strike[0, fields.index("type_attack")] == 1
    assert strike[0, fields.index("damage_div50")] == pytest.approx(6 / 50)
    assert strike[1, fields.index("damage_div50")] == pytest.approx(9 / 50)
    assert bank[names.index("DEFEND_IRONCLAD"), 0, fields.index("block_div50")] == pytest.approx(5 / 50)
    assert bank[names.index("POMMEL_STRIKE"), 0, fields.index("explicit_draw_div10")] == pytest.approx(1 / 10)
    assert not any(word in field for field in fields for word in ("rng", "future", "draw_order"))


def test_ordinal_choice_ids_decode_to_exact_shared_hand_embedding_ids(public_bank):
    _, space, _, metadata = public_bank
    extractor = _extractor(public_bank)
    observations = _observations(space, metadata)
    ids = extractor.decode_choice_card_ids(observations["choice_features"])
    assert ids[0, 1] == metadata["card_names_by_integer_id"].index("POMMEL_STRIKE")
    assert torch.count_nonzero(ids) == 2
    assert extractor.card_embedding.num_embeddings == len(metadata["card_names_by_integer_id"])


def test_static_bank_construction_preserves_engine_card_counter_and_rng(public_bank):
    upstream, space, bank, _ = public_bank
    from sts2_env.cards import base as card_base
    counter = card_base._next_instance_id
    py = random.getstate()
    numpy = np.random.get_state()
    torch_state = torch.get_rng_state()
    rebuilt, _ = build_public_card_bank(upstream, space)
    assert np.array_equal(bank, rebuilt)
    assert card_base._next_instance_id == counter
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_zero_residual_is_exact_original_feature_identity(public_bank):
    _, space, _, metadata = public_bank
    old = PublicFeaturesExtractor(space)
    new = _extractor(public_bank)
    new.load_state_dict(old.state_dict(), strict=False)
    observations = _observations(space, metadata)
    assert torch.equal(old(observations), new(observations))
    assert not torch.count_nonzero(new.choice_residual_projection.weight)


def test_new_projection_has_first_step_gradient_then_choice_encoder_learns(public_bank):
    _, space, _, metadata = public_bank
    extractor = _extractor(public_bank)
    observations = _observations(space, metadata)
    before = extractor(observations)
    before.sum().backward()
    assert torch.count_nonzero(extractor.choice_residual_projection.weight.grad) > 0
    # The first update opens the residual; following updates reach its encoder.
    optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()
    extractor(observations).sum().backward()
    assert torch.count_nonzero(extractor.choice_slot_encoder[0].weight.grad) > 0


def test_no_card_rows_have_zero_residual_after_learning(public_bank):
    _, space, _, metadata = public_bank
    extractor = _extractor(public_bank)
    observations = _observations(space, metadata)
    observations["choice_features"].zero_()
    with torch.no_grad():
        extractor.choice_residual_projection.weight.fill_(0.2)
    base = PublicFeaturesExtractor(space)
    base.load_state_dict({name: value for name, value in extractor.state_dict().items() if name in base.state_dict()})
    assert torch.equal(base(observations), extractor(observations))
    # GENERIC is an explicitly unavailable definition and must not invent one.
    observations["choice_features"][:, 0, 0] = 1
    observations["choice_features"][:, 0, 2] = metadata["card_names_by_integer_id"].index("GENERIC") / len(metadata["card_names_by_integer_id"])
    assert torch.equal(base(observations), extractor(observations))


def test_semantic_bank_hash_and_descriptor_schema_are_checked(public_bank):
    _, space, bank, metadata = public_bank
    changed = bank.copy()
    changed[1, 0, 0] += 1
    with pytest.raises(ValueError, match="recorded hash"):
        ChoiceFeaturesExtractor(space, semantic_bank=changed, semantic_metadata=metadata)
    bad_metadata = copy.deepcopy(metadata)
    bad_metadata["semantic_fields"] = list(reversed(SEMANTIC_FIELDS))
    with pytest.raises(ValueError, match="descriptor schema"):
        ChoiceFeaturesExtractor(space, semantic_bank=bank, semantic_metadata=bad_metadata)


@pytest.mark.parametrize("bad_id", [float("nan"), -0.01, 1.0, 0.3333])
def test_invalid_visible_card_identity_cannot_silently_alias(public_bank, bad_id):
    _, space, _, metadata = public_bank
    observations = _observations(space, metadata)
    observations["choice_features"][0, 1, 2] = bad_id
    with pytest.raises(ValueError, match="invalid card identity"):
        _extractor(public_bank)(observations)


class _PublicToy(gym.Env):
    """Small public-schema environment for genuine PPO/Adam state checks."""
    def __init__(self, space, metadata):
        self.observation_space = space
        self.action_space = gym.spaces.Discrete(space["choice_features"].shape[0])
        self.observation = {key: value[0].numpy() for key, value in _observations(space, metadata, batch=1).items()}
        self.steps = 0

    def action_masks(self):
        return np.array([True, True, True, False, True])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return copy.deepcopy(self.observation), {}

    def step(self, action):
        self.steps += 1
        return copy.deepcopy(self.observation), float(action == 1), self.steps == 4, False, {}


@pytest.fixture
def learned_source(public_bank):
    _, space, _, metadata = public_bank
    torch.set_num_threads(2)
    vec = DummyVecEnv([lambda: _PublicToy(space, metadata), lambda: _PublicToy(space, metadata)])
    model = MaskablePPO("MultiInputPolicy", vec, n_steps=4, batch_size=4, n_epochs=1,
                        policy_kwargs={"features_extractor_class": PublicFeaturesExtractor,
                                       "features_extractor_kwargs": {"features_dim": 256},
                                       "net_arch": [16, 8]}, seed=23, device="cpu")
    model.learn(total_timesteps=16)
    try:
        yield model, vec
    finally:
        vec.close()


def test_migration_preserves_all_original_weights_values_action_probabilities_and_counter(public_bank, learned_source):
    upstream, _, _, _ = public_bank
    source, vec = learned_source
    candidate, audit = migrate_model(source, vec, SimpleNamespace(), upstream)
    assert audit["num_timesteps"] == source.num_timesteps == 16
    for name, value in source.policy.state_dict().items():
        assert torch.equal(value, candidate.policy.state_dict()[name]), name
    identity = verify_initial_identity(source, candidate, vec.reset(), np.array([[1, 1, 1, 0, 1]] * 2))
    assert identity["distribution_exact"] and identity["value_exact"]
    assert identity["distribution_max_abs_diff"] == identity["value_max_abs_diff"] == 0
    assert source._n_updates == candidate._n_updates


def test_migration_keeps_adam_moments_steps_defaults_and_original_groups(public_bank, learned_source):
    upstream, _, _, _ = public_bank
    source, vec = learned_source
    candidate, audit = migrate_model(source, vec, SimpleNamespace(), upstream)
    old_params = dict(source.policy.named_parameters())
    new_params = dict(candidate.policy.named_parameters())
    old_names = {id(value): key for key, value in old_params.items()}
    new_names = {id(value): key for key, value in new_params.items()}
    assert source.policy.optimizer.state
    for old_group, new_group in zip(source.policy.optimizer.param_groups, candidate.policy.optimizer.param_groups):
        assert [old_names[id(value)] for value in old_group["params"]] == [new_names[id(value)] for value in new_group["params"] if new_names[id(value)] in old_params]
        assert {key: value for key, value in old_group.items() if key != "params"} == {key: value for key, value in new_group.items() if key != "params"}
    assert source.policy.optimizer.defaults == candidate.policy.optimizer.defaults
    for name, parameter in old_params.items():
        old_state = source.policy.optimizer.state[parameter]
        new_state = candidate.policy.optimizer.state[new_params[name]]
        assert old_state.keys() == new_state.keys()
        for key, value in old_state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, new_state[key]), (name, key)
            else:
                assert value == new_state[key]
    assert audit["new_parameters_have_empty_optimizer_state"]
    assert all(parameter not in candidate.policy.optimizer.state for name, parameter in new_params.items() if name not in old_params)


def test_migration_does_not_advance_python_numpy_or_torch_rng(public_bank, learned_source):
    upstream, _, _, _ = public_bank
    source, vec = learned_source
    py = random.getstate()
    numpy = np.random.get_state()
    torch_state = torch.get_rng_state()
    migrate_model(source, vec, SimpleNamespace(), upstream)
    assert py == random.getstate()
    assert numpy[0] == np.random.get_state()[0] and np.array_equal(numpy[1], np.random.get_state()[1])
    assert numpy[2:] == np.random.get_state()[2:]
    assert torch.equal(torch_state, torch.get_rng_state())


def test_saved_treatment_restores_bank_predictions_and_adam_without_rebuilding(public_bank, learned_source, tmp_path, monkeypatch):
    upstream, _, _, _ = public_bank
    source, vec = learned_source
    candidate, _ = migrate_model(source, vec, SimpleNamespace(), upstream)
    path = tmp_path / "candidate.zip"
    candidate.save(path)
    monkeypatch.setattr("spire2_rl.choice_encoder.build_public_card_bank", lambda *a: (_ for _ in ()).throw(AssertionError("Save/load must use stored bank")))
    restored = MaskablePPO.load(path, env=vec, device="cpu")
    identity = verify_initial_identity(candidate, restored, vec.reset(), np.array([[1, 1, 1, 0, 1]] * 2))
    assert identity["distribution_exact"] and identity["value_exact"]
    assert restored.num_timesteps == candidate.num_timesteps
    assert torch.equal(restored.policy.features_extractor.choice_semantic_bank,
                       candidate.policy.features_extractor.choice_semantic_bank)
    assert len(restored.policy.optimizer.state) == len(candidate.policy.optimizer.state)
    restored_named = dict(restored.policy.named_parameters())
    for name, parameter in candidate.policy.named_parameters():
        if parameter not in candidate.policy.optimizer.state:
            assert restored_named[name] not in restored.policy.optimizer.state
            continue
        for key, value in candidate.policy.optimizer.state[parameter].items():
            actual = restored.policy.optimizer.state[restored_named[name]][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, actual), (name, key)
            else:
                assert value == actual
    restored.learn(total_timesteps=8, reset_num_timesteps=False)
    assert restored.num_timesteps == candidate.num_timesteps + 8


def test_treatment_can_continue_real_ppo_with_preserved_counter(public_bank, learned_source):
    upstream, _, _, _ = public_bank
    source, vec = learned_source
    candidate, _ = migrate_model(source, vec, SimpleNamespace(), upstream)
    before = source.num_timesteps
    candidate.learn(total_timesteps=8, reset_num_timesteps=False)
    assert candidate.num_timesteps == before + 8
    assert torch.count_nonzero(candidate.policy.features_extractor.choice_residual_projection.weight) > 0
    assert source.num_timesteps == before


def test_migration_rejects_hyperparameter_or_encoder_changes(public_bank, learned_source):
    upstream, _, _, _ = public_bank
    source, vec = learned_source
    with pytest.raises(ValueError, match="cannot change PPO gamma"):
        migrate_model(source, vec, SimpleNamespace(gamma=0.7), upstream)
