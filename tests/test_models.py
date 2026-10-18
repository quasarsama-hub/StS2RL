"""Verify identity semantics and complete observations in the policy encoder."""

import io

import gymnasium as gym
import numpy as np
import pytest
import torch

from spire2_rl.models import FEATURES_SCHEMA, PublicFeaturesExtractor


def _space():
    return gym.spaces.Dict({
        "hand_card_ids": gym.spaces.Box(0, 11, shape=(3,), dtype=np.int64),
        "hand_card_features": gym.spaces.Box(-1, 10, shape=(3, 2), dtype=np.float32),
        "choice_features": gym.spaces.Box(-1, 10, shape=(5, 4), dtype=np.float32),
        "run_scalar": gym.spaces.Box(-1, 10, shape=(2,), dtype=np.float32),
    })


def _observations():
    return {
        "hand_card_ids": torch.tensor([[1., 11., 0.], [3., 5., 8.]]),
        "hand_card_features": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 12,
        "choice_features": torch.arange(40, dtype=torch.float32).reshape(2, 5, 4) / 40,
        "run_scalar": torch.tensor([[0.1, 0.5], [0.4, 0.8]]),
    }


def test_card_relabeling_with_corresponding_embedding_preserves_policy_features():
    torch.manual_seed(25)
    original = PublicFeaturesExtractor(_space(), features_dim=13, embedding_dim=7)
    relabeled = PublicFeaturesExtractor(_space(), features_dim=13, embedding_dim=7)
    relabeled.load_state_dict(original.state_dict())
    # Arbitrary vocabulary numbering has no meaning. Rename cards, but retain
    # the padding ID, and move each learned row to its renamed identity.
    permutation = torch.tensor([0, 7, 9, 1, 5, 11, 3, 4, 8, 2, 10, 6])
    with torch.no_grad():
        relabeled.card_embedding.weight[permutation] = original.card_embedding.weight
    observations = _observations()
    changed = {key: value.clone() for key, value in observations.items()}
    changed["hand_card_ids"] = permutation[changed["hand_card_ids"].long()].float()
    torch.testing.assert_close(original(observations), relabeled(changed), rtol=0, atol=0)


def test_visible_choices_and_added_observation_components_reach_the_encoder():
    schema = _space()
    schema.spaces["additional_visible_state"] = gym.spaces.Box(
        0, 1, shape=(2, 2), dtype=np.float32)
    encoder = PublicFeaturesExtractor(schema, features_dim=9)
    observations = {key: torch.zeros((2, *space.shape))
                    for key, space in schema.spaces.items()}
    with torch.no_grad():
        encoder.projection[0].weight.fill_(0.1)
        encoder.projection[0].bias.fill_(1.0)
    initial = encoder(observations)
    observations["choice_features"][0, 2, 1] = 1
    after_choice = encoder(observations)
    assert torch.all(after_choice[0] > initial[0])
    torch.testing.assert_close(after_choice[1], initial[1])
    observations["additional_visible_state"][1, 1, 0] = 1
    after_addition = encoder(observations)
    assert torch.all(after_addition[1] > after_choice[1])
    torch.testing.assert_close(after_addition[0], after_choice[0])


def test_saved_encoder_restores_dimensions_vocabulary_and_predictions():
    encoder = PublicFeaturesExtractor(_space())
    assert FEATURES_SCHEMA == "public-embedding-v1"
    assert encoder.card_embedding.num_embeddings == 12, "Box.high is inclusive"
    assert encoder.features_dim == 256
    expected = encoder(_observations())
    buffer = io.BytesIO()
    torch.save(encoder.state_dict(), buffer)
    buffer.seek(0)
    restored = PublicFeaturesExtractor(_space())
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored(_observations()).shape == (2, 256)
    torch.testing.assert_close(restored(_observations()), expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad_id", [-1., 12., 1.5, float("nan"), float("inf")])
def test_sb3_float_preprocessing_does_not_hide_invalid_card_ids(bad_id):
    encoder = PublicFeaturesExtractor(_space())
    observations = _observations()
    observations["hand_card_ids"][0, 0] = bad_id
    with pytest.raises(ValueError, match="invalid card identity"):
        encoder(observations)


def test_empty_hand_padding_stays_zero_during_training():
    encoder = PublicFeaturesExtractor(_space())
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    encoder(_observations()).sum().backward()
    assert torch.count_nonzero(encoder.card_embedding.weight.grad[0]) == 0
    optimizer.step()
    assert torch.count_nonzero(encoder.card_embedding.weight[0]) == 0
