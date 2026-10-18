"""A small feature encoder for the complete public observation schema."""

from __future__ import annotations

import math
from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


FEATURES_SCHEMA = "public-embedding-v1"
HAND_IDS_KEY = "hand_card_ids"


class PublicFeaturesExtractor(BaseFeaturesExtractor):
    """Embed card identity and retain every other observed component.

    Card ID zero is empty-hand padding. The vocabulary comes from the declared
    integer Box, rather than an upstream enum or an assumed number of cards.
    Remaining keys, including visible choice features, are flattened in sorted
    order and compressed together into a small learned representation.
    """

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256, embedding_dim: int = 16):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("PublicFeaturesExtractor requires a Dict observation")
        if features_dim < 1 or embedding_dim < 1:
            raise ValueError("Feature and embedding dimensions must be positive")
        hand_space = observation_space.spaces.get(HAND_IDS_KEY)
        if not isinstance(hand_space, gym.spaces.Box) or len(hand_space.shape) != 1:
            raise ValueError("hand_card_ids must be a one-dimensional integer Box")
        if not np.issubdtype(hand_space.dtype, np.integer) or np.any(hand_space.low != 0):
            raise ValueError("hand_card_ids must use integer IDs with zero padding")
        if np.any(hand_space.high < 0):
            raise ValueError("hand_card_ids has an invalid vocabulary bound")
        if any(not isinstance(space, gym.spaces.Box)
               for space in observation_space.spaces.values()):
            raise ValueError("All public observation components must be Box spaces")

        super().__init__(observation_space, features_dim=features_dim)
        self.vocabulary_size = int(hand_space.high.max()) + 1
        self.embedding_dim = embedding_dim
        self.remaining_keys = tuple(sorted(key for key in observation_space.spaces
                                           if key != HAND_IDS_KEY))
        self.component_shapes = {key: space.shape
                                 for key, space in observation_space.spaces.items()}
        self.card_embedding = nn.Embedding(self.vocabulary_size, embedding_dim,
                                           padding_idx=0)
        input_size = math.prod(hand_space.shape) * embedding_dim
        input_size += sum(math.prod(self.component_shapes[key])
                          for key in self.remaining_keys)
        self.projection = nn.Sequential(nn.Linear(input_size, features_dim), nn.ReLU())

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if observations.keys() != self.component_shapes.keys():
            raise ValueError("Observation keys differ from the feature schema")
        card_ids = observations[HAND_IDS_KEY]
        if card_ids.ndim != 2 or tuple(card_ids.shape[1:]) != self.component_shapes[HAND_IDS_KEY]:
            raise ValueError("hand_card_ids has an invalid batch shape")
        # SB3 converts Box observations to floats before feature extraction.
        # Refuse malformed IDs rather than clamping them into a different card.
        if (not torch.isfinite(card_ids).all()
                or (card_ids < 0).any()
                or (card_ids >= self.vocabulary_size).any()
                or (card_ids != card_ids.long()).any()):
            raise ValueError("hand_card_ids contains an invalid card identity")
        parts = [self.card_embedding(card_ids.long()).flatten(start_dim=1)]
        batch_size = card_ids.shape[0]
        for key in self.remaining_keys:
            value = observations[key]
            if value.shape[0] != batch_size or tuple(value.shape[1:]) != self.component_shapes[key]:
                raise ValueError(f"{key} has an invalid batch shape")
            parts.append(value.reshape(batch_size, -1).to(dtype=parts[0].dtype))
        return self.projection(torch.cat(parts, dim=1))
