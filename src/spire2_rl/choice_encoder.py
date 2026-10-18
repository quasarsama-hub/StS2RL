"""An additive, public-card representation trial and exact PPO migration.

The original observation, action layout and feature encoder are unchanged.
Only an initially zero residual exposes the already-visible offered cards as
shared identity embeddings and static public card descriptors.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .emulator import PINNED_UPSTREAM_COMMIT, _load_upstream, _strict_engine
from .models import PublicFeaturesExtractor


CHOICE_ENCODER_SCHEMA = "public-choice-card-residual-v1"
CARD_TYPES = ("ATTACK", "SKILL", "POWER", "STATUS", "CURSE", "QUEST")
TARGET_TYPES = ("SELF", "ANY_ENEMY", "RANDOM_ENEMY", "ALL_ENEMIES",
                "ANY_ALLY", "ALL_ALLIES", "NONE")
# In these public effect definitions the otherwise ambiguous "cards" variable
# is explicitly passed to _draw_cards. No conditional draw is simulated.
EXPLICIT_CARDS_DRAW_IDS = frozenset({"POMMEL_STRIKE", "SHRUG_IT_OFF", "BATTLE_TRANCE",
                                     "OFFERING", "BURNING_PACT"})
SEMANTIC_FIELDS = (
    "known_definition", *("type_" + x.lower() for x in CARD_TYPES),
    "energy_cost_div5", "damage_div50", "block_div50",
    *("target_" + x.lower() for x in TARGET_TYPES),
    "exhaust", "retain", "ethereal", "explicit_draw_div10", "energy_cost_x",
)


def _bank_hash(bank: np.ndarray | torch.Tensor) -> str:
    if isinstance(bank, torch.Tensor):
        bank = bank.detach().cpu().numpy()
    values = np.ascontiguousarray(bank, dtype="<f4")
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_public_card_bank(upstream_path: str | Path, observation_space: gym.spaces.Dict):
    """Read static public card definitions, never a run or a simulated future.

    Bank rows match the pinned emulator's integer hand vocabulary. Index zero
    is padding; the second dimension distinguishes base and upgraded cards.
    Missing GENERIC metadata is disclosed and zeroed, rather than fabricated.
    """
    path = Path(upstream_path).resolve()
    commit = _load_upstream(path, PINNED_UPSTREAM_COMMIT)
    vocabulary = int(observation_space.spaces["hand_card_ids"].high.max()) + 1
    with _strict_engine(path):
        module = importlib.import_module("sts2_env.gym_env.run_env")
        factory = importlib.import_module("sts2_env.cards.factory").create_card
        card_base = importlib.import_module("sts2_env.cards.base")
        mapping = module._CARD_ID_TO_INT_IDX
        if sorted(mapping.values()) != list(range(vocabulary - 1)):
            raise ValueError("Static public card bank does not match hand vocabulary")
        bank = np.zeros((vocabulary, 2, len(SEMANTIC_FIELDS)), dtype=np.float32)
        unavailable = []
        saved_counter = card_base._next_instance_id
        random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            for card_id, index in mapping.items():
                for upgraded in (False, True):
                    try:
                        card = factory(card_id, upgraded=upgraded)
                    except KeyError:
                        if card_id.name != "GENERIC":
                            raise
                        if card_id.name not in unavailable:
                            unavailable.append(card_id.name)
                        continue
                    values = [1.0]
                    values += [float(card.card_type.name == name) for name in CARD_TYPES]
                    values += [float(card.cost) / 5, float(card.base_damage or 0) / 50,
                               float(card.effect_vars.get("block", card.base_block or 0)) / 50]
                    values += [float(card.target_type.name == name) for name in TARGET_TYPES]
                    explicit_draw = card.effect_vars.get("draw", 0)
                    if card_id.name in EXPLICIT_CARDS_DRAW_IDS:
                        explicit_draw = card.effect_vars.get("cards", 0)
                    values += [float(card.exhausts), float(card.is_retain), float(card.is_ethereal),
                               float(explicit_draw) / 10,
                               float(card.has_energy_cost_x)]
                    bank[index + 1, int(upgraded)] = np.asarray(values, dtype=np.float32)
        finally:
            card_base._next_instance_id = saved_counter
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
    names = [None] * vocabulary
    for card_id, index in mapping.items():
        names[index + 1] = card_id.name
    metadata = {
        "schema": CHOICE_ENCODER_SCHEMA,
        "upstream_commit": commit,
        "semantic_fields": list(SEMANTIC_FIELDS),
        "bank_shape": list(bank.shape),
        "bank_sha256": _bank_hash(bank),
        "mapping_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
        "card_names_by_integer_id": names,
        "unavailable_definitions": unavailable,
        "explicit_cards_draw_ids": sorted(EXPLICIT_CARDS_DRAW_IDS),
        "sources": ["sts2_env.cards.factory.create_card", "sts2_env.gym_env.run_env._CARD_ID_TO_INT_IDX"],
        "scope": "Static public base/upgraded definitions only; no run state, hidden draw order, RNG or lookahead.",
        "limitations": "Numeric fields describe printed factory values, not every conditional effect or current combat modifiers. Draw is encoded only for an explicit draw effect variable. Existing upgrade flags are used exactly; pending-card actions that omit that public flag retain base descriptors. No missing flag is inferred from hidden state.",
    }
    return bank, metadata


class ChoiceFeaturesExtractor(PublicFeaturesExtractor):
    """Preserve every original feature and add an initially zero choice branch."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256,
                 embedding_dim: int = 16, *, semantic_bank, semantic_metadata,
                 choice_hidden_dim: int = 8):
        super().__init__(observation_space, features_dim=features_dim, embedding_dim=embedding_dim)
        choice_space = observation_space.spaces.get("choice_features")
        if not isinstance(choice_space, gym.spaces.Box) or len(choice_space.shape) != 2 or choice_space.shape[1] < 6:
            raise ValueError("Choice residual requires visible choice identity and upgrade columns")
        if choice_hidden_dim < 1:
            raise ValueError("Choice hidden dimension must be positive")
        bank = torch.as_tensor(semantic_bank, dtype=torch.float32).clone()
        expected = (self.vocabulary_size, 2, len(SEMANTIC_FIELDS))
        if tuple(bank.shape) != expected or not torch.isfinite(bank).all() or torch.count_nonzero(bank[0]):
            raise ValueError("Public semantic bank has an invalid shape, value or padding row")
        if semantic_metadata.get("bank_sha256") != _bank_hash(bank):
            raise ValueError("Public semantic bank differs from its recorded hash")
        if (semantic_metadata.get("schema") != CHOICE_ENCODER_SCHEMA
                or semantic_metadata.get("semantic_fields") != list(SEMANTIC_FIELDS)):
            raise ValueError("Public semantic bank uses a different descriptor schema")
        self.semantic_metadata = copy.deepcopy(semantic_metadata)
        self.choice_hidden_dim = int(choice_hidden_dim)
        self.register_buffer("choice_semantic_bank", bank)
        self.choice_slot_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim + len(SEMANTIC_FIELDS), self.choice_hidden_dim), nn.Tanh(),
        )
        self.choice_residual_projection = nn.Linear(choice_space.shape[0] * self.choice_hidden_dim,
                                                     self.features_dim, bias=False)
        nn.init.zeros_(self.choice_residual_projection.weight)

    def metadata(self) -> dict[str, Any]:
        result = copy.deepcopy(self.semantic_metadata)
        result["bank_sha256"] = _bank_hash(self.choice_semantic_bank)
        result["choice_hidden_dim"] = self.choice_hidden_dim
        result["initialization"] = "Zero residual projection; original feature modules and parameter names retained."
        return result

    def decode_choice_card_ids(self, choices: torch.Tensor) -> torch.Tensor:
        """Decode exact existing normalized IDs, not enum numbering or ranks."""
        normalized = choices[..., 2]
        scaled = normalized * self.vocabulary_size
        rounded = scaled.round()
        if (not torch.isfinite(normalized).all() or (normalized < 0).any()
                or (rounded >= self.vocabulary_size).any()
                or ((scaled - rounded).abs() > 1e-3).any()):
            raise ValueError("Visible choice contains an invalid card identity")
        return rounded.long()

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        base = super().forward(observations)
        choices = observations["choice_features"]
        ids = self.decode_choice_card_ids(choices)
        upgrades = (choices[..., 5] > 0.5).long()
        semantics = self.choice_semantic_bank[ids, upgrades]
        valid = (choices[..., 0] > 0) & (ids != 0) & (semantics[..., 0] > 0)
        encoded = self.choice_slot_encoder(torch.cat((self.card_embedding(ids), semantics), dim=-1))
        encoded = encoded * valid.unsqueeze(-1).to(encoded.dtype)
        return base + self.choice_residual_projection(encoded.flatten(start_dim=1))


def _same_tree(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right))
    return left == right


def _migrate_optimizer(source_policy, candidate_policy):
    old = source_policy.optimizer
    if not isinstance(old, torch.optim.Adam):
        raise ValueError("Choice migration requires the recorded PPO Adam optimizer")
    if len(old.param_groups) != 1:
        raise ValueError("Recorded PPO Adam must have its original single parameter group")
    old_names = {id(parameter): name for name, parameter in source_policy.named_parameters()}
    new_named = dict(candidate_policy.named_parameters())
    old_named = dict(source_policy.named_parameters())
    additional = [parameter for name, parameter in new_named.items() if name not in old_named]
    if not additional:
        raise ValueError("Choice migration has no new residual parameters")
    # Ordinary SB3 recreates one group in candidate.policy.parameters() order
    # on load. Match that order now, then attach old moments by parameter name;
    # saved optimizer indices will consequently restore to the correct tensors.
    copied_group = {key: copy.deepcopy(value) for key, value in old.param_groups[0].items() if key != "params"}
    copied_group["params"] = list(candidate_policy.parameters())
    optimizer = type(old)([copied_group], **copy.deepcopy(old.defaults))
    copied_states = 0
    for parameter, state in old.state.items():
        target = new_named[old_names[id(parameter)]]
        optimizer.state[target] = copy.deepcopy(state)
        copied_states += 1
    candidate_policy.optimizer = optimizer
    equal = all(_same_tree(state, optimizer.state[new_named[old_names[id(parameter)]]])
                for parameter, state in old.state.items())
    if not equal or any(parameter in optimizer.state for parameter in additional):
        raise ValueError("Old Adam state was not preserved exactly")
    return {"optimizer_states_copied": copied_states, "old_optimizer_states_equal": equal,
            "original_optimizer_groups": len(old.param_groups), "new_parameter_groups": 0,
            "old_group_options_preserved": True,
            "optimizer_parameter_order": "Candidate policy order for safe SB3 save/load; moments transferred by original parameter name.",
            "new_parameters_have_empty_optimizer_state": True,
            "new_parameter_tensors": len(additional)}


def migrate_model(source_model, vec, config, upstream_path: str | Path):
    """Create the treatment arm with exact old weights, Adam state and counter.

    Active episodes intentionally restart, matching the existing resume
    contract. This does not load a new teacher or alter PPO hyperparameters.
    """
    from sb3_contrib import MaskablePPO
    if type(source_model.policy.features_extractor) is not PublicFeaturesExtractor:
        raise ValueError("Source policy must use the original PublicFeaturesExtractor")
    if source_model.observation_space != vec.observation_space or source_model.action_space != vec.action_space:
        raise ValueError("Choice representation must preserve observation and action schemas")
    if source_model.n_envs != vec.num_envs:
        raise ValueError("Choice representation must preserve the number of rollout environments")
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "ent_coef"):
        if hasattr(config, key) and getattr(config, key) != getattr(source_model, key):
            raise ValueError(f"Choice representation cannot change PPO {key}")
    bank, metadata = build_public_card_bank(upstream_path, vec.observation_space)
    kwargs = copy.deepcopy(source_model.policy_kwargs)
    kwargs["features_extractor_class"] = ChoiceFeaturesExtractor
    extractor_kwargs = copy.deepcopy(kwargs.get("features_extractor_kwargs", {}))
    extractor_kwargs.update(semantic_bank=bank.tolist(), semantic_metadata=metadata, choice_hidden_dim=8)
    kwargs["features_extractor_kwargs"] = extractor_kwargs
    random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        candidate = MaskablePPO(
            type(source_model.policy), vec, learning_rate=source_model.learning_rate,
            n_steps=source_model.n_steps, batch_size=source_model.batch_size,
            n_epochs=source_model.n_epochs, gamma=source_model.gamma,
            gae_lambda=source_model.gae_lambda, clip_range=source_model.clip_range,
            clip_range_vf=source_model.clip_range_vf, normalize_advantage=source_model.normalize_advantage,
            ent_coef=source_model.ent_coef, vf_coef=source_model.vf_coef,
            max_grad_norm=source_model.max_grad_norm, rollout_buffer_class=source_model.rollout_buffer_class,
            rollout_buffer_kwargs=copy.deepcopy(source_model.rollout_buffer_kwargs),
            target_kl=source_model.target_kl, stats_window_size=source_model._stats_window_size,
            tensorboard_log=source_model.tensorboard_log, policy_kwargs=kwargs,
            verbose=source_model.verbose, seed=source_model.seed, device=source_model.device,
        )
        copied = source_model.policy.state_dict()
        result = candidate.policy.load_state_dict(copied, strict=False)
        allowed = ("choice_semantic_bank", "choice_slot_encoder.", "choice_residual_projection.")
        if result.unexpected_keys or any(not any(part in key for part in allowed) for key in result.missing_keys):
            raise ValueError("Candidate architecture failed to retain all original parameter names")
        if not all(torch.equal(value.cpu(), candidate.policy.state_dict()[name].cpu())
                   for name, value in copied.items()):
            raise ValueError("Candidate initialization changed existing policy weights")
        # SB3's policy-level orthogonal initialization revisits every Linear
        # after extractor construction, including our initially zero layer.
        # Restore the zero residual after that pass, before making a checkpoint.
        for module in candidate.policy.modules():
            if isinstance(module, ChoiceFeaturesExtractor):
                nn.init.zeros_(module.choice_residual_projection.weight)
        optimizer_audit = _migrate_optimizer(source_model.policy, candidate.policy)
        for key in ("num_timesteps", "_n_updates", "_current_progress_remaining", "_total_timesteps",
                    "_num_timesteps_at_start"):
            setattr(candidate, key, copy.deepcopy(getattr(source_model, key)))
        candidate._last_obs = None
        candidate._last_episode_starts = None
        vec._seeds = [None] * vec.num_envs
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return candidate, {
        "schema": CHOICE_ENCODER_SCHEMA, "source_parameters_copied": len(copied),
        "old_parameter_tensors_equal": True, "initial_feature_residual_zero": True,
        "num_timesteps": candidate.num_timesteps, "rng_preserved": True,
        "semantic_bank_metadata": metadata, **optimizer_audit,
        "resume_semantics": "Old policy, value, Adam and counter retained; active episodes restart on new seeds.",
    }


def verify_initial_identity(source_model, candidate_model, observations, action_masks):
    """Assert identical action distributions and critic values on public states."""
    old_obs, _ = source_model.policy.obs_to_tensor(observations)
    new_obs, _ = candidate_model.policy.obs_to_tensor(observations)
    source_model.policy.set_training_mode(False)
    candidate_model.policy.set_training_mode(False)
    with torch.no_grad():
        old_dist = source_model.policy.get_distribution(old_obs, action_masks=action_masks).distribution.probs
        new_dist = candidate_model.policy.get_distribution(new_obs, action_masks=action_masks).distribution.probs
        old_value = source_model.policy.predict_values(old_obs)
        new_value = candidate_model.policy.predict_values(new_obs)
    distributions_equal = torch.equal(old_dist, new_dist)
    values_equal = torch.equal(old_value, new_value)
    result = {"distribution_exact": distributions_equal, "value_exact": values_equal,
              "distribution_max_abs_diff": float((old_dist - new_dist).abs().max()),
              "value_max_abs_diff": float((old_value - new_value).abs().max())}
    if not distributions_equal or not values_equal:
        raise ValueError(f"Choice representation changed initial behavior: {result}")
    return result
