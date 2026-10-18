"""Potion capacity transport, including the pinned engine's real reward handler."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from spire2_rl.emulator import (DEFAULT_UPSTREAM_PATH, InvalidActionError,
                               SimulatorError, make_env)
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.selection import MonotoneSelections


@pytest.fixture(autouse=True)
def checkout_path(monkeypatch):
    if "SPIRE2_UPSTREAM_PATH" not in os.environ and not DEFAULT_UPSTREAM_PATH.is_dir():
        checkout = Path(__file__).resolve().parents[2] / "upstream" / "sts2-rl-agent"
        if checkout.is_dir():
            monkeypatch.setenv("SPIRE2_UPSTREAM_PATH", str(checkout))


class PublicCapacityEnv(gym.Env):
    """Action transport fake; gameplay resolution is tested with the real engine."""
    scope = "fullrun"

    def __init__(self, *, capacity=3, potions=None, slots=None, phase="CARD_REWARD"):
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict({"choice_features": spaces.Box(
            low=0, high=10, shape=(8, 32), dtype=np.float32)})
        self.backend_metadata = {"scope": "fullrun", "nested": {"original": True}}
        self.raw_env = self
        self.player = SimpleNamespace(max_potion_slots=capacity,
                                      potions=[object()] * 3 if potions is None else potions)
        self._mgr = SimpleNamespace(PHASE_COMBAT="COMBAT", phase=phase,
                                    run_state=SimpleNamespace(player=self.player))
        self.slots = slots or [{"action": "skip_potion"}, {"action": "pick_potion"},
                               {"action": "unrelated"}]
        self.steps = []
        self.fail = None
        self.done = False

    def public_slots(self):
        return self.slots

    def action_masks(self):
        mask = np.zeros(8, dtype=bool)
        mask[:len(self.slots)] = True
        return mask

    def observe(self):
        self.last_observation = {"choice_features": np.arange(256, dtype=np.float32).reshape(8, 32)}
        return self.last_observation

    def reset(self, *, seed=None, options=None):
        self.steps = []
        return self.observe(), {"action_mask": self.action_masks(), "outcome": "running"}

    def step(self, action):
        self.steps.append(action)
        if self.fail is not None:
            raise self.fail
        return self.observe(), 0.37, self.done, False, {
            "action_mask": self.action_masks(), "outcome": "win" if self.done else "running"}


@pytest.mark.parametrize("capacity,potions,masked", [
    (3, [1, 2, 3], True), (3, [1, None, 3], False), (3, [1, 2], False),
    (3, [], False), (0, [], True), (2, [1, 2, None], True),
    (4, [1, 2, 3], False), (np.int64(3), [1, 2, 3], True),
])
def test_masks_exactly_when_all_capacity_slots_are_occupied(capacity, potions, masked):
    base = PublicCapacityEnv(capacity=capacity, potions=potions)
    env = PotionCapacityMask(base)
    before_slots = deepcopy(base.slots)
    obs, info = env.reset()
    expected = base.action_masks()
    expected[1] = not masked
    np.testing.assert_array_equal(info["action_mask"], expected)
    np.testing.assert_array_equal(env.action_masks(), expected)
    assert obs is base.last_observation
    assert base.slots == before_slots
    assert base.player.potions == potions
    assert base.steps == []
    env.close()


def test_slot_identity_features_skip_and_rewards_stay_unchanged():
    slots = [{"action": "unrelated"}, {"action": "skip_potion"},
             {"action": "another"}, {"action": "pick_potion"}]
    base = PublicCapacityEnv(slots=slots)
    env = PotionCapacityMask(base)
    obs, info = env.reset()
    assert np.flatnonzero(info["action_mask"]).tolist() == [0, 1, 2]
    np.testing.assert_array_equal(obs["choice_features"], base.last_observation["choice_features"])
    obs, reward, terminated, truncated, info = env.step(1)
    assert base.steps == [1]
    assert reward == 0.37 and not terminated and not truncated
    assert obs is base.last_observation
    assert env.raw_env is base and env.scope == "fullrun"
    assert base.slots == slots
    assert not info["action_mask"][3]
    env.close()


def test_masked_pickup_rejected_before_mutation_and_mask_is_copied():
    base = PublicCapacityEnv()
    env = PotionCapacityMask(base)
    env.reset()
    with pytest.raises(InvalidActionError, match="masked"):
        env.step(1)
    assert base.steps == []
    mask = env.action_masks()
    mask[:] = False
    assert env.action_masks()[0]
    env.step(0)
    assert base.steps == [0]
    env.close()


@pytest.mark.parametrize("phase,slots", [
    ("COMBAT", [{"action": "skip_potion"}, {"action": "pick_potion"}]),
    ("SHOP", [{"action": "buy_potion"}, {"action": "leave_shop"}]),
    ("EVENT", [{"action": "choose"}, {"action": "confirm_choice"}]),
])
def test_combat_and_unrelated_public_decisions_are_unchanged(phase, slots):
    base = PublicCapacityEnv(phase=phase, slots=slots)
    env = PotionCapacityMask(base)
    obs, info = env.reset()
    np.testing.assert_array_equal(env.action_masks(), base.action_masks())
    np.testing.assert_array_equal(info["action_mask"], base.action_masks())
    assert obs is base.last_observation
    env.step(1)
    assert base.steps == [1]
    env.close()


@pytest.mark.parametrize("capacity,potions", [(True, []), (-1, []), (1.5, []), (3, None)])
def test_malformed_capacity_contract_raises(capacity, potions):
    base = PublicCapacityEnv(capacity=capacity)
    base.player.potions = potions
    env = PotionCapacityMask(base)
    with pytest.raises(SimulatorError, match="capacity or inventory"):
        env.reset()
    env.close()


def test_unexpected_engine_failure_still_raises_and_requires_reset():
    base = PublicCapacityEnv(potions=[None, None, None])
    env = PotionCapacityMask(base)
    env.reset()
    base.fail = SimulatorError("Unrelated engine failure")
    with pytest.raises(SimulatorError, match="Unrelated engine"):
        env.step(1)
    assert base.steps == [1]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.action_masks()
    env.close()


def test_terminal_and_pre_reset_steps_are_rejected():
    base = PublicCapacityEnv()
    env = PotionCapacityMask(base)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset()
    base.done = True
    _, reward, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated and reward == 0.37
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.close()


def test_metadata_is_separate_and_source_identity_is_explicit():
    base = PublicCapacityEnv()
    original = deepcopy(base.backend_metadata)
    env = PotionCapacityMask(base)
    data = env.fingerprint()
    assert data["potion_capacity_interface"] == "mask-full-inventory-v1"
    path = Path(__file__).resolve().parents[1] / "src/spire2_rl/potion_capacity.py"
    assert data["potion_capacity_adapter_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    data["nested"]["original"] = False
    assert env.backend_metadata["nested"]["original"] is True
    assert base.backend_metadata == original
    env.close()


def _real_reward(env, *, empty=False, capacity=3):
    """Set a real PotionReward contract state, without replacing engine methods."""
    from sts2_env.run.reward_objects import PotionReward
    from sts2_env.potions.base import all_potion_models, create_potion
    raw = env.raw_env
    mgr, player = raw._mgr, raw._mgr.run_state.player
    potion_id = all_potion_models()[0].potion_id
    player.max_potion_slots = capacity
    player.potions = [create_potion(potion_id) for _ in range(capacity)]
    if empty and capacity:
        player.potions[-1] = None
    reward = PotionReward(player.player_id, potion_id)
    reward.is_populated = True
    mgr.run_state.pending_choice = None
    mgr._offered_cards, mgr._offered_card_bundles = [], []
    mgr._offered_relic = None
    mgr._event_model = None
    mgr._current_reward = reward
    mgr._offered_potion = create_potion(potion_id)
    mgr._phase = mgr.PHASE_CARD_REWARD
    mgr._pending_rewards = []
    mgr._resume_after_reward_chain = mgr._enter_map_choice
    # Sync each existing transport's mask after setting the test-only state.
    wrappers = []
    inner = env
    while hasattr(inner, "env"):
        wrappers.append(inner)
        inner = inner.env
    guard = wrappers[-1]
    info = guard._info(raw._build_info(), terminated=False, truncated=False)
    for wrapper in reversed(wrappers[:-1]):
        info = wrapper._sync_info(info, done=False)
    return guard._observation(raw._encode_obs()), info, reward


def test_real_engine_advertises_full_pickup_then_rejects_and_consumes_reward():
    env = make_env("fullrun", max_acts=1, max_steps=4000)
    try:
        env.reset(seed=514720099)
        _, info, reward = _real_reward(env)
        mgr = env.raw_env._mgr
        assert env.raw_env.public_slots()[1]["action"] == "pick_potion"
        assert info["action_mask"][1]
        hp = mgr.run_state.player.current_hp
        with pytest.raises(SimulatorError, match="rejected advertised legal pick_potion"):
            env.step(1)
        assert mgr._offered_potion is None and mgr._current_reward is not reward
        assert mgr.run_state.player.current_hp == hp
        assert not mgr.run_state.is_over
    finally:
        env.close()


@pytest.mark.parametrize("capacity,empty", [(3, False), (0, False), (3, True)])
def test_real_reward_wrapper_changes_only_capacity_mask(capacity, empty):
    env = PotionCapacityMask(MonotoneSelections(make_env("fullrun", max_acts=1, max_steps=4000)))
    try:
        env.reset(seed=514720099)
        obs, info, reward = _real_reward(env, empty=empty, capacity=capacity)
        mgr = env.raw_env._mgr
        initial_slots = deepcopy(env.raw_env.public_slots())
        inventory = list(mgr.run_state.player.potions)
        assert info["action_mask"][0]
        assert bool(info["action_mask"][1]) == empty
        assert obs["choice_features"][1, 0] == 1
        assert env.raw_env.public_slots() == initial_slots
        if not empty:
            with pytest.raises(InvalidActionError):
                env.step(1)
            assert mgr._current_reward is reward
            assert mgr.run_state.player.potions == inventory
        action = 1 if empty else 0
        _, value, terminated, truncated, _ = env.step(action)
        assert value == 0 and not terminated and not truncated
        assert mgr._offered_potion is None
        assert len(mgr.run_state.player.held_potions()) == capacity
        assert mgr.phase == mgr.PHASE_MAP_CHOICE
    finally:
        env.close()
