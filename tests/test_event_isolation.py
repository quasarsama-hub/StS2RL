"""Event ownership regressions, including the pinned engine's real failures."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from spire2_rl import event_isolation as isolation
from spire2_rl.emulator import DEFAULT_UPSTREAM_PATH, SimulatorError, make_env
from spire2_rl.event_isolation import IsolatedEvents
from spire2_rl.potion_capacity import PotionCapacityMask
from spire2_rl.selection import MonotoneSelections


class MutableEvent:
    event_id = "Example"

    def __init__(self):
        self.pending_choice = None
        self.count = 0


class RegistryTransport(gym.Env):
    """Transport fake whose resolver must belong to its current reset."""
    scope = "fullrun"

    def __init__(self, module):
        self.module = module
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.backend_metadata = {"scope": "fullrun", "nested": {"original": True}}
        self.raw_env = self
        self.observation = np.array([0.1, 0.2], dtype=np.float32)
        self.info = {"example": True}
        self.mask = np.array([True, True, False])
        self.fail = None
        self.calls = []
        self.registry_seen = None

    def reset(self, *, seed=None, options=None):
        self.registry_seen = self.module._EVENT_REGISTRY
        self.calls.append(("reset", seed, options))
        event = self.registry_seen["Example"]
        assert event.pending_choice is None and event.count == 0
        event.pending_choice = lambda: seed
        if self.fail == "reset":
            raise ValueError("Reset failed")
        self.reset_result = (self.observation, self.info)
        return self.reset_result

    def step(self, action):
        self.registry_seen = self.module._EVENT_REGISTRY
        self.calls.append(("step", action))
        event = self.registry_seen["Example"]
        if self.fail == "step":
            raise ValueError("Step failed")
        event.count += 1
        self.step_result = (self.observation, event.pending_choice(), False, False, self.info)
        return self.step_result

    def action_masks(self):
        self.registry_seen = self.module._EVENT_REGISTRY
        if self.fail == "mask":
            raise ValueError("Mask failed")
        return self.mask


@pytest.fixture
def transport(monkeypatch):
    module = SimpleNamespace(_EVENT_REGISTRY={"Example": MutableEvent()})
    original = module._EVENT_REGISTRY
    monkeypatch.setattr(isolation, "_event_module", lambda: module)
    base = RegistryTransport(module)
    return module, original, base


def test_public_transport_and_source_metadata_are_unchanged(transport):
    module, original, base = transport
    metadata = deepcopy(base.backend_metadata)
    env = IsolatedEvents(base)
    assert env.scope == base.scope and env.raw_env is base
    assert env.action_space is base.action_space
    assert env.observation_space is base.observation_space
    assert base.backend_metadata == metadata
    assert module._EVENT_REGISTRY is original
    assert env.reset(seed=42, options={"example": 1}) is base.reset_result
    assert base.registry_seen is env._registry
    assert env.step(1) is base.step_result
    assert env.action_masks() is base.mask
    assert base.calls == [("reset", 42, {"example": 1}), ("step", 1)]
    assert module._EVENT_REGISTRY is original
    data = env.fingerprint()
    assert data["event_state_isolation"] == "private-registry-per-run-v1"
    assert data["event_registry_count"] == 1
    path = Path(isolation.__file__)
    assert data["event_isolation_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    data["nested"]["original"] = False
    assert env.backend_metadata["nested"]["original"] is True
    assert base.backend_metadata == metadata


def test_two_runs_keep_their_own_resolvers_and_reset_fresh(transport):
    module, original, base = transport
    other_base = RegistryTransport(module)
    first, second = IsolatedEvents(base), IsolatedEvents(other_base)
    first.reset(seed=42)
    first_registry = first._registry
    first_callback = first_registry["Example"].pending_choice
    second.reset(seed=43)
    assert first._registry is not second._registry
    assert first._registry["Example"] is not second._registry["Example"]
    assert first.step(0)[1] == 42
    assert second.step(0)[1] == 43
    first.reset(seed=44)
    assert first._registry is not first_registry
    assert first._registry["Example"] is not first_registry["Example"]
    assert first_callback() == 42
    assert first.step(0)[1] == 44
    assert module._EVENT_REGISTRY is original
    assert original["Example"].count == 0
    assert original["Example"].pending_choice is None


@pytest.mark.parametrize("operation", ["reset", "step", "mask"])
def test_global_registry_is_restored_after_engine_failure(transport, operation):
    module, original, base = transport
    env = IsolatedEvents(base)
    if operation != "reset":
        env.reset(seed=42)
    base.fail = operation
    call = {"reset": lambda: env.reset(seed=43), "step": lambda: env.step(0),
            "mask": env.action_masks}[operation]
    with pytest.raises(ValueError, match="failed"):
        call()
    assert module._EVENT_REGISTRY is original
    base.fail = None
    env.reset(seed=44)
    assert env.step(1)[1] == 44
    assert module._EVENT_REGISTRY is original


def test_nested_engine_contexts_restore_each_previous_registry(transport):
    module, original, base = transport
    first = IsolatedEvents(base)
    second = IsolatedEvents(RegistryTransport(module))
    first.reset(seed=42)
    second.reset(seed=43)
    with first._engine_events():
        assert module._EVENT_REGISTRY is first._registry
        assert second.step(0)[1] == 43
        assert module._EVENT_REGISTRY is first._registry
    assert module._EVENT_REGISTRY is original


def test_pre_reset_calls_and_duplicate_wrapper_are_rejected(transport):
    _, _, base = transport
    env = IsolatedEvents(base)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    with pytest.raises(RuntimeError, match="reset"):
        env.action_masks()
    with pytest.raises(ValueError, match="already installed"):
        IsolatedEvents(env)


def test_event_construction_failures_leave_the_original_registry(transport):
    module, original, base = transport
    env = IsolatedEvents(base)
    class WrongIdentity:
        event_id = "Wrong"
    env._event_types["Example"] = WrongIdentity
    with pytest.raises(SimulatorError, match="identity"):
        env.reset(seed=42)
    assert module._EVENT_REGISTRY is original
    assert not base.calls
    assert env._registry is None


def test_dirty_live_singleton_state_is_never_copied(transport):
    _, original, base = transport
    original["Example"].count = 123
    original["Example"].pending_choice = lambda: "old run"
    env = IsolatedEvents(base)
    env.reset(seed=42)
    assert env.step(0)[1] == 42
    assert env._registry["Example"].count == 1
    assert original["Example"].count == 123
    assert original["Example"].pending_choice() == "old run"


def _upstream():
    path = Path(os.environ.get("SPIRE2_UPSTREAM_PATH", DEFAULT_UPSTREAM_PATH))
    if not path.is_dir():
        path = Path(__file__).resolve().parents[2] / "upstream" / "sts2-rl-agent"
    if not path.is_dir():
        pytest.skip("Pinned emulator checkout is unavailable")
    return path


def _real(*, isolated=False):
    env = PotionCapacityMask(MonotoneSelections(make_env(
        scope="fullrun", max_acts=1, max_steps=4000, upstream_path=_upstream())))
    return IsolatedEvents(env) if isolated else env


def _equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (np.ndarray, list, tuple)):
        np.testing.assert_equal(left, right)
    else:
        assert left == right


def test_real_neow_observed_relic_is_received_despite_other_env_reset():
    control = _real()
    first, second = _real(isolated=True), _real(isolated=True)
    import sts2_env.run.events as events
    original = events._EVENT_REGISTRY
    try:
        expected_reset = deepcopy(control.reset(seed=42))
        expected_mask = control.action_masks()
        expected_step = deepcopy(control.step(0))
        expected_slots = deepcopy(control.raw_env.public_slots())
        assert expected_slots == [{"action": "pick_relic_reward", "relic_id": "LOST_COFFER"}]
        _equal(first.reset(seed=42), expected_reset)
        second.reset(seed=43)
        np.testing.assert_equal(first.action_masks(), expected_mask)
        assert first.raw_env._mgr._event_model is not second.raw_env._mgr._event_model
        _equal(first.step(0), expected_step)
        assert first.raw_env.public_slots() == expected_slots
        assert events._EVENT_REGISTRY is original
    finally:
        control.close()
        first.close()
        second.close()


def test_real_original_neow_changes_displayed_reward_when_another_env_resets():
    first, second = _real(), _real()
    try:
        first.reset(seed=42)
        mgr = first.raw_env._mgr
        shown = mgr._event_options[0].label
        assert shown == "Lost Coffer"
        second.reset(seed=43)
        assert mgr._event_model is second.raw_env._mgr._event_model
        assert mgr._event_options[0].label == shown
        first.step(0)
        assert first.raw_env.public_slots() == [
            {"action": "pick_relic_reward", "relic_id": "NEOWS_TORMENT"}]
    finally:
        first.close()
        second.close()


def test_real_tablet_counter_crosstalk_is_removed_with_identical_actions():
    teacher_path = Path(__file__).resolve().parents[1] / "scripts" / "fair_teacher.py"
    teacher = runpy.run_path(str(teacher_path))["FairTeacher"]()
    original_first, original_second = _real(), _real()
    first, second = _real(isolated=True), _real(isolated=True)
    try:
        original_first.reset(seed=44)
        original_second.reset(seed=44)
        actions = []
        for _ in range(66):
            action = teacher.select(original_first)
            actions.append(action)
            original_first.step(action)
            original_second.step(action)
        left, right = original_first.raw_env._mgr, original_second.raw_env._mgr
        assert left._event_model is right._event_model
        assert left._event_model.event_id == "TabletOfTruth"
        assert (left.run_state.player.max_hp, right.run_state.player.max_hp) == (77, 74)
        _equal(first.reset(seed=44), second.reset(seed=44))
        for action in actions:
            _equal(first.action_masks(), second.action_masks())
            _equal(first.step(action), second.step(action))
        left, right = first.raw_env._mgr, second.raw_env._mgr
        assert left._event_model is not right._event_model
        assert left._event_model.event_id == right._event_model.event_id == "TabletOfTruth"
        assert (left.run_state.player.max_hp, right.run_state.player.max_hp) == (77, 77)
    finally:
        for env in (original_first, original_second, first, second):
            env.close()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_real_sequential_gameplay_and_all_public_spaces_are_unchanged(seed):
    teacher_path = Path(__file__).resolve().parents[1] / "scripts" / "fair_teacher.py"
    teacher = runpy.run_path(str(teacher_path))["FairTeacher"]()
    control, isolated = _real(), _real(isolated=True)
    import sts2_env.run.events as events
    original = events._EVENT_REGISTRY
    try:
        assert isolated.action_space == control.action_space
        assert isolated.observation_space == control.observation_space
        assert isolated.backend_metadata["event_registry_count"] == 68
        reset = deepcopy(control.reset(seed=seed))
        trace = []
        while True:
            mask = control.action_masks()
            action = teacher.select(control)
            result = control.step(action)
            trace.append((mask, action, deepcopy(result)))
            if result[2] or result[3]:
                break
        _equal(isolated.reset(seed=seed), reset)
        for mask, action, result in trace:
            _equal(isolated.action_masks(), mask)
            _equal(isolated.step(action), result)
            assert events._EVENT_REGISTRY is original
    finally:
        control.close()
        isolated.close()
