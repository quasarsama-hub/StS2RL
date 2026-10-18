"""Mask an impossible potion pickup without changing any public action slot.

The pinned manager advertises potion reward pickup even when every inventory
slot is occupied. Its reward handler then returns failure and consumes that
reward. This wrapper checks only the visible slot capacity and occupancy before
the policy chooses. It does not discard potions, skip rewards, or repair other
engine errors, and it makes no native-game parity claim.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Integral
from pathlib import Path

import gymnasium as gym
import numpy as np

from .emulator import InvalidActionError, SimulatorError


class PotionCapacityMask(gym.Wrapper):
    """Leave observations, slots and rewards intact; mask full-inventory pickup.

    An omitted trailing inventory entry is an empty slot, matching the engine's
    add_potion implementation. Inventory entries beyond max_potion_slots cannot
    make space. The pickup remains in public_slots and its choice features stay
    visible. Only its existing mask bit changes when no available slot exists.
    """

    def __init__(self, env: gym.Env):
        if env.scope != "fullrun":
            raise ValueError("PotionCapacityMask supports fullrun environments only")
        super().__init__(env)
        self.backend_metadata = deepcopy(env.backend_metadata)
        self.backend_metadata.update(
            potion_capacity_interface="mask-full-inventory-v1",
            potion_capacity_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            potion_capacity_semantics=(
                "Noncombat potion reward pickup is masked only when every public inventory "
                "slot within max_potion_slots is occupied. Observations and original action "
                "slot identities remain unchanged. No automatic discard, skip, substituted "
                "action or gameplay reward change occurs."
            ),
        )
        self._mask: np.ndarray | None = None
        self._done = True

    @property
    def scope(self):
        return self.env.scope

    @property
    def raw_env(self):
        """Forward the engine for diagnostics, never policy observations."""
        return self.env.raw_env

    def fingerprint(self) -> dict:
        return deepcopy(self.backend_metadata)

    def _current_mask(self, *, done: bool) -> np.ndarray:
        mask = np.asarray(self.env.action_masks(), dtype=bool).copy()
        if mask.shape != (self.action_space.n,):
            raise SimulatorError("Invalid action-mask shape in potion capacity interface")
        if done:
            if not mask.any():
                mask[0] = True
            return mask

        mgr = self.raw_env._mgr
        if mgr is not None and mgr.phase != mgr.PHASE_COMBAT:
            pickups = [slot for slot, action in enumerate(self.raw_env.public_slots())
                       if action["action"] == "pick_potion"]
            if pickups:
                player = mgr.run_state.player
                capacity, potions = player.max_potion_slots, player.potions
                if (not isinstance(capacity, Integral) or isinstance(capacity, bool)
                        or capacity < 0 or not isinstance(potions, (list, tuple))):
                    raise SimulatorError("Invalid public potion capacity or inventory")
                full = all(index < len(potions) and potions[index] is not None
                           for index in range(int(capacity)))
                if full:
                    for slot in pickups:
                        if slot >= self.action_space.n:
                            raise SimulatorError("Potion pickup exceeds fixed action slots")
                        mask[slot] = False
        if not mask.any():
            raise SimulatorError("Empty active action mask in potion capacity interface")
        return mask

    def _sync_info(self, info: dict, *, done: bool) -> dict:
        self._mask = self._current_mask(done=done)
        self._done = done
        result = dict(info)
        result["action_mask"] = self._mask.copy()
        result["potion_capacity_interface"] = "mask-full-inventory-v1"
        return result

    def reset(self, *, seed=None, options=None):
        self._mask = None
        self._done = True
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, self._sync_info(info, done=False)

    def step(self, action):
        if self._done or self._mask is None:
            raise RuntimeError("Call reset() before stepping an episode")
        if not isinstance(action, (int, np.integer)) or not self.action_space.contains(action):
            raise InvalidActionError(f"Action {action!r} is outside the discrete action space")
        if not self._mask[int(action)]:
            raise InvalidActionError(f"Action {action} is masked in this potion capacity state")
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
            info = self._sync_info(info, done=bool(terminated or truncated))
        except Exception:
            self._mask = None
            self._done = True
            raise
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("Call reset() before requesting legal actions")
        return self._mask.copy()
