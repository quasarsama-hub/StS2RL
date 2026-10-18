"""An add-only policy interface for the emulator's multi-card selections.

The pinned engine creates these choices with an empty selected set, executes
their resolver only on confirmation, and resolves cards in original option
order. Removing deselection therefore preserves every final subset reachable
from a fresh choice while removing selection/deselection UI cycles. This is not
a guarantee that the policy can finish a run or that a native game is equivalent.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
from numbers import Integral
from pathlib import Path

import gymnasium as gym
import numpy as np

from .emulator import InvalidActionError, SimulatorError


class MonotoneSelections(gym.Wrapper):
    """Keep public observations and action slots; mask multi-card deselection.

    Single-card choices, explicit confirmation, gameplay rewards, and engine
    resolution are unchanged. Selected option descriptors remain in the public
    observations even though their deselection actions are masked. Choices must
    have valid count bounds and enough options to satisfy their minimum.

    All final subsets are preserved from the pinned engine's empty initial
    selection. Imported states with preselected defaults do not have that same
    reachability guarantee and are outside this wrapper's intended use.
    """

    def __init__(self, env: gym.Env):
        if env.scope != "fullrun":
            raise ValueError("MonotoneSelections supports fullrun environments only")
        super().__init__(env)
        self.backend_metadata = deepcopy(env.backend_metadata)
        self.backend_metadata.update(
            selection_interface="add-only-v1",
            selection_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            selection_semantics=(
                "Multi-card selections add unselected options and require explicit confirmation. "
                "Every legal final subset remains reachable from an empty initial selection; "
                "public observations, action slot identities, and gameplay rewards are unchanged."
            ),
        )
        self._mask: np.ndarray | None = None
        self._done = True

    @property
    def scope(self):
        return self.env.scope

    @property
    def raw_env(self):
        """Forward the underlying engine for diagnostics, never policy input."""
        return self.env.raw_env

    def fingerprint(self) -> dict:
        return deepcopy(self.backend_metadata)

    @staticmethod
    def _validate_choice(choice) -> None:
        minimum, maximum = choice.min_choices, choice.max_choices
        count = len(choice.options)
        valid_bounds = (
            isinstance(minimum, Integral) and not isinstance(minimum, bool)
            and isinstance(maximum, Integral) and not isinstance(maximum, bool)
            and 0 <= minimum <= maximum and minimum <= count
        )
        selected = choice.selected_indices
        valid_selected = all(
            isinstance(index, Integral) and not isinstance(index, bool)
            and 0 <= index < count for index in selected
        )
        if not valid_bounds or not valid_selected or len(selected) > maximum:
            raise SimulatorError(
                "Invalid multi-card choice contract: "
                f"options={count}, min={minimum!r}, max={maximum!r}, "
                f"selected={selected!r}"
            )

    def _current_mask(self, *, done: bool) -> np.ndarray:
        mask = np.asarray(self.env.action_masks(), dtype=bool).copy()
        if mask.shape != (self.action_space.n,):
            raise SimulatorError("Invalid action-mask shape in selection interface")
        if done:
            # Gym learners may encode a terminal observation, but stepping it
            # remains forbidden. The real guarded backend already has this
            # fallback; preserve it for terminal wrappers returning an empty mask.
            if not mask.any():
                mask[0] = True
            return mask

        raw = self.raw_env
        mgr = raw._mgr
        choice = None
        in_combat = mgr is not None and mgr.phase == mgr.PHASE_COMBAT
        if in_combat:
            combat = mgr.get_combat_state()
            if combat is None:
                raise SimulatorError("Active combat has no state in selection interface")
            choice = combat.pending_choice
        elif mgr is not None:
            choice = mgr.run_state.pending_choice
            if choice is None and mgr.phase == mgr.PHASE_EVENT and mgr._event_model is not None:
                choice = mgr._event_model.pending_choice

        if choice is not None and choice.is_multi:
            self._validate_choice(choice)
            if in_combat:
                layout = importlib.import_module("sts2_env.gym_env.run_env")._LAYOUT
                if len(choice.options) > layout.combat_size - 1:
                    raise SimulatorError("Multi-card choice options exceed fixed combat selection slots")
                for index in choice.selected_indices:
                    # Keep the existing combat option index; never compact the
                    # action layout when a previously selected slot is masked.
                    mask[1 + int(index)] = False
            else:
                for slot, action in enumerate(raw.public_slots()):
                    if (action["action"] == "choose"
                            and action["index"] in choice.selected_indices):
                        mask[slot] = False
        if not mask.any():
            raise SimulatorError("Empty active action mask in selection interface")
        return mask

    def _sync_info(self, info: dict, *, done: bool) -> dict:
        self._mask = self._current_mask(done=done)
        self._done = done
        result = dict(info)
        result["action_mask"] = self._mask.copy()
        result["selection_interface"] = "add-only-v1"
        return result

    def reset(self, *, seed=None, options=None):
        self._mask = None
        self._done = True
        observation, info = self.env.reset(seed=seed, options=options)
        info = self._sync_info(info, done=False)
        return observation, info

    def step(self, action):
        if self._done or self._mask is None:
            raise RuntimeError("Call reset() before stepping an episode")
        if not isinstance(action, (int, np.integer)) or not self.action_space.contains(action):
            raise InvalidActionError(f"Action {action!r} is outside the discrete action space")
        if not self._mask[int(action)]:
            raise InvalidActionError(f"Action {action} is masked in this selection state")
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
