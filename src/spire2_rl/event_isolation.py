"""Give each run its own event instances in the pinned Python emulator.

The upstream registry contains mutable event objects. Sharing them between
same-process environments can overwrite displayed choices, RNGs, pending
resolvers and multipage progress. This additive wrapper gives each reset fresh
instances while leaving all event selection and resolution in the engine.
It repairs that specific ownership defect; native-game parity and isolation of
other upstream global state remain unverified.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path

import gymnasium as gym

from .emulator import SimulatorError, _STRICT_LOCK


def _event_module():
    # make_env() already selects and validates the checkout before this wrapper
    # is constructed. Eager registration also includes the starting Neow event.
    importlib.import_module("sts2_env.events")
    return importlib.import_module("sts2_env.run.events")


class IsolatedEvents(gym.Wrapper):
    """Scope the original event registry to this environment's current run.

    Registry swaps share the existing reentrant engine lock, and always restore
    the exact previous object. Constructing fresh event classes avoids copying
    dirty live state or resolver functions that close over a previous run.
    Public observations, action slots, masks, rewards and info are delegated
    unchanged. This wrapper must surround the whole gameplay transport.
    """

    def __init__(self, env: gym.Env):
        if env.scope != "fullrun":
            raise ValueError("IsolatedEvents supports fullrun environments only")
        if "event_state_isolation" in env.backend_metadata:
            raise ValueError("Event-state isolation is already installed")
        super().__init__(env)
        with _STRICT_LOCK:
            self._events = _event_module()
            registry = self._events._EVENT_REGISTRY
            if not isinstance(registry, dict) or not registry:
                raise SimulatorError("Expected a populated event registry")
            self._event_types = {event_id: type(event) for event_id, event in registry.items()}
            for event_id, event in registry.items():
                if not isinstance(event_id, str) or event_id != event.event_id:
                    raise SimulatorError("Event registry identity does not match its key")
            # Fail before gameplay if a registered type cannot produce a fresh
            # compatible instance. These validation instances are never used.
            self._fresh_registry()
        self._registry: dict | None = None
        self._usable = False
        self.backend_metadata = deepcopy(env.backend_metadata)
        self.backend_metadata.update(
            event_state_isolation="private-registry-per-run-v1",
            event_isolation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            event_registry_count=len(self._event_types),
        )

    @property
    def scope(self):
        return self.env.scope

    @property
    def raw_env(self):
        """Forward diagnostics to the original engine, never policy inputs."""
        return self.env.raw_env

    def fingerprint(self) -> dict:
        return deepcopy(self.backend_metadata)

    def _fresh_registry(self) -> dict:
        registry = {}
        for event_id, event_type in self._event_types.items():
            try:
                event = event_type()
                if event.event_id != event_id:
                    raise ValueError("Fresh event identity does not match its registry key")
                if getattr(event, "pending_choice", None) is not None:
                    raise ValueError("Fresh event unexpectedly has a pending resolver")
            except Exception as exc:
                raise SimulatorError(f"Cannot construct fresh event {event_id}: {exc}") from exc
            registry[event_id] = event
        return registry

    @contextmanager
    def _engine_events(self):
        with _STRICT_LOCK:
            if self._registry is None:
                raise RuntimeError("Call reset() before using isolated events")
            previous = self._events._EVENT_REGISTRY
            self._events._EVENT_REGISTRY = self._registry
            try:
                yield
            finally:
                self._events._EVENT_REGISTRY = previous

    def reset(self, *, seed=None, options=None):
        self._usable = False
        with _STRICT_LOCK:
            self._registry = None
            self._registry = self._fresh_registry()
            with self._engine_events():
                result = self.env.reset(seed=seed, options=options)
        self._usable = True
        return result

    def step(self, action):
        if not self._usable:
            raise RuntimeError("Call reset() before stepping isolated events")
        with self._engine_events():
            return self.env.step(action)

    def action_masks(self):
        if not self._usable:
            raise RuntimeError("Call reset() before requesting isolated event actions")
        with self._engine_events():
            return self.env.action_masks()
