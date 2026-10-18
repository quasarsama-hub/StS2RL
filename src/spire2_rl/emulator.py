"""Guarded access to an external, pinned STS2 Python emulator.

This backend is useful for offline learning experiments. It is not a claim of
parity with the native game. No upstream game code is copied into this package.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


PINNED_UPSTREAM_COMMIT = "04a62c35b0ee264f5e4a3b5c6587f7438f1a7e0c"
UPSTREAM_REPOSITORY = "https://github.com/vvsuby/sts2-rl-agent"
DEFAULT_UPSTREAM_PATH = Path(__file__).resolve().parents[2] / ".external" / "sts2-rl-agent"
_STRICT_LOCK = RLock()
_CHOICE_KINDS = (
    "confirm_choice", "choose", "move", "pick_card", "skip", "pick_relic",
    "buy_card", "buy_relic", "buy_potion", "remove_card", "leave_shop",
    "rest_option", "event_choice", "collect", "pick_potion", "skip_potion",
    "pick_relic_reward", "reroll_card_reward", "use_potion", "skip_relic",
    "pick_card_bundle", "sacrifice_card_reward",
)
CHOICE_FEATURES = 32


class SimulatorError(RuntimeError):
    """An emulator/observation failure, never a gameplay loss or RL transition."""


class InvalidActionError(ValueError):
    """The proposed action is not legal in the current observation."""


@contextmanager
def _strict_engine(upstream_path: str | Path | None = None):
    """Stop upstream from converting exceptions into fabricated game deaths."""
    with _STRICT_LOCK:
        previous = os.environ.get("STS2_STRICT")
        previous_cwd = Path.cwd()
        os.environ["STS2_STRICT"] = "1"
        try:
            # The pinned engine reads docs/CARDS_REFERENCE.md relative to cwd.
            # Keep that requirement inside the same process-global lock.
            if upstream_path is not None:
                os.chdir(upstream_path)
            yield
        finally:
            os.chdir(previous_cwd)
            if previous is None:
                os.environ.pop("STS2_STRICT", None)
            else:
                os.environ["STS2_STRICT"] = previous


def _load_upstream(path: Path, expected_commit: str | None) -> str:
    if not (path / "sts2_env" / "gym_env" / "run_env.py").is_file():
        raise FileNotFoundError(f"STS2 emulator checkout not found at {path}")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SimulatorError("The emulator must be an identifiable Git checkout") from exc
    if expected_commit is not None and (commit != expected_commit or dirty):
        raise SimulatorError(
            f"Expected clean emulator commit {expected_commit}, found {commit}"
            + (" with tracked modifications" if dirty else "")
        )
    loaded = sys.modules.get("sts2_env")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(path):
        raise SimulatorError("A different sts2_env checkout is already imported")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return commit


class _PublicRunDispatch:
    """Adapt public decisions; keep game resolution in the external manager."""

    def public_slots(self) -> list[dict]:
        if self._mgr is None or self._mgr.is_over:
            return []
        actions = self._mgr.get_available_actions()
        # Upstream deliberately masks skipping forced relic collection in its
        # original Gym adapter, although RunManager lists a skip for testing.
        if any(a["action"] == "pick_relic_reward" for a in actions):
            actions = [a for a in actions if a["action"] != "skip_relic"]
        if len(actions) > self.action_space.n:
            raise ValueError("Public decisions exceed the fixed action-space capacity")
        choice = self._mgr.run_state.pending_choice
        if choice is None and self._mgr._event_model is not None:
            choice = self._mgr._event_model.pending_choice
        if choice is not None and choice.is_multi:
            actions = [a for a in actions if a["action"] != "choose"
                       or choice.can_toggle(a["index"])]
        return actions

    def action_masks(self):
        if self._mgr is None or self._mgr.is_over or self._mgr.phase == self._mgr.PHASE_COMBAT:
            return super().action_masks()
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[:len(self.public_slots())] = 1
        return mask

    def _dispatch_public(self, action: int = 0):
        actions = self.public_slots()
        if not 0 <= action < len(actions):
            raise ValueError("Invalid public decision slot")
        result = self._mgr.take_action(actions[action])
        if result.get("success") is False:
            raise RuntimeError(f"Engine rejected advertised legal {actions[action]['action']}")

    _step_map_choice = _dispatch_public
    _step_card_reward = _dispatch_public
    _step_boss_relic = _dispatch_public
    _step_shop = _dispatch_public
    _step_rest_site = _dispatch_public
    _step_event = _dispatch_public
    _step_treasure = _dispatch_public
    _step_noncombat_choice = _dispatch_public

    def _step_combat(self, action: int):
        module = importlib.import_module("sts2_env.gym_env.run_env")
        layout = module._LAYOUT
        mgr = self._mgr
        combat = mgr.get_combat_state()
        if combat is None or combat.is_over:
            raise RuntimeError("Combat phase has no active combat")
        if layout.player_select_start <= action < layout.player_select_start + layout.player_select_size:
            players = [a for a in mgr.get_available_actions() if a["action"] == "select_player"]
            translated = players[action - layout.player_select_start]
        elif combat.pending_choice is not None:
            translated = {"action": "confirm_choice"} if action == 0 else {
                "action": "choose", "index": action - 1,
            }
        elif module.is_potion_action(action):
            slot, target = module.action_to_potion_and_target(action)
            if slot is None or not combat.use_potion(slot, target_index=target):
                raise RuntimeError("Engine rejected advertised legal potion action")
            # Potions bypass RunManager.take_action in upstream; resolve their
            # victory/death exactly once before encoding the next decision.
            if combat.is_over:
                mgr._resolve_combat_end()
            return
        else:
            hand, target = module.action_to_card_and_target(action)
            translated = {"action": "end_turn"} if hand is None else {
                "action": "play_card", "hand_index": hand,
            }
            if target is not None:
                translated["target_index"] = target
        result = mgr.take_action(translated)
        if result.get("success") is False:
            # Replacing a failed play with end_turn changes the policy's
            # action and would assign it misleading gameplay experience.
            raise RuntimeError(f"Engine rejected advertised legal {translated['action']}")


class GuardedEmulator(gym.Wrapper):
    """Stable public observations, legal actions and explicit episode outcomes.

    Full-run choice features describe options the player can currently inspect.
    Future enemy AI states, RNG state, hidden encounters and draw order are not
    observed. Errors raise: they must be logged separately from played episodes.
    """

    def __init__(self, env: gym.Env, *, scope: str, seed: int | None,
                 max_steps: int, backend_metadata: dict[str, Any]):
        super().__init__(env)
        self.scope = scope
        self.backend_metadata = backend_metadata
        self._initial_seed = seed
        self._episode_seed: int | None = None
        self._max_steps = max_steps
        self._steps = 0
        self._done = True
        self._mask: np.ndarray | None = None
        self._upstream = importlib.import_module("sts2_env.gym_env.run_env")
        self._card_lookup = {
            card.name: i for card, i in self._upstream._CARD_ID_TO_INT_IDX.items()
        }
        if scope == "fullrun":
            schema = dict(env.observation_space.spaces)
            schema["choice_features"] = spaces.Box(
                low=-1.0, high=10.0,
                shape=(env.action_space.n, CHOICE_FEATURES), dtype=np.float32,
            )
            self.observation_space = spaces.Dict(schema)

    def _error(self, operation: str, exc: Exception, action: int | None = None):
        self._done = True
        raise SimulatorError(
            f"Emulator {operation} failed (seed={self._episode_seed}, "
            f"step={self._steps}, action={action}): {type(exc).__name__}: {exc}"
        ) from exc

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and self._initial_seed is not None:
            seed = self._initial_seed
        self._initial_seed = None
        self._episode_seed = seed
        self._steps = 0
        self._done = False
        self._mask = None
        try:
            with _strict_engine(self.backend_metadata["upstream_path"]):
                obs, info = self.env.reset(seed=seed, options=options)
                obs = self._observation(obs)
                info = self._info(info, terminated=False, truncated=False)
        except Exception as exc:
            self._error("reset", exc)
        self.action_space.seed(seed)
        return obs, info

    def step(self, action: int):
        if self._done or self._mask is None:
            raise RuntimeError("Call reset() before stepping an episode")
        if not isinstance(action, (int, np.integer)) or not self.action_space.contains(action):
            raise InvalidActionError(f"Action {action!r} is outside the discrete action space")
        if not self._mask[int(action)]:
            raise InvalidActionError(f"Action {action} is masked in this state")
        self._steps += 1
        try:
            with _strict_engine(self.backend_metadata["upstream_path"]):
                obs, reward, terminated, truncated, info = self.env.step(int(action))
                terminated = bool(terminated)
                truncated = not terminated and (bool(truncated) or self._steps >= self._max_steps)
                # Upstream timeout penalty is not a real loss. Preserve any
                # shaping bonus, but remove its artificial terminal penalty.
                if truncated and info.get("truncated_by_max_steps", False):
                    reward -= self._upstream.REWARD_TIMEOUT
                if not np.isfinite(reward):
                    raise ValueError("Non-finite reward")
                obs = self._observation(obs)
                info = self._info(info, terminated=terminated, truncated=truncated)
        except Exception as exc:
            self._error("step", exc, int(action))
        self._done = terminated or truncated
        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("Call reset() before requesting legal actions")
        return self._mask.copy()

    @property
    def raw_env(self) -> gym.Env:
        """Expose the emulator for diagnostics, never as a policy input."""
        return self.env

    def fingerprint(self) -> dict[str, Any]:
        """JSON-serializable configuration saved beside every checkpoint."""
        return json.loads(json.dumps(self.backend_metadata))

    def _info(self, raw: dict, *, terminated: bool, truncated: bool) -> dict:
        mask = np.asarray(raw.get("action_mask", self.env.action_masks()), dtype=bool)
        if mask.shape != (self.action_space.n,) or not mask.any():
            raise ValueError("Invalid or empty action mask")
        self._mask = mask.copy()
        info = dict(raw)
        if self.scope == "combat":
            combat = self.env.combat
            info.update(phase="combat", hp=combat.player.current_hp,
                        max_hp=combat.player.max_hp, floor=0, act=0,
                        player_won=bool(combat.player_won))
        elif info.get("phase") == self.env._mgr.PHASE_COMBAT:
            combat = self.env._mgr.get_combat_state()
            if combat is not None:
                info.update(hp=combat.player.current_hp, max_hp=combat.player.max_hp)
        if self.scope == "fullrun" and not (terminated or truncated):
            if not self.env._mgr.get_available_actions():
                raise ValueError("Active run phase exposes no decisions")
        won = terminated and bool(info.get("player_won", False))
        info.update(
            action_mask=self._mask.copy(), episode_seed=self._episode_seed,
            step=self._steps, outcome=("win" if won else "death" if terminated
                                      else "timeout" if truncated else "running"),
            player_won=won, terminated=terminated, truncated=truncated,
            truncated_by_max_steps=truncated, backend="python_emulator",
            parity_verified=False,
            victory_scope=self.backend_metadata["victory_scope"],
        )
        return info

    def _observation(self, obs):
        if self.scope == "fullrun":
            obs = {key: np.array(value, copy=True) for key, value in obs.items()}
            obs["choice_features"] = self._choice_features()
        else:
            obs = np.array(obs, dtype=np.float32, copy=True)
            np.clip(obs, self.observation_space.low, self.observation_space.high, out=obs)
        if not self.observation_space.contains(obs):
            raise ValueError("Observation violates the pinned schema")
        return obs

    def _choice_features(self) -> np.ndarray:
        result = np.zeros((self.action_space.n, CHOICE_FEATURES), dtype=np.float32)
        mgr = self.env._mgr
        if mgr is None or mgr.is_over:
            return result
        actions = mgr.get_available_actions()
        pending_choice = None
        if mgr.phase == mgr.PHASE_COMBAT:
            combat = mgr.get_combat_state()
            if combat is not None:
                pending_choice = combat.pending_choice
        else:
            pending_choice = mgr.run_state.pending_choice
            if pending_choice is None and mgr.phase == mgr.PHASE_EVENT:
                event = mgr._event_model
                if event is not None:
                    pending_choice = event.pending_choice
        mapped: list[tuple[int, dict]] = []
        if mgr.phase == mgr.PHASE_COMBAT and any(a["action"] in {"choose", "confirm_choice"} for a in actions):
            mapped = [(0 if a["action"] == "confirm_choice" else 1 + int(a["index"]), a)
                      for a in actions if a["action"] in {"choose", "confirm_choice"}]
        elif mgr.phase != mgr.PHASE_COMBAT:
            for i, action in enumerate(self.env.public_slots()):
                a = dict(action)
                # These inventory identities are visible in the current shop;
                # upstream's public action list omits them for relics/potions.
                if a["action"] == "buy_relic":
                    a["relic_id"] = mgr._shop_inventory.relics[a["index"]].relic_id
                elif a["action"] == "buy_potion":
                    a["potion_id"] = mgr._shop_inventory.potions[a["index"]].potion_id
                if a["action"] == "event_choice":
                    a["event_id"] = getattr(mgr._event_model, "event_id", "")
                    # EventOption.description is text shown to the player;
                    # only the current offered options are consulted.
                    offered = next((o for o in mgr._event_options
                                    if o.option_id == a.get("option_id")), None)
                    if offered is not None:
                        a["description"] = offered.description
                mapped.append((i, a))
        for slot, action in mapped:
            if not 0 <= slot < self.action_space.n:
                raise ValueError("Public choice exceeds fixed action layout")
            action = dict(action)
            row = result[slot]
            kind = action["action"]
            if pending_choice is not None and kind in {"choose", "confirm_choice"}:
                # Upstream omits the prompt on run-level choose actions until
                # confirmation becomes legal. Humans can already see both the
                # instruction and its count requirement before selecting.
                action["prompt"] = pending_choice.prompt
                action["selected_count"] = len(pending_choice.selected_indices)
                if kind == "choose":
                    action["selected"] = action["index"] in pending_choice.selected_indices
                row[16] = pending_choice.min_choices / 10.0
                row[17] = pending_choice.max_choices / 10.0
                row[18] = len(pending_choice.selected_indices) / 10.0
                row[19] = float(pending_choice.allow_skip)
                public_context = {
                    "prompt": pending_choice.prompt,
                    "phase": mgr.phase,
                    "event_id": (getattr(mgr._event_model, "event_id", "")
                                 if mgr.phase == mgr.PHASE_EVENT else ""),
                    "source_piles": sorted({o.source_pile for o in pending_choice.options}),
                }
                context_digest = hashlib.blake2b(
                    json.dumps(public_context, sort_keys=True).encode(), digest_size=8,
                ).digest()
                row[20:28] = np.frombuffer(context_digest, dtype=np.uint8).astype(np.float32) / 255
            row[0] = 1
            row[1] = (_CHOICE_KINDS.index(kind) + 1) / len(_CHOICE_KINDS) if kind in _CHOICE_KINDS else 0
            card_id = action.get("card_id")
            if card_id in self._card_lookup:
                row[2] = (self._card_lookup[card_id] + 1) / (len(self._card_lookup) + 1)
            row[3] = float(action.get("price", 0)) / 1000
            row[4] = float(action.get("cost", 0)) / 5
            row[5] = float(bool(action.get("upgraded", False)))
            row[6] = float(bool(action.get("selected", False)))
            row[7] = float(action.get("selected_count", 0)) / 10
            # Stable fingerprints for visible option/relic/potion identity and
            # labels. They identify choices; they do not reveal their outcomes.
            visible = {key: action[key] for key in (
                "option_id", "label", "prompt", "description", "relic_id",
                "potion_id", "point_type", "source_pile", "enchantments", "event_id",
                "card_ids", "rarities",
                "upgraded",
            ) if key in action}
            digest = hashlib.blake2b(json.dumps(visible, sort_keys=True, default=str).encode(), digest_size=8).digest()
            row[8:16] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255
        np.clip(result, -1.0, 10.0, out=result)
        return result


def make_env(scope: str = "fullrun", seed: int | None = None, max_steps: int = 10_000,
             upstream_path: str | Path | None = None, *, max_acts: int = 3,
             ascension: int = 0, character: str = "Ironclad",
             reward_shaping: bool = False, render_mode: str | None = None,
             expected_commit: str | None = PINNED_UPSTREAM_COMMIT) -> GuardedEmulator:
    """Create an unreset environment; ``seed`` applies to its first reset.

    ``fullrun`` defaults to all three modeled acts. ``combat`` is a diagnostic
    Ironclad starter-deck encounter, not a full-run curriculum or native game.
    ``max_acts < 3`` is explicitly recorded as an act-limited victory scope.
    """
    if scope not in {"fullrun", "combat"}:
        raise ValueError("scope must be 'fullrun' or 'combat'")
    if max_steps < 1 or not 1 <= max_acts <= 3 or not 0 <= ascension <= 20:
        raise ValueError("Invalid max_steps, max_acts, or ascension")
    if scope == "combat" and (character != "Ironclad" or ascension != 0):
        raise ValueError("Combat diagnostic supports the Ironclad starter deck at A0")
    path = Path(upstream_path or os.environ.get("SPIRE2_UPSTREAM_PATH", DEFAULT_UPSTREAM_PATH)).resolve()
    commit = _load_upstream(path, expected_commit)
    if scope == "fullrun":
        base_class = importlib.import_module("sts2_env.gym_env.run_env").STS2RunEnv
        class PublicRunEnv(_PublicRunDispatch, base_class):
            pass
        env_class = PublicRunEnv
        env = env_class(
            character_id=character, ascension_level=ascension, max_steps=max_steps,
            # Upstream otherwise fabricates death at its combat-turn limit.
            # A turn cannot exceed the number of player decisions in a run.
            max_combat_turns=max_steps + 1, max_acts=max_acts,
            reward_shaping=reward_shaping, heuristic_shaping=False,
            include_deck_features=True, include_deck_upgrade_counts=True,
            include_map_options=True, include_map_lookahead=True,
            include_enemy_rich_features=True, include_pile_contents=False,
            include_future_intents=False, enable_neow=True, render_mode=render_mode,
        )
    else:
        env_class = importlib.import_module("sts2_env.gym_env.combat_env").STS2CombatEnv
        env = env_class(max_turns=max_steps + 1, render_mode=render_mode)
    metadata = {
        "backend": "python_emulator", "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": commit, "upstream_path": str(path),
        "parity_verified": False, "observation_schema": "public-v2",
        "action_features_version": "public-slot-v3",
        "choice_selection_rule_features": {
            "min_choices": 16, "max_choices": 17, "selected_count": 18,
            "allow_skip": 19, "count_normalizer": 10.0,
            "public_context_fingerprint_start": 20,
            "public_context_fingerprint_size": 8,
        },
        "combat_action_layout": "upstream-fixed",
        "noncombat_action_layout": "ordered-public-decisions",
        "scope": scope, "character": character, "ascension": ascension,
        "max_acts": max_acts, "max_steps": max_steps,
        "reward_shaping": reward_shaping,
        "known_limitations": [
            "Unverified parity with the native game",
            "Visible option labels use stable fingerprints rather than language embeddings",
            "Number of simultaneous public decisions must fit the fixed action-space capacity",
        ],
        "victory_scope": ("combat" if scope == "combat" else "fullrun" if max_acts == 3 else f"act{max_acts}"),
    }
    metadata["observation_options"] = {
        "deck_counts": scope == "fullrun", "deck_upgraded_counts": scope == "fullrun",
        "map_options": scope == "fullrun", "map_lookahead": scope == "fullrun",
        "enemy_rich_features": scope == "fullrun", "choice_features": scope == "fullrun",
        "pile_contents": False, "future_intents": False, "hidden_draw_order": False,
    }
    dependencies = {}
    for package in ("gymnasium", "numpy"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "unknown"
    metadata["dependencies"] = dependencies
    return GuardedEmulator(env, scope=scope, seed=seed, max_steps=max_steps,
                           backend_metadata=metadata)
