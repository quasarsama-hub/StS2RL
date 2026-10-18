"""A thin, fast handle on the native run simulator.

`sts2_gym.Sts2RunEnv` rebuilds a ~40-field info dict on every step, which costs more than
the step itself. Search clones and steps thousands of times per decision, so this wrapper
talks to the native library directly and reads what it needs out of the observation
buffer, which the native side rewrites in place after every call.
"""

from __future__ import annotations

import ctypes

import numpy as np

from . import paths  # noqa: F401  (puts the simulator on sys.path)
from sts2_gym import native  # noqa: E402
from sts2_gym import run_constants as RC  # noqa: E402

_lib = native._lib

OBS_SIZE = native.RUN_OBS_SIZE
MAX_ACTIONS = native.RUN_MAX_ACTIONS
LAYOUT = native.RUN_OBS_LAYOUT
C = native.OBS_SIZE  # where the run block starts; [0, C) is the combat observation

# Combat observation layout (all read from the native library).
HAND = native.OBS_HAND_OFFSET
MAX_HAND = native.OBS_MAX_HAND
CARD_SLOT = native.OBS_CARD_SLOT_SIZE
ENEMY = native.OBS_ENEMY_OFFSET
ENEMY_SLOT = native.OBS_ENEMY_SLOT_SIZE
MAX_ENEMIES = native.MAX_ENEMIES
PLAYER_BUFF = native.OBS_PLAYER_BUFF_OFFSET
MAX_PLAYER_BUFFS = native.OBS_MAX_PLAYER_BUFFS
SEL_KIND = native.OBS_SELECTION_KIND_OFFSET
SEL_COUNT = native.OBS_SELECTION_COUNT_OFFSET
SEL = native.OBS_SELECTION_OFFSET
COMBAT_POTION = native.OBS_HAND_OFFSET + native.OBS_MAX_HAND * native.OBS_CARD_SLOT_SIZE
COMBAT_POTION_SLOTS = 6

INTENT_ATTACK = 0

PHASE_COMBAT = RC.PHASE_COMBAT
PHASE_CARD_REWARD = RC.PHASE_CARD_REWARD
PHASE_MAP = RC.PHASE_MAP
PHASE_REST = RC.PHASE_REST
PHASE_SHOP = RC.PHASE_SHOP
PHASE_RELIC_REWARD = RC.PHASE_RELIC_REWARD
PHASE_COMPLETE = RC.PHASE_COMPLETE
PHASE_EVENT = RC.PHASE_EVENT
PHASE_ANCIENT = RC.PHASE_ANCIENT
NUM_PHASES = 16

# Total floors in a full run: three acts, boss on the 17th floor of each.
FLOORS_PER_ACT = 17
TOTAL_FLOORS = 3 * FLOORS_PER_ACT


class PoolExhausted(RuntimeError):
    pass


class SimError(RuntimeError):
    """The engine threw. The run (or clone) is in an unknown state and must be dropped."""


ENGINE_ERROR = -3
_lib.Sts2Agent_LastError.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
_lib.Sts2Agent_LastError.restype = ctypes.c_int


def last_engine_error() -> str:
    buf = (ctypes.c_ubyte * 8192)()
    n = _lib.Sts2Agent_LastError(buf, 8192)
    return bytes(buf[:n]).decode("utf-8", "replace")


class Run:
    """One native run. Close it (or use `with`) -- the native pool holds 255 handles."""

    __slots__ = ("handle", "_obs_buf", "obs", "_rew", "_term", "_trunc", "_mask_buf", "_mask", "seed")

    def __init__(self, handle: int, seed: str = ""):
        self.handle = handle
        self.seed = seed
        self._obs_buf = (ctypes.c_int * OBS_SIZE)()
        self.obs = np.ctypeslib.as_array(self._obs_buf)
        self._rew = (ctypes.c_float * 1)()
        self._term = (ctypes.c_int * 1)()
        self._trunc = (ctypes.c_int * 1)()
        self._mask_buf = (ctypes.c_int * MAX_ACTIONS)()
        self._mask = np.ctypeslib.as_array(self._mask_buf)

    # -- lifecycle ---------------------------------------------------------------------

    @classmethod
    def new(cls, seed: str | int) -> Run:
        handle = native.run_create()
        if handle < 0:
            raise PoolExhausted("native run pool is full")
        run = cls(handle, str(seed))
        status = native.run_reset(handle, str(seed), run._obs_buf)
        if status != 0:
            run.close()
            raise RuntimeError(f"run reset failed ({status})")
        return run

    def clone(self, resample: bool = True, seed: int = 0) -> Run:
        """Fork. With `resample`, everything the player has not seen is re-rolled.

        Never search on a `resample=False` clone: it knows the real future (draw order,
        rewards, encounters). It exists for replaying a run exactly.
        """
        child = Run.__new__(Run)
        Run.__init__(child, -1, self.seed)
        handle = int(_lib.Sts2Run_Clone(self.handle, 1 if resample else 0, int(seed) & 0x7FFFFFFF, child._obs_buf))
        if handle < 0:
            raise PoolExhausted("native run pool is full (clone)")
        child.handle = handle
        return child

    def close(self) -> None:
        if self.handle >= 0:
            _lib.Sts2Run_Destroy(self.handle)
            self.handle = -1

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- acting ------------------------------------------------------------------------

    def step(self, action: int, target: int = -1) -> bool:
        """Apply one action. Returns False if the simulator refused it; raises SimError if it broke."""
        status = _lib.Sts2Run_Step(
            self.handle, int(action), int(target), self._obs_buf, self._rew, self._term, self._trunc
        )
        if status == ENGINE_ERROR:
            raise SimError(last_engine_error())
        return status == 0

    def legal(self) -> np.ndarray:
        _lib.Sts2Run_ActionMask(self.handle, self._mask_buf, MAX_ACTIONS)
        return np.flatnonzero(self._mask)

    # -- reading -----------------------------------------------------------------------

    @property
    def phase(self) -> int:
        return int(self.obs[C])

    @property
    def done(self) -> bool:
        return int(self.obs[C]) == PHASE_COMPLETE

    @property
    def floor(self) -> int:
        return int(self.obs[C + 1])

    @property
    def act_id(self) -> int:
        return int(self.obs[C + 2])

    @property
    def gold(self) -> int:
        return int(self.obs[C + 4])

    @property
    def hp(self) -> int:
        return int(self.obs[C + 5])

    @property
    def max_hp(self) -> int:
        return int(self.obs[C + 6])

    @property
    def node_type(self) -> int:
        return int(self.obs[C + 8])

    @property
    def event_id(self) -> int:
        return int(self.obs[C + LAYOUT["map_node_type_offset"] + LAYOUT["map_choices"] + 1])

    def won(self) -> bool:
        return bool(_lib.Sts2Run_PlayerWon(self.handle))

    def encounter_id(self) -> int:
        return int(_lib.Sts2Run_EncounterId(self.handle))

    def state_list(self, list_id: int, capacity: int = 64) -> tuple[int, ...]:
        return native.run_state_list(self.handle, list_id, capacity)

    def potions(self) -> tuple[int, ...]:
        return self.state_list(2, 6)

    def deck(self) -> np.ndarray:
        """[n, 4] array of (card id, upgraded, enchantment, enchant amount)."""
        base = C + LAYOUT["deck_offset"]
        block = self.obs[base : base + LAYOUT["max_deck"] * LAYOUT["deck_slot_size"]].reshape(-1, LAYOUT["deck_slot_size"])
        n = int(np.argmin(block[:, 0] != 0)) if block[-1, 0] == 0 else len(block)
        return block[:n].copy()

    def relics(self) -> np.ndarray:
        """[n, 3] array of (relic id, counter, used up)."""
        base = C + LAYOUT["relic_offset"]
        block = self.obs[base : base + LAYOUT["max_relics"] * LAYOUT["relic_slot_size"]].reshape(-1, LAYOUT["relic_slot_size"])
        n = int(np.argmin(block[:, 0] != 0)) if block[-1, 0] == 0 else len(block)
        return block[:n].copy()

    def act_summary(self) -> tuple[int, int, int]:
        """(act id, boss encounter id, map node count) -- cheap, unlike map_graph()."""
        s = native.run_state_list(self.handle, 14, 3)
        return (s[0], s[1], s[2]) if len(s) == 3 else (self.act_id, -1, 0)

    def map_position(self) -> tuple[int, int] | None:
        cur = native.run_state_list(self.handle, 21, 2)
        return (cur[0], cur[1]) if len(cur) == 2 else None

    def map_graph(self) -> dict:
        nodes_flat = native.run_state_list(self.handle, 15, 3 * 1024)
        nodes = {(nodes_flat[i], nodes_flat[i + 1]): nodes_flat[i + 2] for i in range(0, len(nodes_flat) - 2, 3)}
        edges_flat = native.run_state_list(self.handle, 16, 4 * 2048)
        edges = [
            ((edges_flat[i], edges_flat[i + 1]), (edges_flat[i + 2], edges_flat[i + 3]))
            for i in range(0, len(edges_flat) - 3, 4)
        ]
        current = native.run_state_list(self.handle, 21, 2)
        act_summary = native.run_state_list(self.handle, 14, 3)
        return {
            "nodes": nodes,
            "edges": edges,
            "current": (current[0], current[1]) if len(current) == 2 else None,
            "boss": act_summary[1] if len(act_summary) > 1 else -1,
        }

    def map_choice_coords(self) -> list[tuple[int, int] | None]:
        types = self.obs[C + LAYOUT["map_node_type_offset"] : C + LAYOUT["map_node_type_offset"] + LAYOUT["map_choices"]]
        coords = native.run_state_list(self.handle, 7, LAYOUT["map_choices"] * 2)
        out = []
        for i in range(LAYOUT["map_choices"]):
            out.append((coords[2 * i], coords[2 * i + 1]) if types[i] != 0 and 2 * i + 1 < len(coords) else None)
        return out

    # -- combat views ------------------------------------------------------------------

    def hand(self) -> list[tuple[int, int]]:
        """(card id, upgraded) for each card in hand, in hand-index order."""
        out = []
        for i in range(MAX_HAND):
            cid = int(self.obs[HAND + i * CARD_SLOT])
            if cid == 0:
                break
            out.append((cid, int(self.obs[HAND + i * CARD_SLOT + 1])))
        return out

    def enemies(self) -> list[tuple[int, int, int, int, int]]:
        """(hp, max hp, block, intent type, announced magnitude) per enemy slot."""
        out = []
        for i in range(MAX_ENEMIES):
            at = ENEMY + i * ENEMY_SLOT
            max_hp = int(self.obs[at + 1])
            if max_hp == 0:
                break
            out.append((int(self.obs[at]), max_hp, int(self.obs[at + 2]), int(self.obs[at + 3]), int(self.obs[at + 4])))
        return out

    def living_enemy_indices(self) -> list[int]:
        return [i for i, e in enumerate(self.enemies()) if e[0] > 0]

    def combat_potions(self) -> list[int]:
        return [int(self.obs[COMBAT_POTION + 2 * s]) for s in range(COMBAT_POTION_SLOTS)]

    def pending_selection(self) -> int:
        """Number of candidates on an open in-combat card-selection screen (0 if none)."""
        return int(self.obs[SEL_COUNT])

    @property
    def energy(self) -> int:
        return int(self.obs[3])

    @property
    def block(self) -> int:
        return int(self.obs[2])

    @property
    def combat_hp(self) -> int:
        return int(self.obs[0])


def run_score(run: Run, max_act: int = 3) -> float:
    """The quantity everything is trained to maximise, in [0, 1].

    Half of it is progress (floors cleared, out of the floors the run can reach) and half
    is winning. Progress gives the learner a gradient long before it wins anything;
    once it does win, the win term dominates.
    """
    total = FLOORS_PER_ACT * max_act
    floor = min(run.floor, total)
    won = run.done and run.won() and run.hp > 0
    if won:
        return 1.0
    return 0.5 * floor / total
