"""Combat search: single-observer information-set MCTS over resampled clones.

Every simulation forks the real position with `resample=True`, so the draw order and the
enemies' future rolls are re-drawn from what the player could know -- the search never
reads the real future. The tree is shared across those worlds and keyed by *semantic*
actions (which card, which target) rather than hand indices, because after a draw the
same index names different cards in different worlds. Five Strikes in hand are one
action, not five.

Leaves are played out to the end of the fight by a native heuristic rollout and valued by
a caller-supplied function of the post-fight state. The agent passes its run value
function there, so a fight is scored in the same units as every other decision: HP, and
the potions still on the belt, are worth exactly what the run is worth with them.
"""

from __future__ import annotations

import ctypes
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import gamedata as gd
from . import sim

_lib = sim._lib
_lib.Sts2Agent_RolloutCombat.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.Sts2Agent_RolloutCombat.restype = ctypes.c_int

# attack, skill when threatened, skill when safe, power, end turn, other, potion, focus-fire
DEFAULT_ROLLOUT_WEIGHTS = (4.0, 3.0, 0.4, 2.0, 0.6, 1.0, 0.0, 0.75)

MAX_COMBAT_STEPS = 600


def native_rollout(run: sim.Run, seed: int, weights=DEFAULT_ROLLOUT_WEIGHTS, max_steps=MAX_COMBAT_STEPS, max_end_turns=0) -> float:
    """Play the fight out natively. Returns the fraction of enemy HP still standing."""
    buf = (ctypes.c_float * 8)(*weights)
    info = (ctypes.c_int * 4)()
    status = _lib.Sts2Agent_RolloutCombat(run.handle, int(seed) & 0x7FFFFFFF, max_steps, max_end_turns, buf, run._obs_buf, info)
    if status == sim.ENGINE_ERROR:
        raise sim.SimError(sim.last_engine_error())
    return info[0] / info[1] if info[1] > 0 else 0.0


def enemy_hp_fraction(run: sim.Run) -> float:
    enemies = run.enemies()
    total = sum(e[1] for e in enemies)
    return sum(max(0, e[0]) for e in enemies) / total if total else 0.0


# --------------------------------------------------------------------------------------
# semantic actions


def _needs_target(card_id: int) -> bool:
    card = gd.cards().get(card_id)
    return card is not None and card.needs_enemy_target


def legal_moves(run: sim.Run) -> list[tuple[tuple, int, int]]:
    """Every distinct move available in this world, as (key, action, target).

    Keys identify a move by what it is, so the same key can be looked up in another
    determinization of the same information set.
    """
    legal = run.legal()
    obs = run.obs
    moves: list[tuple[tuple, int, int]] = []
    seen: set[tuple] = set()
    n_sel = int(obs[sim.SEL_COUNT])
    if n_sel > 0:
        for a in legal:
            a = int(a)
            if a < n_sel:
                at = sim.SEL + a * sim.CARD_SLOT
                key = ("s", int(obs[at]), int(obs[at + 1]), int(obs[at + 2]))
            else:
                key = ("s", -1)
            if key not in seen:
                seen.add(key)
                moves.append((key, a, -1))
        return moves

    hand_size = 0
    while hand_size < sim.MAX_HAND and obs[sim.HAND + hand_size * sim.CARD_SLOT] != 0:
        hand_size += 1
    targets = None
    for a in legal:
        a = int(a)
        if a < hand_size:
            at = sim.HAND + a * sim.CARD_SLOT
            cid = int(obs[at])
            base = ("c", cid, int(obs[at + 1]), int(obs[at + 2]))
            if _needs_target(cid):
                if targets is None:
                    targets = run.living_enemy_indices()
                for t in targets or [-1]:
                    key = base + (t,)
                    if key not in seen:
                        seen.add(key)
                        moves.append((key, a, t))
            elif base not in seen:
                seen.add(base)
                moves.append((base, a, -1))
        elif a == hand_size:
            moves.append((("e",), a, -1))
        else:
            slot = a - hand_size - 1
            pid = int(obs[sim.COMBAT_POTION + 2 * slot])
            if targets is None:
                targets = run.living_enemy_indices()
            for t in (targets if len(targets) > 1 else [-1]):
                key = ("p", pid, t)
                if key not in seen:
                    seen.add(key)
                    moves.append((key, a, t))
    return moves


def describe(key: tuple) -> str:
    kind = key[0]
    if kind == "e":
        return "End turn"
    if kind == "c":
        name = gd.card_name(key[1]) + ("+" if key[2] else "")
        return f"{name} -> enemy {key[4]}" if len(key) > 4 and key[4] >= 0 else name
    if kind == "p":
        return f"Potion {gd.potion_name(key[1])}" + (f" -> enemy {key[2]}" if key[2] >= 0 else "")
    if kind == "s":
        return "Select (skip)" if key[1] == -1 else f"Select {gd.card_name(key[1])}"
    return str(key)


# --------------------------------------------------------------------------------------
# the tree


@dataclass(slots=True)
class Node:
    visits: int = 0
    value_sum: float = 0.0
    avail: int = 0
    children: dict = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchConfig:
    simulations: int = 64
    c_explore: float = 0.7
    rollout_weights: tuple = DEFAULT_ROLLOUT_WEIGHTS
    max_rollout_steps: int = MAX_COMBAT_STEPS
    # A lost playout is worth more the more enemy HP it removed. Keep this below the gap
    # between surviving on 1 HP and dying, so any survival still beats any death.
    death_shaping: float = 0.1


@dataclass
class SearchResult:
    key: tuple
    action: int
    target: int
    visits: dict
    q: dict
    root_value: float


LeafValue = Callable[[sim.Run], float]


class CombatSearch:
    """IS-MCTS for one combat. Reuses its tree from one decision to the next."""

    def __init__(self, leaf_value: LeafValue, config: SearchConfig | None = None, seed: int = 0):
        self.leaf_value = leaf_value
        self.cfg = config or SearchConfig()
        self.rng = random.Random(seed)
        self.root: Node | None = None
        self.vmin = math.inf
        self.vmax = -math.inf
        self.errors = 0

    def reset(self) -> None:
        self.root = None
        self.vmin = math.inf
        self.vmax = -math.inf

    def advance(self, key: tuple) -> None:
        """Keep the subtree under the move just made."""
        if self.root is not None:
            self.root = self.root.children.get(key)

    def _norm(self, q: float) -> float:
        if self.vmax > self.vmin:
            return (q - self.vmin) / (self.vmax - self.vmin)
        return 0.5

    def _select(self, node: Node, moves) -> tuple:
        log_n = math.log(max(1, node.visits))
        best, best_score = None, -math.inf
        c = self.cfg.c_explore
        for move in moves:
            child = node.children[move[0]]
            score = self._norm(child.q) + c * math.sqrt(math.log(max(child.avail, 1) + 1) / child.visits)
            if score > best_score:
                best, best_score = move, score
        return best

    def _simulate(self, root_run: sim.Run) -> None:
        world = root_run.clone(resample=True, seed=self.rng.getrandbits(31))
        enemy_frac = None
        try:
            node = self.root
            path = [node]
            for _ in range(MAX_COMBAT_STEPS):
                if world.phase != sim.PHASE_COMBAT:
                    break
                moves = legal_moves(world)
                if not moves:
                    break
                for move in moves:
                    child = node.children.get(move[0])
                    if child is not None:
                        child.avail += 1
                untried = [m for m in moves if m[0] not in node.children]
                if untried:
                    key, action, target = self.rng.choice(untried)
                    child = Node(avail=1)
                    node.children[key] = child
                    enemy_frac = enemy_hp_fraction(world)
                    world.step(action, target)
                    path.append(child)
                    if world.phase == sim.PHASE_COMBAT:
                        enemy_frac = native_rollout(world, self.rng.getrandbits(31), self.cfg.rollout_weights, self.cfg.max_rollout_steps)
                    break
                key, action, target = self._select(node, moves)
                enemy_frac = enemy_hp_fraction(world)
                if not world.step(action, target):
                    break
                node = node.children[key]
                path.append(node)
            value = self.leaf_value(world)
            if world.done and not world.won():
                value += self.cfg.death_shaping * (1.0 - (enemy_frac if enemy_frac is not None else 1.0))
        except sim.SimError:
            # A branch that breaks the engine is one the search should stop visiting.
            self.errors += 1
            value = self.vmin if self.vmin != math.inf else 0.0
        finally:
            world.close()
        self.vmin = min(self.vmin, value)
        self.vmax = max(self.vmax, value)
        for n in path:
            n.visits += 1
            n.value_sum += value

    def search(self, run: sim.Run, simulations: int | None = None) -> SearchResult:
        if self.root is None:
            self.root = Node()
        moves = legal_moves(run)
        if len(moves) == 1:
            key, action, target = moves[0]
            return SearchResult(key, action, target, {key: 1}, {key: 0.0}, self.root.q)
        sims = simulations or self.cfg.simulations
        for _ in range(sims):
            self._simulate(run)
        # Robust child: most visits among moves legal HERE, ties broken by value.
        best = None
        best_score = (-1, -math.inf)
        visits, qs = {}, {}
        for key, action, target in moves:
            child = self.root.children.get(key)
            if child is None:
                continue
            visits[key] = child.visits
            qs[key] = child.q
            score = (child.visits, child.q)
            if score > best_score:
                best, best_score = (key, action, target), score
        if best is None:
            best = moves[-1]
        return SearchResult(best[0], best[1], best[2], visits, qs, self.root.q)


# --------------------------------------------------------------------------------------
# default leaf value, used before any value function is trained


def hp_leaf_value(world: sim.Run) -> float:
    """Survive, and keep as much HP as possible. Potions carry a small fixed value."""
    if world.done and not world.won():
        return 0.0
    hp = world.combat_hp if world.phase == sim.PHASE_COMBAT else world.hp
    max_hp = max(1, world.max_hp)
    potions = sum(1 for p in world.potions() if p)
    return 0.2 + 0.8 * hp / max_hp + 0.03 * potions


def play_combat_greedy_rollout(run: sim.Run, seed: int) -> None:
    """Baseline: the rollout policy alone, no search."""
    native_rollout(run, seed)


def batch_rollout_values(run: sim.Run, n: int, seed: int, leaf_value: LeafValue = hp_leaf_value) -> np.ndarray:
    """Monte Carlo value of a position under the rollout policy; for diagnostics."""
    out = np.empty(n)
    rng = random.Random(seed)
    for i in range(n):
        with run.clone(resample=True, seed=rng.getrandbits(31)) as world:
            native_rollout(world, rng.getrandbits(31))
            out[i] = leaf_value(world)
    return out
