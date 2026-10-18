"""Run-state encoding for the value network.

A state becomes a handful of fixed-size integer id arrays (deck, relics, potions, the
current encounter and the act boss) plus a float vector of scalars. Everything is read
from what the game shows the player: the deck, relics and potions, HP and gold, the map's
node icons, the boss portrait, and -- once a fight has started -- which fight it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import gamedata as gd
from . import sim

MAX_DECK = 64
MAX_RELICS = 32
MAX_POTIONS = 6

ACT_ORDINAL = {1: 1, 2: 1, 3: 2, 4: 3}  # overgrowth, underdocks -> act 1; hive 2; glory 3
MAP_TYPES = (1, 2, 3, 4, 5, 7)  # normal, elite, rest, shop, treasure, unknown/event
CARD_TYPE_INDEX = {t: i for i, t in enumerate(gd.CARD_TYPES)}

_SCALAR_NAMES: list[str] = []


def _scalar_names() -> list[str]:
    names = [
        "hp_frac", "hp", "max_hp", "gold", "floor", "floor_in_act", "act1", "act2", "act3",
        "region_overgrowth", "region_underdocks", "region_hive", "region_glory",
        "deck_size", "upgraded_frac", "avg_cost", "total_damage", "total_block", "basic_cards",
        "potions_held", "relics_held",
    ]
    names += [f"type_{t}" for t in gd.CARD_TYPES]
    names += [f"flag_{f}" for f in ("Exhaust", "Ethereal", "Unplayable", "HasEnergyCostX")]
    names += [f"phase_{i}" for i in range(sim.NUM_PHASES)]
    names += [f"node_{i}" for i in range(9)]
    names += ["rows_to_boss", "paths_log"]
    for t in MAP_TYPES:
        names += [f"map{t}_min", f"map{t}_mean", f"map{t}_max"]
    names += ["enemy_hp_total", "enemy_count"]
    return names


SCALAR_NAMES = _scalar_names()
N_SCALARS = len(SCALAR_NAMES)
_S = {name: i for i, name in enumerate(SCALAR_NAMES)}


@dataclass
class Encoded:
    """A batch of encoded states, ready for the network."""

    deck: np.ndarray  # [B, MAX_DECK] card vocab index
    deck_upg: np.ndarray  # [B, MAX_DECK] 0/1
    relics: np.ndarray  # [B, MAX_RELICS] relic vocab index
    potions: np.ndarray  # [B, MAX_POTIONS] potion vocab index
    encounter: np.ndarray  # [B] encounter vocab index (0 = not in a fight)
    boss: np.ndarray  # [B] encounter vocab index of the act boss
    scalars: np.ndarray  # [B, N_SCALARS] float32

    def __len__(self) -> int:
        return len(self.scalars)

    @staticmethod
    def stack(items: list[Encoded]) -> Encoded:
        return Encoded(*(np.concatenate([getattr(e, f) for e in items]) for f in _FIELDS))

    def take(self, idx) -> Encoded:
        return Encoded(*(getattr(self, f)[idx] for f in _FIELDS))

    def to_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {prefix + f: getattr(self, f) for f in _FIELDS}

    @staticmethod
    def from_dict(d, prefix: str = "") -> Encoded:
        return Encoded(*(d[prefix + f] for f in _FIELDS))


_FIELDS = ("deck", "deck_upg", "relics", "potions", "encounter", "boss", "scalars")


# --------------------------------------------------------------------------------------
# map lookahead


@dataclass
class MapStats:
    """Per node: counts of each node type along the paths from it to the boss."""

    mins: dict
    maxs: dict
    means: dict
    paths: dict
    rows_to_boss: dict
    boss: int


def map_stats(graph: dict) -> MapStats:
    nodes = graph["nodes"]
    children: dict = {n: [] for n in nodes}
    for src, dst in graph["edges"]:
        if src in children:
            children[src].append(dst)
    boss_row = max((r for (_, r), t in nodes.items() if t == 6), default=max((r for _, r in nodes), default=0))
    k = len(MAP_TYPES)
    mins, maxs, means, paths, rows = {}, {}, {}, {}, {}
    for node in sorted(nodes, key=lambda n: -n[1]):
        kids = [c for c in children[node] if c in nodes]
        rows[node] = max(0, boss_row - node[1])
        if not kids:
            mins[node] = np.zeros(k)
            maxs[node] = np.zeros(k)
            means[node] = np.zeros(k)
            paths[node] = 1
            continue
        kid_min, kid_max, kid_mean, kid_paths = [], [], [], []
        for c in kids:
            own = np.array([1.0 if nodes[c] == t else 0.0 for t in MAP_TYPES])
            kid_min.append(mins[c] + own)
            kid_max.append(maxs[c] + own)
            kid_mean.append(means[c] + own)
            kid_paths.append(paths[c])
        w = np.array(kid_paths, dtype=float)
        mins[node] = np.min(kid_min, axis=0)
        maxs[node] = np.max(kid_max, axis=0)
        means[node] = np.average(kid_mean, axis=0, weights=w)
        paths[node] = int(w.sum())
    return MapStats(mins, maxs, means, paths, rows, graph.get("boss", -1))


class MapCache:
    """The map is fixed for the whole act and identical in every clone, so compute once."""

    def __init__(self):
        self._cache: dict = {}

    def get(self, run: sim.Run) -> tuple[MapStats, tuple | None]:
        # Keyed on what identifies a map without reading it: a clone that crosses into
        # the next act rolls that act's map from resampled streams, so seed + act alone
        # could hand the real run a stranger's map.
        act, boss, n_nodes = run.act_summary()
        key = (run.seed, act, boss, n_nodes)
        stats = self._cache.get(key)
        if stats is None:
            if len(self._cache) > 64:
                self._cache.clear()
            stats = map_stats(run.map_graph())
            self._cache[key] = stats
        return stats, run.map_position()


# --------------------------------------------------------------------------------------
# encoding


@cache
def _card_arrays():
    cards = gd.cards()
    vocab = gd.card_vocab()
    n = len(vocab)
    cost = np.zeros(n, np.float32)
    dmg = np.zeros(n, np.float32)
    blk = np.zeros(n, np.float32)
    ctype = np.zeros(n, np.int64)
    flags = np.zeros((n, 4), np.float32)
    basic = np.zeros(n, np.float32)
    for cid, card in cards.items():
        i = vocab(cid)
        basic[i] = 1.0 if card.rarity == "Basic" else 0.0
        cost[i] = max(card.cost, 0)
        dmg[i] = card.damage
        blk[i] = card.block
        ctype[i] = CARD_TYPE_INDEX.get(card.type, 1)
        for j, f in enumerate(("Exhaust", "Ethereal", "Unplayable", "HasEnergyCostX")):
            flags[i, j] = 1.0 if f in card.flags else 0.0
    return cost, dmg, blk, ctype, flags, basic


def encode(run: sim.Run, maps: MapCache, *, hp: int | None = None, potions: tuple | None = None, phase: int | None = None, in_combat: bool | None = None) -> Encoded:
    """Encode one state. Keyword overrides let a caller value hypothetical variants."""
    cvocab, rvocab, pvocab, evocab = gd.card_vocab(), gd.relic_vocab(), gd.potion_vocab(), gd.encounter_vocab()
    deck = run.deck()
    relics = run.relics()
    pots = potions if potions is not None else run.potions()
    ph = run.phase if phase is None else phase
    fighting = (ph == sim.PHASE_COMBAT) if in_combat is None else in_combat

    deck_ids = np.zeros((1, MAX_DECK), np.int64)
    deck_upg = np.zeros((1, MAX_DECK), np.int64)
    for i, (cid, upg) in enumerate(deck[:MAX_DECK, :2].tolist()):
        deck_ids[0, i] = cvocab(cid)
        deck_upg[0, i] = upg
    relic_ids = np.zeros((1, MAX_RELICS), np.int64)
    for i, rid in enumerate(relics[:MAX_RELICS, 0].tolist()):
        relic_ids[0, i] = rvocab(rid)
    potion_ids = np.zeros((1, MAX_POTIONS), np.int64)
    for i, pid in enumerate(pots[:MAX_POTIONS]):
        potion_ids[0, i] = pvocab(pid) if pid else 0

    s = np.zeros((1, N_SCALARS), np.float32)
    v = s[0]
    cur_hp = run.hp if hp is None else hp
    max_hp = max(1, run.max_hp)
    act_id = run.act_id
    act = ACT_ORDINAL.get(act_id, 1)
    floor = run.floor
    v[_S["hp_frac"]] = cur_hp / max_hp
    v[_S["hp"]] = cur_hp / 100.0
    v[_S["max_hp"]] = max_hp / 100.0
    v[_S["gold"]] = run.gold / 300.0
    v[_S["floor"]] = floor / sim.TOTAL_FLOORS
    v[_S["floor_in_act"]] = (floor - (act - 1) * sim.FLOORS_PER_ACT) / sim.FLOORS_PER_ACT
    v[_S[f"act{min(act, 3)}"]] = 1.0
    region = {1: "overgrowth", 2: "underdocks", 3: "hive", 4: "glory"}.get(act_id)
    if region:
        v[_S[f"region_{region}"]] = 1.0

    cost, dmg, blk, ctype, flags, basic = _card_arrays()
    idx = deck_ids[0, : len(deck)]
    n = max(1, len(deck))
    v[_S["deck_size"]] = len(deck) / 40.0
    v[_S["upgraded_frac"]] = deck_upg[0, : len(deck)].sum() / n
    v[_S["avg_cost"]] = cost[idx].mean() if len(idx) else 0.0
    v[_S["total_damage"]] = dmg[idx].sum() / 100.0
    v[_S["total_block"]] = blk[idx].sum() / 100.0
    v[_S["basic_cards"]] = basic[idx].sum() / 20.0 if len(idx) else 0.0
    v[_S["potions_held"]] = sum(1 for p in pots if p) / 3.0
    v[_S["relics_held"]] = len(relics) / 10.0
    if len(idx):
        counts = np.bincount(ctype[idx], minlength=len(gd.CARD_TYPES))
        for t, name in enumerate(gd.CARD_TYPES):
            v[_S[f"type_{name}"]] = counts[t] / 20.0
        fl = flags[idx].sum(axis=0) / 10.0
        for j, f in enumerate(("Exhaust", "Ethereal", "Unplayable", "HasEnergyCostX")):
            v[_S[f"flag_{f}"]] = fl[j]
    v[_S[f"phase_{min(ph, sim.NUM_PHASES - 1)}"]] = 1.0
    v[_S[f"node_{min(max(run.node_type, 0), 8)}"]] = 1.0

    boss = 0
    try:
        stats, current = maps.get(run)
        boss = evocab(stats.boss)
        if current is not None and current in stats.mins:
            v[_S["rows_to_boss"]] = stats.rows_to_boss[current] / 16.0
            v[_S["paths_log"]] = np.log1p(stats.paths[current]) / 8.0
            for j, t in enumerate(MAP_TYPES):
                v[_S[f"map{t}_min"]] = stats.mins[current][j] / 5.0
                v[_S[f"map{t}_mean"]] = stats.means[current][j] / 5.0
                v[_S[f"map{t}_max"]] = stats.maxs[current][j] / 5.0
        else:
            v[_S["rows_to_boss"]] = 1.0
    except Exception:  # a map that cannot be read is not worth a crash
        pass

    encounter = 0
    if fighting:
        encounter = evocab(run.encounter_id())
        enemies = run.enemies()
        v[_S["enemy_hp_total"]] = sum(e[0] for e in enemies) / 200.0
        v[_S["enemy_count"]] = len([e for e in enemies if e[0] > 0]) / 3.0

    return Encoded(
        deck=deck_ids,
        deck_upg=deck_upg,
        relics=relic_ids,
        potions=potion_ids,
        encounter=np.array([encounter], np.int64),
        boss=np.array([boss], np.int64),
        scalars=s,
    )


def with_hp_and_potions(base: Encoded, hps: np.ndarray, max_hp: int, potion_sets: list[tuple]) -> Encoded:
    """Copies of a single encoded state with HP and potions varied, for value tables."""
    b = len(hps) * len(potion_sets)
    pvocab = gd.potion_vocab()
    out = Encoded(*(np.repeat(getattr(base, f), b, axis=0) for f in _FIELDS))
    k = 0
    for pots in potion_sets:
        ids = np.zeros(MAX_POTIONS, np.int64)
        for i, pid in enumerate(pots[:MAX_POTIONS]):
            ids[i] = pvocab(pid) if pid else 0
        held = sum(1 for p in pots if p) / 3.0
        for h in hps:
            out.potions[k] = ids
            out.scalars[k, _S["hp_frac"]] = h / max(1, max_hp)
            out.scalars[k, _S["hp"]] = h / 100.0
            out.scalars[k, _S["potions_held"]] = held
            k += 1
    return out


# --------------------------------------------------------------------------------------
# the hand-written prior the value network is residual on


PRIOR_TERMS = ("floor", "hp_frac", "potions_held", "type_Attack", "type_Skill", "type_Power", "basic_cards", "upgraded_frac", "deck_size")


def prior_parts(get):
    """Shared by the numpy heuristic and the network (pass a column getter for either).

    Progress so far, HP, potions -- plus a little for every non-basic card and every
    upgrade, so that an untrained agent takes cards and smiths rather than skipping by
    coin flip. The network learns everything this gets wrong.
    """
    floor, hp_frac, potions = get("floor"), get("hp_frac"), get("potions_held")
    playable = 20.0 * (get("type_Attack") + get("type_Skill") + get("type_Power"))
    nonbasic = playable - 20.0 * get("basic_cards")
    upgraded = get("upgraded_frac") * 40.0 * get("deck_size")
    return floor, hp_frac, potions, nonbasic, upgraded


def prior_score_np(scalars: np.ndarray) -> np.ndarray:
    floor, hp_frac, potions, nonbasic, upgraded = prior_parts(lambda n: scalars[:, _S[n]])
    score = 0.5 * floor + 0.15 * hp_frac + 0.03 * potions + 0.003 * np.clip(nonbasic, 0, 20) + 0.002 * np.clip(upgraded, 0, 20)
    return np.clip(score, 0.01, 0.99)
