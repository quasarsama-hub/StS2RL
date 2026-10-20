"""The full-run agent.

Combat:      information-set MCTS (combat.py), leaves valued by the run value net.
Everything else: for each legal option, fork the run (resampling whatever the player has
             not seen), take the option, and score the state it leads to with the value
             net. Pick the best (or explore, during self-play).

So one learned function -- "how good is this run state?" -- drives card picks, pathing,
shopping, resting, events and potion use, and improving it improves all of them.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from . import combat
from . import features as F
from . import gamedata as gd
from . import sim
from .model import ValueNet

STOCHASTIC_PHASES = {sim.PHASE_MAP, sim.PHASE_EVENT, sim.PHASE_ANCIENT, 10, 11}
PHASE_BUNDLE_SELECT = 12
REWARD_SKIP = 3
BUNDLE_CONFIRM = 2
MAX_OPTIONS = 64
MAX_RUN_STEPS = 8000
LOOP_LIMIT = 4  # the same non-combat state this many times means the agent is going in circles


@dataclass
class AgentConfig:
    combat_sims: int = 64
    elite_sim_mult: float = 2.0  # hard fights get a bigger search budget
    boss_sim_mult: float = 4.0
    c_explore: float = 0.7
    samples: int = 3  # clones per option where the outcome is random
    epsilon: float = 0.0  # chance of a random non-combat choice (self-play exploration)
    max_act: int = 3  # stop the run after this many acts (3 = the whole game)
    record: bool = True


@dataclass
class RunRecord:
    seed: str
    states: list = field(default_factory=list)  # Encoded rows, in visit order
    v_pred: list = field(default_factory=list)  # model's value at each recorded state
    final_floor: int = 0
    won: bool = False
    score: float = 0.0
    acts_cleared: int = 0
    death_encounter: str = ""
    deck: list = field(default_factory=list)
    relics: list = field(default_factory=list)
    seconds: float = 0.0
    decisions: int = 0
    combats: int = 0


EVENT_RESULT_PENDING = -1  # the simulator's event id for an event's result (Proceed) page


def choices(run: sim.Run) -> np.ndarray:
    """The legal moves a player of the real game has.

    The simulator lets a run walk away from any event without choosing (its event "skip",
    action 3). The game has no such button: in recorded real runs that action only stands
    for Proceed on a result page. So it is left out wherever it cannot be a real option --
    on result pages (where any move proceeds) and on events whose first page has at most
    three options. Events that build their option list at run time keep it: at Welcome to
    Wongos, action 3 is the game's own Leave.
    """
    legal = run.legal()
    if run.phase == sim.PHASE_EVENT and len(legal) > 1:
        options = gd.event_options().get(run.event_id)
        if run.event_id == EVENT_RESULT_PENDING or (options is not None and options[1] <= sim.EVENT_SKIP):
            legal = legal[legal != sim.EVENT_SKIP]
    return legal


def heuristic_value(run: sim.Run, max_act: int = 3, maps: F.MapCache | None = None) -> float:
    """Value before any learning -- exactly what an untrained network predicts."""
    if run.done:
        return sim.run_score(run, max_act)
    return float(F.prior_score_np(F.encode(run, maps or F.MapCache()).scalars)[0])


class Agent:
    def __init__(self, model: ValueNet | None = None, config: AgentConfig | None = None, seed: int = 0):
        self.model = model
        self.cfg = config or AgentConfig()
        self.rng = random.Random(seed)
        self.maps = F.MapCache()
        self._combat_key = None
        self._search: combat.CombatSearch | None = None
        self.stats = {"value_calls": 0, "value_time": 0.0, "combat_time": 0.0, "noncombat_time": 0.0}

    # ----------------------------------------------------------------------------------
    # valuing states

    def values(self, runs: list[sim.Run]) -> np.ndarray:
        out = np.empty(len(runs))
        live = []
        for i, r in enumerate(runs):
            if r.done or self._past_last_act(r):
                out[i] = sim.run_score(r, self.cfg.max_act) if r.done else self._act_cap_score(r)
            else:
                live.append(i)
        if live:
            if self.model is None:
                for i in live:
                    out[i] = heuristic_value(runs[i], self.cfg.max_act, self.maps)
            else:
                t = time.perf_counter()
                batch = F.Encoded.stack([F.encode(runs[i], self.maps) for i in live])
                out[live] = self.model.predict(batch)
                self.stats["value_calls"] += 1
                self.stats["value_time"] += time.perf_counter() - t
        return out

    def _past_last_act(self, run: sim.Run) -> bool:
        return F.ACT_ORDINAL.get(run.act_id, 1) > self.cfg.max_act

    def _act_cap_score(self, run: sim.Run) -> float:
        return 1.0  # cleared every act we were asked to play

    # ----------------------------------------------------------------------------------
    # combat

    def _leaf_function(self, run: sim.Run):
        """Value a post-fight state via a table over (potions left, HP), built once per fight."""
        if self.model is None:
            return combat.hp_leaf_value, 0.1
        max_hp = max(1, run.max_hp)
        root_potions = [p for p in run.potions()]
        held = [i for i, p in enumerate(root_potions) if p]
        subsets = []
        for k in range(len(held) + 1):
            for keep in itertools.combinations(held, k):
                subsets.append(tuple(root_potions[i] if i in keep else 0 for i in range(len(root_potions))))
        if len(subsets) > 64:
            subsets = subsets[:1] + subsets[-63:]
        base = F.encode(run, self.maps, phase=sim.PHASE_CARD_REWARD, in_combat=False)
        hps = np.arange(0, max_hp + 1)
        table_in = F.with_hp_and_potions(base, hps, max_hp, subsets)
        table = self.model.predict(table_in).reshape(len(subsets), len(hps))
        index = {pots: i for i, pots in enumerate(subsets)}
        held_set = set(held)
        death_value = 0.5 * min(run.floor, sim.FLOORS_PER_ACT * self.cfg.max_act) / (sim.FLOORS_PER_ACT * self.cfg.max_act)
        # A young network's table is noisy, and combat search takes it literally. Three
        # guards keep a fight sensible regardless: more HP is never worse, surviving is
        # always worth more than dying, and a small HP term breaks ties a flat table
        # would leave to chance.
        table = np.maximum.accumulate(table, axis=1)
        table[:, 1:] = np.maximum(table[:, 1:], death_value + 0.01)
        table = table + 0.02 * hps[None, :] / max_hp
        full = index[subsets[-1]] if subsets else 0
        gap = float(table[full, 1] - death_value) if table.shape[1] > 1 else 0.05
        shaping = 0.5 * max(gap, 1e-3)
        max_act = self.cfg.max_act

        def leaf(world: sim.Run) -> float:
            if world.done:
                return sim.run_score(world, max_act)
            if world.phase == sim.PHASE_COMBAT:
                hp = world.combat_hp
                pots = world.combat_potions()
            else:
                hp = world.hp
                pots = list(world.potions())
            key = tuple(root_potions[i] if (i in held_set and i < len(pots) and pots[i] == root_potions[i]) else 0 for i in range(len(root_potions)))
            row = index.get(key, full)
            return float(table[row, min(max(hp, 0), max_hp)])

        return leaf, shaping

    def _combat_move(self, run: sim.Run, record: RunRecord | None):
        key = (run.floor, run.encounter_id())
        if key != self._combat_key or self._search is None:
            self._combat_key = key
            leaf, shaping = self._leaf_function(run)
            mult = {2: self.cfg.elite_sim_mult, 6: self.cfg.boss_sim_mult}.get(run.node_type, 1.0)
            sims = max(1, int(self.cfg.combat_sims * mult))
            cfg = combat.SearchConfig(simulations=sims, c_explore=self.cfg.c_explore, death_shaping=shaping)
            self._search = combat.CombatSearch(leaf, cfg, seed=self.rng.getrandbits(31))
            if record is not None:
                record.combats += 1
                self._record(run, record)
        res = self._search.search(run)
        self._search.advance(res.key)
        return res.action, res.target

    # ----------------------------------------------------------------------------------
    # everything else

    def _noncombat_move(self, run: sim.Run, legal: np.ndarray, allowed: set[int] | None = None) -> int:
        options = [int(a) for a in legal]
        if len(options) == 1:
            return options[0]
        if run.phase == PHASE_BUNDLE_SELECT:
            return self._bundle_move(run, options)
        if run.phase == sim.PHASE_RELIC_REWARD:
            # Gold, potions and relics are claimed, as the live bridge does; whether to
            # take a CARD is decided on the card-reward screen this opens. Leaving here
            # forfeits everything on the screen, card reward included.
            claims = [a for a in options if a < REWARD_SKIP]
            if claims:
                return claims[0]
        if self.cfg.epsilon > 0 and self.rng.random() < self.cfg.epsilon:
            return self.rng.choice([a for a in options if allowed is None or a in allowed] or options)
        if len(options) > MAX_OPTIONS:
            options = self.rng.sample(options, MAX_OPTIONS)
        k = self.cfg.samples if run.phase in STOCHASTIC_PHASES else 1
        clones, owners = [], []
        try:
            for a in options:
                for _ in range(k):
                    c = run.clone(resample=True, seed=self.rng.getrandbits(31))
                    try:
                        ok = c.step(a)
                    except sim.SimError:
                        ok = False
                    if ok:
                        clones.append(c)
                        owners.append(a)
                    else:
                        c.close()
            if not clones:
                return next((a for a in reversed(options) if allowed is None or a in allowed), options[-1])
            vals = self.values(clones)
        finally:
            for c in clones:
                c.close()
        totals: dict[int, list[float]] = {}
        for a, v in zip(owners, vals):
            totals.setdefault(a, []).append(float(v))

        def score(a):
            return (sum(totals[a]) / len(totals[a]), -options.index(a))

        best = max(totals, key=score)
        if allowed is not None and best not in allowed:
            self.stats["restricted"] = self.stats.get("restricted", 0) + 1  # it wanted a move the game lacks
            kept = [a for a in totals if a in allowed]
            best = max(kept, key=score) if kept else next(a for a in options if a in allowed)
        return best

    def _bundle_move(self, run: sim.Run, options: list[int]) -> int:
        """Scroll Boxes: highlighting a bundle changes nothing a value function can see, so
        a bundle is scored by highlighting AND confirming it, and once one is highlighted
        (the confirm is offered) the choice is made."""
        if BUNDLE_CONFIRM in options:
            return BUNDLE_CONFIRM
        picks = [a for a in options if a != BUNDLE_CONFIRM]
        clones, owners = [], []
        try:
            for a in picks:
                c = run.clone(resample=True, seed=self.rng.getrandbits(31))
                try:
                    ok = c.step(a) and c.step(BUNDLE_CONFIRM)
                except sim.SimError:
                    ok = False
                if ok:
                    clones.append(c)
                    owners.append(a)
                else:
                    c.close()
            if not clones:
                return picks[0]
            vals = self.values(clones)
        finally:
            for c in clones:
                c.close()
        return owners[int(np.argmax(vals))]

    # ----------------------------------------------------------------------------------

    def _record(self, run: sim.Run, record: RunRecord) -> None:
        if not self.cfg.record:
            return
        enc = F.encode(run, self.maps)
        record.states.append(enc)
        record.v_pred.append(float(self.model.predict(enc)[0]) if self.model is not None else math.nan)

    def act(self, run: sim.Run, record: RunRecord | None = None, allowed=None) -> tuple[int, int]:
        """The move to make. `allowed` limits a non-combat choice to some of the legal
        moves (the real-engine bench leaves out moves the game has no button for). Every
        legal move is still evaluated, so the agent's random stream -- and with it every
        later decision -- is the same as without the limit whenever it does not bind."""
        if run.phase == sim.PHASE_COMBAT:
            t = time.perf_counter()
            move = self._combat_move(run, record)
            self.stats["combat_time"] += time.perf_counter() - t
            return move
        self._combat_key = None
        legal = choices(run)
        if len(legal) == 0:
            raise RuntimeError(f"no legal action in phase {run.phase} at floor {run.floor}")
        t = time.perf_counter()
        if record is not None and len(legal) > 1:
            self._record(run, record)
            record.decisions += 1
        a = self._noncombat_move(run, legal, None if allowed is None else {int(x) for x in allowed})
        self.stats["noncombat_time"] += time.perf_counter() - t
        return a, -1

    def play(self, seed: str, verbose: bool = False) -> RunRecord:
        t0 = time.perf_counter()
        record = RunRecord(seed=str(seed))
        self.maps = F.MapCache()
        self._combat_key = None
        run = sim.Run.new(seed)
        seen: dict[bytes, int] = {}
        try:
            for _ in range(MAX_RUN_STEPS):
                if run.done or self._past_last_act(run):
                    break
                action, target = self.act(run, record)
                if run.phase != sim.PHASE_COMBAT:
                    # A screen whose moves leave the state unchanged can trap a greedy
                    # chooser forever. Seeing the same state again and again breaks the tie.
                    key = run.obs.tobytes()
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] > LOOP_LIMIT:
                        others = [int(a) for a in choices(run) if int(a) != action]
                        if others:
                            action, target = self.rng.choice(others), -1
                else:
                    seen.clear()
                floor = run.floor
                if not run.step(action, target):
                    # The mask offered a move the engine then refused (an emulator gap in
                    # some events). Take the next legal move rather than lose the run.
                    for alt in run.legal():
                        if int(alt) != action and run.step(int(alt)):
                            break
                    else:
                        raise RuntimeError(f"simulator refused action {action} (target {target}) in phase {run.phase}")
                if verbose and run.floor != floor:
                    print(f"  floor {run.floor:2d}  hp {run.hp}/{run.max_hp}  gold {run.gold}")
            record.final_floor = run.floor
            capped = self._past_last_act(run)
            record.won = capped or (run.done and run.won() and run.hp > 0)
            record.score = 1.0 if capped else sim.run_score(run, self.cfg.max_act)
            record.acts_cleared = self.cfg.max_act if record.won else F.ACT_ORDINAL.get(run.act_id, 1) - 1
            if not record.won:
                from . import gamedata as gd

                record.death_encounter = gd.encounter_names().get(run.encounter_id(), "?")
            record.deck = [int(c) for c in run.deck()[:, 0]]
            record.relics = [int(r) for r in run.relics()[:, 0]]
        finally:
            run.close()
        record.seconds = time.perf_counter() - t0
        return record
