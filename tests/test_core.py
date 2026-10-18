"""Core invariants: the simulator wrapper, search, features, model, agent and targets."""

from __future__ import annotations

import numpy as np
import pytest

from sts2rl import combat, features as F, sim
from sts2rl.agent import Agent, AgentConfig, heuristic_value
from sts2rl.bench import ACT1_ELITES, default_deck, start_fight
from sts2rl.dataset import Replay
from sts2rl.model import ValueNet


def first_combat(seed="CORE1") -> sim.Run:
    run = sim.Run.new(seed)
    rng = np.random.default_rng(0)
    while run.phase != sim.PHASE_COMBAT:
        run.step(int(rng.choice(run.legal())))
    return run


def test_clone_is_independent_and_faithful():
    run = first_combat()
    before = run.obs.copy()
    with run.clone(resample=False) as a, run.clone(resample=False) as b:
        for _ in range(6):
            if a.phase != sim.PHASE_COMBAT:
                break
            act = int(a.legal()[0])
            a.step(act)
            b.step(act)
        assert np.array_equal(a.obs, b.obs), "identical clones diverged"
    assert np.array_equal(run.obs, before), "stepping a clone moved the original"
    run.close()


def test_resampled_clones_see_different_futures():
    run = first_combat()
    hands = set()
    for s in range(8):
        with run.clone(resample=True, seed=s) as c:
            end_turn = len(c.hand())
            c.step(end_turn)
            hands.add(tuple(c.hand()))
    assert len(hands) > 1, "resampling did not change the next draw"
    run.close()


def test_legal_moves_are_distinct_and_include_end_turn():
    run = first_combat()
    moves = combat.legal_moves(run)
    keys = [k for k, _, _ in moves]
    assert len(keys) == len(set(keys))
    assert ("e",) in keys
    for _, action, _ in moves:
        assert action in run.legal()
    run.close()


def test_search_returns_a_legal_move_and_beats_rollout_policy():
    run = first_combat()
    res = combat.CombatSearch(combat.hp_leaf_value, combat.SearchConfig(simulations=40), seed=1).search(run)
    assert res.action in run.legal()
    run.close()
    # Paired comparison on an elite: search must not lose to its own rollout policy.
    deck = default_deck()
    hp_search, hp_roll = 0, 0
    for i in range(4):
        a = start_fight(f"T{i}", ACT1_ELITES["byrdonis"], deck)
        b = start_fight(f"T{i}", ACT1_ELITES["byrdonis"], deck)
        s = combat.CombatSearch(combat.hp_leaf_value, combat.SearchConfig(simulations=40), seed=i)
        while a.phase == sim.PHASE_COMBAT:
            r = s.search(a)
            a.step(r.action, r.target)
            s.advance(r.key)
        combat.native_rollout(b, i)
        hp_search += 0 if a.done else a.hp
        hp_roll += 0 if b.done else b.hp
        a.close()
        b.close()
    assert hp_search >= hp_roll


def test_encode_shapes_and_map_stats():
    run = sim.Run.new("FEAT1")
    enc = F.encode(run, F.MapCache())
    assert enc.scalars.shape == (1, F.N_SCALARS)
    assert enc.deck.shape == (1, F.MAX_DECK) and (enc.deck[0, :11] > 0).all()
    assert np.isfinite(enc.scalars).all()
    stats = F.map_stats(run.map_graph())
    cur = run.map_position()
    assert stats.rows_to_boss[cur] > 10 and stats.paths[cur] >= 1
    run.close()


def test_untrained_model_equals_heuristic_prior():
    run = sim.Run.new("PRIOR1")
    net = ValueNet()
    v = float(net.predict(F.encode(run, F.MapCache()))[0])
    assert abs(v - heuristic_value(run)) < 1e-4
    run.close()


def test_td_targets():
    rep = Replay.__new__(Replay)
    rep.data = {"run_id": np.array([0, 0, 0, 1, 1]), "score": np.array([0.2, 0.2, 0.2, 1.0, 1.0], np.float32)}
    values = np.array([0.5, 0.4, 0.3, 0.9, 0.8], np.float32)
    g = rep.td_targets(values, lam=0.5)
    assert g[2] == pytest.approx(0.2) and g[4] == pytest.approx(1.0)
    assert g[1] == pytest.approx(0.5 * 0.3 + 0.5 * 0.2)
    assert g[3] == pytest.approx(0.5 * 0.8 + 0.5 * 1.0)


@pytest.mark.parametrize("use_model", [False, True])
def test_agent_completes_a_run(use_model):
    agent = Agent(ValueNet() if use_model else None, AgentConfig(combat_sims=8, samples=1), seed=0)
    rec = agent.play("AGENT1")
    assert rec.final_floor >= 2 and 0.0 <= rec.score <= 1.0
    assert len(rec.states) == len(rec.v_pred) > 0
