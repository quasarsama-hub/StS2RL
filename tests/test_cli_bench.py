"""The sts2-cli real-engine bench, checked offline against a simulator-backed stand-in.

`tests/fake_sts2cli.py` speaks sts2-cli's protocol and keeps its habits (rewards collected
without asking, alive-only enemy lists, per-act floors, relics by title). A run through
the bench must stay in sync with it from Neow to the end, finish exactly where the plain
simulator run of the same seed finishes, and stop at the first floor where the engine is
made to disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sts2rl import sim
from sts2rl.agent import Agent, AgentConfig
from sts2rl.bridge import cli, cli_eval

FAKE = [sys.executable, str(Path(__file__).with_name("fake_sts2cli.py"))]
CFG = dict(combat_sims=8, samples=1, max_act=1, record=False)


def bench(seed: str, agent_seed: int = 7_000_000, **cfg):
    agent = Agent(None, AgentConfig(**{**CFG, **cfg}), seed=agent_seed)
    with cli.CliGame(FAKE, timeout=60) as game:
        return cli.CliSession(game, agent, cli.Names(None), seed).play()


@pytest.mark.parametrize("seed", ["EVAL000000", "EVAL000008", "EVAL000015"])
def test_a_faithful_engine_stays_in_sync_and_matches_the_simulator(seed):
    result = bench(seed)
    assert result.in_sync, result.desync_reason or result.engine_error
    assert result.finished
    twin = Agent(None, AgentConfig(**CFG), seed=7_000_000).play(seed)
    assert (result.final_floor, result.won) == (twin.final_floor, twin.won)


def test_a_run_capped_at_act_1_finishes_when_act_1_is_cleared():
    result = bench("EVAL000013", agent_seed=7_000_013)
    assert result.in_sync and result.finished, result.desync_reason
    assert result.won and result.acts_cleared == 1


def test_crossing_into_act_2_with_the_engines_between_act_heal():
    # sts2-cli heals as soon as the boss dies; the shadow, on the act's ancient.
    result = bench("EVAL000013", agent_seed=7_000_013, max_act=2)
    assert result.in_sync, result.desync_reason
    assert result.final_floor > 18


def test_unpatched_sts2cli_heal_rounding_is_caught(monkeypatch):
    # Upstream sts2-cli rounds the between-act heal up; the game rounds down (and
    # patches/sts2-cli/0001 makes sts2-cli do so). The stand-in heals by its own copy of
    # sts2-cli's formula, so this checks the bench's expectation, not a shared constant.
    monkeypatch.setenv("FAKE_STS2CLI_FAULT", "ceil-heal")
    result = bench("EVAL000013", agent_seed=7_000_013, max_act=2)
    assert not result.in_sync and result.desync_floor == 17
    assert result.desync_reason.startswith("hp (between-act heal): game=73 sim=72")


def test_a_fights_extra_card_reward_is_forfeited_not_desynced():
    # sts2-cli shows only a fight's first card reward; the shadow gives up the second too,
    # and the run counts it.
    result = bench("EVAL000160", agent_seed=7_000_160)
    assert result.in_sync and result.forfeits == 1


def test_the_first_divergence_is_reported_with_its_floor(monkeypatch):
    monkeypatch.setenv("FAKE_STS2CLI_FAULT", "hp:3")
    result = bench("EVAL000000")
    assert not result.in_sync
    assert result.desync_floor == 3
    assert result.desync_reason.startswith("hp: game=")


def test_an_engine_that_dies_is_an_error_not_a_hang():
    dead = [sys.executable, "-c", "print('{\"type\": \"ready\"}', flush=True)"]
    agent = Agent(None, AgentConfig(**CFG), seed=0)
    with cli.CliGame(dead, timeout=10) as game:
        result = cli.CliSession(game, agent, cli.Names(None), "EVAL000000").play()
    assert not result.in_sync and "exited" in result.engine_error


def test_smoke_test_passes_on_a_faithful_engine():
    assert cli.smoke_test(None, command=FAKE) == []


def test_deck_selections_are_followed_through_every_kind_of_change():
    deck = [(1, False), (2, False), (1, False), (3, True)]
    # removal: the cards after the pick move up one
    assert cli.follow_deck(deck, [(1, False), (1, False), (3, True)], 1) == {0: 0, 2: 1, 3: 2}
    # transform: removed, its replacement appended at the end
    assert cli.follow_deck(deck, [(1, False), (1, False), (3, True), (9, False)], 1) == {0: 0, 2: 1, 3: 2}
    # upgrade / Claws: changed in place
    assert cli.follow_deck(deck, [(1, False), (2, True), (1, False), (3, True)], 1) == {0: 0, 2: 2, 3: 3}
    # duplicate: a copy appended, the original left where it was
    assert cli.follow_deck(deck, deck + [(2, False)], 1) == {0: 0, 2: 2, 3: 3}
    # anything else is not followed
    assert cli.follow_deck(deck, [(3, True), (1, False), (1, False)], 1) is None
    # an enchant changes only the picked card, even when identical cards follow it
    pommels = [(5, 0, 0, 0), (7, 0, 0, 0), (7, 0, 0, 0)]
    assert cli.follow_deck(pommels, [(5, 0, 0, 0), (7, 0, 4, 1), (7, 0, 0, 0)], 1) == {0: 0, 2: 2}


def test_the_agent_never_walks_away_from_an_event_the_game_cannot_leave():
    from types import SimpleNamespace

    import numpy as np

    from sts2rl import agent

    def page(event_id):
        return SimpleNamespace(phase=sim.PHASE_EVENT, event_id=event_id, legal=lambda: np.array([0, 1, 3]))

    unrest = next(i for i, (entry, _) in cli.gd.event_options().items() if entry == "UNREST_SITE")
    assert list(agent.choices(page(unrest))) == [0, 1]  # two options; 3 is the simulator's walk-away
    assert list(agent.choices(page(-1))) == [0, 1]  # a result page: every move proceeds
    assert list(agent.choices(page(41))) == [0, 1, 3]  # Welcome to Wongos: 3 is the game's Leave


def test_ids_and_titles():
    assert cli.entry_of("CARD.STRIKE_IRONCLAD") == "STRIKE_IRONCLAD"
    assert cli.card_id("CARD.BASH") != 0
    assert cli.canonical_seed("eval0o1i") == "EVAL0011"
    names = cli.Names(None)
    with sim.Run.new("EVAL000000") as run:
        relic = int(run.relics()[0, 0])
    assert names.relic(relic) == "BURNING_BLOOD.title"  # sts2-cli's own fallback for a missing title


def test_reason_kinds_and_report():
    assert cli_eval.reason_kind("hp: game=3 sim=5; gold: game=1 sim=2") == "gold, hp"
    rows = [
        cli.CliRunResult(seed="A", finished=True, won=True, acts_cleared=3, final_floor=51, score=1.0).to_dict(),
        cli.CliRunResult(seed="B", in_sync=False, desync_floor=7, desync_reason="hp: game=1 sim=2").to_dict(),
    ]
    rows.append(cli.CliRunResult(seed="C", in_sync=False, desync_floor=30, acts_cleared=1, desync_reason="gold: game=1 sim=2").to_dict())
    rows.append(cli.CliRunResult(seed="D", in_sync=False, bench_error="KeyError: 3").to_dict())
    s = cli_eval.summarize(rows, 3)
    assert (s["in_sync"], s["desyncs"], s["bench_errors"], s["win_rate"]) == (1, 2, 1, 1.0)
    # Act 1 was decided in sync for A (finished) and C (desynced in act 2), not for B.
    assert (s["acts"][1]["cleared"], s["acts"][1]["known"]) == (2, 2)
    assert (s["acts"][2]["cleared"], s["acts"][2]["known"]) == (1, 1)
    assert "2 desynced" in cli_eval.format_report(s)
