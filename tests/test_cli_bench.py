"""The sts2-cli real-engine bench, checked offline against a simulator-backed stand-in.

`tests/fake_sts2cli.py` speaks sts2-cli's protocol and keeps its habits (rewards collected
without asking, alive-only enemy lists, per-act floors, relics by title). A run through
the bench must stay in sync with it from Neow to the end, finish exactly where the plain
simulator run of the same seed finishes, and stop at the first floor where the engine is
made to disagree.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from sts2rl import sim
from sts2rl.agent import Agent, AgentConfig
from sts2rl.bridge import cli, cli_eval

FAKE = [sys.executable, str(Path(__file__).with_name("fake_sts2cli.py"))]
CFG = dict(combat_sims=8, samples=1, max_act=1, record=False)
ACT_1_CLEARED = ("EVAL000066", 7_000_066)  # a seed and agent seed whose run beats the Act 1 boss


def bench(seed: str, agent_seed: int = 7_000_000, trace: Path | None = None, **cfg):
    agent = Agent(None, AgentConfig(**{**CFG, **cfg}), seed=agent_seed)
    with cli.CliGame(FAKE, timeout=60) as game:
        return cli.CliSession(game, agent, cli.Names(None), seed, trace_path=trace).play()


# 43 meets an event it could walk away from
@pytest.mark.parametrize("seed", ["EVAL000000", "EVAL000008", "EVAL000013", "EVAL000015", "EVAL000043"])
def test_a_faithful_engine_stays_in_sync_and_matches_the_simulator(seed, tmp_path):
    result = bench(seed, trace=tmp_path / "trace.jsonl")
    assert result.in_sync, result.desync_reason or result.engine_error
    assert result.finished
    assert result.restricted == 0  # the agent never wanted a move the engine does not have
    # The stand-in lists an event's own options only (no seed here meets Welcome to Wongos,
    # the one event with four).
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    events = [r["state"]["options"] for r in rows if r["kind"] == "state" and r["decision"] == "event_choice"]
    assert events and max(len(options) for options in events) <= 3
    twin = Agent(None, AgentConfig(**CFG), seed=7_000_000).play(seed)
    assert (result.final_floor, result.won) == (twin.final_floor, twin.won)


def test_a_run_capped_at_act_1_finishes_when_act_1_is_cleared():
    result = bench(*ACT_1_CLEARED)
    assert result.in_sync and result.finished, result.desync_reason
    assert result.won and result.acts_cleared == 1


def test_crossing_into_act_2_with_the_engines_between_act_heal():
    # sts2-cli heals as soon as the boss dies; the shadow, on the act's ancient.
    result = bench(*ACT_1_CLEARED, max_act=2)
    assert result.in_sync, result.desync_reason
    assert result.final_floor > 18


def test_unpatched_sts2cli_heal_rounding_is_caught(monkeypatch):
    # Upstream sts2-cli rounds the between-act heal up; the game rounds down (and
    # patches/sts2-cli/0001 makes sts2-cli do so). The stand-in heals by its own copy of
    # sts2-cli's formula, so this checks the bench's expectation, not a shared constant.
    monkeypatch.setenv("FAKE_STS2CLI_FAULT", "ceil-heal")
    result = bench(*ACT_1_CLEARED, max_act=2)
    assert not result.in_sync and result.desync_floor == 17
    game, sim_hp = re.match(r"hp \(between-act heal\): game=(\d+) sim=(\d+)", result.desync_reason).groups()
    assert int(game) == int(sim_hp) + 1


def test_a_fights_extra_card_reward_is_forfeited_not_desynced():
    # sts2-cli shows only a fight's first card reward; the shadow gives up the second too,
    # and the run counts it.
    result = bench("EVAL000160", agent_seed=7_000_160)
    assert result.in_sync and result.forfeits >= 1


def test_queued_card_offers_the_engine_drops_are_each_forfeited_and_none_is_left():
    # The Hunt / Glass Eye queue offers behind a reward screen's single card item; sts2-cli
    # shows only a fight's first card reward. The shadow opens and gives up each one.
    class Screen:
        def __init__(self, queued, flagged=0):
            self.phase, self.queued, self.flagged, self.floor = sim.PHASE_RELIC_REWARD, queued, flagged, 5

        def legal(self):
            if self.phase == sim.PHASE_CARD_REWARD:
                return [0, 1, 2, cli.REWARD_SKIP]
            return [0, cli.REWARD_SKIP] if self.queued or self.flagged else [cli.REWARD_SKIP]

        def state_list(self, list_id, capacity):
            return (0, 0, 0, self.flagged)

        def pending_selection(self):
            return 0

        def step(self, action):
            if self.phase == sim.PHASE_CARD_REWARD:
                assert action == cli.REWARD_SKIP
                self.phase = sim.PHASE_RELIC_REWARD
            elif action == cli.REWARD_SKIP:
                self.phase = sim.PHASE_MAP
            else:
                assert not self.flagged
                self.queued -= 1
                self.phase = sim.PHASE_CARD_REWARD
            return True

    session = cli.CliSession(None, Agent(None, AgentConfig(**CFG)), cli.Names(None), "EVAL000000")
    session.room = {"card_reward": True}
    session.shadow = Screen(queued=2)
    session.advance_shadow({"decision": "map_select"})
    assert session.shadow.phase == sim.PHASE_MAP and session.shadow.queued == 0 and session.forfeits == 2
    # A flagged item (the fight's own reward, or Kaleidoscope's, rolled when opened) is
    # given up unopened.
    session.forfeits, session.shadow = 0, Screen(queued=0, flagged=1)
    session.advance_shadow({"decision": "map_select"})
    assert session.shadow.phase == sim.PHASE_MAP and session.forfeits == 1


def test_an_exception_sts2cli_catches_itself_is_an_engine_error_not_a_desync(monkeypatch):
    # sts2-cli's purchase handlers catch the game's exception and answer without a stack
    # trace; a rule refusal reads the same way, so the bench tells them apart by message.
    monkeypatch.setenv("FAKE_STS2CLI_FAULT", "buy-fails")
    result = bench("EVAL000000")
    assert not result.in_sync and not result.desync_reason
    assert "Buy card failed" in result.engine_error


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
    # ... and a change the columns do not show is read as in place, not as a removal
    assert cli.follow_deck(pommels, pommels, 1) == {0: 0, 2: 2}
    # the bench reads every column, enchantment included
    with sim.Run.new("EVAL000000") as run:
        assert {len(c) for c in cli.deck_columns(run)} == {4}


def test_the_agent_never_walks_away_from_an_event_the_game_cannot_leave():
    from types import SimpleNamespace

    import numpy as np

    from sts2rl import agent

    def page(event_id, legal=(0, 1, 3)):
        return SimpleNamespace(phase=sim.PHASE_EVENT, event_id=event_id, legal=lambda: np.array(legal))

    ids = cli.gd.event_ids()
    unrest = next(i for i, (entry, _) in cli.gd.event_options().items() if entry == "UNREST_SITE")
    three = next(i for i, (_, n) in cli.gd.event_options().items() if n == 3)
    assert list(agent.choices(page(unrest))) == [0, 1]  # two options; 3 is the simulator's walk-away
    assert list(agent.choices(page(three, (0, 1, 2, 3)))) == [0, 1, 2]  # three options: 0-2
    assert list(agent.choices(page(-1, (0, 3)))) == [0]  # a result page: every move proceeds
    # Events that build their options at run time have at most three too ...
    assert list(agent.choices(page(ids["SelfHelpBook"]))) == [0, 1]
    assert list(agent.choices(page(ids["LuminousChoir"], (0, 3)))) == [0]
    # ... except Welcome to Wongos, whose fourth is the game's own Leave
    assert list(agent.choices(page(ids["WelcomeToWongos"]))) == [0, 1, 3]
    # and a move that is the only one is never taken away
    assert list(agent.choices(page(ids["SelfHelpBook"], (3,)))) == [3]


def test_a_rest_site_cannot_be_left_until_an_option_is_taken():
    # The game's Proceed is off while a rest option is on (simulator patch 0009).
    agent = Agent(None, AgentConfig(**CFG), seed=0)
    with sim.Run.new("EVAL000000") as run:
        for _ in range(2000):
            if run.phase == sim.PHASE_REST or run.done:
                break
            assert run.step(*agent.act(run))
        assert run.phase == sim.PHASE_REST
        assert cli.REWARD_SKIP not in [int(a) for a in run.legal()]
        assert run.step(0)  # rest
        assert run.phase == sim.PHASE_REST and cli.REWARD_SKIP in [int(a) for a in run.legal()]


def test_the_bench_leaves_a_rest_site_only_when_the_engine_has_no_option_on():
    from types import SimpleNamespace

    session = cli.CliSession(None, Agent(None, AgentConfig(**CFG)), cli.Names(None), "EVAL000000")
    session.shadow = SimpleNamespace(legal=lambda: [0, 1, cli.REWARD_SKIP])
    on = {"options": [{"option_id": "HEAL", "is_enabled": True}, {"option_id": "SMITH", "is_enabled": True}]}
    off = {"options": [{"option_id": "HEAL", "is_enabled": False}, {"option_id": "SMITH", "is_enabled": False}]}
    assert session._expressible("rest_site", on) == [0, 1]
    assert session._expressible("rest_site", off) == [cli.REWARD_SKIP]


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
