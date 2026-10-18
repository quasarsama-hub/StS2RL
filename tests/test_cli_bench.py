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


def bench(seed: str, agent_seed: int = 7_000_000):
    agent = Agent(None, AgentConfig(**CFG), seed=agent_seed)
    with cli.CliGame(FAKE, timeout=60) as game:
        return cli.CliSession(game, agent, cli.Names(None), seed).play()


@pytest.mark.parametrize("seed", ["EVAL000000", "EVAL000008", "EVAL000015"])
def test_a_faithful_engine_stays_in_sync_and_matches_the_simulator(seed):
    result = bench(seed)
    assert result.in_sync, result.desync_reason or result.engine_error
    assert result.finished
    twin = Agent(None, AgentConfig(**CFG), seed=7_000_000).play(seed)
    assert (result.final_floor, result.won) == (twin.final_floor, twin.won)


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


def test_selection_positions_follow_a_shrinking_list():
    # Five cards shown; the simulator re-indexes after the second is taken.
    slots = cli.rekey([0, 2, 3, 4], {0: (1, False), 1: (2, False), 2: (3, False), 3: (4, False)})
    assert slots == [(0, 0), (1, 2), (2, 3), (3, 4)]
    # Deck indices: a removal shifts the later ones down by one.
    assert cli.rekey([0, 1], {3: (7, False), 5: (9, True)}) == [(3, 0), (5, 1)]
    assert cli.rekey([0, 1], {3: (7, False)}) is None


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
    s = cli_eval.summarize(rows, 3)
    assert (s["in_sync"], s["desyncs"], s["synced_win_rate"]) == (1, 1, 1.0)
    assert "1 desynced" in cli_eval.format_report(s)
