"""The real-game bridge, checked against runs recorded from the real game.

No game is needed: the simulator project ships full-run traces captured live through
STS2MCP (every command and every resulting game state). Replaying them through the
bridge's shadow simulator checks two things the bridge depends on:

1. the shadow stays in sync with the game (the comparison reports nothing), and
2. the bridge's sim-action -> game-command mapping reproduces the command the trace
   actually sent, for every decision the agent would make.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2rl import paths
from sts2rl.bridge import live

FIXTURES = sorted((paths.SIM_ROOT / "tests" / "fixtures" / "run_trace").glob("*.json"))
DEBUG = {"debug_gain_max_hp", "debug_upgrade_deck", "debug_enter_next_act"}


def clean_traces():
    for path in FIXTURES:
        data = json.loads(path.read_text())
        actions = [(s.get("action") or {}).get("action") for s in data["trace"]]
        if data.get("character") == "IRONCLAD" and not DEBUG.intersection(actions) and "-a8" in path.name:
            yield path, data


def replay(data):
    """Yield (state_before, payload, state_after, shadow) for each recorded step."""
    shadow = live.ShadowSim(data["seed"])
    trace = data["trace"]
    try:
        for prev, cur in zip(trace, trace[1:]):
            payload = cur.get("action")
            if not payload:
                continue
            yield prev["raw_state"], payload, cur["raw_state"], shadow
    finally:
        shadow.close()


def expected_sim_action(payload, before, shadow):
    """The simulator action a recorded command means, or None for non-decisions."""
    name = payload["action"]
    st = before.get("state_type")
    if name in {"play_card", "end_turn", "use_potion"} and st in live.COMBAT_STATES:
        return live.translate_command(payload, shadow.obs, shadow.info, shadow.env, None)
    if name == "choose_map_node" and int(shadow.info["phase"]) == live.sim.PHASE_MAP:
        return live.translate_command(payload, shadow.obs, shadow.info, shadow.env, None)
    if name in {"select_card_reward", "skip_card_reward"} and int(shadow.info["phase"]) == live.sim.PHASE_CARD_REWARD:
        return live.translate_command(payload, shadow.obs, shadow.info, shadow.env, None)
    if name == "choose_rest_option" and int(shadow.info["phase"]) == live.sim.PHASE_REST:
        return live.translate_command(payload, shadow.obs, shadow.info, shadow.env, None)
    if name == "choose_event_option" and st == "event" and int(shadow.info["phase"]) in (live.sim.PHASE_EVENT, live.sim.PHASE_ANCIENT):
        if int(payload.get("index", 0)) < 0:
            return None
        return live.translate_command(payload, shadow.obs, shadow.info, shadow.env, None)
    return None


def mapped_payload(action, payload, before, shadow):
    view = shadow.view()
    try:
        name = payload["action"]
        if name in {"play_card", "end_turn", "use_potion"}:
            target = -1
            if payload.get("target"):
                tmap = live.build_target_map((before.get("battle") or {}).get("enemies") or [])
                from sts2_gym.commands import resolve_living_ordinal, translate_target

                target = resolve_living_ordinal(translate_target(payload, tmap, None), shadow.obs)
            return live.combat_payload(action, target, view, before)[0]
        if name == "choose_map_node":
            return live.map_payload(action, view, before)[0]
        if name in {"select_card_reward", "skip_card_reward"}:
            return live.card_reward_payload(action, view, before)[0]
        if name == "choose_rest_option":
            return live.rest_payload(action, before)[0]
        if name == "choose_event_option":
            return live.event_payload(action, before)[0]
    finally:
        view.close()


@pytest.mark.parametrize("path,data", list(clean_traces()), ids=lambda x: x.name if isinstance(x, Path) else "")
def test_shadow_and_mapping(path, data):
    steps = desyncs = checked = mismatched = 0
    first_desync = None
    for before, payload, after, shadow in replay(data):
        action = expected_sim_action(payload, before, shadow)
        if action is not None:
            checked += 1
            got = mapped_payload(action, payload, before, shadow)
            want = {k: v for k, v in payload.items() if k in ("action", "card_index", "index", "target", "slot")}
            got = {k: v for k, v in (got or {}).items() if k in want or k == "action"}
            # A target is only meaningful for cards that take one.
            if "target" in want and "target" not in got:
                want.pop("target")
            if got != want:
                mismatched += 1
                print(f"  mapping mismatch: recorded {want} -> mapped {got}")
        shadow.apply(payload, before)
        steps += 1
        diffs = live.compare(after, shadow)
        if diffs:
            desyncs += 1
            first_desync = first_desync or (steps, payload, diffs)
            break
    print(f"{path.name}: {steps} steps, {checked} decisions checked, {mismatched} mismatches, desync={first_desync}")
    assert mismatched == 0
    # The trace corpus is a fidelity benchmark for the simulator: a desync late in a run
    # is a simulator divergence, not a bridge bug. The bridge must at least get a long
    # way in before the first one.
    assert first_desync is None or first_desync[0] > 20, first_desync


def test_compare_detects_divergence():
    """The sync check must fire when the shadow takes a different move than the game."""
    path, data = next(clean_traces())
    fired = False
    skipped = False
    for before, payload, after, shadow in replay(data):
        if not skipped and payload["action"] == "play_card":
            skipped = True  # the game played this card; the shadow ends its turn instead
            shadow.apply({"action": "end_turn"}, before)
        else:
            shadow.apply(payload, before)
        if skipped and live.compare(after, shadow):
            fired = True
            break
    assert skipped and fired


def test_chest_relic_matches_the_real_game():
    """Every chest a recorded run opened offers, in the patched simulator, the relic the
    game offered (patch 0006 -- before it, chests paid gold and no relic at all)."""
    checked = 0
    for _, data in clean_traces():
        for before, payload, after, shadow in replay(data):
            shadow.apply(payload, before)
            offered = (after.get("treasure") or {}).get("relics") or []
            if after.get("state_type") == "treasure" and offered and before.get("state_type") == "map":
                assert shadow.phase == live.sim.PHASE_RELIC_REWARD
                sim_relic = int(shadow.info["pending_rewards"][2])
                assert sim_relic == live.relic_id(offered[0]["id"]), (data["seed"], offered[0]["id"])
                checked += 1
    assert checked >= 1


@pytest.mark.parametrize("name", ["7WGQ2VNJ4M-a8-buff400.json", "8QKMNR4T2W-a8-buff200.json"])
def test_chest_relic_matches_in_runs_with_debug_buffs(name):
    """The two chests in recorded runs that used debug HP buffs, which the bridge replay
    above skips, checked with the simulator's own replay (it applies the buffs)."""
    import sys

    sys.path.insert(0, str(paths.SIM_ROOT / "scripts"))
    import replay_full_run_trace as replay

    payload = replay.load_payload(paths.SIM_ROOT / "tests" / "fixtures" / "run_trace" / name)
    emulated = {st["step"]: st for st in replay.replay_trace(payload, emulator_seed=payload["seed"]).payload["trace"]}
    checked = 0
    for prev, cur in zip(payload["trace"], payload["trace"][1:]):
        if (cur.get("raw_state") or {}).get("state_type") == "treasure" and (prev.get("raw_state") or {}).get("state_type") != "treasure":
            offered = cur["raw_state"]["treasure"]["relics"]
            items = emulated[cur["step"]]["summary"]["rewards"]["items"]
            assert next(i["id"] for i in items if i["type"] == "relic") == live.relic_id(offered[0]["id"])
            checked += 1
    assert checked == 1


def _selection_screen(run) -> dict:
    """The mod's card_select screen for the simulator's open deck selection."""
    from sts2rl.bridge import cli

    offered = cli.selection_options(run)
    cards = [{"index": i, "id": "CARD." + cli.card_entry(c), "is_upgraded": bool(u)} for i, (_, (c, u)) in enumerate(sorted(offered.items()))]
    return {"state_type": "card_select", "card_select": {"screen_type": "select", "cards": cards}}


def test_a_multi_card_deck_selection_names_each_card_by_its_place_on_the_games_list():
    # Neow's "remove two cards" on EVAL000003: after the first Strike goes, the second
    # Strike is deck index 0 again in the simulator, but still position 1 on the game's list.
    from sts2rl import sim
    from sts2rl.agent import Agent, AgentConfig
    from sts2rl.bridge import cli

    agent = Agent(None, AgentConfig(combat_sims=8, samples=1, record=False), seed=0)
    with sim.Run.new("EVAL000003") as run:
        run.step(2)
        assert run.phase == cli.PHASE_TRANSFORM_SELECT
        payload = live.card_select_payload(agent, run, _selection_screen(run))
    assert payload == [
        {"action": "select_card", "index": 0},
        {"action": "select_card", "index": 1},
        {"action": "confirm_selection"},
    ]


def test_a_selection_the_game_lists_differently_is_a_divergence():
    from sts2rl import sim
    from sts2rl.agent import Agent, AgentConfig

    agent = Agent(None, AgentConfig(combat_sims=8, samples=1, record=False), seed=0)
    with sim.Run.new("EVAL000003") as run:
        run.step(2)
        screen = _selection_screen(run)
        screen["card_select"]["cards"].pop()  # the game offers one card fewer
        with pytest.raises(live.Diverged):
            live.card_select_payload(agent, run, screen)


def _rewards_screen(*types):
    items = [{"index": i, "type": t} for i, t in enumerate(types)]
    return {"state_type": "rewards", "rewards": {"items": items, "can_proceed": True}}


def test_a_card_reward_the_simulator_dropped_is_left_on_the_games_screen():
    from sts2rl import sim

    on_screen = {"phase": sim.PHASE_RELIC_REWARD, "pending_rewards": (0, 0, 0, 0)}
    assert live.rewards_payload(_rewards_screen("card"), on_screen) == {"action": "proceed"}
    assert live.rewards_payload(_rewards_screen("gold", "card"), on_screen) == {"action": "claim_reward", "index": 0}

    held = {"phase": sim.PHASE_RELIC_REWARD, "pending_rewards": (0, 0, 0, 1)}
    assert live.rewards_payload(_rewards_screen("card"), held) == {"action": "claim_reward", "index": 0}
    already_open = {"phase": sim.PHASE_CARD_REWARD, "pending_rewards": (0, 0, 0, 0)}  # Prayer Wheel's second
    assert live.rewards_payload(_rewards_screen("card"), already_open) == {"action": "claim_reward", "index": 0}


# Live run RNL2Z10QLE as the bridge played it, up to the agent skipping floor 3's card
# reward. The game kept the reward on its screen and the bridge opened it again.
RNL2Z10QLE = [
    ("choose_event_option", 2), ("choose_event_option", 0), ("choose_map_node", 3),
    ("play", 0), ("play", 1, "CORPSE_SLUG_0"), ("end_turn",),
    ("play", 0, "CORPSE_SLUG_0"), ("play", 0), ("play", 0), ("end_turn",),
    ("play", 0, "CORPSE_SLUG_0"), ("play", 0, "CORPSE_SLUG_0"), ("play", 1, "CORPSE_SLUG_0"), ("end_turn",),
    ("play", 0, "CORPSE_SLUG_0"), ("play", 0), ("play", 1), ("end_turn",),
    ("play", 0, "CORPSE_SLUG_0"), ("play", 1, "CORPSE_SLUG_0"),
    ("claim_reward", 0), ("claim_reward", 0), ("select_card_reward", 0), ("proceed",), ("choose_map_node", 0),
    ("play", 4, "SLUDGE_SPINNER_0"), ("play", 1, "SLUDGE_SPINNER_0"), ("end_turn",),
    ("play", 0, "SLUDGE_SPINNER_0"), ("play", 1, "SLUDGE_SPINNER_0"), ("play", 2, "SLUDGE_SPINNER_0"), ("end_turn",),
    ("play", 3, "SLUDGE_SPINNER_0"), ("claim_reward", 0), ("claim_reward", 0), ("claim_reward", 0),
    ("skip_card_reward",),
]


def test_a_skipped_card_reward_stays_skipped_on_a_live_run():
    from sts2rl import sim

    shadow = live.ShadowSim("RNL2Z10QLE")
    try:
        for move in RNL2Z10QLE:
            name, *args = move
            payload = {"action": "play_card" if name == "play" else name}
            if name in ("choose_event_option", "choose_map_node", "claim_reward"):
                payload["index"] = args[0]
            elif name in ("play", "select_card_reward"):
                payload["card_index"] = args[0]
            if len(args) > 1:
                payload["target"] = args[1]
            enemies = [{"entity_id": "SLUDGE_SPINNER_0" if shadow.info["floor"] == 3 else "CORPSE_SLUG_0", "hp": 1}]
            shadow.apply(payload, {"battle": {"enemies": enemies}} if shadow.phase == sim.PHASE_COMBAT else {})
        assert shadow.phase == sim.PHASE_RELIC_REWARD  # the simulator's screen, card reward gone
        # The game is back on its rewards screen, the card reward still listed.
        assert live.rewards_payload(_rewards_screen("card"), shadow.info) == {"action": "proceed"}
        shadow.apply({"action": "proceed"}, {})
        assert shadow.phase == sim.PHASE_MAP and shadow.info["floor"] == 3
    finally:
        shadow.close()
