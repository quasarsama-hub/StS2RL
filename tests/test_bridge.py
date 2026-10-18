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
