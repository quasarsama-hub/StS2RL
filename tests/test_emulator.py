"""Contract tests for the guarded external emulator, not simulated game rules."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from spire2_rl.emulator import (
    DEFAULT_UPSTREAM_PATH,
    InvalidActionError,
    PINNED_UPSTREAM_COMMIT,
    SimulatorError,
    make_env,
)


@pytest.fixture(autouse=True)
def checkout_path(monkeypatch):
    if "SPIRE2_UPSTREAM_PATH" not in os.environ and not DEFAULT_UPSTREAM_PATH.is_dir():
        workspace_checkout = Path(__file__).resolve().parents[2] / "upstream" / "sts2-rl-agent"
        if workspace_checkout.is_dir():
            monkeypatch.setenv("SPIRE2_UPSTREAM_PATH", str(workspace_checkout))


def _same_obs(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    else:
        np.testing.assert_array_equal(a, b)


def _first_combat(env):
    _, info = env.reset(seed=22)
    for _ in range(20):
        if info["phase"] == env.raw_env._mgr.PHASE_COMBAT:
            return info
        _, _, terminated, truncated, info = env.step(int(np.flatnonzero(info["action_mask"])[0]))
        assert not (terminated or truncated)
    raise AssertionError("Did not reach the first combat")


@pytest.mark.parametrize("scope", ["fullrun", "combat"])
def test_seeded_trace_observations_and_masks_reproduce(scope):
    env = make_env(scope, max_steps=30)
    trace = []
    for attempt in range(2):
        obs, info = env.reset(seed=771)
        for i in range(12):
            assert env.observation_space.contains(obs)
            assert info["action_mask"].dtype == bool
            assert info["action_mask"].shape == (env.action_space.n,)
            action = int(np.flatnonzero(info["action_mask"])[0])
            if attempt == 0:
                trace.append((obs, info["action_mask"].copy(), action))
            else:
                _same_obs(obs, trace[i][0])
                np.testing.assert_array_equal(info["action_mask"], trace[i][1])
                assert action == trace[i][2]
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
    env.close()


@pytest.mark.parametrize("scope", ["fullrun", "combat"])
def test_healthy_step_limit_is_timeout_not_death(scope):
    env = make_env(scope, max_steps=1, seed=7)
    _, info = env.reset()
    action = int(np.flatnonzero(info["action_mask"])[0])
    _, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert truncated
    assert info["outcome"] == "timeout"
    assert not info["player_won"]
    assert info["hp"] > 0
    assert reward == 0
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)
    env.close()


def test_illegal_action_is_rejected_before_mutation():
    env = make_env("fullrun", seed=41)
    _, info = env.reset()
    invalid = int(np.flatnonzero(~info["action_mask"])[0])
    with pytest.raises(InvalidActionError):
        env.step(invalid)
    assert env._steps == 0
    np.testing.assert_array_equal(env.action_masks(), info["action_mask"])
    mask = env.action_masks()
    mask[:] = False
    assert env.action_masks().any(), "Callers cannot overwrite the environment mask"
    env.close()


def test_engine_error_raises_without_false_gameplay_loss(monkeypatch):
    env = make_env("fullrun", seed=4)
    _, info = env.reset()

    def broken_step(action):
        assert os.environ["STS2_STRICT"] == "1"
        raise RuntimeError("injected simulator fault")

    monkeypatch.setenv("STS2_STRICT", "0")
    monkeypatch.setattr(env.raw_env, "step", broken_step)
    with pytest.raises(SimulatorError, match="injected simulator fault"):
        env.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert os.environ["STS2_STRICT"] == "0"
    assert not env.raw_env._mgr.is_over
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.close()


def test_public_reward_choices_have_correct_action_slot_features(monkeypatch):
    env = make_env("fullrun", seed=12)
    env.reset()
    mgr = env.raw_env._mgr
    mgr._phase = mgr.PHASE_CARD_REWARD
    cards = list(env._card_lookup)[:4]
    offers = [{"action": "skip"}] + [
        {"action": "pick_card", "index": i, "card_id": card, "upgraded": i == 1}
        for i, card in enumerate(cards)
    ]
    monkeypatch.setattr(mgr, "get_available_actions", lambda: offers)
    obs = env._observation(env.raw_env._encode_obs())
    assert obs["choice_features"][2, 5] == 1
    assert obs["choice_features"][4, 0] == 1
    assert obs["choice_features"][0, 2] == 0
    first_id = obs["choice_features"][1, 2]
    offers[1]["card_id"] = cards[1]
    new_obs = env._observation(env.raw_env._encode_obs())
    assert new_obs["choice_features"][1, 2] != first_id
    assert env.observation_space.contains(new_obs)
    env.close()


def test_policy_cannot_observe_hidden_draw_order():
    env = make_env("fullrun")
    info = _first_combat(env)
    assert info["phase"] == env.raw_env._mgr.PHASE_COMBAT
    assert not env.raw_env.include_future_intents
    before = env._observation(env.raw_env._encode_obs())
    env.raw_env._mgr.get_combat_state().draw_pile.reverse()
    after = env._observation(env.raw_env._encode_obs())
    _same_obs(before, after)
    assert "future_intents" not in before
    env.close()


def test_fullrun_defaults_and_fingerprint_are_explicit():
    env = make_env()
    data = json.loads(json.dumps(env.fingerprint()))
    assert data["upstream_commit"] == PINNED_UPSTREAM_COMMIT
    assert data["max_acts"] == 3
    assert data["scope"] == "fullrun"
    assert data["victory_scope"] == "fullrun"
    assert data["parity_verified"] is False
    assert data["observation_options"]["future_intents"] is False
    env.close()


def test_incompatible_checkout_is_rejected():
    with pytest.raises(SimulatorError, match="Expected clean emulator commit"):
        make_env(expected_commit="not-the-pinned-revision")


def test_empty_live_phase_is_an_error_not_a_fabricated_legal_action(monkeypatch):
    env = make_env(seed=3)
    _, info = env.reset()
    monkeypatch.setattr(env.raw_env._mgr, "get_available_actions", lambda: [])
    with pytest.raises(ValueError, match="no decisions"):
        env._info(info, terminated=False, truncated=False)
    env.close()


def test_killing_potion_resolves_post_combat_phase():
    env = make_env("fullrun")
    _first_combat(env)
    from sts2_env.core.constants import POTION_ACTION_START
    from sts2_env.potions.base import create_potion

    mgr = env.raw_env._mgr
    combat = mgr.get_combat_state()
    for i, enemy in enumerate(combat.enemies):
        enemy.current_hp = 1 if i == 0 else 0
        enemy.block = 0
    combat.potions.append(create_potion("FirePotion", slot=0))
    info = env._info(env.raw_env._build_info(), terminated=False, truncated=False)
    action = POTION_ACTION_START + 1
    assert info["action_mask"][action]
    _, _, terminated, truncated, info = env.step(action)
    assert combat.is_over and combat.player_won
    assert not (terminated or truncated)
    assert mgr.phase != mgr.PHASE_COMBAT
    assert mgr.get_available_actions()
    assert info["hp"] > 0
    env.close()


def test_engine_rejected_card_does_not_turn_into_end_turn(monkeypatch):
    env = make_env("fullrun")
    info = _first_combat(env)
    legal_cards = np.flatnonzero(info["action_mask"][:env._upstream._LAYOUT.combat_size])
    action = int(next(a for a in legal_cards if a != 0))
    called = []

    def reject(translated):
        called.append(translated["action"])
        return {"success": False}

    monkeypatch.setattr(env.raw_env._mgr, "take_action", reject)
    with pytest.raises(SimulatorError, match="rejected advertised legal play_card"):
        env.step(action)
    assert called == ["play_card"]
    env.close()


def test_bundle_and_sacrifice_are_translated_without_invented_skip(monkeypatch):
    env = make_env("fullrun", seed=12)
    env.reset()
    mgr = env.raw_env._mgr
    mgr._phase = mgr.PHASE_CARD_REWARD
    choices = [
        {"action": "pick_card_bundle", "index": 0, "card_ids": ["BASH"], "upgraded": [False]},
        {"action": "sacrifice_card_reward"},
    ]
    called = []
    monkeypatch.setattr(mgr, "get_available_actions", lambda: choices)
    monkeypatch.setattr(mgr, "take_action", lambda a: called.append(a) or {"success": True})
    info = env._info(env.raw_env._build_info(), terminated=False, truncated=False)
    np.testing.assert_array_equal(np.flatnonzero(info["action_mask"]), [0, 1])
    env.step(0)
    env.step(1)
    assert called == choices
    env.close()


def test_shop_more_than_nine_purchases_remain_reachable(monkeypatch):
    env = make_env("fullrun", seed=15)
    env.reset()
    mgr = env.raw_env._mgr
    mgr._phase = mgr.PHASE_SHOP
    choices = [{"action": "leave_shop"}] + [
        {"action": "buy_card", "index": i, "card_id": "BASH", "price": 50}
        for i in range(13)
    ]
    called = []
    monkeypatch.setattr(mgr, "get_available_actions", lambda: choices)
    monkeypatch.setattr(mgr, "take_action", lambda a: called.append(a) or {"success": True})
    info = env._info(env.raw_env._build_info(), terminated=False, truncated=False)
    assert np.count_nonzero(info["action_mask"]) == 14
    env.step(13)
    assert called == [choices[13]]
    env.close()


@pytest.mark.parametrize("context", ["run", "event", "combat"])
def test_selection_prompt_and_rules_visible_before_confirmation(context):
    env = make_env("fullrun")
    _first_combat(env)
    from sts2_env.core.selection import CardChoiceOption, PendingCardChoice
    from sts2_env.run.events import EventModel

    mgr = env.raw_env._mgr
    combat = mgr.get_combat_state()

    def forbidden_resolver(cards):
        raise AssertionError("Observation must not inspect/execute a selection resolver")

    choice = PendingCardChoice(
        prompt="Choose exactly two cards to upgrade.",
        options=[CardChoiceOption(card=card, source_pile="deck")
                 for card in mgr.run_state.player.deck[:3]],
        resolver=forbidden_resolver, min_choices=2, max_choices=2, allow_skip=False,
    )
    if context == "combat":
        combat.pending_choice = choice
    elif context == "run":
        mgr._phase = mgr.PHASE_CARD_REWARD
        mgr.run_state.pending_choice = choice
    else:
        mgr._phase = mgr.PHASE_EVENT
        mgr._event_model = EventModel()
        mgr._event_model.event_id = "VisibleForge"
        mgr._event_model.pending_choice = choice
    actions = mgr.get_available_actions()
    assert not any(a["action"] == "confirm_choice" for a in actions)
    before = env._observation(env.raw_env._encode_obs())
    rows = before["choice_features"][before["choice_features"][:, 0] == 1]
    assert len(rows) == 3
    np.testing.assert_allclose(rows[:, 16:20], np.tile([0.2, 0.2, 0.0, 0.0], (3, 1)))
    assert np.any(rows[:, 20:28] != 0), "Prompt is visible before confirm becomes legal"

    choice.prompt = "Choose exactly two cards to remove."
    changed_prompt = env._observation(env.raw_env._encode_obs())
    changed_rows = changed_prompt["choice_features"][changed_prompt["choice_features"][:, 0] == 1]
    assert not np.array_equal(rows[:, 20:28], changed_rows[:, 20:28])

    choice.min_choices = 3
    choice.max_choices = 3
    changed_rules = env._observation(env.raw_env._encode_obs())
    changed_rows = changed_rules["choice_features"][changed_rules["choice_features"][:, 0] == 1]
    np.testing.assert_allclose(changed_rows[:, 16:18], 0.3)
    assert not np.array_equal(before["choice_features"], changed_rules["choice_features"])

    choice.selected_indices.add(0)
    selected = env._observation(env.raw_env._encode_obs())
    selected_rows = selected["choice_features"][selected["choice_features"][:, 0] == 1]
    assert selected_rows[0, 6] == 1
    np.testing.assert_allclose(selected_rows[:, 7], 0.1)
    np.testing.assert_allclose(selected_rows[:, 18], 0.1)

    choice.selected_indices.clear()
    choice.allow_skip = True
    skippable = env._observation(env.raw_env._encode_obs())
    skip_rows = skippable["choice_features"][skippable["choice_features"][:, 0] == 1]
    assert np.all(skip_rows[:, 19] == 1)
    assert env.observation_space.contains(skippable)
    assert env.fingerprint()["action_features_version"] == "public-slot-v3"
    env.close()
