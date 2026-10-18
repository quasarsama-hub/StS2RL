"""A visible-information heuristic baseline, not a trained policy.

Uses the pinned external heuristic combat scorer without lookahead, plus public
choice/map/rest/shop rules. Event and boss-relic fallbacks use the first legal
option. It must be measured before collecting imitation demonstrations.
"""
from collections import Counter
import numpy as np


class FairTeacher:
    def __init__(self):
        from sts2_env.heuristic_bot import HeuristicBot
        self.bot = HeuristicBot(use_lookahead=False)
        self.decisions = Counter()
        self.phases = Counter()

    def pending(self, env):
        mgr = env.raw_env._mgr
        if mgr.phase == mgr.PHASE_COMBAT:
            return mgr.get_combat_state().pending_choice
        if mgr.run_state.pending_choice is not None:
            return mgr.run_state.pending_choice
        if mgr.phase == mgr.PHASE_EVENT and mgr._event_model is not None:
            return mgr._event_model.pending_choice
        return None

    def choice_priority(self, choice, index, deck_size):
        card = choice.options[index].card
        name = card.card_id.name
        value = self.bot._card_pick_value(name, deck_size)
        prompt = choice.prompt.lower()
        unwanted = any(word in prompt for word in ["remove", "exhaust", "discard", "cook"])
        if unwanted:
            if name in {"SLIMED", "BURN", "DAZED", "WOUND", "VOID", "SHAME", "REGRET", "DECAY", "PAIN", "PARASITE"}:
                return 100.0
            if "CURSE" in name:
                return 100.0
            return -value
        if "upgrade" in prompt:
            if card.upgraded:
                return -100.0
            return value + (20.0 if name == "BASH" else 0.0)
        if "0 cost" in prompt:
            return value + max(0, card.cost) * 5.0
        return value

    def choice_action(self, env, choice, valid):
        mgr = env.raw_env._mgr
        if mgr.phase == mgr.PHASE_COMBAT:
            named = [(slot, {"action": "confirm_choice"} if slot == 0 else
                      {"action": "choose", "index": slot - 1}) for slot in valid]
        else:
            slots = env.raw_env.public_slots()
            named = [(slot, slots[slot]) for slot in valid]
        confirms = [slot for slot, action in named if action["action"] == "confirm_choice"]
        if choice.is_multi and confirms:
            self.decisions["multi_confirm_minimum_or_skip"] += 1
            return confirms[0]
        candidates = [(slot, action) for slot, action in named if action["action"] == "choose"]
        if candidates:
            deck_size = len(mgr.run_state.player.deck)
            selected = max(candidates, key=lambda item: self.choice_priority(choice, item[1]["index"], deck_size))
            self.decisions["public_prompt_card_choice"] += 1
            return selected[0]
        if confirms:
            self.decisions["only_choice_confirmation"] += 1
            return confirms[0]
        raise RuntimeError("Pending choice exposes no recognized legal choice actions")

    def macro_score(self, env, action, position):
        mgr = env.raw_env._mgr
        state = mgr.run_state
        player = state.player
        hp = player.current_hp / max(1, player.max_hp)
        deck_size = len(player.deck)
        kind = action["action"]
        if kind in {"pick_relic_reward", "collect"}:
            return 100.0
        if kind == "pick_card":
            return self.bot._card_pick_value(action["card_id"], deck_size)
        if kind == "pick_card_bundle":
            values = [self.bot._card_pick_value(name, deck_size) for name in action["card_ids"]]
            return sum(values) / max(1, len(values))
        if kind == "skip":
            return 6.0 if deck_size > 16 else -1.0
        if kind == "pick_potion":
            return 5.0 if any(potion is None for potion in player.potions) else -1.0
        if kind == "skip_potion":
            return 1.0 if all(potion is not None for potion in player.potions) else -1.0
        if kind == "reroll_card_reward":
            return 0.0
        if kind == "sacrifice_card_reward":
            return 0.0
        if kind == "move":
            node = action.get("point_type", "UNKNOWN")
            if node == "REST_SITE":
                return 12.0 if state.act_floor >= 14 else 8.0 if hp < 0.5 else 4.0 if hp < 0.7 else 2.0
            if node == "TREASURE":
                return 5.5
            if node == "ELITE":
                return 4.5 if hp > 0.7 and state.act_floor >= 4 else 3.0 if hp > 0.55 else 0.5
            if node == "MONSTER":
                return 1.0 if hp < 0.35 else 3.0
            if node == "SHOP":
                return 4.5 if player.gold >= 200 else 3.0 if player.gold >= 100 else 1.5
            if node == "UNKNOWN":
                return 4.0 if hp >= 0.5 else 2.5
            return 1.0
        if kind == "rest_option":
            option = action.get("option_id")
            if option == "HEAL":
                return 10.0 if hp < 0.6 else 5.0
            if option == "SMITH":
                return 8.0 if hp >= 0.6 else 3.0
            return 4.0
        if kind == "leave_shop":
            return 0.5
        if kind == "buy_relic":
            return 25.0
        if kind == "buy_card":
            value = self.bot._card_pick_value(action.get("card_id", ""), deck_size)
            return 15.0 if value >= 30 else 5.0 if value > 0 else -1.0
        if kind == "buy_potion":
            return 3.0
        if kind == "remove_card":
            return 18.0 if deck_size > 12 else 5.0
        if kind in {"event_choice", "pick_relic"}:
            return 5.0 - position * 0.001
        return None

    def select(self, env):
        valid = np.flatnonzero(env.action_masks()).tolist()
        if not valid:
            raise RuntimeError("No legal actions")
        mgr = env.raw_env._mgr
        self.phases[mgr.phase] += 1
        choice = self.pending(env)
        if choice is not None:
            return self.choice_action(env, choice, valid)
        if mgr.phase == mgr.PHASE_COMBAT:
            self.decisions["no_lookahead_combat_score"] += 1
            return max(valid, key=lambda slot: self.bot._score_combat(env.raw_env, slot))
        slots = env.raw_env.public_slots()
        candidates = [(slot, self.macro_score(env, slots[slot], slot)) for slot in valid]
        scored = [(slot, score) for slot, score in candidates if score is not None]
        if scored:
            selected = max(scored, key=lambda item: item[1])[0]
            kind = slots[selected]["action"]
            self.decisions["first_legal_event_or_boss" if kind in {"event_choice", "pick_relic"}
                           else "public_macro_score"] += 1
            return selected
        self.decisions["unrecognized_first_legal_fallback"] += 1
        return valid[0]
