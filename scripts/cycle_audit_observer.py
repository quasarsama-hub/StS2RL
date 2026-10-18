"""Transparent, public-state logging for frozen Act 1 diagnostic runs.

Flags identify decisions to inspect; unused cards/energy need not be mistakes.
No RNG, future intents, draw order, hidden encounters or hidden cards are read.
"""
from __future__ import annotations

import importlib


class CycleAuditObserver:
    def __init__(self, env, trace_limit=12):
        if not isinstance(trace_limit, int) or trace_limit < 0:
            raise ValueError("trace_limit must be a nonnegative integer")
        self.env, self.trace_limit = env, trace_limit
        self._clear()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _clear(self):
        self.episode_audit = {key: {} for key in (
            "macro_action_counts", "map_room_choices", "picked_card_counts", "rest_option_counts")}
        self.episode_audit.update(combat_entries=[], death_context=None,
                                  flagged_decisions=[], observer_errors=[])
        self._combat_ref, self._fight_entry, self._steps = None, None, 0
        self.current_observation = None

    def _safe(self, stage, callback):
        try:
            return callback()
        except Exception as exc:
            self.episode_audit["observer_errors"].append({
                "stage": stage, "step": self._steps,
                "error": f"{type(exc).__name__}: {exc}"})
            return None

    def _count(self, key, value):
        counts = self.episode_audit[key]
        value = str(value)
        counts[value] = counts.get(value, 0) + 1

    def _snapshot(self, mgr, combat=None):
        player = combat.player if combat is not None else mgr.run_state.player
        room = mgr.current_room
        snapshot = {
            "floor": int(mgr.run_state.total_floor),
            "room_type": getattr(getattr(room, "room_type", None), "name", "UNKNOWN"),
            "enemies": [str(enemy.monster_id) for enemy in combat.enemies] if combat is not None else [],
            "hp": float(player.current_hp), "max_hp": float(player.max_hp),
            "deck_size": len(mgr.run_state.player.deck),
            "potions": [None if potion is None else str(potion.potion_id)
                        for potion in (combat.potions if combat is not None else player.potions)],
        }
        if combat is not None:
            snapshot.update(energy=int(combat.energy), block=int(player.block),
                            hand=[str(card.card_id.name) for card in combat.hand])
        return snapshot

    def _entry(self):
        mgr = self.raw_env._mgr
        combat = mgr.get_combat_state() if mgr.phase == mgr.PHASE_COMBAT else None
        if combat is not None and combat is not self._combat_ref:
            self._combat_ref = combat
            self._fight_entry = self._snapshot(mgr, combat)
            entries = self.episode_audit["combat_entries"]
            if len(entries) < 64:
                entries.append(dict(self._fight_entry))

    def _flag(self, kind, snapshot, *, fatal=False):
        if self.trace_limit == 0:
            return
        flags = self.episode_audit["flagged_decisions"]
        row = {"kind": kind, "step": self._steps, **snapshot}
        if len(flags) < self.trace_limit:
            flags.append(row)
        elif fatal:
            flags[-1] = row

    def reset(self, *args, **kwargs):
        self._clear()
        result = self.env.reset(*args, **kwargs)
        self.current_observation = result[0]
        self._safe("reset_combat_entry", self._entry)
        return result

    def action_masks(self):
        return self.env.action_masks()

    def _before(self, action):
        mgr = self.raw_env._mgr
        combat = mgr.get_combat_state() if mgr.phase == mgr.PHASE_COMBAT else None
        self._entry()
        if combat is None:
            return mgr, combat, dict(self.raw_env.public_slots()[int(action)]), None
        snapshot = self._snapshot(mgr, combat)
        if combat.pending_choice is None and int(action) == 0:
            module = importlib.import_module("sts2_env.gym_env.run_env")
            mask = self.env.action_masks()
            legal_cards = [slot for slot in range(1, min(len(mask), module._LAYOUT.combat_size))
                           if mask[slot] and not module.is_potion_action(slot)]
            snapshot["playable_card_actions"] = legal_cards
            if legal_cards or snapshot["energy"] > 0:
                self._flag("end_turn_with_cards_or_energy", snapshot)
        return mgr, combat, None, snapshot

    def _after(self, before, result):
        if before is None:
            self._entry()
            return
        mgr, combat, macro, snapshot = before
        if macro is not None:
            kind = macro["action"]
            self._count("macro_action_counts", kind)
            if kind == "move":
                self._count("map_room_choices", macro.get("point_type", "UNKNOWN"))
            elif kind in {"pick_card", "buy_card", "pick_card_bundle"}:
                for card in macro.get("card_ids", [macro.get("card_id", "UNKNOWN")]):
                    self._count("picked_card_counts", card)
            elif kind == "rest_option":
                self._count("rest_option_counts", macro.get("option_id", "UNKNOWN"))
        self._entry()
        if result[4].get("outcome") == "death":
            death = self._snapshot(mgr, combat)
            entry = self._fight_entry if combat is not None else None
            death["hp_entering_fight"] = entry["hp"] if entry is not None else None
            self.episode_audit["death_context"] = death
            if combat is not None:
                self._flag("fatal_fight", {**snapshot, "death_context": death}, fatal=True)

    def step(self, action):
        before = self._safe("before_step", lambda: self._before(action))
        result = self.env.step(action)  # Original game errors propagate unchanged.
        self.current_observation = result[0]
        self._steps += 1
        self._safe("after_step", lambda: self._after(before, result))
        return result
