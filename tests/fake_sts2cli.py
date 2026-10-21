"""A stand-in for sts2-cli, backed by the simulator, for testing the bench offline.

It speaks sts2-cli's protocol (JSON lines; `start_run`, `action`, `quit`) and reproduces
the engine's habits the bench depends on (each from sts2-cli's RunSimulator.cs): gold,
potions and relics are collected without asking, in reward order; chests open themselves;
"proceed" pages are skipped; a selection with one forced pick is made without a screen;
the between-act heal lands as soon as the boss dies; enemies are listed alive-only and
targeted by alive ordinal, potions by slot; the floor is counted per act; relics and
potions are reported by title, cards by "CARD.<ENTRY>". Events show only real options --
the simulator's walk-away action has no button in the game.

FAKE_STS2CLI_FAULT=hp:<floor> makes it report one HP less than the truth from that floor
on, so a test can check the bench notices; FAKE_STS2CLI_FAULT=ceil-heal rounds the
between-act heal up, as unpatched sts2-cli does; FAKE_STS2CLI_FAULT=buy-fails answers a
purchase the way sts2-cli does when the game's own code throws.
"""

from __future__ import annotations

import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sts2rl import features as F  # noqa: E402
from sts2rl import sim  # noqa: E402
from sts2rl.agent import BUNDLE_CONFIRM, PHASE_BUNDLE_SELECT, REWARD_SKIP  # noqa: E402
from sts2rl.bridge import cli  # noqa: E402

NAMES = cli.Names(None)
FAULT = os.environ.get("FAKE_STS2CLI_FAULT", "")


class Refusal(Exception):
    """An error sts2-cli answers in its own words (no stack trace)."""


def out(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


class Engine:
    def __init__(self, seed: str):
        self.run = sim.Run.new(seed)
        self.act_start = {1: self.run.floor}
        self.rested = False
        self.card_rewards_shown = 0  # in this room
        self.fight_rewards = False  # the reward screen up now is a fight's
        self.ancients_seen = {1}  # acts whose ancient has been entered (Neow: at once)
        self.advance()

    # -- the engine's own housekeeping --------------------------------------------------

    def advance(self) -> None:
        run = self.run
        for _ in range(64):
            ph = run.phase
            legal = [int(a) for a in run.legal()]
            if ph == sim.PHASE_COMBAT:
                self.fight_rewards = True
            elif ph in (sim.PHASE_MAP, sim.PHASE_EVENT, sim.PHASE_ANCIENT, sim.PHASE_SHOP, sim.PHASE_REST):
                self.fight_rewards = False
            if ph == sim.PHASE_MAP:
                self.card_rewards_shown = 0
            if ph == sim.PHASE_ANCIENT:
                self.ancients_seen.add(F.ACT_ORDINAL.get(run.act_id, 1))
            if ph == sim.PHASE_RELIC_REWARD:
                gold, potion, relic, _ = run.state_list(6, 4)
                if gold or potion or relic:
                    run.step(0)  # gold, potions and relics are taken by themselves, in order
                    continue
                cards = [a for a in legal if a < REWARD_SKIP]
                if cards and not (self.in_combat_room() and self.card_rewards_shown):
                    run.step(cards[0])  # a card reward is a screen -- only a fight's first
                    continue
                run.step(REWARD_SKIP)
                continue
            if ph == sim.PHASE_CARD_REWARD and self.in_combat_room() and self.card_rewards_shown:
                run.step(REWARD_SKIP)  # Prayer Wheel / White Star's extra one: never shown
                continue
            if ph == cli.PHASE_TREASURE:
                run.step(REWARD_SKIP)
                continue
            if ph in (sim.PHASE_EVENT, sim.PHASE_ANCIENT, sim.PHASE_REST) and len(legal) == 1:
                run.step(legal[0])  # a proceed page, or an event with no options
                continue
            if ph == sim.PHASE_EVENT and run.event_id == cli.EVENT_RESULT_PENDING:
                run.step(0)  # a finished event goes straight to the map
                continue
            if ph == sim.PHASE_REST and self.rested:
                run.step(REWARD_SKIP)  # sts2-cli forces the map after any rest option
                continue
            if ph != sim.PHASE_REST and ph != cli.PHASE_TRANSFORM_SELECT:
                self.rested = False
            if ((ph == sim.PHASE_COMBAT and run.pending_selection() > 0) or ph == cli.PHASE_TRANSFORM_SELECT) and len(legal) == 1:
                run.step(legal[0])  # one forced pick: HeadlessCardSelector returns it at once
                continue
            break
        act = F.ACT_ORDINAL.get(run.act_id, 1)
        self.act_start.setdefault(act, run.floor)

    def in_combat_room(self) -> bool:
        # sts2-cli puts only a fight's first card reward on screen; an event's (Neow's
        # Kaleidoscope, Glass Eye) are each shown, through its pending-reward path.
        return self.fight_rewards

    # -- what it shows -----------------------------------------------------------------

    def player(self) -> dict:
        run = self.run
        hp = run.hp
        act = F.ACT_ORDINAL.get(run.act_id, 1)
        if run.phase == sim.PHASE_MAP and act not in self.ancients_seen:
            # HealBetweenActs, applied when the boss died; the simulator heals on the
            # ancient. Decimal * 0.8m, then (int): rounded down (Math.Ceiling unpatched).
            heal = Decimal(run.max_hp - run.hp) * Decimal("0.8")
            hp += math.ceil(heal) if FAULT == "ceil-heal" else int(heal)
        if FAULT.startswith("hp:") and run.floor >= int(FAULT[3:]):
            hp -= 1
        pots = run.combat_potions() if run.phase == sim.PHASE_COMBAT else list(run.potions())
        return {
            "hp": hp,
            "max_hp": run.max_hp,
            "gold": run.gold,
            "relics": [{"name": NAMES.relic(int(r[0]))} for r in run.relics()],
            # Player.Potions skips empty slots: the index counts potions held, not slots.
            "potions": [{"index": i, "name": NAMES.potion(p)} for i, p in enumerate(p for p in pots if p)],
            "deck": [self.deck_card(c) for c in run.deck()],
        }

    @staticmethod
    def deck_card(c) -> dict:
        card = {"id": "CARD." + cli.card_entry(int(c[0])), "upgraded": bool(c[1])}
        if int(c[2]):  # sts2-cli adds the enchantment's title (and a non-zero amount) only when there is one
            card["enchantment"] = f"ENCHANTMENT_{int(c[2])}"
            if int(c[3]):
                card["enchantment_amount"] = int(c[3])
        return card

    def context(self) -> dict:
        act = F.ACT_ORDINAL.get(self.run.act_id, 1)
        return {"act": act, "floor": self.run.floor - self.act_start.get(act, 0)}

    def card(self, cid: int, upgraded: bool = False, index: int = 0) -> dict:
        from sts2rl import gamedata as gd

        c = gd.cards().get(cid)
        game_id = "CARD." + cli.card_entry(cid) if c else "CARD.UNSEEN"
        return {"index": index, "id": game_id, "upgraded": upgraded, "target_type": c.target if c else "None"}

    def selection(self) -> list[tuple[int, bool]]:
        listed = [(c, u) for _, (c, u) in sorted(cli.selection_options(self.run).items())]
        if any(c is None for c, _ in listed):
            # The observation carries ten candidates; the real engine lists them all.
            # Taking the first one, on a copy, slides the next into view -- for as long as
            # the screen stays open (a pick-one screen reveals just one more).
            window = sim.native.OBS_MAX_SELECTION_CANDIDATES
            with self.run.clone(resample=False) as probe:
                for j in range(window, len(listed)):
                    if not probe.step(0) or probe.pending_selection() < window:
                        break
                    listed[j] = cli.selection_candidates(probe)[window - 1]
        return [(c or 0, bool(u)) for c, u in listed]

    def state(self) -> dict:
        run = self.run
        ph = run.phase
        base = {"type": "decision", "context": self.context(), "player": self.player()}
        if ph == sim.PHASE_COMPLETE:
            return {**base, "decision": "game_over", "victory": bool(run.won() and run.hp > 0)}
        if (ph == sim.PHASE_COMBAT and run.pending_selection() > 0) or ph == cli.PHASE_TRANSFORM_SELECT:
            cards = [self.card(c, u, i) for i, (c, u) in enumerate(self.selection())]
            return {**base, "decision": "card_select", "cards": cards, "min_select": 0, "max_select": len(cards)}
        if ph == sim.PHASE_COMBAT:
            hand = [self.card(c, bool(u), i) for i, (c, u) in enumerate(run.hand())]
            enemies = [{"index": k, "hp": e[0], "max_hp": e[1], "block": e[2]} for k, e in enumerate(e for e in run.enemies() if e[0] > 0)]
            return {**base, "decision": "combat_play", "hand": hand, "enemies": enemies, "energy": run.energy}
        if ph == sim.PHASE_CARD_REWARD:
            ids = [int(run.obs[sim.C + 9 + i]) for i in range(3) if int(run.obs[sim.C + 9 + i])]
            fourth = int(run.obs[sim.C + sim.LAYOUT["fourth_reward_offset"]])  # Lasting Candy
            cards = [self.card(c, False, i) for i, c in enumerate(ids + ([fourth] if fourth else []))]
            self.card_rewards_shown += 1
            return {**base, "decision": "card_reward", "cards": cards, "can_skip": True}
        if ph == sim.PHASE_MAP:
            choices = [{"col": c[0], "row": c[1]} for c in run.map_choice_coords() if c is not None]
            return {**base, "decision": "map_select", "choices": choices}
        if ph == sim.PHASE_REST:
            options = [{"index": i, "option_id": cli.REST_IDS[a][0], "is_enabled": True} for i, a in enumerate(a for a in run.legal() if int(a) in cli.REST_IDS)]
            return {**base, "decision": "rest_site", "options": options}
        if ph == sim.PHASE_SHOP:
            cards = [{"index": i, "name": NAMES.card(c), "is_stocked": bool(c)} for i, c in enumerate(run.state_list(8, 7))]
            relics = [{"index": i, "name": NAMES.relic(r), "is_stocked": bool(r)} for i, r in enumerate(run.state_list(9, 3))]
            potions = [{"index": i, "name": NAMES.potion(p), "is_stocked": bool(p)} for i, p in enumerate(run.state_list(10, 3))]
            return {**base, "decision": "shop", "cards": cards, "relics": relics, "potions": potions}
        if ph in (sim.PHASE_EVENT, sim.PHASE_ANCIENT):
            from sts2rl import gamedata as gd

            legal = {int(a) for a in run.legal()}
            known = gd.event_options().get(run.event_id) if ph == sim.PHASE_EVENT else None
            if ph == sim.PHASE_EVENT and run.event_id != gd.event_ids()["WelcomeToWongos"] and not (known and known[1] > cli.EVENT_SKIP):
                legal.discard(cli.EVENT_SKIP)  # the simulator's walk-away has no button
            # The event's own option list: its fixed count when the simulator's table has
            # one, else whatever is offered.
            n = known[1] if known else (max(legal) + 1 if legal else 0)
            if ph == sim.PHASE_ANCIENT:
                keys = [f"NEOW.pages.INITIAL.options.{cli.relic_entry(r)}" for r in run.state_list(3, 3) if r]
            else:
                entry = cli.event_entry_by_id().get(run.event_id, "SOME_EVENT")
                keys = [f"{entry}.pages.INITIAL.options.OPTION_{i}" for i in range(n)]
            options = [{"index": i, "text_key": keys[i] if i < len(keys) else "", "is_locked": i not in legal} for i in range(n)]
            return {**base, "decision": "event_choice", "options": options}
        if ph == PHASE_BUNDLE_SELECT:
            return {**base, "decision": "bundle_select", "bundles": [{"index": 0}, {"index": 1}]}
        return {**base, "decision": "unknown", "message": f"phase {ph}"}

    # -- what it does ------------------------------------------------------------------

    def step(self, action: int, target: int = -1) -> None:
        if not self.run.step(action, target):
            raise ValueError(f"action {action} refused in phase {self.run.phase}")

    def act(self, name: str, args: dict) -> None:
        run = self.run
        if name in ("play_card", "use_potion"):
            hand = run.hand()
            if name == "play_card":
                action = int(args["card_index"])
            else:
                held = [slot for slot, p in enumerate(run.combat_potions()) if p]
                action = len(hand) + 1 + held[int(args["potion_index"])]
            living = run.living_enemy_indices()
            target = -1
            if "target_index" in args:
                target = living[int(args["target_index"])]
            elif len(living) == 1:
                target = living[0]
            self.step(action, target)
        elif name == "end_turn":
            self.step(len(run.hand()))
        elif name in ("select_cards", "skip_select"):
            # Indices name positions in the list as it was first shown; the simulator's
            # list (or deck) changes as cards are taken, so each position is followed.
            deck_mode = run.phase == cli.PHASE_TRANSFORM_SELECT and not cli.offer_cards(run)
            where = {pos: a for pos, a in enumerate(sorted(cli.selection_options(run)))}
            for position in ([int(p) for p in args["indices"].split(",")] if name == "select_cards" else []):
                action = where.pop(position)
                deck_before = cli.deck_columns(run)
                count = len(cli.selection_options(run))
                self.step(action)
                if not self._selection_open():
                    break
                if deck_mode:
                    moved = cli.follow_deck(deck_before, cli.deck_columns(run), action)
                    where = {pos: moved[a] for pos, a in where.items()}
                elif len(cli.selection_options(run)) == count - 1:
                    where = {pos: a - (a > action) for pos, a in where.items()}
            if self._selection_open():
                done = [int(a) for a in run.legal() if int(a) not in cli.selection_options(run)]
                if done:
                    self.step(done[0])
        elif name == "select_card_reward":
            index = int(args["card_index"])
            self.step(5 if index == 3 else index)  # the fourth card is Lasting Candy's, action 5
        elif name == "skip_card_reward":
            self.step(REWARD_SKIP)
        elif name == "select_map_node":
            coords = run.map_choice_coords()
            self.step(coords.index((int(args["col"]), int(args["row"]))))
        elif name == "choose_option":
            index = int(args["option_index"])
            if run.phase == sim.PHASE_REST:
                self.rested = True
                self.step([int(a) for a in run.legal() if int(a) in cli.REST_IDS][index])
            else:
                self.step(index)
        elif name in ("buy_card", "buy_relic", "buy_potion"):
            if FAULT == "buy-fails" and name != "buy_potion":
                # sts2-cli's DoBuyCard / DoBuyRelic catch the game's exception themselves.
                raise Refusal(f"Buy {name[4:]} failed: Object reference not set to an instance of an object.")
            base = {"buy_card": 0, "buy_relic": 7, "buy_potion": 10}[name]
            key = {"buy_card": "card_index", "buy_relic": "relic_index", "buy_potion": "potion_index"}[name]
            self.step(base + int(args[key]))
        elif name == "remove_card":
            self.step(cli.SHOP_REMOVE)
        elif name == "leave_room":
            self.step(cli.SHOP_LEAVE if run.phase == sim.PHASE_SHOP else REWARD_SKIP)
        elif name == "select_bundle":
            self.step(int(args["bundle_index"]))
            self.step(BUNDLE_CONFIRM)
        else:
            raise ValueError(f"Unknown action: {name}")
        self.advance()

    def _selection_open(self) -> bool:
        run = self.run
        return (run.phase == sim.PHASE_COMBAT and run.pending_selection() > 0) or run.phase == cli.PHASE_TRANSFORM_SELECT


def main() -> None:
    out({"type": "ready", "version": "fake"})
    engine = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd = json.loads(line)
        try:
            if cmd["cmd"] == "start_run":
                engine = Engine(cmd["seed"])
                out(engine.state())
            elif cmd["cmd"] == "action":
                engine.act(cmd["action"], cmd.get("args") or {})
                out(engine.state())
            elif cmd["cmd"] == "quit":
                out({"type": "quit_result", "success": True})
                break
            else:
                out({"type": "error", "message": f"Unknown command: {cmd['cmd']}"})
        except Refusal as exc:
            out({"type": "error", "message": str(exc)})
        except Exception as exc:  # the real engine answers errors in-band too
            out({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
