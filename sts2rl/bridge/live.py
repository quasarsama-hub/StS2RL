"""Play the real game through the STS2MCP mod, with the simulator as a shadow.

    python -m sts2rl.bridge.live --model models/best.pt --seed MYSEED01 --abandon-existing
    python -m sts2rl.bridge.live --model models/best.pt --attach     # a run you just started

How it works
------------
The simulator reproduces the game's RNG exactly, so a run started on the same seed,
character and ascension (Ironclad, A8 -- what the simulator models) unfolds identically
in both places. The bridge keeps a *shadow* simulator in lockstep with the game: every
command sent to the game is also applied to the shadow, through the same translation
layer the simulator project uses to replay real-game traces.

At each decision the agent reasons on the shadow state exactly as it does in training
(its searches fork the shadow with everything unseen resampled, so it never peeks), picks
a simulator action, and the bridge sends the game the command that means the same thing.
Screens with nothing to decide (claiming gold, opening a chest) are clicked through.

After every step the bridge compares the game with the shadow -- floor, HP, gold, the hand,
the enemies. If they ever disagree, the simulator has diverged from the real game and its
predictions can no longer be trusted, so the bridge stops (or, with --keep-going, falls
back to simple scripted play) and writes the trace for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths, sim
from .. import gamedata as gd
from ..agent import Agent, AgentConfig

sys.path.insert(0, str(paths.SIM_ROOT / "scripts"))

import start_real_game_run as srg  # noqa: E402
import trace_real_game as trg_post  # noqa: E402
import trace_real_game_run as trg  # noqa: E402
from sts2_gym import Sts2RunEnv, native  # noqa: E402
from sts2_gym.commands import (  # noqa: E402
    UnsupportedCommandError,
    build_target_map,
    card_id_by_replay_id,
    execute_command,
    translate_command,
)

COMBAT_STATES = {"monster", "elite", "boss"}
DEFAULT_URL = "http://localhost:15526"


# --------------------------------------------------------------------------------------
# id translation between the game's JSON and the simulator


def _snake_upper(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


_relic_by_entry: dict[str, int] | None = None
_potion_by_entry: dict[str, int] | None = None


def relic_id(entry: str) -> int:
    global _relic_by_entry
    if _relic_by_entry is None:
        _relic_by_entry = {r.entry or _snake_upper(r.name): rid for rid, r in gd.relics().items()}
    return _relic_by_entry.get(entry, 0)


def potion_id(entry: str) -> int:
    global _potion_by_entry
    if _potion_by_entry is None:
        _potion_by_entry = {_snake_upper(p.name): pid for pid, p in gd.potions().items()}
    return _potion_by_entry.get(entry, 0)


def card_id(entry: str) -> int:
    return card_id_by_replay_id().get(entry, 0)


def living_game_enemies(state: dict) -> list[dict]:
    return [e for e in ((state.get("battle") or {}).get("enemies") or []) if int(e.get("hp") or 0) > 0]


# --------------------------------------------------------------------------------------
# the shadow simulator


class ShadowSim:
    def __init__(self, seed: str):
        self.seed = seed
        self.env = Sts2RunEnv(seed=seed, max_floors=200, max_episode_steps=10**7)
        self.obs, self.info = self.env.reset()

    @property
    def phase(self) -> int:
        return int(self.info["phase"])

    def view(self) -> sim.Run:
        """A faithful copy of the shadow for the agent to reason on (it resamples itself)."""
        run = sim.Run.__new__(sim.Run)
        sim.Run.__init__(run, -1, self.seed)
        run.handle = native.run_clone(self.env._run_handle, False, 0, run._obs_buf)
        return run

    def step_action(self, action: int, target: int = -1) -> None:
        self.obs, _, _, _, self.info = self.env.step(action, target=target)

    def apply(self, payload: dict, state_before: dict) -> None:
        """Mirror a command the game just accepted."""
        name = payload.get("action")
        if name == "shop_purchase":
            slot = shop_slot_for_item(state_before, payload["index"], self.info)
            if slot is not None:
                self.step_action(slot)
            return
        if name == "combat_select_card":
            self.step_action(int(payload.get("card_index", 0)))
            return
        if name in {"combat_confirm_selection", "advance_dialogue", "claim_treasure_relic"}:
            if name == "claim_treasure_relic" and self.phase == 10:
                self.step_action(0)
            return
        target_map = build_target_map(((state_before.get("battle") or {}).get("enemies")) or [])
        action = translate_command(payload, self.obs, self.info, self.env, None)
        if action is None:
            return
        _, _, _, self.obs, self.info = execute_command(self.env, payload, self.obs, self.info, target_map=target_map)

    def close(self) -> None:
        self.env.close()


def shop_slot_for_item(state: dict, item_index: int, info: dict) -> int | None:
    items = (state.get("shop") or {}).get("items") or []
    item = next((i for i in items if i.get("index") == item_index), None)
    if item is None:
        return None
    cat = item.get("category")
    if cat == "card":
        cid = card_id(item.get("card_id", ""))
        for slot, sid in enumerate(info.get("shop_cards", ())):
            if int(sid) == cid:
                return slot
    elif cat == "relic":
        rid = relic_id(item.get("relic_id", ""))
        for k, sid in enumerate(info.get("shop_relics", ())):
            if int(sid) == rid:
                return 7 + k
    elif cat == "potion":
        pid = potion_id(item.get("potion_id", ""))
        for k, sid in enumerate(info.get("shop_potions", ())):
            if int(sid) == pid:
                return 10 + k
    elif cat == "card_removal":
        return 13
    return None


# --------------------------------------------------------------------------------------
# simulator action -> game command


class NoMapping(Exception):
    pass


def combat_payload(action: int, target: int, view: sim.Run, state: dict) -> list[dict]:
    hand = view.hand()
    player = state.get("player") or {}
    game_hand = player.get("hand") or []
    living_sim = view.living_enemy_indices()
    living = living_game_enemies(state)

    def entity_for(sim_index: int) -> str | None:
        if not living:
            return None
        if sim_index in living_sim:
            ordinal = living_sim.index(sim_index)
            if ordinal < len(living):
                return living[ordinal].get("entity_id")
        return living[0].get("entity_id")

    if view.pending_selection() > 0:
        if state.get("state_type") != "hand_select":
            raise NoMapping("simulator has a selection open, game does not")
        return [{"action": "combat_select_card", "card_index": action}, {"action": "combat_confirm_selection"}]
    if action < len(hand):
        if action >= len(game_hand) or card_id(game_hand[action].get("id", "")) != hand[action][0]:
            raise NoMapping(f"hand slot {action} differs between game and simulator")
        payload = {"action": "play_card", "card_index": action}
        if game_hand[action].get("target_type") == "AnyEnemy":
            ent = entity_for(target)
            if ent:
                payload["target"] = ent
        return [payload]
    if action == len(hand):
        return [{"action": "end_turn"}]
    slot = action - len(hand) - 1
    payload = {"action": "use_potion", "slot": slot}
    ent = entity_for(target) if target >= 0 else None
    if ent:
        payload["target"] = ent
    return [payload]


def map_payload(action: int, view: sim.Run, state: dict) -> list[dict]:
    coords = view.map_choice_coords()
    want = coords[action] if action < len(coords) else None
    for opt in (state.get("map") or {}).get("next_options") or []:
        if want is not None and (opt.get("col"), opt.get("row")) == want:
            return [{"action": "choose_map_node", "index": opt["index"]}]
    raise NoMapping(f"map choice {action} at {want} is not offered by the game")


def card_reward_payload(action: int, view: sim.Run, state: dict) -> list[dict]:
    from sts2_gym.run_constants import REWARD_SKIP_ACTION

    if action == REWARD_SKIP_ACTION:
        return [{"action": "skip_card_reward"}]
    cards = (state.get("card_reward") or {}).get("cards") or []
    sim_id = int(view.obs[sim.C + 9 + action]) if action < 3 else 0
    for card in cards:
        if card.get("index") == action and card_id(card.get("id", "")) == sim_id:
            return [{"action": "select_card_reward", "card_index": action}]
    raise NoMapping(f"card reward {action} ({gd.card_name(sim_id)}) not on the game's screen")


REST_IDS = {0: ("HEAL", "REST"), 1: ("SMITH",), 2: ("CLONE",), 4: ("LIFT",), 5: ("DIG",), 6: ("HATCH",), 7: ("COOK",), 8: ("KINDLE",)}


def rest_payload(action: int, state: dict) -> list[dict]:
    from sts2_gym.run_constants import REWARD_SKIP_ACTION

    if action == REWARD_SKIP_ACTION:
        return [{"action": "proceed"}]
    for opt in (state.get("rest_site") or {}).get("options") or []:
        if str(opt.get("id", "")).upper() in REST_IDS.get(action, ()) and opt.get("is_enabled", True):
            return [{"action": "choose_rest_option", "index": opt["index"]}]
    raise NoMapping(f"rest option {action} not offered")


def shop_payload(action: int, view: sim.Run, state: dict, info: dict) -> list[dict]:
    from sts2_gym.run_constants import SHOP_SKIP_ACTION

    if action == SHOP_SKIP_ACTION:
        return [{"action": "proceed"}]
    for item in (state.get("shop") or {}).get("items") or []:
        if item.get("is_stocked", True) and shop_slot_for_item(state, item["index"], info) == action:
            return [{"action": "shop_purchase", "index": item["index"]}]
    raise NoMapping(f"shop slot {action} has no matching item in the game")


def event_payload(action: int, state: dict) -> list[dict]:
    event = state.get("event") or {}
    if event.get("in_dialogue"):
        return [{"action": "advance_dialogue"}]
    for opt in event.get("options") or []:
        if opt.get("index") == action and not opt.get("is_locked"):
            return [{"action": "choose_event_option", "index": action}]
    raise NoMapping(f"event option {action} not available")


def card_select_payload(actions: list[int], view: sim.Run, state: dict) -> list[dict]:
    """Deck selections: the game lists the eligible cards in deck order; the simulator's
    actions are deck indices masked to the same set, so position k means the same card."""
    card_select = state.get("card_select") or {}
    legal = [int(a) for a in view.legal()]
    if card_select.get("screen_type") == "choose":
        return [{"action": "select_card", "index": legal.index(actions[0]) if actions[0] in legal else actions[0]}]
    out = []
    for a in actions:
        if a not in legal:
            raise NoMapping(f"deck index {a} is not selectable")
        out.append({"action": "select_card", "index": legal.index(a)})
    out.append({"action": "confirm_selection"})
    return out


# --------------------------------------------------------------------------------------
# sync check


def compare(state: dict, shadow: ShadowSim) -> list[str]:
    """What the game shows that the shadow does not. Empty means in sync."""
    diffs = []
    info, obs = shadow.info, shadow.obs
    player = state.get("player") or {}
    run = state.get("run") or {}
    st = state.get("state_type")
    if run.get("floor") is not None and int(run["floor"]) != int(info["floor"]):
        diffs.append(f"floor game={run['floor']} sim={info['floor']}")
    if player.get("hp") is not None and int(player["hp"]) != int(info["player_hp"]):
        diffs.append(f"hp game={player['hp']} sim={info['player_hp']}")
    if player.get("max_hp") is not None and int(player["max_hp"]) != int(info["player_max_hp"]):
        diffs.append(f"max_hp game={player['max_hp']} sim={info['player_max_hp']}")
    if st in COMBAT_STATES and int(info["phase"]) == sim.PHASE_COMBAT:
        game_hand = [card_id(c.get("id", "")) for c in (player.get("hand") or [])]
        sim_hand = [int(obs[sim.HAND + i * sim.CARD_SLOT]) for i in range(sim.MAX_HAND)]
        sim_hand = [c for c in sim_hand if c]
        if sorted(game_hand) != sorted(sim_hand):
            diffs.append(f"hand game={[gd.card_name(c) for c in game_hand]} sim={[gd.card_name(c) for c in sim_hand]}")
        game_hp = sorted(int(e.get("hp") or 0) for e in living_game_enemies(state))
        sim_hp = []
        for i in range(sim.MAX_ENEMIES):
            at = sim.ENEMY + i * sim.ENEMY_SLOT
            if int(obs[at + 1]) and int(obs[at]) > 0:
                sim_hp.append(int(obs[at]))
        if game_hp != sorted(sim_hp):
            diffs.append(f"enemy hp game={game_hp} sim={sorted(sim_hp)}")
    elif st in {"map", "shop", "rest_site", "event", "card_reward"} and player.get("gold") is not None:
        if int(player["gold"]) != int(info["gold"]):
            diffs.append(f"gold game={player['gold']} sim={info['gold']}")
    return diffs


# --------------------------------------------------------------------------------------
# the session


@dataclass
class Session:
    base_url: str
    agent: Agent
    shadow: ShadowSim
    strict: bool = True
    delay: float = 0.3
    log_path: Path | None = None
    steps: list = field(default_factory=list)
    desynced: bool = False
    queue: list = field(default_factory=list)

    def log(self, **row) -> None:
        self.steps.append(row)
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    # The agent's choice on the shadow, turned into game commands for this screen.
    def decide(self, state: dict) -> list[dict]:
        st = state.get("state_type")
        if self.desynced:
            return [trg.choose_action(state, 0)] if trg.choose_action(state, 0) else []
        if st == "rewards":
            return [trg.choose_reward_action(state)]  # claiming is not a decision
        if st == "treasure":
            relics = (state.get("treasure") or {}).get("relics") or []
            return [{"action": "claim_treasure_relic", "index": 0}] if relics else [{"action": "proceed"}]
        if st == "event" and (state.get("event") or {}).get("in_dialogue"):
            return [{"action": "advance_dialogue"}]
        view = self.shadow.view()
        try:
            if st in COMBAT_STATES or st == "hand_select":
                if view.phase != sim.PHASE_COMBAT:
                    raise NoMapping("game is in combat, simulator is not")
                action, target = self.agent.act(view)
                return combat_payload(action, target, view, state)
            action, _ = self.agent.act(view)
            if st == "map":
                return map_payload(action, view, state)
            if st == "card_reward":
                return card_reward_payload(action, view, state)
            if st == "rest_site":
                return rest_payload(action, state)
            if st == "shop":
                return shop_payload(action, view, state, self.shadow.info)
            if st == "event":
                return event_payload(action, state)
            if st == "card_select":
                picks = [action]
                # Multi-card screens: keep choosing on a copy until the simulator leaves.
                with view.clone(resample=False) as probe:
                    probe.step(action)
                    while probe.phase == view.phase and len(picks) < 8 and len(probe.legal()):
                        nxt, _ = self.agent.act(probe)
                        picks.append(nxt)
                        probe.step(nxt)
                return card_select_payload(picks, view, state)
            if st == "bundle_select":
                return [{"action": "select_bundle", "index": action}, {"action": "confirm_bundle_selection"}]
            fallback = trg.choose_action(state, 0)
            return [fallback] if fallback else []
        except NoMapping as exc:
            self.log(event="no_mapping", state_type=st, reason=str(exc))
            print(f"  ! could not express the agent's move on this screen ({exc}); using scripted play")
            fallback = trg.choose_action(state, 0)
            return [fallback] if fallback else []
        finally:
            view.close()

    def post(self, payload: dict, state: dict) -> tuple[dict, dict]:
        before = trg.compact_state(state)
        result = trg_post.post_action(self.base_url, payload)
        tries = 0
        while result.get("status") == "error" and tries < 6:
            tries += 1
            time.sleep(max(self.delay, 0.5))
            result = trg_post.post_action(self.base_url, payload)
        if result.get("status") == "error":
            raise RuntimeError(f"game refused {payload}: {result}")
        min_hand = 5 if payload["action"] == "end_turn" else 1
        after = trg.wait_for_state_to_change(
            self.base_url, before, self.delay, min_combat_hand=min_hand,
            require_new_state_type=payload["action"] == "choose_map_node",
        )
        after, _ = trg.recover_stranded_run(self.base_url, payload, before, after, self.delay, min_combat_hand=min_hand)
        return result, after

    def play(self, max_steps: int = 20000) -> dict:
        state = trg.wait_for_actionable_state(self.base_url)
        for step in range(max_steps):
            if trg.is_terminal_state(state):
                break
            payloads = self.decide(state)
            if not payloads:
                print(f"  ! nothing to do on screen {state.get('state_type')!r}; waiting")
                time.sleep(1.0)
                state = trg.wait_for_actionable_state(self.base_url)
                continue
            for payload in payloads:
                before_state = state
                result, state = self.post(payload, state)
                if not self.desynced:
                    try:
                        self.shadow.apply(payload, before_state)
                    except (UnsupportedCommandError, sim.SimError) as exc:
                        self._desync(f"shadow could not apply {payload}: {exc}", state)
                diffs = [] if self.desynced else compare(state, self.shadow)
                self.log(step=step, payload=payload, state_type=before_state.get("state_type"), diffs=diffs, raw_state=state)
                if diffs:
                    self._desync("; ".join(diffs), state)
                if trg.is_terminal_state(state) or (self.desynced and self.strict):
                    break
            if self.desynced and self.strict:
                break
            run = state.get("run") or {}
            print(f"floor {run.get('floor')}  {state.get('state_type'):12s}  {payloads[0]}")
        return state

    def _desync(self, why: str, state: dict) -> None:
        if self.desynced:
            return
        self.desynced = True
        self.log(event="desync", reason=why)
        print(f"\n!!! The simulator no longer matches the game: {why}")
        if self.strict:
            print("    Stopping so you can take over. The trace is in", self.log_path)
        else:
            print("    Continuing with scripted play (--keep-going).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="value-net checkpoint, e.g. runs/main/best.pt")
    ap.add_argument("--url", default=DEFAULT_URL)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", help="start a new custom run on this seed (needs the custom-run STS2MCP fork)")
    group.add_argument("--attach", action="store_true", help="take over a run that is still at its opening Neow screen")
    ap.add_argument("--ascension", type=int, default=8, help="must match the simulator (A8)")
    ap.add_argument("--abandon-existing", action="store_true")
    ap.add_argument("--sims", type=int, default=200, help="combat simulations per move (more = stronger, slower)")
    ap.add_argument("--keep-going", action="store_true", help="on desync, continue with scripted play instead of stopping")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--log", help="JSONL trace path (default runs/live/<seed>.jsonl)")
    args = ap.parse_args()

    from ..model import ValueNet

    if args.ascension != 8:
        print("warning: the simulator models Ascension 8; other levels will desync quickly.")
    if args.seed:
        srg.start_seeded_run(args.url, args.seed, "IRONCLAD", args.abandon_existing, ascension=args.ascension)
        seed = srg.canonical_seed(args.seed)
    else:
        seed = srg.current_run_seed(args.url)
        if not seed:
            sys.exit("No run in progress to attach to. Start an Ironclad A8 run and stop at the Neow screen.")
        state = srg.get_state(args.url)
        if int((state.get("run") or {}).get("floor") or 0) > 1:
            sys.exit("This run is past its first floor; the simulator can only shadow a run from the start.")
    print(f"Run seed {seed}. Loading {args.model} ...")
    model = ValueNet.load(args.model)
    agent = Agent(model, AgentConfig(combat_sims=args.sims, samples=4, record=False), seed=0)
    log_path = Path(args.log) if args.log else paths.RUNS_DIR / "live" / f"{seed}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shadow = ShadowSim(seed)
    session = Session(args.url, agent, shadow, strict=not args.keep_going, delay=args.delay, log_path=log_path)
    try:
        final = session.play()
    finally:
        shadow.close()
    print("\nfinished on screen", final.get("state_type"), "floor", (final.get("run") or {}).get("floor"))
    print("trace:", log_path)


if __name__ == "__main__":
    main()
