"""Full runs on the REAL game engine, headless, through sts2-cli -- with the simulator
as a shadow.

sts2-cli (github.com/wuhao21/sts2-cli, MIT) loads the game's own `sts2.dll` from your
Steam install, stubs out Godot, and plays a run over stdin/stdout as JSON lines. No
window, no mod, no clicking: hundreds of real-engine runs can be scripted. What it cannot
do is fork a run, which search needs -- so, as with the live bridge, the agent reasons on a
shadow simulator kept in lockstep with the engine and the engine only executes the moves.

Every screen sts2-cli shows is matched to the shadow's phase. Screens that are not
decisions (sts2-cli collects gold, potions and relics by itself, opens chests, and skips
"proceed" pages) are stepped through on the shadow the same way. Then the two are compared
-- HP, gold, floor, deck, relics, potions, and in combat the hand, energy and every
enemy's HP -- and the first disagreement ends the run with its reason. That is the point
of the bench: the first desync on each seed is a precise, reproducible report of where
the simulator stops matching the game you own.

See `sts2rl.bridge.cli_eval` for batch runs, and scripts/setup_sts2cli.sh for setup.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from .. import features as F
from .. import gamedata as gd
from .. import sim
from ..agent import BUNDLE_CONFIRM, PHASE_BUNDLE_SELECT, REWARD_SKIP, Agent

PHASE_TRANSFORM_SELECT = 9
PHASE_TREASURE = 10
SHOP_REMOVE = 13
SHOP_LEAVE = 14
COMBAT_ROOMS = {"Monster", "Elite", "Boss"}
AUTO_LIMIT = 32  # non-decision steps in a row before the shadow is declared lost
MAX_PICKS = 10
PHASE_NAMES = {
    0: "combat", 1: "card reward", 2: "map", 3: "rest site", 4: "shop", 5: "rewards", 6: "run over",
    7: "event", 8: "ancient", 9: "card select", 10: "treasure", 11: "crystal sphere", 12: "bundle select",
    13: "fake merchant",
}
# The rest site's sparse action ids (3 is leave) and the game's option ids for each.
REST_IDS = {0: ("HEAL", "REST"), 1: ("SMITH",), 2: ("CLONE",), 4: ("LIFT",), 5: ("DIG",), 6: ("HATCH",), 7: ("COOK",), 8: ("KINDLE",)}


class CliError(RuntimeError):
    """sts2-cli died, timed out, or refused a command."""


class Desync(Exception):
    """The engine and the shadow disagree; the shadow's predictions are no longer valid."""


# --------------------------------------------------------------------------------------
# ids and names
#
# sts2-cli reports cards by id ("CARD.BASH") but relics and potions only by their English
# title. Titles come from the localization tables sts2-cli extracts from your game, so the
# simulator's ids are translated to titles through the very same tables.


def entry_of(game_id: str) -> str:
    return str(game_id).rsplit(".", 1)[-1].upper()


def _snake_upper(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


@cache
def _card_by_entry() -> dict[str, int]:
    return {(c.entry or _snake_upper(c.name)): cid for cid, c in gd.cards().items()}


def card_id(game_id: str) -> int:
    return _card_by_entry().get(entry_of(game_id), 0)


def relic_entry(relic_id: int) -> str:
    r = gd.relics().get(relic_id)
    return (r.entry or _snake_upper(r.name)) if r else f"RELIC_{relic_id}"


def potion_entry(potion_id: int) -> str:
    p = gd.potions().get(potion_id)
    return _snake_upper(p.name) if p else f"POTION_{potion_id}"


def card_entry(cid: int) -> str:
    c = gd.cards().get(cid)
    return (c.entry or _snake_upper(c.name)) if c else f"CARD_{cid}"


class Names:
    """Simulator id -> the title sts2-cli prints, read from sts2-cli's localization."""

    def __init__(self, loc_dir: Path | None = None):
        self.tables: dict[str, dict[str, str]] = {}
        for table in ("cards", "relics", "potions"):
            path = Path(loc_dir) / f"{table}.json" if loc_dir else None
            self.tables[table] = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}

    def _title(self, table: str, entry: str) -> str:
        # sts2-cli's own fallback for a missing key is the key itself.
        key = f"{entry}.title"
        return self.tables[table].get(key, key)

    def card(self, cid: int) -> str:
        return self._title("cards", card_entry(cid))

    def relic(self, rid: int) -> str:
        return self._title("relics", relic_entry(rid))

    def potion(self, pid: int) -> str:
        return self._title("potions", potion_entry(pid))


# --------------------------------------------------------------------------------------
# the transport


def find_dotnet() -> str:
    for p in (os.path.expanduser("~/.dotnet/dotnet"), os.path.expanduser("~/.dotnet-arm64/dotnet")):
        if os.access(p, os.X_OK):
            return p
    return shutil.which("dotnet") or "dotnet"


def cli_command(cli_dir: Path) -> list[str]:
    """Launch the built headless program directly (faster than `dotnet run`)."""
    cli_dir = Path(cli_dir)
    built = sorted((cli_dir / "src" / "Sts2Headless" / "bin").glob("*/net*/Sts2Headless.dll"))
    if built:
        return [find_dotnet(), str(built[-1])]
    return [find_dotnet(), "run", "--no-build", "--project", str(cli_dir / "src" / "Sts2Headless" / "Sts2Headless.csproj")]


class CliGame:
    """One sts2-cli process: send a JSON command, get the JSON state it answers with."""

    def __init__(self, command: list[str], cwd: Path | None = None, timeout: float = 120.0, stderr_path: Path | None = None):
        self.timeout = timeout
        self._stderr = open(stderr_path, "w") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1, encoding="utf-8",
        )
        self._lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._read()
        if ready.get("type") != "ready":
            self.close()
            raise CliError(f"sts2-cli did not start: {ready}")

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _read(self) -> dict:
        deadline = time.monotonic() + self.timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise CliError(f"sts2-cli gave no answer in {self.timeout:.0f}s")
            try:
                line = self._lines.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise CliError(f"sts2-cli exited (code {self.proc.poll()})")
            line = line.strip()
            if not line.startswith("{"):
                continue  # build chatter and engine logging, not protocol
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict):
                return msg

    def send(self, command: dict) -> dict:
        if self.proc.poll() is not None:
            raise CliError(f"sts2-cli exited (code {self.proc.returncode})")
        try:
            self.proc.stdin.write(json.dumps(command) + "\n")
            self.proc.stdin.flush()
        except OSError as exc:  # it died between the check and the write
            raise CliError(f"sts2-cli exited ({exc})") from exc
        return self._read()

    def start_run(self, seed: str, ascension: int = 8, character: str = "Ironclad") -> dict:
        return self.send({"cmd": "start_run", "character": character, "seed": seed, "ascension": ascension, "lang": "en"})

    def act(self, action: str, **args) -> dict:
        cmd = {"cmd": "action", "action": action}
        if args:
            cmd["args"] = args
        return self.send(cmd)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if self._stderr is not subprocess.DEVNULL:
            self._stderr.close()

    def __enter__(self) -> CliGame:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------------------
# reading the shadow


def canonical_seed(seed: str) -> str:
    """What the game does to a typed seed (SeedHelper.Canonicalize)."""
    return str(seed).upper().replace("O", "0").replace("I", "1").strip()


def selection_candidates(run: sim.Run) -> list[tuple[int, int]]:
    n = min(run.pending_selection(), sim.native.OBS_MAX_SELECTION_CANDIDATES)
    return [(int(run.obs[sim.SEL + i * sim.CARD_SLOT]), int(run.obs[sim.SEL + i * sim.CARD_SLOT + 1])) for i in range(n)]


def offer_cards(run: sim.Run) -> list[int]:
    return [c for c in run.state_list(17, 16) if c]


def selection_options(run: sim.Run) -> dict[int, tuple[int, bool | None]]:
    """simulator action -> the card it picks, on the open selection as it stands now.

    Three simulator screens answer the engine's one card_select: an in-combat selection
    (actions index its candidates; one past the end declines), a grid of offered cards
    (likewise; the grid carries no upgrade flag), and a selection over the deck (actions
    are deck indices).
    """
    if run.phase == sim.PHASE_COMBAT:
        shown = {i: (c, bool(u)) for i, (c, u) in enumerate(selection_candidates(run))}
        # The observation carries ten candidates; any past them are unseen, not absent.
        return {i: shown.get(i, (None, None)) for i in range(run.pending_selection())}
    offer = offer_cards(run)
    if offer:
        return {i: (c, None) for i, c in enumerate(offer)}
    deck = run.deck()
    return {int(a): (int(deck[a][0]), bool(deck[a][1])) for a in run.legal() if int(a) < len(deck)}


def same_card(a: tuple, b: tuple) -> bool:
    """Equal cards; None (unseen) matches anything."""
    if a[0] is None or b[0] is None:
        return True
    return a[0] == b[0] and (a[1] is None or b[1] is None or a[1] == b[1])


def rekey(positions: list[int], options: dict) -> list[tuple[int, int]] | None:
    """Pair the cards still selectable, in order, with the engine positions not yet taken.

    The simulator applies each pick at once -- a removed card leaves the deck and the
    indices after it shift, a taken candidate leaves its list -- while the engine's
    indices keep naming the list as first shown. Order is preserved, so the k-th card
    still on offer is the k-th position not yet picked.
    """
    items = sorted(options.items())
    if len(items) != len(positions):
        return None
    return [(a, pos) for (a, _), pos in zip(items, positions)]


def shadow_matches(decision: str, run: sim.Run) -> bool:
    """Is the shadow on the kind of screen the engine is showing?"""
    ph = run.phase
    in_select = ph == sim.PHASE_COMBAT and run.pending_selection() > 0
    return {
        "combat_play": ph == sim.PHASE_COMBAT and not in_select,
        "card_select": in_select or ph == PHASE_TRANSFORM_SELECT,
        "card_reward": ph == sim.PHASE_CARD_REWARD,
        "map_select": ph == sim.PHASE_MAP,
        "rest_site": ph == sim.PHASE_REST,
        "shop": ph == sim.PHASE_SHOP,
        "event_choice": ph in (sim.PHASE_EVENT, sim.PHASE_ANCIENT),
        "bundle_select": ph == PHASE_BUNDLE_SELECT,
        "game_over": ph == sim.PHASE_COMPLETE,
    }.get(decision, False)


def describe(run: sim.Run) -> str:
    return f"{PHASE_NAMES.get(run.phase, run.phase)} (floor {run.floor})"


# --------------------------------------------------------------------------------------
# one run


@dataclass
class Move:
    """One agent decision: the engine commands that express it, and the shadow's steps."""

    commands: list[tuple[str, dict]]
    steps: list[tuple[int, int]]


@dataclass
class CliRunResult:
    seed: str
    in_sync: bool = True  # the shadow matched the engine for the whole run
    finished: bool = False  # the engine reached game over (or the act cap)
    won: bool = False
    final_floor: int = 0
    acts_cleared: int = 0
    score: float = 0.0
    desync_floor: int | None = None
    desync_reason: str = ""
    engine_error: str = ""
    decisions: int = 0
    auto_steps: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CliSession:
    game: CliGame
    agent: Agent
    names: Names
    seed: str
    ascension: int = 8
    trace_path: Path | None = None
    max_decisions: int = 20000
    shadow: sim.Run | None = None
    floor_offset: dict = field(default_factory=dict)  # act -> shadow floor minus engine floor
    _trace: object = None

    # -- bookkeeping -------------------------------------------------------------------

    def log(self, **row) -> None:
        if self._trace is not None:
            self._trace.write(json.dumps(row, default=str) + "\n")
            self._trace.flush()

    def _send(self, action: str, args: dict) -> dict:
        state = self.game.act(action, **args)
        self.log(kind="command", action=action, args=args, answer=state.get("decision") or state.get("type"))
        if state.get("type") == "error":
            raise CliError(f"engine refused {action} {args}: {state.get('message')}")
        return state

    # -- the loop ----------------------------------------------------------------------

    def play(self) -> CliRunResult:
        t0 = time.perf_counter()
        seed = canonical_seed(self.seed)
        result = CliRunResult(seed=seed)
        self._trace = open(self.trace_path, "w") if self.trace_path else None
        self.shadow = sim.Run.new(seed)
        self.agent.maps = F.MapCache()
        try:
            state = self.game.start_run(seed, ascension=self.ascension)
            if state.get("type") == "error":
                raise CliError(f"start_run failed: {state.get('message')}")
            for _ in range(self.max_decisions):
                self.log(kind="state", decision=state.get("decision"), state=state)
                if state.get("type") == "error":
                    raise CliError(state.get("message", "error"))
                decision = state.get("decision")
                result.auto_steps += self.advance_shadow(state)
                diffs = self.compare(state)
                if diffs:
                    raise Desync("; ".join(diffs))
                if decision == "game_over" or self.agent._past_last_act(self.shadow):
                    result.finished = True
                    break
                move = self.decide(state)
                result.decisions += 1
                for action, args in move.commands:
                    state = self._send(action, args)
                for action, target in move.steps:
                    if not self.shadow.step(action, target):
                        raise Desync(f"the simulator refused action {action} on {describe(self.shadow)}")
            else:
                raise CliError(f"no game over after {self.max_decisions} decisions")
        except Desync as exc:
            result.in_sync = False
            result.desync_floor = self.shadow.floor
            result.desync_reason = str(exc)
            self.log(kind="desync", floor=self.shadow.floor, reason=str(exc))
        except (CliError, sim.SimError) as exc:
            result.in_sync = False
            result.engine_error = f"{type(exc).__name__}: {exc}"
            self.log(kind="error", reason=result.engine_error)
        finally:
            run = self.shadow
            result.final_floor = run.floor
            capped = self.agent._past_last_act(run)
            result.won = result.finished and (capped or (run.done and run.won() and run.hp > 0))
            result.score = 1.0 if result.won else sim.run_score(run, self.agent.cfg.max_act)
            result.acts_cleared = self.agent.cfg.max_act if result.won else F.ACT_ORDINAL.get(run.act_id, 1) - 1
            run.close()
            result.seconds = time.perf_counter() - t0
            self.log(kind="result", **result.to_dict())
            if self._trace is not None:
                self._trace.close()
        return result

    # -- non-decisions -----------------------------------------------------------------

    def advance_shadow(self, state: dict) -> int:
        """Step the shadow through screens the engine resolved by itself."""
        decision = state.get("decision")
        for n in range(AUTO_LIMIT):
            if shadow_matches(decision, self.shadow):
                return n
            action = self._auto_action(decision)
            self.log(kind="auto", phase=self.shadow.phase, action=action)
            if not self.shadow.step(action):
                raise Desync(f"the simulator refused auto step {action} on {describe(self.shadow)}")
        raise Desync(f"engine shows {decision}; simulator still on {describe(self.shadow)} after {AUTO_LIMIT} steps")

    def _auto_action(self, decision: str) -> int:
        run = self.shadow
        legal = [int(a) for a in run.legal()]
        if run.phase == sim.PHASE_RELIC_REWARD:
            # sts2-cli takes gold, then relics, then potions by itself (a potion with the
            # belt full is lost, in both), and shows a card reward as a screen of its own.
            # It shows only the first card reward of a fight, so any further one is left
            # on the shadow's screen too.
            gold, potion, relic, cards = run.state_list(6, 4)
            items = ["gold"] * bool(gold) + ["potion"] * bool(potion) + ["relic"] * bool(relic) + ["card"] * int(cards)
            wanted = ["gold", "relic", "potion"] + (["card"] if decision == "card_reward" else [])
            for item in wanted:
                if item in items and items.index(item) in legal:
                    return items.index(item)
            return REWARD_SKIP
        if run.phase == PHASE_TREASURE and REWARD_SKIP in legal:
            return REWARD_SKIP
        if len(legal) == 1 and run.phase != sim.PHASE_COMBAT:
            return legal[0]  # a proceed page: nothing to decide
        raise Desync(f"engine shows {decision}; simulator is on {describe(run)} with {len(legal)} options")

    # -- the sync check ----------------------------------------------------------------

    def compare(self, state: dict) -> list[str]:
        run = self.shadow
        decision = state.get("decision")
        player = state.get("player") or {}
        ctx = state.get("context") or {}
        diffs: list[str] = []
        in_combat = run.phase == sim.PHASE_COMBAT

        def check(name, game, mine):
            if game is not None and game != mine:
                diffs.append(f"{name}: game={game} sim={mine}")

        if decision == "game_over":
            check("victory", bool(state.get("victory")), bool(run.won() and run.hp > 0))
            return diffs
        # In combat the run-level hp, max hp and gold are the fight's own (WriteObservation).
        check("hp", player.get("hp"), run.hp)
        check("max hp", player.get("max_hp"), run.max_hp)
        check("gold", player.get("gold"), run.gold)
        act = ctx.get("act")
        if act is not None:
            check("act", int(act), F.ACT_ORDINAL.get(run.act_id, 1))
            if ctx.get("floor") is not None:
                offset = self.floor_offset.setdefault(int(act), run.floor - int(ctx["floor"]))
                check("floor", int(ctx["floor"]) + offset, run.floor)
        if not in_combat and "deck" in player:
            game_deck = Counter((card_id(c.get("id", "")), bool(c.get("upgraded"))) for c in player["deck"])
            sim_deck = Counter((int(c[0]), bool(c[1])) for c in run.deck())
            if game_deck != sim_deck:
                extra = game_deck - sim_deck
                missing = sim_deck - game_deck
                diffs.append(
                    "deck: game has " + ", ".join(f"{gd.card_name(c)}{'+' if u else ''}" for c, u in extra.elements())
                    + " / sim has " + ", ".join(f"{gd.card_name(c)}{'+' if u else ''}" for c, u in missing.elements())
                )
        if "relics" in player:
            game_relics = Counter(r.get("name") for r in player["relics"] or [])
            sim_relics = Counter(self.names.relic(int(r[0])) for r in run.relics())
            if game_relics != sim_relics:
                diffs.append(f"relics: game has {sorted((game_relics - sim_relics).elements())} / sim has {sorted((sim_relics - game_relics).elements())}")
        if "potions" in player:
            game_pots = {int(p["index"]): p.get("name") for p in player["potions"] or [] if p}
            pots = run.combat_potions() if in_combat else list(run.potions())
            sim_pots = {i: self.names.potion(p) for i, p in enumerate(pots) if p}
            check("potions", game_pots, sim_pots)
        if decision == "combat_play":
            game_hand = Counter(card_id(c.get("id", "")) for c in state.get("hand") or [])
            sim_hand = Counter(c for c, _ in run.hand())
            if game_hand != sim_hand:
                diffs.append(f"hand: game={sorted(gd.card_name(c) for c in game_hand.elements())} sim={sorted(gd.card_name(c) for c in sim_hand.elements())}")
            check("energy", state.get("energy"), run.energy)
            game_hp = sorted(int(e.get("hp") or 0) for e in state.get("enemies") or [])
            sim_hp = sorted(e[0] for e in run.enemies() if e[0] > 0)
            check("enemy hp", game_hp, sim_hp)
        elif decision == "map_select":
            game_nodes = {(int(c["col"]), int(c["row"])) for c in state.get("choices") or []}
            sim_nodes = {c for c in run.map_choice_coords() if c is not None}
            # Before an act's first move the engine lists the start node AND its children;
            # the simulator offers only the start node (the act's ancient).
            if sim_nodes != game_nodes and not (len(sim_nodes) == 1 and sim_nodes <= game_nodes):
                diffs.append(f"map choices: game={sorted(game_nodes)} sim={sorted(sim_nodes)}")
        elif decision == "card_reward":
            # Lasting Candy's fourth card lives elsewhere in the observation; the three
            # ordinary ones are compared.
            game_cards = [card_id(c.get("id", "")) for c in state.get("cards") or []][:3]
            sim_cards = [int(run.obs[sim.C + 9 + i]) for i in range(3) if int(run.obs[sim.C + 9 + i])]
            check("card reward", [gd.card_name(c) for c in game_cards], [gd.card_name(c) for c in sim_cards])
        elif decision == "shop":
            game_stock = Counter(
                [("card", c.get("name")) for c in state.get("cards") or [] if c.get("is_stocked")]
                + [("relic", r.get("name")) for r in state.get("relics") or [] if r.get("is_stocked")]
                + [("potion", p.get("name")) for p in state.get("potions") or [] if p.get("is_stocked")]
            )
            sim_stock = Counter(
                [("card", self.names.card(c)) for c in run.state_list(8, 7) if c]
                + [("relic", self.names.relic(r)) for r in run.state_list(9, 3) if r]
                + [("potion", self.names.potion(p)) for p in run.state_list(10, 3) if p]
            )
            if game_stock != sim_stock:
                diffs.append(f"shop stock: game has {sorted((game_stock - sim_stock).elements())} / sim has {sorted((sim_stock - game_stock).elements())}")
        elif decision == "event_choice" and run.phase == sim.PHASE_ANCIENT:
            offered = [entry_of(o.get("text_key") or "") for o in state.get("options") or []]
            sim_ids = [r for r in run.state_list(3, 3) if r]
            sim_offered = [relic_entry(r) for r in sim_ids]
            # A few Neow-only ids have no row in the relic table; those offers go unchecked.
            if sim_ids and all(r in gd.relics() for r in sim_ids) and offered != sim_offered:
                diffs.append(f"ancient offers: game={offered} sim={sim_offered}")
        return diffs

    # -- decisions ---------------------------------------------------------------------

    def decide(self, state: dict) -> Move:
        decision = state["decision"]
        run = self.shadow
        if decision == "card_select":
            return self._card_select(state)
        action, target = self.agent.act(run)
        self.log(kind="decision", decision=decision, action=action, target=target)
        if decision == "combat_play":
            return self._combat(action, target, state)
        if decision == "map_select":
            coords = run.map_choice_coords()
            want = coords[action] if action < len(coords) else None
            if want is None or want not in {(int(c["col"]), int(c["row"])) for c in state.get("choices") or []}:
                raise Desync(f"map choice {action} at {want} is not offered by the engine")
            return Move([("select_map_node", {"col": want[0], "row": want[1]})], [(action, -1)])
        if decision == "card_reward":
            if action == REWARD_SKIP:
                return Move([("skip_card_reward", {})], [(action, -1)])
            cards = state.get("cards") or []
            mine = int(run.obs[sim.C + 9 + action]) if action < 3 else 0
            if action >= len(cards) or card_id(cards[action].get("id", "")) != mine:
                raise Desync(f"card reward {action} ({gd.card_name(mine)}) is not on the engine's screen")
            return Move([("select_card_reward", {"card_index": action})], [(action, -1)])
        if decision == "rest_site":
            if action == REWARD_SKIP:
                return Move([("leave_room", {})], [(action, -1)])
            for opt in state.get("options") or []:
                if str(opt.get("option_id", "")).upper() in REST_IDS.get(action, ()) and opt.get("is_enabled", True):
                    return Move([("choose_option", {"option_index": int(opt["index"])})], [(action, -1)])
            raise Desync(f"rest option {action} is not offered by the engine")
        if decision == "shop":
            return self._shop(action, state)
        if decision == "event_choice":
            options = state.get("options") or []
            if action >= len(options) or options[action].get("is_locked"):
                raise Desync(f"event option {action} is not available in the engine ({len(options)} options)")
            return Move([("choose_option", {"option_index": action})], [(action, -1)])
        if decision == "bundle_select":
            if action == BUNDLE_CONFIRM or action >= len(state.get("bundles") or []):
                raise Desync(f"bundle action {action} has no engine equivalent")
            return Move([("select_bundle", {"bundle_index": action})], [(action, -1), (BUNDLE_CONFIRM, -1)])
        raise Desync(f"no mapping for the engine's {decision!r} screen")

    def _combat(self, action: int, target: int, state: dict) -> Move:
        run = self.shadow
        hand = run.hand()
        living = run.living_enemy_indices()
        ordinal = living.index(target) if target in living else None
        if action < len(hand):
            game_hand = state.get("hand") or []
            if action >= len(game_hand) or card_id(game_hand[action].get("id", "")) != hand[action][0]:
                raise Desync(f"hand slot {action} differs between engine and simulator")
            args = {"card_index": action}
            if game_hand[action].get("target_type") == "AnyEnemy" and ordinal is not None:
                args["target_index"] = ordinal
            return Move([("play_card", args)], [(action, target)])
        if action == len(hand):
            return Move([("end_turn", {})], [(action, target)])
        slot = action - len(hand) - 1
        args = {"potion_index": slot}
        if ordinal is not None:
            args["target_index"] = ordinal
        return Move([("use_potion", args)], [(action, target)])

    def _shop(self, action: int, state: dict) -> Move:
        run = self.shadow
        if action == SHOP_LEAVE:
            return Move([("leave_room", {})], [(action, -1)])
        if action == SHOP_REMOVE:
            return Move([("remove_card", {})], [(action, -1)])
        if action < 7:
            kind, items, title, verb, key = "card", state.get("cards"), self.names.card(run.state_list(8, 7)[action]), "buy_card", "card_index"
        elif action < 10:
            kind, items, title, verb, key = "relic", state.get("relics"), self.names.relic(run.state_list(9, 3)[action - 7]), "buy_relic", "relic_index"
        else:
            kind, items, title, verb, key = "potion", state.get("potions"), self.names.potion(run.state_list(10, 3)[action - 10]), "buy_potion", "potion_index"
        for item in items or []:
            if item.get("is_stocked") and item.get("name") == title:
                return Move([(verb, {key: int(item["index"])})], [(action, -1)])
        raise Desync(f"shop {kind} {title!r} (slot {action}) is not stocked by the engine")

    def _card_select(self, state: dict) -> Move:
        """Choose cards on an open selection -- possibly several, one simulator step each
        -- and send them to the engine in one `select_cards`. The shadow is stepped here.

        Three simulator screens answer the engine's one: an in-combat selection (actions
        index its candidate list; one past the end declines), a grid of offered cards
        (likewise), and a selection over the deck (actions are deck indices). A pick is
        matched to the engine's list by the card itself, never by position alone.
        """
        run = self.shadow
        game_cards = [(card_id(c.get("id", "")), bool(c.get("upgraded"))) for c in state.get("cards") or []]
        max_select = max(1, int(state.get("max_select") or 1))
        in_combat = run.phase == sim.PHASE_COMBAT
        offered = selection_options(run)
        listed = [card for _, card in sorted(offered.items())]
        if len(listed) != len(game_cards) or not all(same_card(a, b) for a, b in zip(listed, game_cards)):
            raise Desync(
                f"selection: game offers {[gd.card_name(c) for c, _ in game_cards]} "
                f"sim offers {[gd.card_name(c) if c else '?' for c, _ in listed]}"
            )
        # A pick is matched by POSITION -- which copy of a duplicated card is taken
        # matters (pile order) -- and the card itself is checked at every step.
        slots = rekey(list(range(len(game_cards))), offered)

        # The picks are made on the shadow itself, one step at a time, and the engine gets
        # them all at once. (Not on a copy: a copy of an open multi-card selection loses
        # whether the player may stop early.)
        picks: list[int] = []  # positions in the engine's list
        steps: list[tuple[int, int]] = []
        signature = (run.phase, tuple(run.state_list(20, 3)))

        def still_open() -> bool:
            if in_combat:
                return run.phase == sim.PHASE_COMBAT and run.pending_selection() > 0
            return (run.phase, tuple(run.state_list(20, 3))) == signature

        while len(steps) < MAX_PICKS:
            before = selection_options(run)
            action, _ = self.agent.act(run)
            card = before.get(action)
            if card is not None:
                position = dict(slots)[action]
                if not same_card(game_cards[position], card):
                    raise Desync(f"selection slot {position}: game has {gd.card_name(game_cards[position][0])}, sim picks {gd.card_name(card[0])}")
                picks.append(position)
            steps.append((action, -1))
            if not run.step(action):
                raise Desync(f"the simulator refused selection {action}")
            if card is None or not still_open():
                break  # declined, finished early, or the screen closed itself
            after = selection_options(run)
            slots = rekey([pos for _, pos in slots if pos not in picks], after)
            if slots is None or not all(same_card(game_cards[pos], after[a]) for a, pos in slots):
                left = [gd.card_name(game_cards[pos][0]) for pos in range(len(game_cards)) if pos not in picks]
                raise Desync(
                    f"selection: after taking {[gd.card_name(game_cards[p][0]) for p in picks]} the engine has "
                    f"{left} left, the simulator {[gd.card_name(c) if c else '?' for _, (c, _) in sorted(after.items())]}"
                )
            if len(picks) >= max_select:
                # The engine resolves on its last allowed pick; the shadow's screen
                # closes the same way, through its done/decline action.
                done = [int(a) for a in run.legal() if int(a) not in after]
                if not done:
                    raise Desync(f"the simulator wants more than the engine's {max_select} picks")
                steps.append((done[0], -1))
                run.step(done[0])
                break
        self.log(kind="decision", decision="card_select", picks=picks, steps=steps)
        if not picks:
            return Move([("skip_select", {})], [])
        return Move([("select_cards", {"indices": ",".join(str(p) for p in picks)})], [])


def play_run(game: CliGame, agent: Agent, seed: str, names: Names, trace_path: Path | None = None) -> CliRunResult:
    return CliSession(game, agent, names, seed, trace_path=trace_path).play()


def smoke_test(cli_dir: Path | None, seed: str = "SMOKE00001", command: list[str] | None = None) -> list[str]:
    """Start one run on the engine and check its opening screen against the simulator.

    Returns the problems found (empty means the engine runs and agrees with the shadow).
    """
    names = Names(Path(cli_dir) / "localization_eng" if cli_dir else None)
    seed = canonical_seed(seed)
    with CliGame(command or cli_command(cli_dir), cwd=cli_dir, timeout=180) as game:
        state = game.start_run(seed, ascension=8)
    if state.get("type") == "error":
        return [f"start_run failed: {state.get('message')}"]
    with sim.Run.new(seed) as run:
        session = CliSession(game, Agent(None), names, seed)
        session.shadow = run
        if not shadow_matches(state.get("decision"), run):
            return [f"engine opens on {state.get('decision')}, simulator on {describe(run)}"]
        return session.compare(state)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Check that sts2-cli runs and opens a run the way the simulator does.")
    ap.add_argument("--smoke", required=True, metavar="CLI_DIR")
    problems = smoke_test(Path(ap.parse_args().smoke))
    for p in problems:
        print("  mismatch:", p)
    print("engine OK: the opening screen matches the simulator" if not problems else "engine runs, but does not match")
    sys.exit(1 if problems else 0)
