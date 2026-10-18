"""Watch the agent play one simulated run, decision by decision.

    python -m sts2rl.watch --model runs/main/best.pt --seed MYSEED
    python -m sts2rl.watch --seed MYSEED --sims 200 --quiet-combat

Non-combat choices are printed with the value the network gave each option, so you can
see why it took the card it took.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import combat, sim
from . import gamedata as gd
from .agent import Agent, AgentConfig
from .model import ValueNet

PHASE_NAMES = {0: "combat", 1: "card reward", 2: "map", 3: "rest site", 4: "shop", 5: "rewards", 7: "event",
               8: "ancient", 9: "card select", 10: "treasure", 11: "crystal sphere", 12: "bundle select"}
NODE_NAMES = {0: "?", 1: "monster", 2: "ELITE", 3: "rest", 4: "shop", 5: "treasure", 6: "BOSS", 7: "event", 8: "ancient"}
REST_NAMES = {0: "rest (heal)", 1: "smith (upgrade)", 2: "clone", 3: "leave", 4: "lift", 5: "dig", 6: "hatch", 7: "cook", 8: "kindle"}


def describe_option(run: sim.Run, action: int) -> str:
    ph = run.phase
    if ph == sim.PHASE_CARD_REWARD:
        if action < 3:
            return "take " + gd.card_name(int(run.obs[sim.C + 9 + action]))
        return {3: "skip", 4: "sacrifice", 5: "take 4th card", 6: "reroll"}.get(action, str(action))
    if ph == sim.PHASE_MAP:
        t = int(run.obs[sim.C + sim.LAYOUT["map_node_type_offset"] + action])
        return f"go to {NODE_NAMES.get(t, t)}"
    if ph == sim.PHASE_REST:
        return REST_NAMES.get(action, str(action))
    if ph == sim.PHASE_SHOP:
        if action == 13:
            return "remove a card"
        if action == 14:
            return "leave"
        item = int(run.obs[sim.C + sim.LAYOUT["shop_offset"] + action * sim.LAYOUT["shop_slot_size"]])
        name = gd.card_name(item) if action < 7 else gd.relic_name(item) if action < 10 else gd.potion_name(item)
        return f"buy {name}"
    if ph == 9:
        deck = run.deck()
        return gd.card_name(int(deck[action, 0])) if action < len(deck) else f"#{action}"
    return f"option {action}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--seed", default="WATCH00001")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--quiet-combat", action="store_true")
    args = ap.parse_args()

    model = ValueNet.load(args.model) if args.model else None
    agent = Agent(model, AgentConfig(combat_sims=args.sims, record=False), seed=0)
    run = sim.Run.new(args.seed)
    last_floor = -1
    try:
        while not run.done and not agent._past_last_act(run):
            if run.floor != last_floor:
                last_floor = run.floor
                print(f"\n=== floor {run.floor}  HP {run.hp}/{run.max_hp}  gold {run.gold}  deck {len(run.deck())} cards")
            ph = run.phase
            if ph == sim.PHASE_COMBAT:
                hand = [gd.card_name(c) + ("+" if u else "") for c, u in run.hand()]
                action, target = agent.act(run)
                key = next((k for k, a, t in combat.legal_moves(run) if a == action and t == target), None)
                if not args.quiet_combat:
                    enemies = [f"{e[0]}hp" for e in run.enemies() if e[0] > 0]
                    print(f"  [{run.combat_hp}hp {run.energy}e vs {', '.join(enemies)}] hand {hand} -> {combat.describe(key) if key else action}")
                run.step(action, target)
                continue
            legal = [int(a) for a in run.legal()]
            if len(legal) > 1 and ph in PHASE_NAMES:
                clones = []
                for a in legal:
                    c = run.clone(resample=True, seed=a)
                    clones.append(c if c.step(a) else None)
                ok = [c for c in clones if c is not None]
                vals = agent.values(ok) if ok else np.array([])
                it = iter(vals)
                scored = [(a, next(it) if c is not None else float("nan")) for a, c in zip(legal, clones)]
                for c in ok:
                    c.close()
            action, target = agent.act(run)
            if len(legal) > 1 and ph in PHASE_NAMES:
                opts = "  ".join(f"{describe_option(run, a)}={v:.3f}" + (" <" if a == action else "") for a, v in scored)
                print(f"  {PHASE_NAMES[ph]}: {opts}")
            if not run.step(action, target):
                run.step(int(run.legal()[-1]))
        won = run.done and run.won() and run.hp > 0 or agent._past_last_act(run)
        print(f"\nRun over on floor {run.floor}: {'WON' if won else 'died'}  score {sim.run_score(run):.3f}")
        print("final deck:", ", ".join(sorted(gd.card_name(int(c)) for c in run.deck()[:, 0])))
        print("relics:", ", ".join(gd.relic_name(int(r)) for r in run.relics()[:, 0]))
    finally:
        run.close()


if __name__ == "__main__":
    main()
