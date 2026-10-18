"""Combat benchmark: the same fights, played by different combat policies.

    python -m sts2rl.bench --sims 0,32,100 --fights 24

Fights are built directly with `Sts2Run_StartCombat` (a chosen deck, relics, HP and
encounter), so a policy change can be measured on hard fights without playing whole runs.
Every policy sees the same seeds, so the comparison is paired.
"""

from __future__ import annotations

import argparse
import statistics
import time

from . import combat, sim
from . import gamedata as gd
from sts2_gym import native

STARTER = [472] * 5 + [131] * 4 + [30]
BURNING_BLOOD = 36

ACT1_ELITES = {"bygone-effigy": 62, "byrdonis": 68, "phrog-parasite": 65, "phantasmal-gardeners": 72, "skulking-colony": 86, "terror-eel": 67}
ACT1_BOSSES = {"ceremonial-beast": 74, "kin": 82, "vantom": 83, "lagavulin-matriarch": 77, "soul-fysh": 79, "waterfall-giant": 84}


def _card_id(name: str) -> int:
    for cid, card in gd.cards().items():
        if card.name == name:
            return cid
    raise KeyError(name)


def default_deck() -> list[int]:
    """A modest act-1 deck: the starter plus a handful of common Ironclad picks."""
    extra = ["PommelStrike", "ShrugItOff", "TwinStrike", "Inflame", "IronWave"]
    return STARTER + [_card_id(n) for n in extra]


def start_fight(seed: str, encounter: int, deck: list[int], hp: int = 70, max_hp: int = 80, potions=(), relics=(BURNING_BLOOD,)) -> sim.Run:
    run = sim.Run.new(seed)
    status = native.run_start_combat(run.handle, list(deck), encounter, list(relics), hp, max_hp, list(potions), 99, 3, run._obs_buf)
    if status != 0:
        run.close()
        raise RuntimeError(f"start combat failed ({status}) for encounter {encounter}")
    return run


def play(run: sim.Run, sims: int, seed: int, c_explore: float = 0.7) -> None:
    if sims <= 0:
        combat.native_rollout(run, seed)
        return
    search = combat.CombatSearch(combat.hp_leaf_value, combat.SearchConfig(simulations=sims, c_explore=c_explore), seed=seed)
    for _ in range(combat.MAX_COMBAT_STEPS):
        if run.phase != sim.PHASE_COMBAT:
            break
        res = search.search(run)
        run.step(res.action, res.target)
        search.advance(res.key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default="0,32,100")
    ap.add_argument("--fights", type=int, default=12, help="seeds per encounter")
    ap.add_argument("--encounters", default="elites", choices=["elites", "bosses", "all"])
    ap.add_argument("--c", type=float, default=0.7)
    args = ap.parse_args()
    pool = {"elites": ACT1_ELITES, "bosses": ACT1_BOSSES, "all": {**ACT1_ELITES, **ACT1_BOSSES}}[args.encounters]
    deck = default_deck()
    for sims in [int(x) for x in args.sims.split(",")]:
        hps, deaths, t0 = [], 0, time.perf_counter()
        per_enc = {}
        for name, enc in pool.items():
            enc_hps = []
            for i in range(args.fights):
                run = start_fight(f"BENCH{i}", enc, deck)
                play(run, sims, seed=1000 + i, c_explore=args.c)
                dead = run.done and not run.won()
                hp = 0 if dead else run.hp
                deaths += dead
                hps.append(hp)
                enc_hps.append(hp)
                run.close()
            per_enc[name] = statistics.mean(enc_hps)
        dt = time.perf_counter() - t0
        n = len(hps)
        print(f"sims={sims:4d}  mean HP {statistics.mean(hps):5.1f}  deaths {deaths}/{n}  {dt / n:.2f}s/fight  " + "  ".join(f"{k}:{v:.0f}" for k, v in per_enc.items()))


if __name__ == "__main__":
    main()
