"""Static game data, read from the simulator's generated tables.

The simulator's `Generated/*.g.cs` files are produced from the decompiled game, so this
module re-reads them rather than keeping its own copy: a patch that renumbers or retunes a
card changes what the agent sees the moment the submodule is updated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from . import paths

CARD_TYPES = ("Attack", "Skill", "Power", "Status", "Curse", "Quest")
RARITIES = ("Basic", "Common", "Uncommon", "Rare", "Ancient", "Special", "Token", "Status", "Curse", "Event", "Quest", "None")
TARGETS = ("Self", "AnyEnemy", "AllEnemies", "RandomEnemy", "None", "AnyAlly", "AllAllies")
CARD_FLAGS = (
    "Exhaust",
    "Ethereal",
    "Unplayable",
    "Innate",
    "Retain",
    "HasEnergyCostX",
    "Sly",
    "Eternal",
    "StrikeTag",
    "DefendTag",
)


@dataclass(frozen=True)
class Card:
    id: int
    name: str
    entry: str
    cost: int
    damage: int
    block: int
    upgrade_damage: int
    upgrade_block: int
    upgrade_cost: int
    type: str
    rarity: str
    target: str
    star_cost: int
    flags: frozenset[str]

    @property
    def needs_enemy_target(self) -> bool:
        return self.target == "AnyEnemy"


@dataclass(frozen=True)
class Relic:
    id: int
    name: str
    entry: str
    rarity: str


@dataclass(frozen=True)
class Potion:
    id: int
    name: str
    rarity: str


_FIELD_RE = re.compile(r"(\w+): (\"[^\"]*\"|-?\d+|\w+\.\w+|true|false)")


def _parse_defs(path, ctor: str) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith(f"new {ctor}("):
            continue
        fields = {}
        for key, value in _FIELD_RE.findall(line):
            fields[key] = value.strip('"').split(".")[-1]
        rows.append(fields)
    return rows


@cache
def cards() -> dict[int, Card]:
    out = {}
    for f in _parse_defs(paths.GENERATED / "Cards.g.cs", "CardDef"):
        out[int(f["Id"])] = Card(
            id=int(f["Id"]),
            name=f["Name"],
            entry=f.get("Entry", ""),
            cost=int(f.get("Cost", 0)),
            damage=int(f.get("BaseDamage", 0)),
            block=int(f.get("BaseBlock", 0)),
            upgrade_damage=int(f.get("UpgradeDamage", 0)),
            upgrade_block=int(f.get("UpgradeBlock", 0)),
            upgrade_cost=int(f.get("UpgradeCost", 0)),
            type=f.get("Type", "Skill"),
            rarity=f.get("Rarity", "None"),
            target=f.get("Target", "None"),
            star_cost=int(f.get("StarCost", 0)),
            flags=frozenset(k for k in CARD_FLAGS if f.get(k) == "true"),
        )
    return out


@cache
def relics() -> dict[int, Relic]:
    return {
        int(f["Id"]): Relic(int(f["Id"]), f["Name"], f.get("Entry", ""), f.get("Rarity", "None"))
        for f in _parse_defs(paths.GENERATED / "Relics.g.cs", "RelicDef")
    }


@cache
def potions() -> dict[int, Potion]:
    return {
        int(f["Id"]): Potion(int(f["Id"]), f["Name"], f.get("Rarity", "None"))
        for f in _parse_defs(paths.GENERATED / "Potions.g.cs", "PotionDef")
    }


@cache
def card_pool(name: str) -> tuple[int, ...]:
    text = (paths.GENERATED / "CardPools.g.cs").read_text(encoding="utf-8")
    match = re.search(rf"ReadOnlySpan<int>\s+{name}\s*=>\s*\[([^\]]*)\]", text)
    if match is None:
        raise KeyError(f"no card pool named {name}")
    return tuple(int(x) for x in match.group(1).split(",") if x.strip())


@cache
def encounter_names() -> dict[int, str]:
    from sts2_gym.env import ENCOUNTER_NAMES

    return dict(ENCOUNTER_NAMES)


class Vocab:
    """Dense, stable indices for sparse game ids. Index 0 is reserved for 'none'."""

    def __init__(self, ids):
        self.ids = sorted(set(ids))
        self.index = {v: i + 1 for i, v in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids) + 1

    def __call__(self, game_id: int) -> int:
        return self.index.get(game_id, 0)


@cache
def card_vocab() -> Vocab:
    return Vocab(cards())


@cache
def relic_vocab() -> Vocab:
    return Vocab(relics())


@cache
def potion_vocab() -> Vocab:
    return Vocab(potions())


@cache
def encounter_vocab() -> Vocab:
    return Vocab(encounter_names())


def card_name(card_id: int) -> str:
    card = cards().get(card_id)
    return card.name if card else f"card#{card_id}"


def relic_name(relic_id: int) -> str:
    relic = relics().get(relic_id)
    return relic.name if relic else f"relic#{relic_id}"


def potion_name(potion_id: int) -> str:
    potion = potions().get(potion_id)
    return potion.name if potion else f"potion#{potion_id}"
