"""Random playable deck generation.

The generator is intentionally conservative: it does not try to build strong
lists, but it keeps every included Pokemon usable by adding its evolution chain
and matching basic energy for all printed typed attack costs.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
import os
import random
from typing import Iterable, Literal, Sequence

from cg.api import (
    CardType,
    EnergyType,
    _native_all_attack,
    _native_all_card_data,
)


BASIC_ENERGY_BY_TYPE = {
    int(EnergyType.GRASS): 1,
    int(EnergyType.FIRE): 2,
    int(EnergyType.WATER): 3,
    int(EnergyType.LIGHTNING): 4,
    int(EnergyType.PSYCHIC): 5,
    int(EnergyType.FIGHTING): 6,
    int(EnergyType.DARKNESS): 7,
    int(EnergyType.METAL): 8,
}

FOSSIL_ITEM_IDS = {1099, 1136, 1138, 1150, 1151}
TEAM_ROCKET_ENERGY_ID = 15
NEO_UPPER_ENERGY_ID = 10
SAMPLING_METHODS = ("coverage", "competitive")
SamplingMethod = Literal["coverage", "competitive"]
TRAINER_ROLES = (
    "pokemon_search",
    "draw",
    "energy_search",
    "energy_accel",
    "switch",
    "gust",
    "evolution",
    "stadium",
    "tool",
)

TYPE_TOKEN_TO_ENERGY = {
    "{G}": int(EnergyType.GRASS),
    "{R}": int(EnergyType.FIRE),
    "{W}": int(EnergyType.WATER),
    "{L}": int(EnergyType.LIGHTNING),
    "{P}": int(EnergyType.PSYCHIC),
    "{F}": int(EnergyType.FIGHTING),
    "{D}": int(EnergyType.DARKNESS),
    "{M}": int(EnergyType.METAL),
}

OWNER_TAGS = {
    "Team Rocket": "owner:team_rocket",
    "Iono": "owner:iono",
    "Lillie": "owner:lillie",
    "Cynthia": "owner:cynthia",
    "Marnie": "owner:marnie",
    "Ethan": "owner:ethan",
    "Steven": "owner:steven",
    "N's": "owner:n",
    "N’s": "owner:n",
    "Hop": "owner:hop",
}


@dataclass(frozen=True)
class RandomDeckConfig:
    """Balance knobs for random deck generation."""

    min_pokemon: int = 16
    max_pokemon: int = 23
    min_energy: int = 11
    max_energy: int = 16
    max_energy_types: int = 2
    max_copies: int = 4
    max_build_attempts: int = 200
    competitive_opening_samples: int = 96
    competitive_min_basic_open_rate: float = 0.72
    competitive_min_energy_open_rate: float = 0.62
    competitive_min_setup_open_rate: float = 0.50
    competitive_min_draw: int = 6
    competitive_max_draw: int = 10
    competitive_min_pokemon_search: int = 6
    competitive_max_pokemon_search: int = 10
    competitive_min_switch_gust: int = 2
    competitive_max_switch_gust: int = 4
    competitive_max_stadium: int = 2
    competitive_max_tool: int = 3
    competitive_max_energy_search: int = 4
    competitive_max_energy_accel: int = 4


@dataclass(frozen=True)
class CoverageReport:
    """Coverage of eligible card ids across a generated deck corpus."""

    covered: int
    total: int
    missing: tuple[int, ...]

    @property
    def ratio(self) -> float:
        return self.covered / self.total if self.total else 1.0


@dataclass(frozen=True)
class DeckValidation:
    """Validation result for the structural guarantees this generator provides."""

    ok: bool
    errors: tuple[str, ...]


class RandomDeckGenerator:
    """Build random 60-card lists that satisfy structural playability checks."""

    def __init__(self, config: RandomDeckConfig | None = None):
        self.config = config or RandomDeckConfig()
        self.cards = {c.cardId: c for c in _native_all_card_data()}
        self.attacks = {a.attackId: a for a in _native_all_attack()}
        self.by_name: dict[str, list[int]] = defaultdict(list)
        for card in self.cards.values():
            self.by_name[card.name].append(card.cardId)
        self.card_text = self._load_card_text()

        self.pokemon_ids = [
            cid
            for cid, card in self.cards.items()
            if int(card.cardType) == int(CardType.POKEMON)
        ]
        self.basic_pokemon_ids = [
            cid for cid in self.pokemon_ids if self.cards[cid].basic
        ]
        self.trainer_ids = [
            cid
            for cid, card in self.cards.items()
            if int(card.cardType)
            in {
                int(CardType.ITEM),
                int(CardType.TOOL),
                int(CardType.SUPPORTER),
                int(CardType.STADIUM),
            }
        ]
        self.special_energy_ids = [
            cid
            for cid, card in self.cards.items()
            if int(card.cardType) == int(CardType.SPECIAL_ENERGY)
        ]

        self._chain_cache: dict[int, bool] = {}
        self.eligible_ids = frozenset(
            cid
            for cid in self.cards
            if self._card_is_eligible(cid)
        )
        self.card_tags = {
            cid: frozenset(self._infer_tags(cid))
            for cid in self.cards
        }
        self.card_roles = {
            cid: frozenset(self._infer_roles(cid))
            for cid in self.cards
        }

    def generate_deck(
        self,
        seed: int | None = None,
        forced_cards: Iterable[int] = (),
        prefer_uncovered: Iterable[int] = (),
        sampling_method: SamplingMethod = "coverage",
        main_card_id: int | None = None,
    ) -> list[int]:
        """Generate one 60-card deck.

        Args:
            seed: Optional deterministic seed.
            forced_cards: Card ids that must be present if eligible.
            prefer_uncovered: Card ids to prioritize as singletons/families.
            sampling_method: ``coverage`` for broad card exposure, or
                ``competitive`` for a tighter archetype-style list.
        """

        if sampling_method == "competitive":
            return self.generate_competitive_deck(seed=seed, main_card_id=main_card_id)
        if main_card_id is not None:
            forced_cards = tuple(forced_cards) + (main_card_id,)
        if sampling_method not in SAMPLING_METHODS:
            raise ValueError(f"unknown sampling_method {sampling_method!r}")

        rng = random.Random(seed)
        forced = [cid for cid in dict.fromkeys(forced_cards) if cid in self.eligible_ids]
        preferred = [cid for cid in dict.fromkeys(prefer_uncovered) if cid in self.eligible_ids]

        last_errors: tuple[str, ...] = ()
        for _ in range(self.config.max_build_attempts):
            deck = Counter()
            families: set[str] = set()

            self._add_required_support_for_special_energy(deck, forced, rng, families)
            for cid in forced:
                self._add_forced_card(deck, cid, rng, families)

            if self._basic_pokemon_count(deck) == 0:
                self._add_pokemon_package(deck, self._choose_basic_anchor(preferred, rng), rng, True, families)

            pokemon_target = rng.randint(self.config.min_pokemon, self.config.max_pokemon)
            self._fill_pokemon(deck, pokemon_target, preferred, rng, families)
            self._add_energy(deck, preferred, rng)
            self._fill_trainers(deck, preferred, rng)
            self._finish_to_60(deck, preferred, rng, set(forced))

            result = self.validate(deck.elements())
            if result.ok and all(deck[cid] > 0 for cid in forced):
                out = list(deck.elements())
                rng.shuffle(out)
                return out
            last_errors = result.errors

        detail = "; ".join(last_errors) if last_errors else "no valid build found"
        raise RuntimeError(f"could not generate a random deck: {detail}")

    def generate_decks(
        self,
        count: int,
        seed: int = 0,
        cover_all: bool = True,
        sampling_method: SamplingMethod = "coverage",
        main_card_id: int | None = None,
    ) -> list[list[int]]:
        """Generate a corpus of random decks, optionally biased toward coverage."""

        if sampling_method == "competitive":
            return [
                self.generate_competitive_deck(
                    seed=seed + i * 104_729,
                    main_card_id=main_card_id,
                )
                for i in range(count)
            ]
        if main_card_id is not None:
            cover_all = False
        if sampling_method not in SAMPLING_METHODS:
            raise ValueError(f"unknown sampling_method {sampling_method!r}")

        decks: list[list[int]] = []
        covered: set[int] = set()
        targets = set(self.eligible_ids)
        for i in range(count):
            preferred = sorted(targets - covered) if cover_all else []
            forced = self._forced_for_next_deck(preferred, seed + i)
            deck = self.generate_deck(
                seed=seed + i * 104_729,
                forced_cards=tuple(forced) + ((main_card_id,) if main_card_id is not None else ()),
                prefer_uncovered=preferred,
            )
            decks.append(deck)
            covered.update(deck)
        return decks

    def generate_competitive_deck(
        self,
        seed: int | None = None,
        main_card_id: int | None = None,
    ) -> list[int]:
        """Generate a tighter random deck using archetype and synergy heuristics."""

        rng = random.Random(seed)
        if main_card_id is not None:
            self._validate_competitive_main(main_card_id)
        last_errors: tuple[str, ...] = ()
        for _ in range(self.config.max_build_attempts):
            deck = Counter()
            families: set[str] = set()
            main = main_card_id if main_card_id is not None else self._choose_main_attacker(rng)
            archetype_tags = set(self.card_tags[main])
            archetype_types = self._archetype_energy_types(main)
            for energy_type in archetype_types:
                archetype_tags.add(f"type:{energy_type}")

            if not self._add_competitive_pokemon_package(deck, main, rng, True, families):
                continue
            self._add_competitive_support_pokemon(deck, archetype_tags, archetype_types, rng, families)
            self._add_competitive_energy(deck, archetype_types, archetype_tags, rng)
            self._add_competitive_trainers(deck, archetype_tags, rng)
            self._finish_competitive_to_60(deck, archetype_tags, rng)

            result = self.validate(deck.elements())
            if (
                result.ok
                and self._passes_competitive_profile(deck, archetype_tags)
                and self._passes_opening_curve(deck, rng)
            ):
                out = list(deck.elements())
                rng.shuffle(out)
                return out
            last_errors = result.errors

        detail = "; ".join(last_errors) if last_errors else "opening curve rejected all builds"
        raise RuntimeError(f"could not generate a competitive random deck: {detail}")

    def generate_family_decks(
        self,
        main_card_id: int,
        count: int,
        seed: int = 0,
    ) -> list[list[int]]:
        """Generate competitive variants centered on one Pokemon family."""

        return self.generate_decks(
            count=count,
            seed=seed,
            cover_all=False,
            sampling_method="competitive",
            main_card_id=main_card_id,
        )

    def coverage(self, decks: Iterable[Iterable[int]]) -> CoverageReport:
        """Return card-id coverage against all structurally eligible cards."""

        seen = set(chain.from_iterable(decks)) & self.eligible_ids
        missing = tuple(sorted(self.eligible_ids - seen))
        return CoverageReport(len(seen), len(self.eligible_ids), missing)

    def validate(self, deck: Iterable[int]) -> DeckValidation:
        """Validate deck size, copy limits, evolution chains, and typed energy."""

        cards = list(deck)
        errors: list[str] = []
        counts = Counter(cards)
        if len(cards) != 60:
            errors.append(f"deck has {len(cards)} cards, expected 60")

        unknown = sorted(cid for cid in counts if cid not in self.cards)
        if unknown:
            errors.append(f"unknown card ids: {unknown[:8]}")

        ace_specs = 0
        named_copies: dict[str, int] = defaultdict(int)
        for cid, count in counts.items():
            card = self.cards.get(cid)
            if not card:
                continue
            if int(card.cardType) != int(CardType.BASIC_ENERGY) and count > self.config.max_copies:
                errors.append(f"{card.name} ({cid}) has {count} copies, max {self.config.max_copies}")
            if int(card.cardType) != int(CardType.BASIC_ENERGY):
                named_copies[card.name] += count
            if card.aceSpec:
                ace_specs += count
        for name, count in sorted(named_copies.items()):
            if count > self.config.max_copies:
                errors.append(f"{name} has {count} copies across printings, max {self.config.max_copies}")
        if ace_specs > 1:
            errors.append(f"deck has {ace_specs} ACE SPEC cards, max 1")

        if not any(
            self.cards[cid].basic and int(self.cards[cid].cardType) == int(CardType.POKEMON)
            for cid in counts
            if cid in self.cards
        ):
            errors.append("deck has no Basic Pokemon")

        names_in_deck = {self.cards[cid].name for cid in counts if cid in self.cards}
        for cid in sorted(counts):
            card = self.cards.get(cid)
            if not card or int(card.cardType) != int(CardType.POKEMON):
                continue
            if card.evolvesFrom and card.evolvesFrom not in names_in_deck:
                errors.append(f"{card.name} ({cid}) needs {card.evolvesFrom} in deck")
            for energy_type in sorted(self._required_attack_energy_types(cid)):
                if BASIC_ENERGY_BY_TYPE[energy_type] not in counts:
                    energy_name = EnergyType(energy_type).name
                    errors.append(f"{card.name} ({cid}) needs {energy_name} energy")

        return DeckValidation(not errors, tuple(errors))

    def describe_deck(self, deck: Iterable[int]) -> dict[str, int]:
        """Return simple deck category counts."""

        counts = Counter(deck)
        return {
            "cards": sum(counts.values()),
            "unique": len(counts),
            "pokemon": self._pokemon_count(counts),
            "basic_pokemon": self._basic_pokemon_count(counts),
            "energy": self._energy_count(counts),
            "trainers": sum(counts.values()) - self._pokemon_count(counts) - self._energy_count(counts),
        }

    def card_name(self, card_id: int) -> str:
        card = self.cards.get(card_id)
        return card.name if card else f"unknown:{card_id}"

    def _load_card_text(self) -> dict[int, str]:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "data", "EN_Card_Data.csv")
        text_by_id: dict[int, list[str]] = defaultdict(list)
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    raw_id = row.get("Card ID", "")
                    if not raw_id.isdigit():
                        continue
                    pieces = [
                        row.get("Card Name", ""),
                        row.get("Stage (Pokémon)/Type (Energy and Trainer)", ""),
                        row.get("Rule", ""),
                        row.get("Category", ""),
                        row.get("Previous stage", ""),
                        row.get("Type", ""),
                        row.get("Move Name", ""),
                        row.get("Cost", ""),
                        row.get("Effect Explanation", ""),
                    ]
                    text_by_id[int(raw_id)].append(" ".join(p for p in pieces if p and p != "n/a"))
        except OSError:
            return {}
        return {cid: " ".join(parts) for cid, parts in text_by_id.items()}

    def _fold_text(self, text: str) -> str:
        return (
            text.lower()
            .replace("pok\u00e9mon", "pokemon")
            .replace("pok\u00c3\u00a9mon", "pokemon")
            .replace("\u2019", "'")
            .replace("\u00e2\u20ac\u2122", "'")
        )

    def _infer_tags(self, cid: int) -> set[str]:
        card = self.cards[cid]
        text = self.card_text.get(cid, "")
        folded = self._fold_text(f"{card.name} {text}")
        tags = set()

        for needle, tag in OWNER_TAGS.items():
            if needle.lower() in folded:
                tags.add(tag)

        for token, energy_type in TYPE_TOKEN_TO_ENERGY.items():
            if token.lower() in folded:
                tags.add(f"type:{energy_type}")

        if int(card.cardType) == int(CardType.POKEMON):
            tags.add("pokemon")
            tags.add("basic" if card.basic else "evolution")
            if card.stage1:
                tags.add("stage1")
            if card.stage2:
                tags.add("stage2")
            if card.ex:
                tags.add("ex")
            if card.megaEx:
                tags.add("mega")
            if card.tera:
                tags.add("tera")
            if int(card.energyType) in BASIC_ENERGY_BY_TYPE:
                tags.add(f"type:{int(card.energyType)}")
            if card.evolvesFrom:
                tags.add(f"evolves_from:{card.evolvesFrom.lower()}")
        elif int(card.cardType) in {
            int(CardType.ITEM),
            int(CardType.TOOL),
            int(CardType.SUPPORTER),
            int(CardType.STADIUM),
        }:
            tags.add("trainer")
            tags.add(CardType(int(card.cardType)).name.lower())
        elif int(card.cardType) in {int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)}:
            tags.add("energy")
            if int(card.energyType) in BASIC_ENERGY_BY_TYPE:
                tags.add(f"type:{int(card.energyType)}")

        if "draw" in folded:
            tags.add("draw")
        if "search your deck" in folded or "look at the top" in folded:
            tags.add("search")
        if "supporter" in folded:
            tags.add("supporter_search")
        if "pokemon" in folded or "pokémon" in folded:
            tags.add("pokemon_search")
        if "basic pokemon" in folded or "basic pokémon" in folded:
            tags.add("basic_search")
        if "energy" in folded:
            tags.add("energy_support")
        if "attach" in folded and "energy" in folded:
            tags.add("energy_accel")
        if "switch your active" in folded:
            tags.add("switch")
        if "opponent" in folded and "benched" in folded and "active" in folded:
            tags.add("gust")
        if "evolv" in folded or "rare candy" in folded:
            tags.add("evolution_support")
        if "stage 2" in folded:
            tags.add("stage2_support")
        if "stadium" in folded:
            tags.add("stadium_support")
        if "tool" in folded:
            tags.add("tool_support")
        if "mega" in folded:
            tags.add("mega_support")
        if "tera" in folded:
            tags.add("tera_support")
        if "discard" in folded:
            tags.add("discard")
        if "ancient" in folded:
            tags.add("ancient")
        if "future" in folded:
            tags.add("future")
        return tags

    def _infer_roles(self, cid: int) -> set[str]:
        card = self.cards[cid]
        text = self.card_text.get(cid, "")
        folded = self._fold_text(f"{card.name} {text}")
        roles: set[str] = set()

        has_search = "search your deck" in folded
        mentions_pokemon = "pokemon" in folded or "pokÃ©mon" in folded
        mentions_basic_pokemon = "basic pokemon" in folded or "basic pokÃ©mon" in folded
        mentions_energy = "energy" in folded

        if has_search and mentions_pokemon:
            roles.add("pokemon_search")
        if has_search and mentions_basic_pokemon:
            roles.add("pokemon_search")
        if "draw" in folded:
            roles.add("draw")
        if has_search and mentions_energy:
            roles.add("energy_search")
        if mentions_energy and "attach" in folded:
            roles.add("energy_accel")
        if "switch your active" in folded or card.name in {"Switch", "Scramble Switch"}:
            roles.add("switch")
        if "opponent" in folded and "benched" in folded and "active" in folded:
            roles.add("gust")
        if card.name in {"Rare Candy", "Salvatore"} or "evolve" in folded or "evolves from" in folded:
            roles.add("evolution")
        if int(card.cardType) == int(CardType.STADIUM):
            roles.add("stadium")
        if int(card.cardType) == int(CardType.TOOL):
            roles.add("tool")

        return roles

    def _card_is_eligible(self, cid: int) -> bool:
        card = self.cards[cid]
        if int(card.cardType) == int(CardType.POKEMON):
            return self._chainable(cid)
        return True

    def _chainable(self, cid: int) -> bool:
        if cid in self._chain_cache:
            return self._chain_cache[cid]

        card = self.cards[cid]
        if int(card.cardType) != int(CardType.POKEMON) or card.basic or not card.evolvesFrom:
            self._chain_cache[cid] = True
            return True

        candidates = self.by_name.get(card.evolvesFrom, [])
        ok = any(self._can_be_evolution_source(prev) for prev in candidates)
        self._chain_cache[cid] = ok
        return ok

    def _can_be_evolution_source(self, cid: int) -> bool:
        card = self.cards[cid]
        if cid in FOSSIL_ITEM_IDS:
            return True
        if int(card.cardType) != int(CardType.POKEMON):
            return False
        return self._chainable(cid)

    def _evolution_chain(self, cid: int, rng: random.Random) -> list[int]:
        card = self.cards[cid]
        if int(card.cardType) != int(CardType.POKEMON) or card.basic or not card.evolvesFrom:
            return [cid]

        candidates = [
            prev
            for prev in self.by_name.get(card.evolvesFrom, [])
            if self._can_be_evolution_source(prev)
        ]
        if not candidates:
            return [cid]

        rng.shuffle(candidates)
        prev = min(candidates, key=lambda x: self._chain_choice_penalty(x, cid))
        return self._evolution_chain(prev, rng) + [cid]

    def _chain_choice_penalty(self, prev: int, target: int) -> tuple[int, int]:
        prev_types = self._required_attack_energy_types(prev)
        target_types = self._required_attack_energy_types(target)
        new_types = prev_types - target_types
        return (len(new_types), prev)

    def _required_attack_energy_types(self, cid: int) -> set[int]:
        card = self.cards[cid]
        if int(card.cardType) != int(CardType.POKEMON):
            return set()

        required: set[int] = set()
        for attack_id in card.attacks:
            attack = self.attacks.get(attack_id)
            if not attack:
                continue
            for energy in attack.energies:
                etype = int(energy)
                if etype in BASIC_ENERGY_BY_TYPE:
                    required.add(etype)
        return required

    def _family_key(self, cid: int, rng: random.Random) -> str:
        chain_ids = self._evolution_chain(cid, rng)
        return self.cards[chain_ids[0]].name

    def _package_counts(self, chain_ids: Sequence[int], rng: random.Random, main: bool) -> Counter:
        counts = Counter()
        if len(chain_ids) == 1:
            counts[chain_ids[0]] = rng.choice([2, 3, 3, 4]) if main else rng.choice([1, 2])
        elif len(chain_ids) == 2:
            counts[chain_ids[0]] = rng.choice([3, 4]) if main else 2
            counts[chain_ids[1]] = rng.choice([2, 3]) if main else 1
        else:
            counts[chain_ids[0]] = rng.choice([3, 4]) if main else 2
            counts[chain_ids[1]] = rng.choice([2, 3]) if main else 1
            counts[chain_ids[2]] = rng.choice([1, 2]) if main else 1
        return counts

    def _choose_main_attacker(self, rng: random.Random) -> int:
        scored: list[tuple[int, int]] = []
        for cid in self.pokemon_ids:
            if cid not in self.eligible_ids:
                continue
            types = self._archetype_energy_types(cid)
            if not types or len(types) > self.config.max_energy_types:
                continue
            card = self.cards[cid]
            tags = self.card_tags[cid]
            roles = self.card_roles[cid]
            score = 1 + max(0, self._max_attack_damage(cid))
            score += (card.hp or 0) // 10
            score += 45 if card.megaEx else 0
            score += 30 if card.ex else 0
            score += 18 if card.stage2 else 10 if card.stage1 else 0
            score += 18 if "energy_accel" in roles else 0
            score += 14 if "draw" in roles else 0
            score += 12 if "pokemon_search" in roles else 0
            scored.append((max(1, score), cid))
        return self._weighted_choice(scored, rng)

    def _validate_competitive_main(self, cid: int) -> None:
        if cid not in self.cards:
            raise ValueError(f"unknown main_card_id {cid}")
        card = self.cards[cid]
        if int(card.cardType) != int(CardType.POKEMON):
            raise ValueError(f"main_card_id {cid} is not a Pokemon")
        if cid not in self.eligible_ids:
            raise ValueError(f"main_card_id {cid} cannot form a legal evolution chain")
        types = self._archetype_energy_types(cid)
        if not types or len(types) > self.config.max_energy_types:
            raise ValueError(f"main_card_id {cid} is not compatible with competitive deck generation")

    def _archetype_energy_types(self, cid: int) -> tuple[int, ...]:
        required = sorted(self._required_attack_energy_types(cid))
        if required:
            return tuple(required[: self.config.max_energy_types])
        card = self.cards[cid]
        if int(card.energyType) in BASIC_ENERGY_BY_TYPE:
            return (int(card.energyType),)
        return ()

    def _max_attack_damage(self, cid: int) -> int:
        card = self.cards[cid]
        best = 0
        for attack_id in card.attacks:
            attack = self.attacks.get(attack_id)
            if attack:
                best = max(best, attack.damage or 0)
        return best

    def _weighted_choice(self, scored: Sequence[tuple[int, int]], rng: random.Random) -> int:
        if not scored:
            return rng.choice(self.basic_pokemon_ids)
        total = sum(weight for weight, _ in scored)
        pick = rng.randrange(total)
        running = 0
        for weight, cid in scored:
            running += weight
            if pick < running:
                return cid
        return scored[-1][1]

    def _competitive_package_counts(self, chain_ids: Sequence[int], main: bool) -> Counter:
        counts = Counter()
        if len(chain_ids) == 1:
            counts[chain_ids[0]] = 4 if main else 2
        elif len(chain_ids) == 2:
            counts[chain_ids[0]] = 4 if main else 2
            counts[chain_ids[1]] = 3 if main else 2
        else:
            counts[chain_ids[0]] = 4 if main else 2
            counts[chain_ids[1]] = 3 if main else 1
            counts[chain_ids[2]] = 2 if main else 1
        return counts

    def _add_competitive_pokemon_package(
        self,
        deck: Counter,
        cid: int,
        rng: random.Random,
        main: bool,
        families: set[str],
    ) -> bool:
        if cid not in self.eligible_ids:
            return False
        card = self.cards[cid]
        if int(card.cardType) != int(CardType.POKEMON):
            return False

        chain_ids = self._evolution_chain(cid, rng)
        family = self.cards[chain_ids[0]].name
        if family in families:
            return False

        package = self._competitive_package_counts(chain_ids, main)
        current_types = self._deck_required_energy_types(deck)
        package_types = set().union(*(self._required_attack_energy_types(x) for x in package))
        if len(current_types | package_types) > self.config.max_energy_types:
            return False
        if not self._can_add_many(deck, package):
            return False

        self._add_many(deck, package)
        families.add(family)
        return True

    def _add_competitive_support_pokemon(
        self,
        deck: Counter,
        archetype_tags: set[str],
        archetype_types: Sequence[int],
        rng: random.Random,
        families: set[str],
    ) -> None:
        target = rng.randint(14, 19)
        attempts = 0
        while self._pokemon_count(deck) < target and attempts < 8:
            attempts += 1
            candidates = []
            sample = list(self.pokemon_ids)
            rng.shuffle(sample)
            for cid in sample:
                if cid not in self.eligible_ids:
                    continue
                if self._family_key(cid, rng) in families:
                    continue
                types = self._archetype_energy_types(cid)
                if types and not set(types).issubset(set(archetype_types)):
                    continue
                score = self._support_pokemon_score(cid, archetype_tags)
                if score > 0:
                    candidates.append((score, cid))
            if not candidates:
                break
            cid = self._weighted_choice(sorted(candidates, reverse=True)[:80], rng)
            self._add_competitive_pokemon_package(deck, cid, rng, False, families)

    def _support_pokemon_score(self, cid: int, archetype_tags: set[str]) -> int:
        card = self.cards[cid]
        tags = self.card_tags[cid]
        roles = self.card_roles[cid]
        overlap = tags & archetype_tags
        score = 2 * len(overlap)
        score += 18 if "draw" in roles else 0
        score += 16 if "pokemon_search" in roles else 0
        score += 12 if "energy_accel" in roles else 0
        score += 8 if "basic" in tags else 0
        score += 4 if card.ex else 0
        score -= 10 if card.stage2 and "stage2" not in archetype_tags else 0
        return score

    def _add_competitive_energy(
        self,
        deck: Counter,
        archetype_types: Sequence[int],
        archetype_tags: set[str],
        rng: random.Random,
    ) -> None:
        target = rng.randint(12, 15)
        types = list(archetype_types) or sorted(self._deck_required_energy_types(deck))
        if not types:
            types = [rng.choice(sorted(BASIC_ENERGY_BY_TYPE))]

        primary = types[0]
        for i in range(target):
            etype = primary if len(types) == 1 or i % 3 else types[min(1, len(types) - 1)]
            self._try_add(deck, BASIC_ENERGY_BY_TYPE[etype], 1)

        specials = [
            cid
            for cid in self.special_energy_ids
            if self._special_energy_compatible(cid, deck)
            and self._trainer_synergy_score(cid, archetype_tags) > 0
        ]
        rng.shuffle(specials)
        for cid in specials[: rng.randint(0, 2)]:
            if self._energy_count(deck) < 16:
                self._try_add(deck, cid, 1)

    def _add_competitive_trainers(
        self,
        deck: Counter,
        archetype_tags: set[str],
        rng: random.Random,
    ) -> None:
        if "stage2" in archetype_tags:
            self._try_add_competitive_trainer(deck, 1079, 4, archetype_tags)

        for role, target in [
            ("pokemon_search", self.config.competitive_min_pokemon_search),
            ("draw", self.config.competitive_min_draw),
            ("energy_search", 1),
        ]:
            self._fill_trainer_role(deck, archetype_tags, role, target, rng)

        self._fill_switch_gust(deck, archetype_tags, rng)

        while self._trainer_count(deck) < 27 and sum(deck.values()) < 60:
            candidates = self._best_trainers(archetype_tags, rng, allow_zero_score=True)
            if not candidates:
                break
            added = False
            for cid in candidates:
                before = deck[cid]
                self._try_add_competitive_trainer(
                    deck,
                    cid,
                    self._competitive_trainer_copies(cid, "generic", rng),
                    archetype_tags,
                )
                if deck[cid] > before:
                    added = True
                    break
            if not added:
                break

    def _fill_trainer_role(
        self,
        deck: Counter,
        archetype_tags: set[str],
        role: str,
        target: int,
        rng: random.Random,
    ) -> None:
        attempts = 0
        while self._role_count(deck, role) < target and attempts < 20:
            attempts += 1
            candidates = self._best_trainers(archetype_tags, rng, role=role)
            added = False
            for cid in candidates[:30]:
                before = deck[cid]
                self._try_add_competitive_trainer(
                    deck,
                    cid,
                    self._competitive_trainer_copies(cid, role, rng),
                    archetype_tags,
                )
                if deck[cid] > before:
                    added = True
                    break
            if not added:
                break

    def _fill_switch_gust(
        self,
        deck: Counter,
        archetype_tags: set[str],
        rng: random.Random,
    ) -> None:
        attempts = 0
        while self._switch_gust_count(deck) < self.config.competitive_min_switch_gust and attempts < 20:
            attempts += 1
            role = "gust" if self._role_count(deck, "gust") < self._role_count(deck, "switch") else "switch"
            candidates = self._best_trainers(archetype_tags, rng, role=role)
            added = False
            for cid in candidates[:30]:
                before = deck[cid]
                self._try_add_competitive_trainer(
                    deck,
                    cid,
                    self._competitive_trainer_copies(cid, role, rng),
                    archetype_tags,
                )
                if deck[cid] > before:
                    added = True
                    break
            if not added:
                break

    def _best_trainers(
        self,
        archetype_tags: set[str],
        rng: random.Random,
        role: str | None = None,
        allow_zero_score: bool = False,
    ) -> list[int]:
        scored = [
            (self._trainer_synergy_score(cid, archetype_tags), rng.random(), cid)
            for cid in self.trainer_ids
            if self._trainer_compatible(cid, archetype_tags)
            and (role is None or role in self.card_roles[cid])
        ]
        if not allow_zero_score:
            scored = [row for row in scored if row[0] > 0]
        scored.sort(reverse=True)
        return [cid for _, _, cid in scored]

    def _trainer_synergy_score(self, cid: int, archetype_tags: set[str]) -> int:
        tags = self.card_tags.get(cid, frozenset())
        roles = self.card_roles.get(cid, frozenset())
        score = len(tags & archetype_tags) * 3
        for role, weight in [
            ("pokemon_search", 14),
            ("draw", 12),
            ("energy_search", 8),
            ("energy_accel", 8),
            ("switch", 7),
            ("gust", 7),
            ("evolution", 7),
            ("stadium", 2),
            ("tool", 2),
        ]:
            if role in roles:
                score += weight
        if "evolution" in roles and "stage2" not in archetype_tags:
            score -= 4
        if "stage2_support" in tags and "stage2" not in archetype_tags:
            score -= 10
        if "mega_support" in tags and "mega" not in archetype_tags:
            score -= 8
        if "tera_support" in tags and "tera" not in archetype_tags:
            score -= 5
        return score

    def _trainer_compatible(self, cid: int, archetype_tags: set[str]) -> bool:
        if cid in FOSSIL_ITEM_IDS:
            return False
        card = self.cards[cid]
        tags = self.card_tags[cid]
        if card.aceSpec and "ace_spec_taken" in archetype_tags:
            return False
        if cid == 1079 and "stage2" not in archetype_tags:
            return False
        if "stage2_support" in tags and "stage2" not in archetype_tags:
            return False
        if "mega_support" in tags and "mega" not in archetype_tags:
            return False
        return True

    def _try_add_competitive_trainer(
        self,
        deck: Counter,
        cid: int,
        count: int,
        archetype_tags: set[str],
    ) -> int:
        added = 0
        for _ in range(count):
            if not self._competitive_role_room(deck, cid, archetype_tags):
                break
            before = deck[cid]
            self._try_add(deck, cid, 1)
            if deck[cid] == before:
                break
            added += 1
        return added

    def _competitive_role_room(self, deck: Counter, cid: int, archetype_tags: set[str]) -> bool:
        roles = self.card_roles[cid]
        if "draw" in roles and self._role_count(deck, "draw") >= self.config.competitive_max_draw:
            return False
        if (
            "pokemon_search" in roles
            and self._role_count(deck, "pokemon_search") >= self.config.competitive_max_pokemon_search
        ):
            return False
        if (
            roles & {"switch", "gust"}
            and self._switch_gust_count(deck) >= self.config.competitive_max_switch_gust
        ):
            return False
        if (
            "energy_search" in roles
            and self._role_count(deck, "energy_search") >= self.config.competitive_max_energy_search
        ):
            return False
        if (
            "energy_accel" in roles
            and self._role_count(deck, "energy_accel") >= self.config.competitive_max_energy_accel
        ):
            return False
        if "stadium" in roles and self._role_count(deck, "stadium") >= self.config.competitive_max_stadium:
            return False
        if "tool" in roles and self._role_count(deck, "tool") >= self.config.competitive_max_tool:
            return False
        if "evolution" in roles:
            max_evolution = 4 if "stage2" in archetype_tags else 2
            if self._trainer_role_count(deck, "evolution") >= max_evolution:
                return False
        return True

    def _competitive_trainer_copies(
        self,
        cid: int,
        role: str,
        rng: random.Random,
        preferred: int | None = None,
    ) -> int:
        card = self.cards[cid]
        if card.aceSpec:
            return 1
        if preferred is not None:
            return preferred
        if role in {"pokemon_search", "draw"}:
            return rng.choice([3, 4])
        if role in {"switch", "gust", "energy_search", "energy_accel", "evolution"}:
            return rng.choice([2, 3])
        if int(card.cardType) in {int(CardType.STADIUM), int(CardType.TOOL)}:
            return rng.choice([1, 2])
        return rng.choice([2, 3])

    def _finish_competitive_to_60(
        self,
        deck: Counter,
        archetype_tags: set[str],
        rng: random.Random,
    ) -> None:
        while sum(deck.values()) < 60:
            candidates = self._best_trainers(archetype_tags, rng, allow_zero_score=True)
            if not candidates:
                break
            added = False
            for cid in candidates:
                before = deck[cid]
                self._try_add_competitive_trainer(deck, cid, 1, archetype_tags)
                if deck[cid] > before:
                    added = True
                    break
            if not added:
                break

        required_types = sorted(self._deck_required_energy_types(deck))
        i = 0
        while sum(deck.values()) < 60:
            etype = required_types[i % len(required_types)] if required_types else rng.choice(sorted(BASIC_ENERGY_BY_TYPE))
            self._try_add(deck, BASIC_ENERGY_BY_TYPE[etype], 1)
            i += 1

        while sum(deck.values()) > 60:
            removable = [
                cid
                for cid in deck
                if deck[cid] > self._minimum_needed_copies(deck, cid)
                and int(self.cards[cid].cardType) != int(CardType.POKEMON)
            ]
            if not removable:
                break
            cid = min(removable, key=lambda x: self._trainer_synergy_score(x, archetype_tags))
            deck[cid] -= 1
            if deck[cid] <= 0:
                del deck[cid]

    def _role_count(self, deck: Counter, role: str) -> int:
        return sum(
            count
            for cid, count in deck.items()
            if role in self.card_roles.get(cid, frozenset())
        )

    def _trainer_role_count(self, deck: Counter, role: str) -> int:
        return sum(
            count
            for cid, count in deck.items()
            if role in self.card_roles.get(cid, frozenset())
            and int(self.cards[cid].cardType)
            in {
                int(CardType.ITEM),
                int(CardType.TOOL),
                int(CardType.SUPPORTER),
                int(CardType.STADIUM),
            }
        )

    def _switch_gust_count(self, deck: Counter) -> int:
        return self._role_count(deck, "switch") + self._role_count(deck, "gust")

    def _passes_competitive_profile(self, deck: Counter, archetype_tags: set[str]) -> bool:
        pokemon = self._pokemon_count(deck)
        energy = self._energy_count(deck)
        trainers = self._trainer_count(deck)
        if not (14 <= pokemon <= 20 and 12 <= energy <= 16 and 24 <= trainers <= 32):
            return False
        if not (
            self.config.competitive_min_draw
            <= self._role_count(deck, "draw")
            <= self.config.competitive_max_draw
        ):
            return False
        if not (
            self.config.competitive_min_pokemon_search
            <= self._role_count(deck, "pokemon_search")
            <= self.config.competitive_max_pokemon_search
        ):
            return False
        if not (
            self.config.competitive_min_switch_gust
            <= self._switch_gust_count(deck)
            <= self.config.competitive_max_switch_gust
        ):
            return False
        if self._role_count(deck, "stadium") > self.config.competitive_max_stadium:
            return False
        if self._role_count(deck, "tool") > self.config.competitive_max_tool:
            return False
        if self._role_count(deck, "energy_search") > self.config.competitive_max_energy_search:
            return False
        if self._role_count(deck, "energy_accel") > self.config.competitive_max_energy_accel:
            return False
        if "stage2" in archetype_tags and not (3 <= deck[1079] <= 4):
            return False
        return True

    def _passes_opening_curve(self, deck: Counter, rng: random.Random) -> bool:
        cards = list(deck.elements())
        required_types = self._deck_required_energy_types(deck)
        if not cards or not required_types:
            return False

        basic_hits = 0
        energy_hits = 0
        setup_hits = 0
        samples = self.config.competitive_opening_samples
        for _ in range(samples):
            opening = rng.sample(cards, 8)
            first_seven = opening[:7]
            has_basic = any(self._is_basic_pokemon(cid) for cid in first_seven)
            has_energy = any(self._is_relevant_energy(cid, required_types, deck) for cid in opening)
            has_setup = any(
                self.card_roles[cid] & {"draw", "pokemon_search", "energy_search"}
                for cid in opening
            )
            has_second_basic = sum(1 for cid in opening if self._is_basic_pokemon(cid)) >= 2
            basic_hits += int(has_basic)
            energy_hits += int(has_energy)
            setup_hits += int(has_basic and (has_setup or has_second_basic))

        return (
            basic_hits / samples >= self.config.competitive_min_basic_open_rate
            and energy_hits / samples >= self.config.competitive_min_energy_open_rate
            and setup_hits / samples >= self.config.competitive_min_setup_open_rate
        )

    def _add_pokemon_package(
        self,
        deck: Counter,
        cid: int,
        rng: random.Random,
        main: bool,
        families: set[str],
    ) -> bool:
        if cid not in self.eligible_ids:
            return False
        card = self.cards[cid]
        if int(card.cardType) != int(CardType.POKEMON):
            return False

        chain_ids = self._evolution_chain(cid, rng)
        family = self.cards[chain_ids[0]].name
        if family in families and not any(deck[x] == 0 for x in chain_ids):
            return False

        package = self._package_counts(chain_ids, rng, main)
        package[cid] = max(1, package[cid])
        if not self._can_add_many(deck, package):
            return False

        current_types = self._deck_required_energy_types(deck)
        package_types = set().union(*(self._required_attack_energy_types(x) for x in package))
        if len(current_types | package_types) > self.config.max_energy_types:
            return False

        self._add_many(deck, package)
        families.add(family)
        return True

    def _add_forced_card(
        self,
        deck: Counter,
        cid: int,
        rng: random.Random,
        families: set[str],
    ) -> None:
        card = self.cards[cid]
        if int(card.cardType) == int(CardType.POKEMON):
            self._add_pokemon_package(deck, cid, rng, True, families)
            return
        if int(card.cardType) in {int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)}:
            self._try_add(deck, cid, 1)
            return
        self._try_add(deck, cid, 1)

    def _add_required_support_for_special_energy(
        self,
        deck: Counter,
        forced: Sequence[int],
        rng: random.Random,
        families: set[str],
    ) -> None:
        if TEAM_ROCKET_ENERGY_ID in forced:
            candidates = [
                cid
                for cid in self.pokemon_ids
                if "Team Rocket" in self.cards[cid].name and self._chainable(cid)
            ]
            if candidates:
                self._add_pokemon_package(deck, rng.choice(candidates), rng, True, families)
        if NEO_UPPER_ENERGY_ID in forced:
            candidates = [
                cid
                for cid in self.pokemon_ids
                if self.cards[cid].stage2 and self._chainable(cid)
            ]
            if candidates:
                self._add_pokemon_package(deck, rng.choice(candidates), rng, True, families)

    def _fill_pokemon(
        self,
        deck: Counter,
        target: int,
        preferred: Sequence[int],
        rng: random.Random,
        families: set[str],
    ) -> None:
        attempts = 0
        while self._pokemon_count(deck) < target and attempts < 400:
            attempts += 1
            cid = self._choose_pokemon_anchor(deck, preferred, rng, families)
            self._add_pokemon_package(deck, cid, rng, self._pokemon_count(deck) < 8, families)

    def _choose_basic_anchor(self, preferred: Sequence[int], rng: random.Random) -> int:
        preferred_basics = [
            cid
            for cid in preferred
            if cid in self.basic_pokemon_ids
        ]
        if preferred_basics:
            return rng.choice(preferred_basics[: min(80, len(preferred_basics))])
        return rng.choice(self.basic_pokemon_ids)

    def _choose_pokemon_anchor(
        self,
        deck: Counter,
        preferred: Sequence[int],
        rng: random.Random,
        families: set[str],
    ) -> int:
        current_types = self._deck_required_energy_types(deck)
        pools = [
            [cid for cid in preferred if cid in self.pokemon_ids],
            self.pokemon_ids,
        ]
        for pool in pools:
            candidates = []
            sample = list(pool)
            rng.shuffle(sample)
            for cid in sample[:300]:
                if cid not in self.eligible_ids:
                    continue
                if self._family_key(cid, rng) in families and rng.random() < 0.85:
                    continue
                types = self._required_attack_energy_types(cid)
                if len(current_types | types) <= self.config.max_energy_types:
                    candidates.append(cid)
            if candidates:
                return rng.choice(candidates)
        return rng.choice(self.basic_pokemon_ids)

    def _add_energy(self, deck: Counter, preferred: Sequence[int], rng: random.Random) -> None:
        target = rng.randint(self.config.min_energy, self.config.max_energy)
        required_types = sorted(self._deck_required_energy_types(deck))
        if not required_types:
            required_types = [int(self.cards[self._first_pokemon_id(deck)].energyType)]
            required_types = [t for t in required_types if t in BASIC_ENERGY_BY_TYPE]
        if not required_types:
            required_types = [rng.choice(sorted(BASIC_ENERGY_BY_TYPE))]

        forced_energy = [
            cid
            for cid in preferred
            if cid in self.special_energy_ids and self._special_energy_compatible(cid, deck)
        ][:2]
        for cid in forced_energy:
            self._try_add(deck, cid, 1)

        remaining = max(0, target - self._energy_count(deck))
        for i in range(remaining):
            etype = required_types[i % len(required_types)]
            self._try_add(deck, BASIC_ENERGY_BY_TYPE[etype], 1)

        if rng.random() < 0.55:
            candidates = [
                cid for cid in self.special_energy_ids if self._special_energy_compatible(cid, deck)
            ]
            rng.shuffle(candidates)
            for cid in candidates[: rng.randint(0, 2)]:
                if self._energy_count(deck) < self.config.max_energy:
                    self._try_add(deck, cid, 1)

    def _fill_trainers(self, deck: Counter, preferred: Sequence[int], rng: random.Random) -> None:
        target_trainers = 60 - self._pokemon_count(deck) - self._energy_count(deck)
        trainer_target = max(20, min(32, target_trainers))

        preferred_trainers = [
            cid
            for cid in preferred
            if cid in self.trainer_ids and cid not in FOSSIL_ITEM_IDS
        ]
        rng.shuffle(preferred_trainers)
        for cid in preferred_trainers:
            if self._trainer_count(deck) >= trainer_target:
                break
            self._try_add(deck, cid, 1)

        attempts = 0
        while self._trainer_count(deck) < trainer_target and attempts < 300:
            attempts += 1
            cid = rng.choice(self.trainer_ids)
            if cid in FOSSIL_ITEM_IDS:
                continue
            self._try_add(deck, cid, 1)

    def _finish_to_60(
        self,
        deck: Counter,
        preferred: Sequence[int],
        rng: random.Random,
        protected: set[int],
    ) -> None:
        candidates = list(preferred) + self.trainer_ids + self.special_energy_ids
        rng.shuffle(candidates)
        attempts = 0
        while sum(deck.values()) < 60 and attempts < 1000:
            attempts += 1
            cid = candidates[attempts % len(candidates)]
            card = self.cards[cid]
            if cid in FOSSIL_ITEM_IDS:
                continue
            if int(card.cardType) == int(CardType.SPECIAL_ENERGY) and not self._special_energy_compatible(cid, deck):
                continue
            self._try_add(deck, cid, 1)

        required_types = sorted(self._deck_required_energy_types(deck))
        i = 0
        while sum(deck.values()) < 60:
            etype = required_types[i % len(required_types)] if required_types else int(EnergyType.COLORLESS)
            energy_id = BASIC_ENERGY_BY_TYPE.get(etype, rng.choice(list(BASIC_ENERGY_BY_TYPE.values())))
            self._try_add(deck, energy_id, 1)
            i += 1

        while sum(deck.values()) > 60:
            removable = [
                cid
                for cid in deck
                if cid not in protected
                if deck[cid] > self._minimum_needed_copies(deck, cid)
            ]
            if not removable:
                break
            cid = rng.choice(removable)
            deck[cid] -= 1
            if deck[cid] <= 0:
                del deck[cid]

    def _forced_for_next_deck(self, preferred: Sequence[int], seed: int) -> list[int]:
        if not preferred:
            return []
        rng = random.Random(seed)
        pokemon = [cid for cid in preferred if cid in self.pokemon_ids]
        trainers = [cid for cid in preferred if cid in self.trainer_ids]
        energies = [
            cid
            for cid in preferred
            if int(self.cards[cid].cardType)
            in {int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)}
        ]
        forced: list[int] = []
        if pokemon:
            forced.append(rng.choice(pokemon[: min(120, len(pokemon))]))
        forced.extend(trainers[:3])
        forced.extend(energies[:1])
        return forced

    def _can_add_many(self, deck: Counter, additions: Counter) -> bool:
        probe = deck.copy()
        for cid, count in additions.items():
            if not self._try_add(probe, cid, count):
                return False
        return True

    def _add_many(self, deck: Counter, additions: Counter) -> None:
        for cid, count in additions.items():
            self._try_add(deck, cid, count)

    def _try_add(self, deck: Counter, cid: int, count: int) -> bool:
        if cid not in self.cards or count <= 0:
            return False
        card = self.cards[cid]
        if card.aceSpec:
            existing = sum(deck[x] for x in deck if self.cards[x].aceSpec)
            if existing > deck[cid]:
                return False
            count = min(count, 1 - deck[cid])
            if count <= 0:
                return False
        if int(card.cardType) == int(CardType.BASIC_ENERGY):
            deck[cid] += count
            return True

        same_name = sum(
            n for other, n in deck.items()
            if self.cards[other].name == card.name
            and int(self.cards[other].cardType) != int(CardType.BASIC_ENERGY)
        )
        room = self.config.max_copies - same_name
        if room <= 0:
            return False
        deck[cid] += min(count, room)
        return True

    def _minimum_needed_copies(self, deck: Counter, cid: int) -> int:
        card = self.cards[cid]
        if int(card.cardType) == int(CardType.BASIC_ENERGY):
            required_types = self._deck_required_energy_types(deck)
            return 1 if int(card.energyType) in required_types else 0
        if int(card.cardType) != int(CardType.POKEMON):
            return 0
        if card.evolvesFrom:
            return 1
        if card.basic and self._basic_pokemon_count(deck) == deck[cid]:
            return 1
        needed_as_prevo = any(
            self.cards[evo].evolvesFrom == card.name
            for evo in deck
            if int(self.cards[evo].cardType) == int(CardType.POKEMON)
        )
        return 1 if needed_as_prevo else 0

    def _deck_required_energy_types(self, deck: Counter) -> set[int]:
        required: set[int] = set()
        for cid in deck:
            required |= self._required_attack_energy_types(cid)
        return required

    def _special_energy_compatible(self, cid: int, deck: Counter) -> bool:
        if cid == TEAM_ROCKET_ENERGY_ID:
            return any(
                int(self.cards[x].cardType) == int(CardType.POKEMON)
                and "Team Rocket" in self.cards[x].name
                for x in deck
            )
        if cid == NEO_UPPER_ENERGY_ID:
            return any(
                int(self.cards[x].cardType) == int(CardType.POKEMON)
                and self.cards[x].stage2
                for x in deck
            )
        return True

    def _first_pokemon_id(self, deck: Counter) -> int:
        for cid in deck:
            if int(self.cards[cid].cardType) == int(CardType.POKEMON):
                return cid
        return self.basic_pokemon_ids[0]

    def _is_basic_pokemon(self, cid: int) -> bool:
        card = self.cards.get(cid)
        return bool(
            card
            and int(card.cardType) == int(CardType.POKEMON)
            and card.basic
        )

    def _is_relevant_energy(self, cid: int, required_types: set[int], deck: Counter) -> bool:
        card = self.cards.get(cid)
        if not card:
            return False
        if int(card.cardType) == int(CardType.BASIC_ENERGY):
            return int(card.energyType) in required_types
        if int(card.cardType) == int(CardType.SPECIAL_ENERGY):
            return self._special_energy_compatible(cid, deck)
        return False

    def _pokemon_count(self, deck: Counter) -> int:
        return sum(
            count
            for cid, count in deck.items()
            if int(self.cards[cid].cardType) == int(CardType.POKEMON)
        )

    def _basic_pokemon_count(self, deck: Counter) -> int:
        return sum(
            count
            for cid, count in deck.items()
            if int(self.cards[cid].cardType) == int(CardType.POKEMON)
            and self.cards[cid].basic
        )

    def _energy_count(self, deck: Counter) -> int:
        return sum(
            count
            for cid, count in deck.items()
            if int(self.cards[cid].cardType)
            in {int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)}
        )

    def _trainer_count(self, deck: Counter) -> int:
        return sum(
            count
            for cid, count in deck.items()
            if int(self.cards[cid].cardType)
            in {
                int(CardType.ITEM),
                int(CardType.TOOL),
                int(CardType.SUPPORTER),
                int(CardType.STADIUM),
            }
        )


_DEFAULT_GENERATOR: RandomDeckGenerator | None = None


def default_generator() -> RandomDeckGenerator:
    global _DEFAULT_GENERATOR
    if _DEFAULT_GENERATOR is None:
        _DEFAULT_GENERATOR = RandomDeckGenerator()
    return _DEFAULT_GENERATOR


def generate_random_deck(
    seed: int | None = None,
    forced_cards: Iterable[int] = (),
    sampling_method: SamplingMethod = "coverage",
    main_card_id: int | None = None,
) -> list[int]:
    """Generate one structurally playable random deck."""

    return default_generator().generate_deck(
        seed=seed,
        forced_cards=forced_cards,
        sampling_method=sampling_method,
        main_card_id=main_card_id,
    )


def generate_random_decks(
    count: int,
    seed: int = 0,
    cover_all: bool = True,
    sampling_method: SamplingMethod = "coverage",
    main_card_id: int | None = None,
) -> list[list[int]]:
    """Generate many random decks, biased toward broad card coverage by default."""

    return default_generator().generate_decks(
        count=count,
        seed=seed,
        cover_all=cover_all,
        sampling_method=sampling_method,
        main_card_id=main_card_id,
    )


def generate_family_decks(
    main_card_id: int,
    count: int,
    seed: int = 0,
) -> list[list[int]]:
    """Generate competitive deck variants centered on one Pokemon family."""

    return default_generator().generate_family_decks(
        main_card_id=main_card_id,
        count=count,
        seed=seed,
    )


def validate_random_deck(deck: Iterable[int]) -> DeckValidation:
    """Validate a deck against the random-generator structural guarantees."""

    return default_generator().validate(deck)


def random_deck_coverage(decks: Iterable[Iterable[int]]) -> CoverageReport:
    """Report eligible card coverage across a generated deck corpus."""

    return default_generator().coverage(decks)
