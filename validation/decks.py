"""Known-legal 60-card decks for validation.

Each deck is a flat list of 60 card IDs. `expand` turns (card_id, count) pairs
into the flat list the engine expects.
"""
from __future__ import annotations

from ptcg.cg.api import CardType, _native_all_card_data


def expand(pairs: list[tuple[int, int]]) -> list[int]:
    """Expand (card_id, count) pairs into a flat list of card IDs."""
    deck: list[int] = []
    for card_id, count in pairs:
        deck.extend([card_id] * count)
    if len(deck) != 60:
        raise ValueError(f"deck has {len(deck)} cards, expected 60")
    return deck


# Mega Lucario ex deck.
MEGA_LUCARIO = expand([
    (673, 2),   # Makuhita
    (674, 2),   # Hariyama
    (675, 2),   # Lunatone
    (676, 3),   # Solrock
    (677, 3),   # Riolu
    (678, 4),   # Mega Lucario ex
    (1102, 4),  # Dusk Ball
    (1123, 2),  # Switch
    (1141, 4),  # Premium Power Pro
    (1142, 4),  # Fighting Gong
    (1152, 4),  # Poke Pad
    (1159, 1),  # Hero Cape
    (1182, 2),  # Boss's Orders
    (1192, 4),  # Carmine
    (1227, 4),  # Lillie's Determination
    (1252, 2),  # Gravity Mountain
    (6, 13),    # Basic Fighting Energy
])
MEGA_LUCARIO_EX_ID = 678


# Dragapult ex deck.
DRAGAPULT_EX = expand([
    (119, 4),   # Dreepy
    (120, 4),   # Drakloak
    (121, 3),   # Dragapult ex
    (140, 1),   # Fezandipiti ex
    (184, 1),   # Latias ex
    (235, 2),   # Budew
    (1071, 1),  # Meowth ex
    (1079, 2),  # Rare Candy
    (1080, 1),  # Unfair Stamp
    (1086, 4),  # Buddy-Buddy Poffin
    (1097, 2),  # Night Stretcher
    (1120, 4),  # Crushing Hammer
    (1121, 4),  # Ultra Ball
    (1152, 3),  # Poke Pad
    (1156, 1),  # Lucky Helmet
    (1182, 3),  # Boss's Orders
    (1198, 4),  # Crispin
    (1210, 2),  # Brock's Scouting
    (1227, 4),  # Lillie's Determination
    (1256, 2),  # Team Rocket Watchtower
    (2, 4),     # Basic Fire Energy
    (5, 4),     # Basic Psychic Energy
])


# Default deck shipped in data/sample_submission/deck.csv
SAMPLE_SUBMISSION = expand([
    (1158, 1),
    (721, 2),
    (722, 4),
    (723, 4),
    (1145, 4),
    (1205, 2),
    (1227, 4),
    (1235, 4),
    (3, 35),    # Basic Grass Energy
])


ALL_DECKS = {
    "mega_lucario": MEGA_LUCARIO,
    "dragapult_ex": DRAGAPULT_EX,
    "sample_submission": SAMPLE_SUBMISSION,
}


def generate_random_deck(
    seed: int | None = None,
    forced_cards=(),
    sampling_method: str = "coverage",
    main_card_id: int | None = None,
) -> list[int]:
    """Generate one structurally playable random 60-card deck."""
    from .random_decks import generate_random_deck as _generate_random_deck

    return _generate_random_deck(
        seed=seed,
        forced_cards=forced_cards,
        sampling_method=sampling_method,
        main_card_id=main_card_id,
    )


def generate_random_decks(
    count: int,
    seed: int = 0,
    cover_all: bool = True,
    sampling_method: str = "coverage",
    main_card_id: int | None = None,
) -> list[list[int]]:
    """Generate random decks, biased toward broad card coverage by default."""
    from .random_decks import generate_random_decks as _generate_random_decks

    return _generate_random_decks(
        count=count,
        seed=seed,
        cover_all=cover_all,
        sampling_method=sampling_method,
        main_card_id=main_card_id,
    )


def generate_mega_lucario_family_decks(count: int, seed: int = 0) -> list[list[int]]:
    """Generate competitive variants centered on the Mega Lucario ex line."""
    return generate_pokemon_family_decks(MEGA_LUCARIO_EX_ID, count=count, seed=seed)


def resolve_pokemon_card_id(main_pokemon: int | str) -> int:
    """Resolve a Pokemon card id from an integer id or unique name fragment."""

    if isinstance(main_pokemon, int):
        return main_pokemon

    needle = main_pokemon.casefold()
    matches = [
        card
        for card in _native_all_card_data()
        if int(card.cardType) == int(CardType.POKEMON)
        and needle in card.name.casefold()
    ]
    if not matches:
        raise ValueError(f"no Pokemon card matches {main_pokemon!r}")
    exact = [card for card in matches if card.name.casefold() == needle]
    if len(exact) == 1:
        return exact[0].cardId
    if len(matches) == 1:
        return matches[0].cardId

    choices = ", ".join(f"{card.name} ({card.cardId})" for card in matches[:12])
    suffix = "" if len(matches) <= 12 else f", and {len(matches) - 12} more"
    raise ValueError(f"Pokemon name {main_pokemon!r} is ambiguous: {choices}{suffix}")


def generate_pokemon_family_decks(
    main_pokemon: int | str,
    count: int,
    seed: int = 0,
) -> list[list[int]]:
    """Generate competitive variants centered on any main Pokemon family."""
    from .random_decks import generate_family_decks

    return generate_family_decks(
        main_card_id=resolve_pokemon_card_id(main_pokemon),
        count=count,
        seed=seed,
    )


if __name__ == "__main__":
    for name, deck in ALL_DECKS.items():
        print(f"{name}: {len(deck)} cards, {len(set(deck))} unique ids")
