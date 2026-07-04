"""Canonical, order-independent projection of a cabt observation.

Used to differentially test the custom C++ engine against the cabt oracle. We
compare the *acting player's view* and deliberately preserve cabt's information
masking (opponent hand / face-down cards stay hidden). Multiset-like fields are
normalised order-independently; bench/active stay positional because actions
index into them.

These functions operate on the raw obs dicts (camelCase, exactly as cabt's
GetBattleData JSON), so they have no dependency on the cg dataclass layer.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

CONDITION_KEYS = ("poisoned", "burned", "asleep", "paralyzed", "confused")
FLAG_KEYS = (
    "supporterPlayed",
    "teamRocketSupporterPlayed",
    "ancientSupporterPlayed",
    "stadiumPlayed",
    "energyAttached",
    "retreated",
    "canariPlayed",
    "tarragonPlayed",
)


def _energy_counts(energies: list[int] | None) -> tuple:
    """Resolved energy types -> sorted ((type, count), ...) multiset."""
    return tuple(sorted(Counter(energies or []).items()))


def _ids(cards: list[dict] | None) -> tuple:
    """Sorted multiset of card ids (for order-independent zones)."""
    return tuple(sorted(c["id"] for c in (cards or []) if c is not None))


def _pokemon(p: dict | None) -> dict | None:
    if p is None:
        return None
    return {
        "id": p["id"],
        "hp": p["hp"],
        "maxHp": p["maxHp"],
        "energies": _energy_counts(p.get("energies")),
        "tools": _ids(p.get("tools")),
        "preEvo": _ids(p.get("preEvolution")),
        "appearThisTurn": p.get("appearThisTurn", False),
    }


def _player(ps: dict, reveal_hand: bool) -> dict:
    active = ps.get("active") or []
    prizes = ps.get("prize") or []
    hand = ps.get("hand")
    hand_count = len(hand) if (reveal_hand and hand is not None) else ps.get("handCount")
    prize_face_up = list(ps.get("prizeFaceUp") or [])
    if len(prize_face_up) < len(prizes):
        prize_face_up.extend([False] * (len(prizes) - len(prize_face_up)))
    return {
        "active": _pokemon(active[0]) if active and active[0] is not None else None,
        "bench": tuple(_pokemon(b) for b in (ps.get("bench") or [])),
        "benchMax": ps.get("benchMax"),
        "deckCount": ps.get("deckCount"),
        "handCount": hand_count,
        # own hand is visible -> compare as multiset; opponent stays masked
        "hand": _ids(hand) if (reveal_hand and hand is not None) else None,
        "discard": _ids(ps.get("discard")),
        "prize": (),
        "prizeCount": len(prizes),
        "prizeFaceUp": prize_face_up,
        "conditions": tuple(k for k in CONDITION_KEYS if ps.get(k)),
    }


def canonical_state(current: dict | None) -> dict | None:
    """Project a cabt `current` state into a comparable, order-normalised dict."""
    if current is None:
        return None
    yi = current["yourIndex"]
    players = current["players"]
    return {
        "turn": current["turn"],
        "turnActionCount": current["turnActionCount"],
        "yourIndex": yi,
        "firstPlayer": current["firstPlayer"],
        "flags": {k: current.get(k, False) for k in FLAG_KEYS},
        "result": current["result"],
        "stadium": _ids(current.get("stadium")),
        "players": [
            _player(players[0], reveal_hand=(yi == 0)),
            _player(players[1], reveal_hand=(yi == 1)),
        ],
    }


# --- legal-move (option) projection ---------------------------------------
# Resolve area+index references to card ids so options can be matched across
# engines that don't share cabt's internal array ordering. This is the seed of
# the S1 "action bridge".

_AREA_NAMES = {
    1: "DECK", 2: "HAND", 3: "DISCARD", 4: "ACTIVE", 5: "BENCH", 6: "PRIZE",
    7: "STADIUM", 8: "ENERGY", 9: "TOOL", 10: "PRE_EVOLUTION", 11: "PLAYER",
    12: "LOOKING",
}


def _zone(current: dict, select: dict, area: int, player_index: int | None):
    """Return the card sequence for an (area, player) reference, or None."""
    if area == 1:  # DECK is carried on the select, not the player state
        return select.get("deck")
    if area == 7:
        return current.get("stadium")
    if area == 12:
        return current.get("looking")
    if player_index is None:
        return None
    ps = current["players"][player_index]
    return {
        2: ps.get("hand"), 3: ps.get("discard"), 4: ps.get("active"),
        5: ps.get("bench"), 6: ps.get("prize"),
    }.get(area)


def _ref_id(current: dict, select: dict, area: int | None, index: int | None,
            player_index: int | None):
    """Resolve a card reference to its card id (None if face-down/unknown)."""
    if area is None or index is None:
        return None
    seq = _zone(current, select, area, player_index)
    if seq is None or index >= len(seq) or seq[index] is None:
        return None
    card = seq[index]
    return card.get("id") if isinstance(card, dict) else None


def _energy_card_id(current: dict, area: int | None, index: int | None,
                    energy_index: int | None, player_index: int | None):
    if area is None or index is None or energy_index is None or player_index is None:
        return None
    ps = current["players"][player_index]
    target = None
    if area == 4:
        active = ps.get("active") or []
        if 0 <= index < len(active):
            target = active[index]
    elif area == 5:
        bench = ps.get("bench") or []
        if 0 <= index < len(bench):
            target = bench[index]
    if not isinstance(target, dict):
        return None
    energy_cards = target.get("energyCards") or []
    if 0 <= energy_index < len(energy_cards):
        card = energy_cards[energy_index]
        return card.get("id") if isinstance(card, dict) else None
    return None


def option_descriptor(current: dict | None, select: dict, op: dict) -> tuple:
    """A semantic, engine-agnostic key for one legal option.

    Resolves area/index references to card ids where possible so the *set* of
    legal moves can be compared between cabt and our engine. Falls back to the
    raw fields when state isn't available (e.g. setup before `current`).
    """
    t = op["type"]
    if current is None:
        return (t,) + tuple(sorted((k, v) for k, v in op.items() if k != "type"))
    pi = op.get("playerIndex")
    if t == 7:  # PLAY (hand index)
        return ("PLAY", _ref_id(current, select, 2, op.get("index"), current["yourIndex"]))
    if t == 8:  # ATTACH card -> in-play pokemon
        return ("ATTACH",
                _ref_id(current, select, op.get("area"), op.get("index"), current["yourIndex"]),
                _AREA_NAMES.get(op.get("inPlayArea")), op.get("inPlayIndex"))
    if t == 9:  # EVOLVE
        return ("EVOLVE",
                _ref_id(current, select, op.get("area"), op.get("index"), current["yourIndex"]),
                _AREA_NAMES.get(op.get("inPlayArea")), op.get("inPlayIndex"))
    if t in (10, 11):  # ABILITY / DISCARD (in-play card)
        name = "ABILITY" if t == 10 else "DISCARD_INPLAY"
        return (name, _AREA_NAMES.get(op.get("area")), op.get("index"),
                _ref_id(current, select, op.get("area"), op.get("index"), pi))
    if t == 13:  # ATTACK
        return ("ATTACK", op.get("attackId"))
    if t == 12:  # RETREAT
        return ("RETREAT",)
    if t == 14:  # END
        return ("END",)
    if t in (1, 2):  # YES / NO
        return ("YES" if t == 1 else "NO",)
    if t == 0:  # NUMBER
        return ("NUMBER", op.get("number"))
    if t == 16:  # SPECIAL_CONDITION
        return ("SPECIAL_CONDITION", op.get("specialConditionType"))
    if t in (5, 6) and op.get("area") in (4, 5) and "energyIndex" in op:
        # Attached energy choices can be encoded by cabt as ENERGY_CARD when an
        # effect moves/discards a concrete card, but native represents the same
        # selection as an ENERGY ref with the attached card id.
        area = op.get("area")
        index = op.get("index")
        energy_index = op.get("energyIndex")
        label = _AREA_NAMES.get(area)
        card_id = _energy_card_id(current, area, index, energy_index, pi)
        energy_ref_index = 0 if energy_index is None else int(energy_index)
        if area == 5:
            return ("ENERGY", label, (index + 1) * 1000 + energy_ref_index,
                    index, card_id)
        return ("ENERGY", label, energy_ref_index, card_id)
    if t == 6:  # ENERGY attached to an in-play Pokemon
        return ("ENERGY", _AREA_NAMES.get(op.get("area")), op.get("index"),
                _ref_id(current, select, op.get("area"), op.get("index"), pi))
    if t in (3, 4, 5):  # CARD / TOOL_CARD / ENERGY_CARD
        kind = {3: "CARD", 4: "TOOL_CARD", 5: "ENERGY_CARD"}[t]
        return (kind, _AREA_NAMES.get(op.get("area")), op.get("index"),
                _ref_id(current, select, op.get("area"), op.get("index"), pi))
    if t == 15:  # SKILL order
        return ("SKILL", op.get("cardId"), op.get("serial"))
    # fallback: raw, sorted
    return (t,) + tuple(sorted((k, v) for k, v in op.items() if k != "type"))


def canonical_options(current: dict | None, select: dict | None) -> list[tuple]:
    """Ordered descriptors for the legal options (order = cabt option order)."""
    if select is None:
        return []
    return [option_descriptor(current, select, op) for op in select.get("option", [])]


# --- structural diff ------------------------------------------------------

def diff(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Recursively compare two canonical structures; return (path, a, b) diffs."""
    out: list[tuple[str, Any, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            out += diff(a.get(k, _MISSING), b.get(k, _MISSING), f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((f"{path}[len]", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path or ".", a, b))
    return out


_MISSING = object()
