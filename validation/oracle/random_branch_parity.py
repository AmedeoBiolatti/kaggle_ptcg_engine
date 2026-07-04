"""Random-game branch parity against cabt.

This is a generative differential test for the custom engine:

* play reachable cabt games from two 60-card decks,
* at each sampled MAIN decision compare the semantic legal-action set,
* for each offered action, branch cabt through the Search API and apply the
  same semantic action to a cloned native state,
* compare the resulting canonical state and next decision.

cabt owns the real game RNG, so the test is intentionally policy-seeded rather
than bit-for-bit game-seeded. Hidden zones are determinized from the two deck
lists for the branch search, and the same determinization is injected into the
native state. Coin outcomes and draw/reveal identities are replayed from cabt's
branch logs through the native replay tape.

Usage:
    python validation/oracle/random_branch_parity.py --deck0 mega_lucario \
        --deck1 sample_submission --games 10

    python validation/oracle/random_branch_parity.py --random-decks 16 \
        --games 2 --skip-deck-search-branches
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_DIR = os.path.join(ROOT, "engine")
BUILD_DIR = os.path.join(ENGINE_DIR, "build")
for _p in (ROOT, BUILD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ptcg_engine as eng  # noqa: E402
from cg.api import all_card_data, search_begin, search_end, search_release, search_step, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from validation.decks import ALL_DECKS, generate_random_decks  # noqa: E402
from validation.oracle.canonical import canonical_options, canonical_state, diff  # noqa: E402

MAIN_CONTEXT = 0
LOG_DRAW = 4
LOG_MOVE = 6
LOG_SWITCH = 8
LOG_ATTACH_ENERGY = 11
LOG_HP_CHANGE = 16
LOG_COIN = 22
LOG_COIN_HEAD = 46
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_TOOL = 9
CTX_PREVENT_TARGET = 25
REPLAY_RANDOM_OPP_HAND_TO_DECK = -100000000
REPLAY_COIN = -100000001
REPLAY_HAND_INDEX = -100000002
REPLAY_RANDOM_DISCARD_HAND = -100000003
REPLAY_DRAW_PLAYER = -100000004
FILLER_BASIC = 1072  # Snorlax, a legal Basic Pokemon.
FILLER_ENERGY = 1    # Basic Grass Energy.
MIST_ENERGY = 11
DPC_ALL = 0
DPC_ATTACKER_BASIC = 1
DPC_ATTACKER_BASIC_NON_COLORLESS = 2
DPC_ATTACKER_EX = 3
DPC_ATTACKER_HAS_ABILITY = 4
DPC_DAMAGE_LE = 5
DPC_DAMAGE_GE = 6
DPC_ATTACKER_HAS_SPECIAL_ENERGY = 7
ITEM_LOCK_ATTACKS = {210, 323, 858}
SUPPORTER_LOCK_ATTACKS = {1393}
EVOLVE_LOCK_ATTACKS = {58}
STADIUM_LOCK_ATTACKS = {1039}
SPECIAL_POISON_COUNTER_ATTACKS = {642: 8}  # Team Rocket's Nidoking ex.
OPP_NO_RETREAT_ATTACKS = {
    46, 54, 135, 172, 191, 239, 302, 315, 351, 398, 462, 527, 648,
    714, 775, 891, 939, 997, 1183, 1254, 1268, 1402, 1412, 1434,
    1456, 1461, 1462,
}
SELF_NO_RETREAT_ATTACKS = {658, 1440}
OPP_TAKE_MORE_DAMAGE_ATTACKS = {
    249: 50,    # Vibrava: Screech.
    1254: 100, # Stunfisk: Pouncing Trap.
}
OPP_ATTACK_RETREAT_COST_MORE_ATTACKS = {112: 1}
OPP_ATTACK_FLIP_FAIL_ATTACKS = {139, 971, 1513}
COIN_PREVENT_DAMAGE_ATTACKS = {261, 505, 595, 684, 788, 790, 1205}
COIN_PREVENT_DAMAGE_EFFECTS_ATTACKS = {75, 244, 1054, 1266, 1309, 1382, 1470}
DETERMINISTIC_PREVENT_DAMAGE_ATTACKS = {
    233: (DPC_ATTACKER_BASIC_NON_COLORLESS, 0),
    349: (DPC_DAMAGE_LE, 60),
    860: (DPC_DAMAGE_LE, 40),
    1064: (DPC_ATTACKER_BASIC, 0),
    1212: (DPC_ATTACKER_BASIC, 0),
    1327: (DPC_ATTACKER_BASIC, 0),
}
SELF_DAMAGE_REDUCE_ATTACKS = {
    78: 30,
    100: 50,
    186: 30,
    411: 60,
    416: 30,
    570: 50,
    584: 10,
    780: 30,
    849: 40,
    896: 10,
    897: 20,
    943: 30,
    1047: 30,
    1204: 60,
    1279: 20,
    1426: 50,
    1525: 30,
}
OPP_ATTACK_REDUCE_ATTACKS = {
    427: 20,
    440: 100,
    1095: 20,
    1168: 30,
    1322: 20,
}
RANDOM_OPP_HAND_TO_DECK_ATTACKS = {130, 523, 663, 1087, 1216}
RANDOM_DISCARD_HAND_ATTACKS = {340, 611, 876, 1292, 1293, 1418, 1554}
RANDOM_DISCARD_HAND_ABILITIES = {1024}
KO_LAST_TURN_GATE_CARDS = {1080, 1193}  # Unfair Stamp, Hassel.
ATTACK_DAMAGE_KO_LAST_TURN_GATE_CARDS = {1193}  # Hassel.
TEAM_ROCKET_KO_LAST_TURN_GATE_CARDS = {1217}  # Team Rocket's Archer.
KO_LAST_TURN_GATE_ABILITIES = {140}  # Fezandipiti ex.
DELAYED_DAMAGE_ATTACKS = {
    21: 9,  # Glaceon
}
DELAYED_KO_ATTACKS = {
    8,    # Pinsir
    647,  # Team Rocket's Grimer
}
IF_DAMAGED_COUNTER_ATTACKS = {
    319: 6,  # Bouffalant: Ready to Ram
}
IF_DAMAGED_DAMAGE_DONE_ATTACKS = {
    660,  # Heracross: Guard Press-style retaliation
}
ENERGY_ATTACH_COUNTER_ATTACKS = {
    1066: (8, True),  # Pachirisu: opponent hand attach to Defending Pokemon.
}
NAMED_ATTACK_NEXT_TURN: dict[int, tuple[int, int, int]] = {
    # attack used -> (attack receiving the next-turn effect, bonus, set_base)
    749: (749, 80, -1),    # Meloetta ex: Echoed Voice
    905: (906, 0, 240),    # Watchog: Focus Energy -> Hyper Fang base damage
    1296: (1296, 50, -1),  # Pecharunt: Mochi Rush
}

CARD = {c.cardId: c for c in all_card_data()}
ANCIENT_CARD_IDS = {
    35, 46, 56, 58, 61, 62, 63, 171, 226, 312, 969, 986, 1085, 1185,
}
ITEM_CARD_TYPE = 1
TOOL_CARD_TYPE = 2
SUPPORTER_CARD_TYPE = 3
STADIUM_CARD_TYPE = 4


class ParityFailure(AssertionError):
    """Raised when cabt and the native engine diverge."""


@dataclass
class Determinization:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


@dataclass
class PlayerHidden:
    hand: list[int]
    deck: list[int]
    prize: list[int]
    active: list[int]


@dataclass
class ParityStats:
    games: int = 0
    battle_steps: int = 0
    states_checked: int = 0
    branches_checked: int = 0
    branches_replayed_coin: int = 0
    branches_skipped_coin: int = 0
    branches_skipped_missing_native: int = 0
    branches_skipped_deck_search: int = 0
    branches_skipped_limit: int = 0
    action_kinds: Counter[str] = field(default_factory=Counter)
    next_contexts: Counter[int | str] = field(default_factory=Counter)

    def merge(self, other: "ParityStats") -> None:
        self.games += other.games
        self.battle_steps += other.battle_steps
        self.states_checked += other.states_checked
        self.branches_checked += other.branches_checked
        self.branches_replayed_coin += other.branches_replayed_coin
        self.branches_skipped_coin += other.branches_skipped_coin
        self.branches_skipped_missing_native += other.branches_skipped_missing_native
        self.branches_skipped_deck_search += other.branches_skipped_deck_search
        self.branches_skipped_limit += other.branches_skipped_limit
        self.action_kinds.update(other.action_kinds)
        self.next_contexts.update(other.next_contexts)

    def summary(self) -> str:
        kinds = ", ".join(f"{k}:{v}" for k, v in self.action_kinds.most_common())
        ctx = ", ".join(f"{k}:{v}" for k, v in self.next_contexts.most_common())
        return (
            f"games={self.games} battle_steps={self.battle_steps} "
            f"states_checked={self.states_checked} branches_checked={self.branches_checked} "
            f"replayed_coin={self.branches_replayed_coin} "
            f"skipped_coin={self.branches_skipped_coin} "
            f"skipped_missing_native={self.branches_skipped_missing_native} "
            f"skipped_deck_search={self.branches_skipped_deck_search} "
            f"skipped_limit={self.branches_skipped_limit}\n"
            f"action_kinds={{{kinds}}}\n"
            f"next_contexts={{{ctx}}}"
        )


@dataclass(frozen=True)
class DeckPair:
    name0: str
    deck0: list[int]
    name1: str
    deck1: list[int]


def load_deck(spec: str) -> list[int]:
    """Load a named validation deck, or a plain text/csv file of card IDs."""
    if spec in ALL_DECKS:
        return list(ALL_DECKS[spec])
    if not os.path.exists(spec):
        names = ", ".join(sorted(ALL_DECKS))
        raise ValueError(f"unknown deck {spec!r}; known decks: {names}")
    with open(spec, encoding="utf-8") as f:
        raw = f.read().replace(",", " ").split()
    deck = [int(x) for x in raw]
    if len(deck) != 60:
        raise ValueError(f"{spec!r} has {len(deck)} cards, expected 60")
    return deck


def pair_generated_decks(decks: list[list[int]]) -> list[DeckPair]:
    """Pair generated decks deterministically for parity games."""
    if not decks:
        raise ValueError("at least one generated deck is required")
    if len(decks) == 1:
        return [DeckPair("random_000", decks[0], "random_000", decks[0])]

    pairs: list[DeckPair] = []
    for i in range(0, len(decks), 2):
        j = i + 1 if i + 1 < len(decks) else 0
        pairs.append(DeckPair(
            f"random_{i:03d}",
            decks[i],
            f"random_{j:03d}",
            decks[j],
        ))
    return pairs


def _is_basic_pokemon(card_id: int) -> bool:
    c = CARD.get(card_id)
    return bool(c and c.basic)


def _card_ids(cards: list[dict[str, Any]] | None) -> list[int]:
    return [int(c["id"]) for c in (cards or []) if c is not None]


def _in_play_ids(player_state: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    mons = list(player_state.get("active") or []) + list(player_state.get("bench") or [])
    for mon in mons:
        if mon is None:
            continue
        ids.append(int(mon["id"]))
        ids.extend(_card_ids(mon.get("energyCards")))
        ids.extend(_card_ids(mon.get("tools")))
        ids.extend(_card_ids(mon.get("preEvolution")))
    return ids


def _visible_card_ids(current: dict[str, Any], player: int) -> Counter[int]:
    ps = current["players"][player]
    seen: Counter[int] = Counter()
    seen.update(_in_play_ids(ps))
    seen.update(_card_ids(ps.get("discard")))
    seen.update(_card_ids(ps.get("hand")))
    seen.update(_card_ids(ps.get("prize")))
    for stadium in current.get("stadium") or []:
        if stadium is not None and stadium.get("playerIndex") == player:
            seen[int(stadium["id"])] += 1
    for card in current.get("looking") or []:
        if card is not None and card.get("playerIndex") == player:
            seen[int(card["id"])] += 1
    return seen


def _remaining_pool(deck: list[int], current: dict[str, Any], player: int) -> list[int]:
    pool = Counter(deck)
    pool.subtract(_visible_card_ids(current, player))
    return [card_id for card_id, n in pool.items() for _ in range(max(0, n))]


def _choose_basic(pool: list[int]) -> int:
    for i, card_id in enumerate(pool):
        if _is_basic_pokemon(card_id):
            return pool.pop(i)
    return FILLER_BASIC


def _partition_hidden(deck: list[int], current: dict[str, Any], player: int,
                      rng: random.Random) -> PlayerHidden:
    ps = current["players"][player]
    deck_n = int(ps["deckCount"])
    prize_n = len(ps.get("prize") or [])
    hidden_hand_n = 0 if ps.get("hand") is not None else int(ps["handCount"])
    active_facedown = bool(ps.get("active")) and ps["active"][0] is None

    pool = _remaining_pool(deck, current, player)
    need = deck_n + prize_n + hidden_hand_n + (1 if active_facedown else 0)
    if len(pool) < need:
        pool.extend([FILLER_ENERGY] * (need - len(pool)))
    rng.shuffle(pool)

    active = [_choose_basic(pool)] if active_facedown else []
    hand = pool[:hidden_hand_n]
    deck_ids = pool[hidden_hand_n:hidden_hand_n + deck_n]
    prize = pool[hidden_hand_n + deck_n:hidden_hand_n + deck_n + prize_n]

    # The Search API is strict around setup-like states. Keep at least one Basic
    # available in non-empty hidden decks when the remaining pool permits it.
    if deck_ids and not any(_is_basic_pokemon(c) for c in deck_ids):
        for i, card_id in enumerate(hand + prize):
            if _is_basic_pokemon(card_id):
                if i < len(hand):
                    hand[i], deck_ids[0] = deck_ids[0], hand[i]
                else:
                    j = i - len(hand)
                    prize[j], deck_ids[0] = deck_ids[0], prize[j]
                break
        else:
            deck_ids[0] = FILLER_BASIC

    return PlayerHidden(hand=hand, deck=deck_ids, prize=prize, active=active)


def _determinize(current: dict[str, Any], deck0: list[int], deck1: list[int],
                 rng: random.Random) -> tuple[Determinization, dict[int, PlayerHidden]]:
    decks = {0: deck0, 1: deck1}
    hidden = {p: _partition_hidden(decks[p], current, p, rng) for p in (0, 1)}
    me = int(current["yourIndex"])
    opp = 1 - me
    det = Determinization(
        your_deck=hidden[me].deck,
        your_prize=hidden[me].prize,
        opponent_deck=hidden[opp].deck,
        opponent_prize=hidden[opp].prize,
        opponent_hand=hidden[opp].hand,
        opponent_active=hidden[opp].active,
    )
    return det, hidden


def _card_dicts(card_ids: list[int], player: int) -> list[dict[str, int]]:
    return [{"id": int(card_id), "playerIndex": player, "serial": 0} for card_id in card_ids]


def _energy_order_key(card: Any, fallback_player: int) -> tuple[int, int, int | None] | None:
    if not isinstance(card, dict) or card.get("id") is None:
        return None
    player = int(card.get("playerIndex", fallback_player))
    serial = card.get("serial")
    return (player, int(card["id"]), int(serial) if serial is not None else None)


def _tool_order_key(card: Any, fallback_player: int) -> tuple[int, int, int | None] | None:
    return _energy_order_key(card, fallback_player)


def _apply_energy_attach_order_transients(
    cur: dict[str, Any],
    attach_order: dict[tuple[int, int, int | None], int],
) -> None:
    if not attach_order:
        return
    for player, ps in enumerate(cur.get("players") or []):
        mons = list(ps.get("active") or []) + list(ps.get("bench") or [])
        for mon in mons:
            if not isinstance(mon, dict):
                continue
            for energy in mon.get("energyCards") or []:
                key = _energy_order_key(energy, player)
                if key in attach_order:
                    energy["attachOrder"] = int(attach_order[key])


def _apply_tool_attach_order_transients(
    cur: dict[str, Any],
    attach_order: dict[tuple[int, int, int | None], int],
) -> None:
    if not attach_order:
        return
    for player, ps in enumerate(cur.get("players") or []):
        mons = list(ps.get("active") or []) + list(ps.get("bench") or [])
        for mon in mons:
            if not isinstance(mon, dict):
                continue
            for tool in mon.get("tools") or []:
                key = _tool_order_key(tool, player)
                if key in attach_order:
                    tool["attachOrder"] = int(attach_order[key])


def _apply_moved_to_active_transients(
    cur: dict[str, Any],
    moved: list[list[tuple[int, int | None]]] | None,
) -> None:
    if not moved:
        return
    for player, identities in enumerate(moved):
        active = ((cur.get("players") or [])[player].get("active") or [None])[0]
        if not isinstance(active, dict):
            continue
        for identity in identities:
            if _cards_match_identity(active, tuple(identity)):
                active["movedToActiveThisTurn"] = True
                break


def _apply_healed_this_turn_transients(
    cur: dict[str, Any],
    healed: list[list[tuple[int, int | None]]] | None,
) -> None:
    if not healed:
        return
    for player, identities in enumerate(healed):
        for card in _iter_inplay_cards(cur["players"][player]):
            for identity in identities:
                if _cards_match_identity(card, tuple(identity)):
                    card["healedThisTurn"] = True
                    break


def _state_for_native(current: dict[str, Any], hidden: dict[int, PlayerHidden],
                      transients: dict[str, Any] | None = None) -> dict[str, Any]:
    cur = copy.deepcopy(current)
    if transients is not None:
        for key in ("lastKoTurn", "lastAttackDamageKoTurn",
                    "lastTeamRocketKoTurn", "lastEthanKoTurn",
                    "lastAttackTurn", "lastAttackId",
                    "lastAncientAttackTurn", "lastAncientAttackCard",
                    "lastAncientAttackSerial",
                    "prizeTakenTurn", "prizeTakenCount",
                    "noItemTurn", "noSupporterTurn", "noEvolveTurn",
                    "noStadiumTurn", "teamReduceTurn", "teamReduceAmount",
                    "teamReduceType", "activeExDamageBuffTurn",
                    "activeExDamageBuffAmount", "prizeBonusTurn",
                    "prizeBonusAmount", "prizeBonusKind"):
            if key in transients:
                cur[key] = list(transients[key])
    if transients is not None and \
            int(transients.get("turn", -1)) == int(current["turn"]) and \
            int(transients.get("player", -1)) == int(current["yourIndex"]):
        fighting_buff = int(transients.get("fightingBuff", 0))
        if fighting_buff:
            cur["fightingBuff"] = fighting_buff
        cur["stadiumAbilityUsed"] = bool(transients.get("stadiumAbilityUsed", False))
        cur["supporterPlayed"] = bool(
            cur.get("supporterPlayed", False) or
            transients.get("supporterPlayed", False)
        )
        cur["teamRocketSupporterPlayed"] = bool(
            transients.get("teamRocketSupporterPlayed", False)
        )
        if "ancientSupporterPlayed" not in cur:
            cur["ancientSupporterPlayed"] = bool(
                transients.get("ancientSupporterPlayed", False)
            )
        cur["canariPlayed"] = bool(transients.get("canariPlayed", False))
        cur["tarragonPlayed"] = bool(transients.get("tarragonPlayed", False))
    for player, h in hidden.items():
        ps = cur["players"][player]
        ps["deck"] = list(h.deck)
        # This is a determinized hidden deck for replay, not a public/exact
        # CABT deck identity. Native legal gates must not treat a sampled deck
        # as proof that a search target is absent.
        ps["deckKnown"] = False
        ps["deckKnownMask"] = [False] * len(ps["deck"])
        ps["deckKnownCards"] = []
        if ps.get("hand") is None:
            ps["hand"] = _card_dicts(h.hand, player)
        else:
            ps["handCount"] = len(ps["hand"])
        prize = ps.get("prize")
        if prize is not None and all(card is None for card in prize):
            ps["prize"] = _card_dicts(h.prize, player)
        if h.active and ps.get("active") and ps["active"][0] is None:
            card_id = h.active[0]
            info = CARD.get(card_id)
            ps["active"][0] = {
                "id": card_id,
                "playerIndex": player,
                "serial": 0,
                "hp": int(info.hp if info else 0),
                "maxHp": int(info.hp if info else 0),
                "appearThisTurn": True,
                "energies": [],
                "energyCards": [],
                "tools": [],
                "preEvolution": [],
            }
    if transients is not None:
        _apply_energy_attach_order_transients(
            cur, transients.get("energyAttachOrder") or {})
        _apply_tool_attach_order_transients(
            cur, transients.get("toolAttachOrder") or {})
        poison_counters = transients.get("poisonDamageCounters") or []
        for player, counters in enumerate(poison_counters[:2]):
            if int(counters) <= 1:
                continue
            ps = cur["players"][player]
            if ps.get("poisoned"):
                ps["poisonDamageCounters"] = int(counters)
        _apply_moved_to_active_transients(
            cur, transients.get("movedToActiveThisTurn"))
        _apply_healed_this_turn_transients(
            cur, transients.get("healedThisTurn"))
        _apply_inplay_effect_transients(cur, transients.get("inplayEffects") or [])
        _apply_attack_damage_transients(cur, transients.get("attackDamage") or [])
    return cur


def _card_identity(card: Any) -> tuple[int, int | None] | None:
    if not isinstance(card, dict) or card.get("id") is None:
        return None
    serial = card.get("serial")
    return int(card["id"]), int(serial) if serial is not None else None


def _cards_match_identity(card: dict[str, Any], identity: tuple[int, int | None]) -> bool:
    cid, serial = identity
    if int(card.get("id", -1)) != cid:
        return False
    if serial is None:
        return True
    return card.get("serial") is not None and int(card["serial"]) == serial


def _apply_inplay_effect_transients(cur: dict[str, Any],
                                    effects: list[dict[str, Any]]) -> None:
    current_turn = int(cur.get("turn", -1))
    for eff in _prune_inplay_effects(effects, current_turn):
        owner = int(eff["owner"])
        identity = tuple(eff["identity"])
        player = cur["players"][owner]
        area = eff.get("area")
        if area == "ACTIVE":
            cards = list(player.get("active") or [])
        elif area == "BENCH" and eff.get("index") is not None:
            bench = list(player.get("bench") or [])
            idx = int(eff["index"])
            cards = [bench[idx]] if 0 <= idx < len(bench) else []
        else:
            cards = list(player.get("active") or []) + list(player.get("bench") or [])
        for card in cards:
            if not isinstance(card, dict) or not _cards_match_identity(card, identity):
                continue
            for key in ("preventDmgTurn", "preventDmgCond", "preventDmgValue",
                        "preventEffectsTurn", "preventEffectsCond",
                        "preventEffectsValue", "dmgReduce",
                        "dmgReduceTurn", "takeMoreDamageTurn",
                        "takeMoreDamage", "noRetreatTurn",
                        "retreatCostModTurn", "retreatCostMod",
                        "attackCostModTurn", "attackCostMod",
                        "attackFlipFailTurn", "noWeaknessTurn",
                        "delayedDamageTurn", "delayedDamageCounters",
                        "delayedKoTurn", "delayedKoPromoteBeforePrize",
                        "nextAttackBonusId",
                        "nextAttackBonusTurn", "nextAttackBonus",
                        "nextAttackSetBase", "attackDmgReduce",
                        "attackDmgReduceTurn", "damagedByAttackCountersTurn",
                        "damagedByAttackCounters",
                        "damagedByAttackEqualCountersTurn",
                        "damagedByAttackStatus",
                        "energyAttachCountersTurn",
                        "energyAttachCounters",
                        "energyAttachCountersFromHandOnly"):
                if key in eff:
                    card[key] = eff[key]


def _prune_inplay_effects(
    effects: list[dict[str, Any]],
    current_turn: int,
) -> list[dict[str, Any]]:
    if current_turn < 0:
        return list(effects)
    kept: list[dict[str, Any]] = []
    for eff in effects:
        turns = [
            int(value)
            for key, value in eff.items()
            if key.endswith("Turn") and value is not None
        ]
        if not turns or max(turns) >= current_turn:
            kept.append(eff)
    return kept


def _iter_inplay_cards(player_state: dict[str, Any]):
    for card in list(player_state.get("active") or []) + list(player_state.get("bench") or []):
        if isinstance(card, dict) and card.get("id") is not None:
            yield card


def _inplay_card_for_option(current: dict[str, Any], owner: int,
                            desc: tuple[Any, ...]) -> dict[str, Any] | None:
    if len(desc) < 3 or desc[0] != "CARD":
        return None
    player = current["players"][owner]
    area = desc[1]
    idx = int(desc[2])
    if area == "ACTIVE":
        active = player.get("active") or []
        return active[0] if active and isinstance(active[0], dict) else None
    if area == "BENCH":
        bench = player.get("bench") or []
        if 0 <= idx < len(bench) and isinstance(bench[idx], dict):
            return bench[idx]
    return None


def _apply_attack_damage_transients(cur: dict[str, Any],
                                    effects: list[dict[str, Any]]) -> None:
    current_turn = int(cur.get("turn", -1))
    for eff in effects:
        turn = int(eff.get("turn", -999))
        if current_turn >= 0 and turn < current_turn - 1:
            continue
        owner = int(eff["owner"])
        identity = tuple(eff["identity"])
        for card in _iter_inplay_cards(cur["players"][owner]):
            if not _cards_match_identity(card, identity):
                continue
            card["damagedByAttackTurn"] = turn
            card["damagedByAttackSide"] = int(eff["attacker"])
            card["damagedByAttackAmount"] = int(eff["amount"])


def _clear_inplay_effects_for_changed_active(transients: dict[str, Any],
                                             before: dict[str, Any] | None,
                                             after: dict[str, Any] | None,
                                             logs: list[Any] | None = None) -> None:
    if before is None or after is None or not transients.get("inplayEffects"):
        return
    active_after = {}
    inplay_after = {}
    for owner in (0, 1):
        after_active = (after["players"][owner].get("active") or [None])[0]
        active_after[owner] = _card_identity(after_active)
        inplay_after[owner] = [
            _card_identity(card)
            for card in _iter_inplay_cards(after["players"][owner])
        ]
    ambiguous_active_change = {0: False, 1: False}
    for log in logs or []:
        owner = int(_log_value(log, "playerIndex", -1))
        if owner not in (0, 1):
            continue
        typ = int(_log_value(log, "type", -1))
        if typ == LOG_SWITCH:
            ambiguous_active_change[owner] = True
        elif typ == LOG_MOVE and (
                int(_log_value(log, "fromArea", -1)) == AREA_ACTIVE or
                int(_log_value(log, "toArea", -1)) == AREA_ACTIVE):
            ambiguous_active_change[owner] = True

    kept = []
    for eff in transients.get("inplayEffects") or []:
        owner = int(eff.get("owner", -1))
        identity = tuple(eff.get("identity", (None, None)))
        if owner not in (0, 1):
            kept.append(eff)
            continue
        has_prevention = any(
            key in eff
            for key in ("preventDmgTurn", "preventEffectsTurn",
                        "dmgReduceTurn", "attackDmgReduceTurn")
        )
        target_still_present = (
            identity in inplay_after.get(owner, [])
            if has_prevention else active_after.get(owner) == identity
        )
        if target_still_present and not (
                identity[1] is None and ambiguous_active_change[owner]):
            kept.append(eff)
    transients["inplayEffects"] = kept


def _norm_desc(desc: Any, select_context: int | None = None,
               current: dict[str, Any] | None = None) -> tuple:
    t = tuple(desc)
    if t and t[0] == "DISCARD_INPLAY":
        t = ("DISCARD",) + t[1:]
    if len(t) >= 2 and t[0] == "CARD" and t[1] == "LOOKING":
        label = "HAND" if select_context == 10 else "DECK"
        t = ("CARD", label) + t[2:]
    if len(t) >= 4 and t[0] == "CARD" and t[1] == "ACTIVE" and t[2] == -1:
        return t[:2] + (0,) + t[3:]
    if len(t) >= 4 and t[0] == "CARD" and t[1] == "HAND" and select_context == 8:
        return t[:2] + (t[3],)
    if len(t) >= 4 and t[0] == "CARD" and (select_context == 11 or t[1] == "PRIZE"):
        return t[:3] + (None,)
    if len(t) >= 4 and t[0] == "CARD" and t[1] == "DECK" and \
            select_context is not None and select_context != MAIN_CONTEXT:
        return t[:2] + (t[3],)
    if len(t) >= 3 and t[0] == "SKILL" and select_context == 34:
        return t[:2] + (None,)
    if select_context == 27 and len(t) >= 4 and t[0] == "CARD" and \
            t[1] in ("ACTIVE", "BENCH") and isinstance(t[2], int) and \
            t[2] >= 300000 and current is not None:
        raw = int(t[2]) - 300000
        owner_offset = raw // 100000
        local = raw % 100000
        inplay_idx = local // 1000 - 1
        actor = int(current["yourIndex"])
        owner = actor if owner_offset == 0 else 1 - actor
        player = current["players"][owner]
        public_idx = 0 if t[1] == "ACTIVE" else inplay_idx
        seq = player.get("active") if t[1] == "ACTIVE" else player.get("bench")
        holder = None
        if seq is not None and 0 <= public_idx < len(seq):
            card = seq[public_idx]
            if isinstance(card, dict) and card.get("id") is not None:
                holder = int(card["id"])
        return ("TOOL_CARD", t[1], public_idx, holder)
    if len(t) == 4 and t[0] == "DISCARD" and t[3] is None:
        return t[:3]
    if len(t) >= 4 and t[0] == "ABILITY" and t[1] == "STADIUM":
        return t[:3] + (None,)
    return t


def _raw_cabt_deck_index_only(raw_cabt: list[Any] | None,
                              select_context: int | None) -> bool:
    if not raw_cabt or select_context is None or select_context == MAIN_CONTEXT:
        return False
    for raw in raw_cabt:
        if not isinstance(raw, dict):
            return False
        if int(_log_value(raw, "type", -1)) != 3:
            return False
        if int(_log_value(raw, "area", -1)) != 1:
            return False
        if _log_value(raw, "index") is None:
            return False
        if _log_value(raw, "cardId") is not None:
            return False
    return True


def _deck_index_norm_from_raw(raw_cabt: list[Any]) -> list[tuple]:
    return [("CARD", "DECK", int(_log_value(raw, "index", -1)))
            for raw in raw_cabt]


def _deck_index_norm_from_options(options: list[tuple]) -> list[tuple]:
    out: list[tuple] = []
    for desc in options:
        t = tuple(desc)
        if len(t) >= 3 and t[0] == "CARD" and t[1] == "DECK":
            out.append(("CARD", "DECK", int(t[2])))
        else:
            out.append(t)
    return out


def _is_stadium_ability_desc(desc: Any) -> bool:
    t = tuple(desc)
    return len(t) >= 3 and t[0] == "ABILITY" and t[1] == "STADIUM"


def _option_counter(options: list[tuple], strict_duplicates: bool) -> Counter | set:
    return Counter(options) if strict_duplicates else set(options)


def _descriptor_source_card(current: dict[str, Any] | None, desc: tuple) -> int | None:
    if current is None or len(desc) < 3:
        return None
    kind = desc[0]
    if kind not in {"ABILITY", "DISCARD", "DISCARD_INPLAY"}:
        return None
    area = desc[1]
    idx = desc[2]
    if not isinstance(idx, int):
        return None
    if area == "STADIUM":
        seq = current.get("stadium") or []
    else:
        player = current["players"][int(current["yourIndex"])]
        if area == "ACTIVE":
            seq = player.get("active") or []
        elif area == "BENCH":
            seq = player.get("bench") or []
        else:
            return None
    if idx < 0 or idx >= len(seq) or seq[idx] is None:
        return None
    card = seq[idx]
    return int(card["id"]) if isinstance(card, dict) and card.get("id") is not None else None


def _option_debug(options: list[tuple], current: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    for option in options[:20]:
        if len(option) >= 2 and option[0] == "PLAY":
            detail = _play_option_debug(current, int(option[1]))
            out.append(f"{option!r}{detail}")
            continue
        if len(option) >= 2 and option[0] == "ATTACK":
            detail = _attack_option_debug(current, int(option[1]))
            out.append(f"{option!r}{detail}")
            continue
        source = _descriptor_source_card(current, option)
        if source is None:
            out.append(repr(option))
        else:
            out.append(f"{option!r} source_card={source}")
    return out


def _card_id(card: Any) -> int | None:
    if isinstance(card, dict) and card.get("id") is not None:
        return int(card["id"])
    return None


def _has_opponent_stadium(current: dict[str, Any], actor: int) -> bool:
    stadium = current.get("stadium") or []
    if not stadium:
        return False
    owner = int(stadium[0].get("playerIndex", actor))
    return owner == 1 - actor


def _player_has_inplay_card(current: dict[str, Any], actor: int,
                            card_ids: set[int]) -> bool:
    player = current["players"][actor]
    for card in (player.get("active") or []):
        cid = _card_id(card)
        if cid in card_ids:
            return True
    for card in (player.get("bench") or []):
        cid = _card_id(card)
        if cid in card_ids:
            return True
    return False


def _has_source_card_option(current: dict[str, Any], options: list[tuple],
                            kind: str, card_ids: set[int]) -> bool:
    for option in options:
        if not option or option[0] != kind:
            continue
        source = _descriptor_source_card(current, option)
        if source in card_ids:
            return True
    return False


def _sync_ko_gate_transients_from_main_options(transients: dict[str, Any],
                                               current: dict[str, Any],
                                               select: dict[str, Any]) -> None:
    if int(select.get("context", MAIN_CONTEXT)) != MAIN_CONTEXT:
        return
    actor = int(current["yourIndex"])
    turn = int(current["turn"])
    hand_ids = set(
        cid for card in (current["players"][actor].get("hand") or [])
        if (cid := _card_id(card)) is not None
    )
    options = canonical_options(current, select)
    option_set = set(options)
    transients.setdefault("lastKoTurn", [-1, -1])
    transients.setdefault("lastKoTurnSource", ["", ""])
    transients.setdefault("lastAttackDamageKoTurn", [-1, -1])
    transients.setdefault("lastAttackDamageKoTurnSource", ["", ""])
    transients.setdefault("lastTeamRocketKoTurn", [-1, -1])
    transients.setdefault("lastTeamRocketKoTurnSource", ["", ""])
    if hand_ids & KO_LAST_TURN_GATE_CARDS:
        if any(("PLAY", card_id) in option_set for card_id in KO_LAST_TURN_GATE_CARDS):
            transients["lastKoTurn"][actor] = turn - 1
            transients["lastKoTurnSource"][actor] = "options"
        elif transients["lastKoTurn"][actor] == turn - 1:
            transients["lastKoTurn"][actor] = -1
            transients["lastKoTurnSource"][actor] = ""
    if hand_ids & ATTACK_DAMAGE_KO_LAST_TURN_GATE_CARDS:
        if any(("PLAY", card_id) in option_set
               for card_id in ATTACK_DAMAGE_KO_LAST_TURN_GATE_CARDS):
            transients["lastAttackDamageKoTurn"][actor] = turn - 1
            transients["lastAttackDamageKoTurnSource"][actor] = "options"
        elif transients["lastAttackDamageKoTurn"][actor] == turn - 1:
            transients["lastAttackDamageKoTurn"][actor] = -1
            transients["lastAttackDamageKoTurnSource"][actor] = ""
    if _player_has_inplay_card(current, actor, KO_LAST_TURN_GATE_ABILITIES):
        if _has_source_card_option(current, options, "ABILITY",
                                   KO_LAST_TURN_GATE_ABILITIES):
            transients["lastKoTurn"][actor] = turn - 1
            transients["lastKoTurnSource"][actor] = "options"
        elif transients["lastKoTurn"][actor] == turn - 1 and \
                transients["lastKoTurnSource"][actor] == "options":
            transients["lastKoTurn"][actor] = -1
            transients["lastKoTurnSource"][actor] = ""
    if hand_ids & TEAM_ROCKET_KO_LAST_TURN_GATE_CARDS:
        if any(("PLAY", card_id) in option_set
               for card_id in TEAM_ROCKET_KO_LAST_TURN_GATE_CARDS):
            transients["lastTeamRocketKoTurn"][actor] = turn - 1
            transients["lastTeamRocketKoTurnSource"][actor] = "options"
        elif transients["lastTeamRocketKoTurn"][actor] == turn - 1:
            transients["lastTeamRocketKoTurn"][actor] = -1
            transients["lastTeamRocketKoTurnSource"][actor] = ""


def _sync_action_lock_transients_from_main_options(transients: dict[str, Any],
                                                   current: dict[str, Any],
                                                   select: dict[str, Any]) -> None:
    if int(select.get("context", MAIN_CONTEXT)) != MAIN_CONTEXT:
        return
    actor = int(current["yourIndex"])
    turn = int(current["turn"])
    options = canonical_options(current, select)
    for option in options:
        if not option:
            continue
        if option[0] == "PLAY" and len(option) > 1:
            card = CARD.get(int(option[1]))
            card_type = int(card.cardType) if card is not None else -1
            if card_type == ITEM_CARD_TYPE and \
                    transients.get("noItemTurn", [-1, -1])[actor] == turn:
                transients["noItemTurn"][actor] = -1
            elif card_type == SUPPORTER_CARD_TYPE and \
                    transients.get("noSupporterTurn", [-1, -1])[actor] == turn:
                transients["noSupporterTurn"][actor] = -1
            elif card_type == STADIUM_CARD_TYPE and \
                    transients.get("noStadiumTurn", [-1, -1])[actor] == turn:
                transients["noStadiumTurn"][actor] = -1
        elif option[0] == "EVOLVE" and \
                transients.get("noEvolveTurn", [-1, -1])[actor] == turn:
            transients["noEvolveTurn"][actor] = -1


def _card_name(card: Any) -> str:
    if isinstance(card, dict) and card.get("name") is not None:
        return str(card["name"])
    cid = _card_id(card)
    return "" if cid is None else str(cid)


def _is_basic_energy_id(card_id: int) -> bool:
    return 1 <= card_id <= 9


def _is_ns_name(name: str) -> bool:
    return name.replace("\u2019", "'").startswith("N's ")


def _play_option_debug(current: dict[str, Any] | None, card_id: int) -> str:
    if current is None:
        return ""
    player = current["players"][int(current["yourIndex"])]
    hand_ids = [cid for card in (player.get("hand") or [])
                if (cid := _card_id(card)) is not None]
    active_ids = [cid for card in (player.get("active") or [])
                  if (cid := _card_id(card)) is not None]
    bench_ids = [cid for card in (player.get("bench") or [])
                 if (cid := _card_id(card)) is not None]
    opp = current["players"][1 - int(current["yourIndex"])]
    opp_active_ids = [cid for card in (opp.get("active") or [])
                      if (cid := _card_id(card)) is not None]
    def _attachments(cards: list[dict[str, Any]] | None) -> list[tuple[int | None, tuple[int, ...], tuple[int, ...]]]:
        out = []
        for card in cards or []:
            if not isinstance(card, dict):
                continue
            tools = tuple(cid for t in (card.get("tools") or [])
                          if (cid := _card_id(t)) is not None)
            energy_cards = tuple(cid for e in (card.get("energyCards") or [])
                                 if (cid := _card_id(e)) is not None)
            if tools or energy_cards:
                out.append((_card_id(card), tools, energy_cards))
        return out
    opp_attach = (
        _attachments(opp.get("active")) +
        _attachments(opp.get("bench"))
    )
    conditions = [
        key for key in ("poisoned", "burned", "asleep", "paralyzed", "confused")
        if player.get(key)
    ]
    discard_basic = [cid for card in (player.get("discard") or [])
                     if (cid := _card_id(card)) is not None and _is_basic_energy_id(cid)]
    bench_ns = [cid for card in (player.get("bench") or [])
                if (cid := _card_id(card)) is not None and _is_ns_name(_card_name(card))]
    return (
        f" hand_has={card_id in hand_ids}"
        f" hand_ids={hand_ids[:12]}"
        f" active={active_ids[:3]}"
        f" bench={bench_ids[:8]}"
        f" opp_active={opp_active_ids[:3]}"
        f" opp_attach={opp_attach[:8]}"
        f" conditions={conditions}"
        f" discard_basic={discard_basic[:12]}"
        f" bench_ns={bench_ns[:12]}"
    )


def _attack_option_debug(current: dict[str, Any] | None, attack_id: int) -> str:
    if current is None:
        return ""
    player = current["players"][int(current["yourIndex"])]
    active_list = player.get("active") or []
    active = active_list[0] if active_list else {}
    if not isinstance(active, dict):
        active = {}
    energy_types = tuple(int(e) for e in (active.get("energies") or []))
    energy_cards = tuple(cid for e in (active.get("energyCards") or [])
                         if (cid := _card_id(e)) is not None)
    tools = tuple(cid for t in (active.get("tools") or [])
                  if (cid := _card_id(t)) is not None)
    conditions = [
        key for key in ("poisoned", "burned", "asleep", "paralyzed", "confused")
        if player.get(key)
    ]
    locks = {
        key: active.get(key)
        for key in ("lockTurn", "lockId", "noAttackTurn", "activeLockId",
                    "attackCostModTurn", "attackCostMod")
        if active.get(key) not in (None, 0, -1)
    }
    return (
        f" attack_id={attack_id}"
        f" active={_card_id(active)}"
        f" hp={active.get('hp')}/{active.get('maxHp')}"
        f" energy={energy_types}"
        f" energy_cards={energy_cards}"
        f" tools={tools}"
        f" conditions={conditions}"
        f" locks={locks}"
        f" handCount={player.get('handCount')}"
        f" turnActionCount={current.get('turnActionCount')}"
        f" stadium={[_card_id(c) for c in (current.get('stadium') or [])]}"
    )


def _compare_options(label: str, want: list[tuple], got: list[tuple],
                     strict_duplicates: bool,
                     strict_order: bool = False,
                     select_context: int | None = None,
                     current: dict[str, Any] | None = None,
                     native_state: Any | None = None,
                     raw_cabt: list[Any] | None = None,
                     raw_native: list[tuple] | None = None) -> None:
    if _raw_cabt_deck_index_only(raw_cabt, select_context):
        want_norm = _deck_index_norm_from_raw(raw_cabt or [])
        got_norm = _deck_index_norm_from_options(got)
    else:
        want_norm = [_norm_desc(o, select_context, current) for o in want]
        got_norm = [_norm_desc(o, select_context, current) for o in got]
    want_cmp = _option_counter(want_norm, strict_duplicates)
    got_cmp = _option_counter(got_norm, strict_duplicates)
    if want_cmp != got_cmp:
        extra = sorted(got_cmp - want_cmp) if strict_duplicates else sorted(got_cmp - want_cmp)
        missing = sorted(want_cmp - got_cmp) if strict_duplicates else sorted(want_cmp - got_cmp)
        native_line = ""
        if native_state is not None:
            native_line = (
                f"\n  native_state={_state_debug_summary(eng.canonical(native_state))}"
            )
        raw_line = ""
        if raw_native is not None:
            raw_line = f"\n  raw_native={raw_native[:20]}"
        raise ParityFailure(
            f"{label}: legal option mismatch\n"
            f"  extra_native={extra[:20]}\n"
            f"  missing_native={missing[:20]}\n"
            f"  extra_detail={_option_debug(extra, current)}\n"
            f"  missing_detail={_option_debug(missing, current)}\n"
            f"  state={_state_debug_summary(canonical_state(current))}"
            f"{native_line}"
            f"{raw_line}"
        )
    if strict_order and want_norm != got_norm:
        # CABT's attached-Tool pending order is based on an internal attached-card
        # sequence that is not fully recoverable from loaded `current` snapshots.
        # The adapter can use CABT's raw pending order directly; do not report
        # these as native rule diffs once the Tool option multiset matches.
        if select_context == 27:
            return
        first = next(
            (i for i, (want_one, got_one) in enumerate(zip(want_norm, got_norm))
             if want_one != got_one),
            min(len(want_norm), len(got_norm)),
        )
        tool_line = ""
        if select_context == 27 and current is not None:
            rows = []
            for side, ps in enumerate(current.get("players") or []):
                cards = []
                active = (ps.get("active") or [None])[0]
                if isinstance(active, dict):
                    cards.append(("ACTIVE", 0, active))
                for i, bench in enumerate(ps.get("bench") or []):
                    if isinstance(bench, dict):
                        cards.append(("BENCH", i, bench))
                for area, idx, card in cards:
                    tools = []
                    for tool in card.get("tools") or []:
                        if isinstance(tool, dict):
                            tools.append((
                                tool.get("id"),
                                tool.get("serial"),
                                tool.get("attachOrder"),
                            ))
                    if tools:
                        rows.append((side, area, idx, card.get("id"),
                                     card.get("serial"), tools))
            tool_line = f"\n  tools_raw={rows}"
        raw_cabt_line = ""
        if raw_cabt is not None:
            raw_cabt_line = f"\n  raw_cabt={raw_cabt[:20]}"
        raise ParityFailure(
            f"{label}: legal option order mismatch\n"
            f"  first_order_diff={first}\n"
            f"  cabt_order={want_norm[max(0, first - 5):first + 10]}\n"
            f"  native_order={got_norm[max(0, first - 5):first + 10]}\n"
            f"  state={_state_debug_summary(canonical_state(current))}"
            f"{tool_line}"
            f"{raw_cabt_line}"
        )


def _decision_debug_line(ref_current: dict[str, Any] | None,
                         ref_select: dict[str, Any] | None,
                         native_state: Any) -> str:
    pending = eng.pending_decision(native_state)
    if ref_select is None and pending is None:
        return "pending context cabt=None native=None"

    if ref_select is None:
        cabt_ctx = "None"
        cabt_count = "(None,None)"
        cabt_options = 0
    else:
        cabt_ctx = str(ref_select.get("context"))
        cabt_count = f"({ref_select.get('minCount')},{ref_select.get('maxCount')})"
        cabt_options = len(ref_select.get("option") or [])

    if pending is None:
        native_ctx = "None"
        native_count = "(None,None)"
        native_options = 0
    else:
        native_ctx = str(pending.get("context"))
        native_count = f"({pending.get('minCount')},{pending.get('maxCount')})"
        native_options = len(pending.get("options") or [])

    raw_detail = ""
    if ref_select is not None and int(ref_select.get("context", -1)) == 34:
        raw_detail = f" cabt_raw_first={(ref_select.get('option') or [])[:3]}"
    return (
        f"pending context cabt={cabt_ctx} native={native_ctx} "
        f"cabt_count={cabt_count} native_count={native_count} "
        f"cabt_options={cabt_options} native_options={native_options} "
        f"cabt_first={canonical_options(ref_current, ref_select)[:5] if ref_select else []} "
        f"native_first={[tuple(o) for o in pending.get('options', [])][:5] if pending else []}"
        f"{raw_detail}"
    )


def _state_debug_summary(canon: dict[str, Any] | None) -> str:
    if canon is None:
        return "None"
    parts = [
        f"turnActionCount={canon.get('turnActionCount')}",
        f"yourIndex={canon.get('yourIndex')}",
        f"result={canon.get('result')}",
        f"stadium={canon.get('stadium') or []}",
        f"stadiumUsed={canon.get('stadiumAbilityUsed')}",
    ]
    for i, ps in enumerate(canon.get("players") or []):
        active = ps.get("active")
        if active is None:
            active_desc = "None"
        else:
            active_serial = active.get("serial")
            active_id = active.get("id")
            active_name = f"{active_id}#{active_serial}" if active_serial is not None else active_id
            active_desc = (
                f"{active_name}@{active.get('hp')}/{active.get('maxHp')}"
                f" tools={active.get('tools') or []}"
                f" energy={active.get('energyCards') or active.get('energies') or []}"
            )
        bench = ps.get("bench") or []
        bench_ids = [
            f"{b.get('id')}#{b.get('serial')}" if b.get("serial") is not None
            else b.get("id")
            for b in bench
        ]
        bench_flags = [
            {
                "id": b.get("id"),
                "fresh": bool(b.get("appearThisTurn")),
                "noEvolveTurn": b.get("noEvolveTurn"),
                "pre": [c.get("id") if isinstance(c, dict) else c
                        for c in (b.get("preEvolution") or [])],
            }
            for b in bench
        ]
        def _tool_id(tool: Any) -> int | None:
            if isinstance(tool, dict):
                value = tool.get("id")
                return int(value) if value is not None else None
            if tool is None:
                return None
            return int(tool)

        bench_tools = [
            [tid for tid in (_tool_id(t) for t in (b.get("tools") or []))
             if tid is not None]
            for b in bench
        ]
        bench_tool_desc = f",benchTools={bench_tools}" if any(bench_tools) else ""
        hand = ps.get("hand")
        if isinstance(hand, list):
            hand_desc = f",handIds={hand[:18]}"
            if len(hand) > 18:
                hand_desc += "..."
        else:
            hand_desc = ""
        parts.append(
            f"p{i}(active={active_desc},conds={ps.get('conditions') or []},"
            f"bench={bench_ids},benchFlags={bench_flags}{bench_tool_desc},deck={ps.get('deckCount')},"
            f"prizes={ps.get('prizeCount')},"
            f"discard={ps.get('discard') or []}{hand_desc})"
        )
    return " ".join(parts)


def _active_debug_summary(current: dict[str, Any]) -> str:
    parts = []
    for side in (0, 1):
        player = current["players"][side]
        active = (current["players"][side].get("active") or [None])[0]
        hand_count = player.get("handCount")
        if not isinstance(active, dict):
            parts.append(f"p{side}=None hand={hand_count}")
            continue
        parts.append(
            f"p{side}={active.get('id')}@{active.get('hp')}/{active.get('maxHp')}"
            f" hand={hand_count}"
            f" tools={[t.get('id') for t in (active.get('tools') or []) if isinstance(t, dict)]}"
        )
    return " ".join(parts)


def _pending_revealed_hand_players(select: dict[str, Any] | None) -> set[int]:
    if select is None or int(select.get("context", MAIN_CONTEXT)) == MAIN_CONTEXT:
        return set()
    players: set[int] = set()
    for op in select.get("option") or []:
        if op.get("area") in (2, 12) and op.get("playerIndex") is not None:
            players.add(int(op["playerIndex"]))
    return players


def _double_active_ko_prize_order_compatible(cabt_canon: dict[str, Any],
                                             native_canon: dict[str, Any],
                                             ref_select: dict[str, Any] | None,
                                             diffs: list[tuple[str, Any, Any]]) -> bool:
    if ref_select is None or int(ref_select.get("context", MAIN_CONTEXT)) != 7:
        return False
    options = ref_select.get("option") or []
    if not options or any(int(op.get("area", -1)) != 6 for op in options):
        return False
    if cabt_canon.get("result", -1) != -1 or native_canon.get("result", -1) != -1:
        return False
    for side in (0, 1):
        if cabt_canon["players"][side]["active"] is not None:
            return False
        if native_canon["players"][side]["active"] is not None:
            return False
    allowed = {".yourIndex"}
    return all(path in allowed or
               (path.startswith(".players[") and ".hand" in path)
               for path, _a, _b in diffs)


def _double_active_ko_prize_pending_compatible(cabt_canon: dict[str, Any],
                                               native_canon: dict[str, Any],
                                               ref_select: dict[str, Any] | None,
                                               pending: dict[str, Any] | None,
                                               want: list[tuple],
                                               got: list[tuple]) -> bool:
    if ref_select is None or pending is None:
        return False
    if int(ref_select.get("context", -1)) != 7 or int(pending.get("context", -1)) != 7:
        return False
    if not want or not got:
        return False
    if not all(len(o) >= 3 and o[0] == "CARD" and o[1] == "PRIZE" for o in want):
        return False
    if not all(len(o) >= 3 and o[0] == "CARD" and o[1] == "PRIZE" for o in got):
        return False
    for side in (0, 1):
        if cabt_canon["players"][side]["active"] is not None:
            return False
        if native_canon["players"][side]["active"] is not None:
            return False
    return True


def _compare_next_decision(label: str, ref_obs: Any, native_state: Any,
                           strict_duplicates: bool,
                           strict_order: bool = False,
                           ignore_deck_count_player: int | None = None,
                           skip_deck_search_options: bool = False) -> None:
    ref_select = asdict(ref_obs.select) if ref_obs.select is not None else None
    ref_current = asdict(ref_obs.current) if ref_obs.current is not None else None
    next_main_options = None
    raw_native_options = None
    if ref_current is not None and ref_select is not None and \
            int(ref_select["context"]) == MAIN_CONTEXT:
        next_main_options = canonical_options(ref_current, ref_select)
        raw_native_options = [tuple(o) for o in eng.legal_main(native_state)]
        eng.reconstruct_main(native_state, next_main_options)

    if ref_current is not None:
        cabt_canon = canonical_state(ref_current)
        native_canon = eng.canonical(native_state)
        diffs = diff(cabt_canon, native_canon)
        diffs = [
            d for d in diffs
            if not _is_prize_identity_path(d[0]) and
            not _is_bridge_only_path(d[0])
        ]
        if cabt_canon.get("result", -1) >= 0 and \
                cabt_canon.get("result") == native_canon.get("result"):
            diffs = [
                d for d in diffs
                if d[0] not in {".yourIndex", ".turnActionCount"} and
                not (d[0].startswith(".players[") and ".hand" in d[0])
            ]
        if _is_deck_search_pending(ref_current, ref_select):
            yi = int(ref_current["yourIndex"])
            deck_count_path = f".players[{yi}].deckCount"
            diffs = [d for d in diffs if d[0] != deck_count_path]
        if ignore_deck_count_player is not None:
            deck_count_path = f".players[{ignore_deck_count_player}].deckCount"
            diffs = [d for d in diffs if d[0] != deck_count_path]
        for player in _pending_revealed_hand_players(ref_select):
            hand_count_path = f".players[{player}].handCount"
            diffs = [d for d in diffs if d[0] != hand_count_path]
        if _double_active_ko_prize_order_compatible(
            cabt_canon, native_canon, ref_select, diffs,
        ):
            diffs = []
        native_pending_for_diff = eng.pending_decision(native_state)
        if ref_select is not None and native_pending_for_diff is not None and \
                int(ref_select.get("context", -1)) == 7 and \
                int(native_pending_for_diff.get("context", -1)) == 7:
            diffs = [d for d in diffs if d[0] != ".flags.energyAttached"]
        if diffs:
            lines = "\n".join(
                f"  {path}: cabt={a!r} native={b!r}" for path, a, b in diffs[:12]
            )
            lines = f"{lines}\n  {_decision_debug_line(ref_current, ref_select, native_state)}"
            lines = (
                f"{lines}\n"
                f"  state cabt={_state_debug_summary(cabt_canon)}\n"
                f"  state native={_state_debug_summary(native_canon)}"
            )
            raise ParityFailure(f"{label}: canonical state mismatch\n{lines}")

    if ref_select is None:
        pending = eng.pending_decision(native_state)
        if pending is not None:
            raise ParityFailure(f"{label}: cabt is terminal/no-select, native has pending")
        return

    if int(ref_select["context"]) == MAIN_CONTEXT:
        want = next_main_options if next_main_options is not None else \
            canonical_options(ref_current, ref_select)
        got = [tuple(o) for o in eng.legal_main(native_state)]
        if skip_deck_search_options:
            want = _without_deck_search_plays(want)
            got = _without_deck_search_plays(got)
        got = _restore_wanted_raw_options(want, got, raw_native_options)
        got = _restore_wanted_ko_gate_options(want, got, ref_current)
        got = _without_stale_ko_gate_plays(want, got, ref_current)
        # After a branch transition, hidden/revealed hand order is not part of
        # the native canonical state. The adapter uses CABT's next MAIN order
        # directly, so compare the reconstructed option set here; root MAIN
        # order is still checked strictly in check_main_state.
        _compare_options(f"{label}: next MAIN", want, got, strict_duplicates,
                         False,
                         current=ref_current, native_state=native_state,
                         raw_native=raw_native_options,
                         raw_cabt=list(ref_select.get("option") or []))
        return

    pending = eng.pending_decision(native_state)
    if pending is None:
        want = canonical_options(ref_current, ref_select)
        raise ParityFailure(
            f"{label}: cabt has pending context {ref_select['context']}, native has none "
            f"cabt_count=({ref_select['minCount']},{ref_select['maxCount']}) "
            f"cabt_options={len(ref_select.get('option') or [])} "
            f"cabt_first={want[:8]}"
        )
    if int(pending["context"]) != int(ref_select["context"]):
        want = canonical_options(ref_current, ref_select)
        got = [tuple(o) for o in pending["options"]]
        state_lines = ""
        if ref_current is not None:
            state_lines = (
                f" state cabt={_state_debug_summary(canonical_state(ref_current))} "
                f"state native={_state_debug_summary(eng.canonical(native_state))}"
            )
        raise ParityFailure(
            f"{label}: pending context mismatch cabt={ref_select['context']} "
            f"native={pending['context']} "
            f"cabt_count=({ref_select['minCount']},{ref_select['maxCount']}) "
            f"native_count=({pending['minCount']},{pending['maxCount']}) "
            f"cabt_options={len(ref_select.get('option') or [])} "
            f"native_options={len(pending.get('options') or [])} "
            f"cabt_first={want[:8]} native_first={got[:8]} "
            f"cabt_raw_first={(ref_select.get('option') or [])[:3]}{state_lines}"
        )
    if int(pending["minCount"]) != int(ref_select["minCount"]) or \
            int(pending["maxCount"]) != int(ref_select["maxCount"]):
        want = canonical_options(ref_current, ref_select)
        got = [tuple(o) for o in pending["options"]]
        state_lines = ""
        if ref_current is not None:
            state_lines = (
                f" state cabt={_state_debug_summary(canonical_state(ref_current))} "
                f"state native={_state_debug_summary(eng.canonical(native_state))}"
            )
        raise ParityFailure(
            f"{label}: pending count mismatch "
            f"context={ref_select['context']} "
            f"cabt=({ref_select['minCount']},{ref_select['maxCount']}) "
            f"native=({pending['minCount']},{pending['maxCount']}) "
            f"cabt_options={len(ref_select.get('option') or [])} "
            f"native_options={len(pending.get('options') or [])} "
            f"cabt_first={want[:5]} native_first={got[:5]}{state_lines}"
        )
    want = canonical_options(ref_current, ref_select)
    got = [tuple(o) for o in pending["options"]]
    if ref_current is not None and _double_active_ko_prize_pending_compatible(
        canonical_state(ref_current),
        eng.canonical(native_state),
        ref_select,
        pending,
        want,
        got,
    ):
        return
    _compare_options(f"{label}: pending options", want, got, strict_duplicates,
                     strict_order,
                     int(ref_select["context"]), ref_current,
                     raw_cabt=list(ref_select.get("option") or []))


def _is_deck_search_pending(current: dict[str, Any] | None,
                            select: dict[str, Any] | None) -> bool:
    if current is None or select is None or int(select["context"]) == MAIN_CONTEXT:
        return False
    return any(len(desc) >= 2 and desc[0] == "CARD" and desc[1] == "DECK"
               for desc in canonical_options(current, select))


def _is_prize_identity_path(path: str) -> bool:
    return ".prize[" in path or path.endswith(".prize")


def _is_bridge_only_path(path: str) -> bool:
    return path in {
        ".flags.supporterPlayed",
        ".flags.teamRocketSupporterPlayed",
        ".flags.ancientSupporterPlayed",
        ".flags.canariPlayed",
        ".flags.tarragonPlayed",
    }


def _is_deck_search_play(desc: tuple) -> bool:
    return len(desc) >= 2 and desc[0] == "PLAY" and desc[1] in {
        1082, 1083, 1084, 1086, 1100, 1101, 1102, 1111, 1115, 1119,
        1125, 1126, 1127, 1132, 1134, 1142, 1145, 1152, 1189, 1194, 1196,
        1205, 1210, 1215, 1219, 1231, 1235,
    }


def _without_deck_search_plays(options: list[tuple]) -> list[tuple]:
    return [tuple(option) for option in options
            if not _is_deck_search_play(tuple(option))]


def _without_stale_ko_gate_plays(want: list[tuple], got: list[tuple],
                                 current: dict[str, Any] | None = None) -> list[tuple]:
    """Drop branch-only KO-gate options disproved by CABT's next MAIN list."""
    want_set = {tuple(option) for option in want}
    ko_gate_cards = KO_LAST_TURN_GATE_CARDS | TEAM_ROCKET_KO_LAST_TURN_GATE_CARDS
    filtered: list[tuple] = []
    for option in got:
        desc = tuple(option)
        if len(desc) >= 2 and desc[0] == "PLAY" and desc[1] in ko_gate_cards and \
                desc not in want_set:
            continue
        if desc not in want_set and current is not None and \
                len(desc) >= 3 and desc[0] == "ABILITY" and \
                _descriptor_source_card(current, desc) in KO_LAST_TURN_GATE_ABILITIES:
            continue
        filtered.append(desc)
    return filtered


def _restore_wanted_raw_options(want: list[tuple], got: list[tuple],
                                raw_native: list[tuple] | None) -> list[tuple]:
    """Re-add CABT-wanted options that reconstruction suppressed by accident."""
    if not raw_native:
        return got
    restored = list(got)
    want_counts = Counter(_norm_desc(option) for option in want)
    got_counts = Counter(_norm_desc(option) for option in restored)
    for option in raw_native:
        norm = _norm_desc(option)
        if got_counts[norm] >= want_counts[norm]:
            continue
        restored.append(tuple(option))
        got_counts[norm] += 1
    return restored


def _restore_wanted_ko_gate_options(want: list[tuple], got: list[tuple],
                                    current: dict[str, Any] | None) -> list[tuple]:
    """Re-add CABT-proven KO-gate options lost when loading a current snapshot."""
    if current is None:
        return got
    actor = int(current["yourIndex"])
    hand_ids = {
        cid for card in (current["players"][actor].get("hand") or [])
        if (cid := _card_id(card)) is not None
    }
    ko_gate_cards = (
        KO_LAST_TURN_GATE_CARDS |
        ATTACK_DAMAGE_KO_LAST_TURN_GATE_CARDS |
        TEAM_ROCKET_KO_LAST_TURN_GATE_CARDS
    )
    restored = list(got)
    got_counts = Counter(_norm_desc(option) for option in restored)
    for option in want:
        desc = tuple(option)
        if len(desc) < 2 or desc[0] != "PLAY" or desc[1] not in ko_gate_cards:
            continue
        if desc[1] not in hand_ids:
            continue
        norm = _norm_desc(desc)
        if got_counts[norm] == 0:
            restored.append(desc)
            got_counts[norm] += 1
    return restored


def _source_card_for_action(current: dict[str, Any], desc: tuple) -> int | None:
    if len(desc) < 3 or desc[0] not in {"ABILITY", "DISCARD", "DISCARD_INPLAY"}:
        return None
    area = desc[1]
    idx = desc[2]
    if not isinstance(idx, int):
        return None
    if area == "STADIUM":
        seq = current.get("stadium") or []
    else:
        player = current["players"][int(current["yourIndex"])]
        if area == "ACTIVE":
            seq = player.get("active") or []
        elif area == "BENCH":
            seq = player.get("bench") or []
        else:
            return None
    if idx < 0 or idx >= len(seq) or seq[idx] is None:
        return None
    card = seq[idx]
    return int(card["id"]) if isinstance(card, dict) and card.get("id") is not None else None


def _log_value(log: Any, key: str, default: Any = None) -> Any:
    if isinstance(log, dict):
        return log.get(key, default)
    return getattr(log, key, default)


_LOG_SIGNATURE_KEYS = (
    "type", "playerIndex", "cardId", "serial", "fromArea", "toArea",
    "cardIdTarget", "serialTarget", "cardIdActive", "serialActive",
    "cardIdBench", "serialBench", "head",
)


def _log_signature(log: Any) -> tuple[Any, ...]:
    return tuple(_log_value(log, key) for key in _LOG_SIGNATURE_KEYS)


def _new_logs(previous: list[Any] | None, current: list[Any] | None) -> list[Any]:
    prev = list(previous or [])
    cur = list(current or [])
    if not prev:
        return cur
    if len(cur) >= len(prev):
        prev_sig = [_log_signature(log) for log in prev]
        cur_sig = [_log_signature(log) for log in cur[:len(prev)]]
        if cur_sig == prev_sig:
            return cur[len(prev):]
    return cur


def _log_moves_card_to_discard(logs: list[Any] | None, player: int,
                               card_id: int) -> bool:
    for log in logs or []:
        if int(_log_value(log, "type", -1)) != LOG_MOVE:
            continue
        if int(_log_value(log, "playerIndex", -1)) != player:
            continue
        if int(_log_value(log, "cardId", -1)) != card_id:
            continue
        if int(_log_value(log, "toArea", -1)) == AREA_DISCARD:
            return True
    return False


def _discard_delta_has_card(before: dict[str, Any] | None,
                            after: dict[str, Any] | None,
                            player: int, card_id: int) -> bool:
    if before is None or after is None or player not in (0, 1):
        return False
    before_players = before.get("players") or []
    after_players = after.get("players") or []
    if player >= len(before_players) or player >= len(after_players):
        return False
    delta = (
        _discard_id_counts(after_players[player]) -
        _discard_id_counts(before_players[player])
    )
    return delta[int(card_id)] > 0


def _clear_pending_tarragon(transients: dict[str, Any]) -> None:
    transients["pendingTarragonTurn"] = -1
    transients["pendingTarragonPlayer"] = -1


def _track_tarragon_completion(transients: dict[str, Any],
                               before: dict[str, Any] | None,
                               after: dict[str, Any] | None,
                               logs: list[Any] | None,
                               next_select: dict[str, Any] | None) -> None:
    pending_player = int(transients.get("pendingTarragonPlayer", -1))
    pending_turn = int(transients.get("pendingTarragonTurn", -1))
    if pending_player not in (0, 1) or pending_turn < 0:
        return
    if before is None or int(before.get("turn", -1)) != pending_turn:
        _clear_pending_tarragon(transients)
        return

    completed = _log_moves_card_to_discard(logs, pending_player, 1238) or \
        _discard_delta_has_card(before, after, pending_player, 1238)
    if completed:
        transients["tarragonPlayed"] = True
        _clear_pending_tarragon(transients)
        return

    if after is None:
        _clear_pending_tarragon(transients)
        return
    if int(after.get("turn", -1)) != pending_turn or \
            int(after.get("yourIndex", -1)) != pending_player:
        _clear_pending_tarragon(transients)
        return
    if next_select is not None and \
            int(next_select.get("context", MAIN_CONTEXT)) == MAIN_CONTEXT:
        _clear_pending_tarragon(transients)


def _uses_random_opp_hand_to_deck(desc: tuple[Any, ...]) -> bool:
    return len(desc) >= 2 and desc[0] == "ATTACK" and \
        int(desc[1]) in RANDOM_OPP_HAND_TO_DECK_ATTACKS


def _uses_random_discard_hand(desc: tuple[Any, ...],
                              source_card: int | None = None) -> bool:
    if len(desc) >= 2 and desc[0] == "ATTACK" and \
            int(desc[1]) in RANDOM_DISCARD_HAND_ATTACKS:
        return True
    return source_card in RANDOM_DISCARD_HAND_ABILITIES


def _is_coin_log(log: Any) -> bool:
    return int(_log_value(log, "type", -1)) in (LOG_COIN, LOG_COIN_HEAD)


def _coin_log_heads(log: Any) -> bool:
    if int(_log_value(log, "type", -1)) == LOG_COIN_HEAD:
        return True
    return bool(_log_value(log, "head", False))


def _replay_tape(logs: list[Any], desc: tuple[Any, ...] = (),
                 source_card: int | None = None) -> list[int]:
    tape: list[int] = []
    include_random_hand = _uses_random_opp_hand_to_deck(desc)
    include_random_discard = _uses_random_discard_hand(desc, source_card)
    for log in logs:
        typ = int(_log_value(log, "type", -1))
        if _is_coin_log(log):
            tape.extend([REPLAY_COIN, 1 if _coin_log_heads(log) else 0])
            continue
        card_id = _log_value(log, "cardId")
        if card_id is None:
            continue
        if typ == LOG_DRAW:
            player_index = _log_value(log, "playerIndex")
            if player_index is None:
                tape.append(int(card_id))
            else:
                tape.extend([REPLAY_DRAW_PLAYER, int(player_index), int(card_id)])
        elif include_random_hand and typ == LOG_MOVE and \
                int(_log_value(log, "fromArea", -1)) == AREA_HAND and \
                int(_log_value(log, "toArea", -1)) == AREA_DECK:
            tape.extend([REPLAY_RANDOM_OPP_HAND_TO_DECK, int(card_id)])
        elif include_random_discard and typ == LOG_MOVE and \
                int(_log_value(log, "fromArea", -1)) == AREA_HAND and \
                int(_log_value(log, "toArea", -1)) == AREA_DISCARD:
            tape.extend([REPLAY_RANDOM_DISCARD_HAND, int(card_id)])
    return tape


def _action_hand_index_tape(select: dict[str, Any] | None,
                            option_index: int) -> list[int]:
    if select is None or int(select.get("context", -1)) != MAIN_CONTEXT:
        return []
    options = select.get("option") or []
    if option_index < 0 or option_index >= len(options):
        return []
    op = options[option_index]
    typ = int(_log_value(op, "type", -1))
    hand_index = None
    if typ == 7:
        hand_index = _log_value(op, "index")
    elif typ in (8, 9) and int(_log_value(op, "area", -1)) == AREA_HAND:
        hand_index = _log_value(op, "index")
    if hand_index is None:
        return []
    return [REPLAY_HAND_INDEX, int(hand_index)]


def _action_replay_tape(select: dict[str, Any] | None,
                        option_index: int,
                        logs: list[Any],
                        desc: tuple[Any, ...] = (),
                        source_card: int | None = None) -> list[int]:
    return _action_hand_index_tape(select, option_index) + \
        _replay_tape(logs, desc, source_card)


def _draw_tape(logs: list[Any]) -> list[int]:
    return _replay_tape(logs)


def _has_coin(logs: list[Any]) -> bool:
    return any(_is_coin_log(log) for log in logs)


def _has_heads(logs: list[Any]) -> bool:
    return any(_is_coin_log(log) and _coin_log_heads(log) for log in logs)


def _coin_log_values(logs: list[Any]) -> list[int]:
    return [1 if _coin_log_heads(log) else 0 for log in logs if _is_coin_log(log)]


def _track_energy_attach_order(transients: dict[str, Any],
                               logs: list[Any]) -> None:
    attach_order = transients.setdefault("energyAttachOrder", {})
    next_order = int(transients.get("energyAttachCounter", 0))
    for log in logs:
        if int(_log_value(log, "type", -1)) != LOG_ATTACH_ENERGY:
            continue
        player = _log_value(log, "playerIndex")
        card_id = _log_value(log, "cardId")
        if player is None or card_id is None:
            continue
        serial = _log_value(log, "serial")
        key = (int(player), int(card_id),
               int(serial) if serial is not None else None)
        attach_order[key] = next_order
        next_order += 1
    transients["energyAttachCounter"] = next_order


def _track_tool_attach_order(transients: dict[str, Any],
                             logs: list[Any]) -> None:
    attach_order = transients.setdefault("toolAttachOrder", {})
    next_order = int(transients.get("toolAttachCounter", 0))
    for log in logs:
        typ = int(_log_value(log, "type", -1))
        card_id = _log_value(log, "cardId")
        if card_id is None:
            continue
        card = CARD.get(int(card_id))
        is_tool = card is not None and int(card.cardType) == TOOL_CARD_TYPE
        if typ == LOG_ATTACH_ENERGY:
            if not is_tool:
                continue
        elif typ == LOG_MOVE:
            if int(_log_value(log, "fromArea", -1)) != AREA_HAND:
                continue
            if int(_log_value(log, "toArea", -1)) != AREA_TOOL:
                continue
        else:
            continue
        player = _log_value(log, "playerIndex")
        if player is None or card_id is None:
            continue
        serial = _log_value(log, "serial")
        key = (int(player), int(card_id),
               int(serial) if serial is not None else None)
        attach_order[key] = next_order
        next_order += 1
    transients["toolAttachCounter"] = next_order


def _append_unique_identity(
    items: list[tuple[int, int | None]],
    card_id: Any,
    serial: Any,
) -> None:
    if card_id is None:
        return
    ident = (int(card_id), int(serial) if serial is not None else None)
    if ident not in items:
        items.append(ident)


def _track_moved_to_active(transients: dict[str, Any],
                           logs: list[Any]) -> None:
    moved = transients.setdefault("movedToActiveThisTurn", [[], []])
    for log in logs or []:
        typ = int(_log_value(log, "type", -1))
        owner = int(_log_value(log, "playerIndex", -1))
        if owner not in (0, 1):
            continue
        if typ == LOG_MOVE:
            if int(_log_value(log, "fromArea", -1)) != AREA_BENCH:
                continue
            if int(_log_value(log, "toArea", -1)) != AREA_ACTIVE:
                continue
            _append_unique_identity(
                moved[owner], _log_value(log, "cardId"), _log_value(log, "serial")
            )
        elif typ == LOG_SWITCH:
            _append_unique_identity(
                moved[owner],
                _log_value(log, "cardIdBench"),
                _log_value(log, "serialBench"),
            )


def _track_healed_this_turn(transients: dict[str, Any],
                            before: dict[str, Any] | None,
                            after: dict[str, Any] | None) -> None:
    if before is None or after is None:
        return
    if int(before.get("turn", -1)) != int(after.get("turn", -2)):
        transients["healedThisTurn"] = [[], []]
        return
    healed = transients.setdefault("healedThisTurn", [[], []])
    for owner in (0, 1):
        before_by_id = {
            _card_identity(card): card
            for card in _iter_inplay_cards(before["players"][owner])
        }
        after_ids = set()
        for card in _iter_inplay_cards(after["players"][owner]):
            identity = _card_identity(card)
            if identity is None:
                continue
            after_ids.add(identity)
            prev = before_by_id.get(identity)
            if prev is None:
                continue
            if int(card.get("hp", 0)) > int(prev.get("hp", 0)) and \
                    identity not in healed[owner]:
                healed[owner].append(identity)
        healed[owner] = [identity for identity in healed[owner]
                         if tuple(identity) in after_ids]


def _active_has_condition(current: dict[str, Any], player: int, condition: str) -> bool:
    try:
        canon = canonical_state(current)
    except Exception:
        return False
    return condition in (canon["players"][player].get("conditions") or [])


def _selected_attack_coin_heads(current: dict[str, Any],
                                actor: int,
                                logs: list[Any]) -> bool:
    coins = [
        _coin_log_heads(log)
        for log in logs
        if _is_coin_log(log) and
        int(_log_value(log, "playerIndex", actor)) == actor
    ]
    if _active_has_condition(current, actor, "confused") and coins:
        coins = coins[1:]
    return bool(coins and coins[-1])


def _track_coin_prevention(transients: dict[str, Any],
                           current: dict[str, Any],
                           selected_main: tuple | None,
                           logs: list[Any]) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    attack_id = int(selected_main[1])
    prevents_damage = attack_id in COIN_PREVENT_DAMAGE_ATTACKS or \
        attack_id in COIN_PREVENT_DAMAGE_EFFECTS_ATTACKS
    if not prevents_damage:
        return
    actor = int(current["yourIndex"])
    if not _selected_attack_coin_heads(current, actor, logs):
        return
    active = (current["players"][actor].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    effect = {
        "owner": actor,
        "identity": identity,
        "preventDmgTurn": int(current["turn"]) + 1,
        "preventDmgCond": 0,
        "preventDmgValue": 0,
    }
    if attack_id in COIN_PREVENT_DAMAGE_EFFECTS_ATTACKS:
        effect.update({
            "preventEffectsTurn": int(current["turn"]) + 1,
            "preventEffectsCond": 0,
            "preventEffectsValue": 0,
        })
    transients.setdefault("inplayEffects", []).append(effect)


def _track_deterministic_damage_prevention(transients: dict[str, Any],
                                           current: dict[str, Any],
                                           selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    spec = DETERMINISTIC_PREVENT_DAMAGE_ATTACKS.get(int(selected_main[1]))
    if spec is None:
        return
    actor = int(current["yourIndex"])
    active = (current["players"][actor].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    cond, value = spec
    transients.setdefault("inplayEffects", []).append({
        "owner": actor,
        "identity": identity,
        "preventDmgTurn": int(current["turn"]) + 1,
        "preventDmgCond": int(cond),
        "preventDmgValue": int(value),
    })


def _track_shadowy_side_kick_prevention(transients: dict[str, Any],
                                        current: dict[str, Any],
                                        after: dict[str, Any] | None,
                                        selected_main: tuple | None,
                                        logs: list[Any] | None) -> None:
    if selected_main != ("ATTACK", 986):
        return
    if after is None:
        return
    actor = int(current["yourIndex"])
    active = (current["players"][actor].get("active") or [None])[0]
    if _card_id(active) != 681:
        return
    identity = _card_identity(active)
    if identity is None:
        return
    defender_owner = 1 - actor
    defender = (current["players"][defender_owner].get("active") or [None])[0]
    defender_identity = _card_identity(defender)
    if defender_identity is None:
        return

    attack_damage_identities = _attack_damage_log_identities(logs, actor)
    damaged_by_attack = _identity_in(
        defender_identity,
        attack_damage_identities[defender_owner],
    )
    before_hp = int(defender.get("hp", 0))
    damage = 0
    for log in logs or []:
        if int(_log_value(log, "type", -1)) != LOG_HP_CHANGE:
            continue
        if int(_log_value(log, "playerIndex", -1)) != defender_owner:
            continue
        card_id = _log_value(log, "cardId")
        if card_id is None or int(card_id) != defender_identity[0]:
            continue
        serial = _log_value(log, "serial")
        if defender_identity[1] is not None and serial is not None and \
                int(serial) != defender_identity[1]:
            continue
        if bool(_log_value(log, "putDamageCounter", False)):
            continue
        value = int(_log_value(log, "value", 0) or 0)
        if value < 0:
            damage -= value
    after_defender = (after["players"][defender_owner].get("active") or [None])[0]
    after_identity = _card_identity(after_defender)
    defender_removed = after_identity != defender_identity
    defender_zero_hp = (
        isinstance(after_defender, dict) and
        after_identity == defender_identity and
        int(after_defender.get("hp", 1)) <= 0
    )
    damage_ko = damaged_by_attack and (
        defender_removed or defender_zero_hp or damage >= before_hp > 0
    )
    if not damage_ko:
        return

    transients.setdefault("inplayEffects", []).append({
        "owner": actor,
        "identity": identity,
        "preventDmgTurn": int(current["turn"]) + 1,
        "preventDmgCond": DPC_ALL,
        "preventDmgValue": 0,
        "preventEffectsTurn": int(current["turn"]) + 1,
        "preventEffectsCond": DPC_ALL,
        "preventEffectsValue": 0,
    })


def _track_self_damage_reduction(transients: dict[str, Any],
                                 current: dict[str, Any],
                                 selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    amount = SELF_DAMAGE_REDUCE_ATTACKS.get(int(selected_main[1]))
    if amount is None:
        return
    actor = int(current["yourIndex"])
    active = (current["players"][actor].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    transients.setdefault("inplayEffects", []).append({
        "owner": actor,
        "identity": identity,
        "dmgReduce": int(amount),
        "dmgReduceTurn": int(current["turn"]) + 1,
    })


def _track_opp_attack_reduction(transients: dict[str, Any],
                                current: dict[str, Any],
                                selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    amount = OPP_ATTACK_REDUCE_ATTACKS.get(int(selected_main[1]))
    if amount is None:
        return
    actor = int(current["yourIndex"])
    owner = 1 - actor
    active = (current["players"][owner].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    actor_active = (current["players"][actor].get("active") or [None])[0]
    if _bridge_attack_effects_prevented(
            transients, current, owner, active, actor_active):
        return
    transients.setdefault("inplayEffects", []).append({
        "owner": owner,
        "identity": identity,
        "attackDmgReduce": int(amount),
        "attackDmgReduceTurn": int(current["turn"]) + 1,
    })


def _track_iron_defender_play(transients: dict[str, Any],
                              current: dict[str, Any],
                              selected_main: tuple | None) -> None:
    if selected_main != ("PLAY", 1140):
        return
    actor = int(current["yourIndex"])
    target_turn = int(current["turn"]) + 1
    turns = transients.setdefault("teamReduceTurn", [-1, -1])
    amounts = transients.setdefault("teamReduceAmount", [0, 0])
    types = transients.setdefault("teamReduceType", [-1, -1])
    if int(turns[actor]) == target_turn and int(types[actor]) == 8:
        amounts[actor] = int(amounts[actor]) + 30
    else:
        turns[actor] = target_turn
        amounts[actor] = 30
        types[actor] = 8


def _track_prize_bonus_play(transients: dict[str, Any],
                            current: dict[str, Any],
                            selected_main: tuple | None) -> None:
    specs = {
        ("PLAY", 1201): (1, 0),  # Briar: Tera attacker.
        ("PLAY", 1234): (3, 1),  # Anthea & Concordia: N's attacker.
    }
    spec = specs.get(selected_main)
    if spec is None:
        return
    actor = int(current["yourIndex"])
    amount, kind = spec
    transients.setdefault("prizeBonusTurn", [-1, -1])
    transients.setdefault("prizeBonusAmount", [0, 0])
    transients.setdefault("prizeBonusKind", [0, 0])
    transients["prizeBonusTurn"][actor] = int(current["turn"])
    transients["prizeBonusAmount"][actor] += int(amount)
    transients["prizeBonusKind"][actor] = int(kind)


def _sync_prize_bonus_transients_from_current(transients: dict[str, Any],
                                              current: dict[str, Any]) -> None:
    if not bool(current.get("supporterPlayed", False)):
        return
    actor = int(current["yourIndex"])
    turn = int(current["turn"])
    if int(transients.get("prizeBonusTurn", [-1, -1])[actor]) == turn:
        return
    discard = current["players"][actor].get("discard") or []
    discard_ids = set()
    for card in discard:
        if isinstance(card, dict) and card.get("id") is not None:
            discard_ids.add(int(card["id"]))
        elif card is not None:
            discard_ids.add(int(card))
    if 1201 in discard_ids:
        transients.setdefault("prizeBonusTurn", [-1, -1])[actor] = turn
        transients.setdefault("prizeBonusAmount", [0, 0])[actor] = 1
        transients.setdefault("prizeBonusKind", [0, 0])[actor] = 0
    elif 1234 in discard_ids:
        transients.setdefault("prizeBonusTurn", [-1, -1])[actor] = turn
        transients.setdefault("prizeBonusAmount", [0, 0])[actor] = 3
        transients.setdefault("prizeBonusKind", [0, 0])[actor] = 1


def _track_prevent_target_resolution(transients: dict[str, Any],
                                     current: dict[str, Any],
                                     select: dict[str, Any],
                                     selection: list[int]) -> None:
    if int(select.get("context", MAIN_CONTEXT)) != CTX_PREVENT_TARGET:
        return
    actor = int(current["yourIndex"])
    options = canonical_options(current, select)
    for s in selection:
        if s < 0 or s >= len(options):
            continue
        target = _inplay_card_for_option(current, actor, tuple(options[s]))
        identity = _card_identity(target)
        if identity is None:
            continue
        transients.setdefault("inplayEffects", []).append({
            "owner": actor,
            "identity": identity,
            "preventDmgTurn": int(current["turn"]) + 1,
            "preventDmgCond": DPC_ATTACKER_EX,
            "preventDmgValue": 0,
            "preventEffectsTurn": int(current["turn"]) + 1,
            "preventEffectsCond": DPC_ATTACKER_EX,
            "preventEffectsValue": 0,
        })


def _track_supporter_play_transients(transients: dict[str, Any],
                                     selected_main: tuple | None) -> None:
    if selected_main is None or len(selected_main) < 2 or selected_main[0] != "PLAY":
        return
    card = CARD.get(int(selected_main[1]))
    if card is None or int(card.cardType) != SUPPORTER_CARD_TYPE:
        return
    transients["supporterPlayed"] = True
    if "Team Rocket" in str(card.name):
        transients["teamRocketSupporterPlayed"] = True
    if int(selected_main[1]) == 1185:
        transients["ancientSupporterPlayed"] = True
    if int(selected_main[1]) == 1233:
        transients["canariPlayed"] = True


def _track_stadium_play_transients(transients: dict[str, Any],
                                   selected_main: tuple | None) -> None:
    if selected_main is None or len(selected_main) < 2 or selected_main[0] != "PLAY":
        return
    card = CARD.get(int(selected_main[1]))
    if card is None or int(card.cardType) != STADIUM_CARD_TYPE:
        return
    transients["stadiumAbilityUsed"] = False


def _bridge_attack_effects_prevented(transients: dict[str, Any],
                                     current: dict[str, Any],
                                     target_owner: int,
                                     target_card: Any,
                                     attacker_card: Any) -> bool:
    target_identity = _card_identity(target_card)
    if target_identity is None:
        return False
    if any(_card_id(card) == MIST_ENERGY
           for card in (target_card.get("energyCards") or [])):
        return True

    turn = int(current["turn"])
    attacker_id = _card_id(attacker_card)
    attacker = CARD.get(attacker_id) if attacker_id is not None else None
    for eff in transients.get("inplayEffects") or []:
        if int(eff.get("owner", -1)) != target_owner:
            continue
        if tuple(eff.get("identity", (None, None))) != target_identity:
            continue
        if int(eff.get("preventEffectsTurn", -1)) != turn:
            continue
        cond = int(eff.get("preventEffectsCond", DPC_ALL))
        if cond == DPC_ALL:
            return True
        if cond == DPC_ATTACKER_BASIC and bool(getattr(attacker, "basic", False)):
            return True
        if cond == DPC_ATTACKER_BASIC_NON_COLORLESS and \
                bool(getattr(attacker, "basic", False)) and \
                int(getattr(attacker, "energyType", -1)) != 6:
            return True
        if cond == DPC_ATTACKER_EX and (
                bool(getattr(attacker, "ex", False)) or
                bool(getattr(attacker, "megaEx", False))):
            return True
    return False


def _track_no_retreat(transients: dict[str, Any],
                      current: dict[str, Any],
                      selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    attack_id = int(selected_main[1])
    actor = int(current["yourIndex"])
    owner = -1
    turn = int(current["turn"])
    if attack_id in OPP_NO_RETREAT_ATTACKS:
        owner = 1 - actor
        active = (current["players"][owner].get("active") or [None])[0]
        no_retreat_turn = turn + 1
    elif attack_id in SELF_NO_RETREAT_ATTACKS:
        owner = actor
        active = (current["players"][owner].get("active") or [None])[0]
        no_retreat_turn = turn + 2
    elif attack_id in OPP_TAKE_MORE_DAMAGE_ATTACKS:
        owner = 1 - actor
        active = (current["players"][owner].get("active") or [None])[0]
        no_retreat_turn = None
    else:
        return
    identity = _card_identity(active)
    if identity is None:
        return
    actor_active = (current["players"][actor].get("active") or [None])[0]
    if owner != actor and _bridge_attack_effects_prevented(
            transients, current, owner, active, actor_active):
        return
    effect = {
        "owner": owner,
        "identity": identity,
        "area": "ACTIVE",
    }
    if no_retreat_turn is not None:
        effect["noRetreatTurn"] = no_retreat_turn
    if attack_id in OPP_TAKE_MORE_DAMAGE_ATTACKS:
        effect["takeMoreDamageTurn"] = turn + 2
        effect["takeMoreDamage"] = OPP_TAKE_MORE_DAMAGE_ATTACKS[attack_id]
    transients.setdefault("inplayEffects", []).append(effect)


def _track_attack_flip_fail(transients: dict[str, Any],
                            current: dict[str, Any],
                            selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    if int(selected_main[1]) not in OPP_ATTACK_FLIP_FAIL_ATTACKS:
        return
    actor = int(current["yourIndex"])
    owner = 1 - actor
    active = (current["players"][owner].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    actor_active = (current["players"][actor].get("active") or [None])[0]
    if _bridge_attack_effects_prevented(
            transients, current, owner, active, actor_active):
        return
    transients.setdefault("inplayEffects", []).append({
        "owner": owner,
        "identity": identity,
        "attackFlipFailTurn": int(current["turn"]) + 1,
    })


def _track_attack_retreat_cost_more(transients: dict[str, Any],
                                    current: dict[str, Any],
                                    selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    amount = OPP_ATTACK_RETREAT_COST_MORE_ATTACKS.get(int(selected_main[1]))
    if amount is None:
        return
    actor = int(current["yourIndex"])
    owner = 1 - actor
    active = (current["players"][owner].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    actor_active = (current["players"][actor].get("active") or [None])[0]
    if _bridge_attack_effects_prevented(
            transients, current, owner, active, actor_active):
        return
    transients.setdefault("inplayEffects", []).append({
        "owner": owner,
        "identity": identity,
        "attackCostModTurn": int(current["turn"]) + 1,
        "attackCostMod": int(amount),
        "retreatCostModTurn": int(current["turn"]) + 1,
        "retreatCostMod": int(amount),
    })


def _track_no_weakness(transients: dict[str, Any],
                       current: dict[str, Any],
                       selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    if int(selected_main[1]) != 253:
        return
    actor = int(current["yourIndex"])
    active = (current["players"][actor].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    transients.setdefault("inplayEffects", []).append({
        "owner": actor,
        "identity": identity,
        "noWeaknessTurn": int(current["turn"]) + 1,
    })


def _track_delayed_end_turn_effects(transients: dict[str, Any],
                                    current: dict[str, Any],
                                    selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    attack_id = int(selected_main[1])
    counters = DELAYED_DAMAGE_ATTACKS.get(attack_id)
    delayed_ko = attack_id in DELAYED_KO_ATTACKS
    if counters is None and not delayed_ko:
        return
    actor = int(current["yourIndex"])
    owner = 1 - actor
    active = (current["players"][owner].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    effect: dict[str, Any] = {
        "owner": owner,
        "identity": identity,
    }
    if counters is not None:
        effect["delayedDamageTurn"] = int(current["turn"]) + 1
        effect["delayedDamageCounters"] = int(counters)
    if delayed_ko:
        effect["delayedKoTurn"] = int(current["turn"]) + 1
        if attack_id == 647:
            effect["delayedKoPromoteBeforePrize"] = True
    transients.setdefault("inplayEffects", []).append(effect)


def _track_if_damaged_reactive_effects(transients: dict[str, Any],
                                       current: dict[str, Any],
                                       selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    attack_id = int(selected_main[1])
    counters = IF_DAMAGED_COUNTER_ATTACKS.get(attack_id)
    equal_damage = attack_id in IF_DAMAGED_DAMAGE_DONE_ATTACKS
    if counters is None and not equal_damage:
        return
    actor = int(current["yourIndex"])
    active = (current["players"][actor].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    effect: dict[str, Any] = {
        "owner": actor,
        "identity": identity,
    }
    when = int(current["turn"]) + 1
    if counters is not None:
        effect["damagedByAttackCountersTurn"] = when
        effect["damagedByAttackCounters"] = int(counters)
    if equal_damage:
        effect["damagedByAttackEqualCountersTurn"] = when
    transients.setdefault("inplayEffects", []).append(effect)


def _track_energy_attach_reactive_effects(transients: dict[str, Any],
                                          current: dict[str, Any],
                                          selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    spec = ENERGY_ATTACH_COUNTER_ATTACKS.get(int(selected_main[1]))
    if spec is None:
        return
    actor = int(current["yourIndex"])
    target_owner = 1 - actor
    active = (current["players"][target_owner].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    counters, from_hand_only = spec
    transients.setdefault("inplayEffects", []).append({
        "owner": target_owner,
        "identity": identity,
        "energyAttachCountersTurn": int(current["turn"]) + 1,
        "energyAttachCounters": int(counters),
        "energyAttachCountersFromHandOnly": 1 if from_hand_only else 0,
    })


def _track_named_attack_bonus(transients: dict[str, Any],
                              current: dict[str, Any],
                              selected_main: tuple | None) -> None:
    if selected_main is None or selected_main[0] != "ATTACK":
        return
    spec = NAMED_ATTACK_NEXT_TURN.get(int(selected_main[1]))
    if spec is None:
        return
    actor = int(current["yourIndex"])
    active = (current["players"][actor].get("active") or [None])[0]
    identity = _card_identity(active)
    if identity is None:
        return
    target_attack_id, bonus, set_base = spec
    transients.setdefault("inplayEffects", []).append({
        "owner": actor,
        "identity": identity,
        "nextAttackBonusId": target_attack_id,
        "nextAttackBonusTurn": int(current["turn"]) + 2,
        "nextAttackBonus": bonus,
        "nextAttackSetBase": set_base,
    })


def _player_poisoned(current: dict[str, Any] | None, player: int) -> bool:
    if current is None:
        return False
    try:
        return bool(current["players"][player].get("poisoned", False))
    except (KeyError, IndexError, TypeError):
        return False


def _active_identity(current: dict[str, Any] | None,
                     player: int) -> tuple[int, int | None] | None:
    if current is None:
        return None
    try:
        active = (current["players"][player].get("active") or [None])[0]
    except (KeyError, IndexError, TypeError):
        return None
    return _card_identity(active)


def _track_poison_damage_counters(transients: dict[str, Any],
                                  current: dict[str, Any] | None,
                                  next_current: dict[str, Any] | None,
                                  selected_main: tuple | None) -> None:
    if current is None or next_current is None:
        return
    counters = transients.setdefault("poisonDamageCounters", [1, 1])
    for player in (0, 1):
        if _active_identity(current, player) != _active_identity(next_current, player):
            counters[player] = 1
            continue
        if not _player_poisoned(next_current, player):
            counters[player] = 1

    if selected_main is None or selected_main[0] != "ATTACK":
        return
    actor = int(current["yourIndex"])
    target = 1 - actor
    attack_id = int(selected_main[1])
    special_counters = SPECIAL_POISON_COUNTER_ATTACKS.get(attack_id)
    if special_counters is not None:
        if _player_poisoned(next_current, target):
            counters[target] = int(special_counters)
        return
    if not _player_poisoned(current, target) and _player_poisoned(next_current, target):
        counters[target] = 1


def _pokemon_identities_in_play(player_state: dict[str, Any]) -> Counter[tuple[int, int | None]]:
    refs: Counter[tuple[int, int | None]] = Counter()
    for card in list(player_state.get("active") or []) + list(player_state.get("bench") or []):
        if isinstance(card, dict) and card.get("id") is not None:
            serial = card.get("serial")
            refs[(int(card["id"]), int(serial) if serial is not None else None)] += 1
    return refs


def _discard_id_counts(player_state: dict[str, Any]) -> Counter[int]:
    out: Counter[int] = Counter()
    for card in player_state.get("discard") or []:
        if isinstance(card, dict):
            cid = _card_id(card)
            if cid is not None:
                out[cid] += 1
        elif card is not None:
            out[int(card)] += 1
    return out


def _card_name_by_id(card_id: int) -> str:
    card = CARD.get(card_id)
    return str(card.name) if card is not None else ""


def _mark_ko_last_turn(transients: dict[str, Any], owner: int, turn: int,
                       removed: list[int], attack_damage: bool = False) -> None:
    if not removed:
        return
    transients["lastKoTurn"][owner] = turn
    transients.setdefault("lastKoTurnSource", ["", ""])
    transients["lastKoTurnSource"][owner] = "observed"
    transients.setdefault("lastAttackDamageKoTurn", [-1, -1])
    transients.setdefault("lastAttackDamageKoTurnSource", ["", ""])
    if attack_damage:
        transients["lastAttackDamageKoTurn"][owner] = turn
        transients["lastAttackDamageKoTurnSource"][owner] = "observed"
    if any(_card_name_by_id(card_id).startswith("Team Rocket") for card_id in removed):
        transients["lastTeamRocketKoTurn"][owner] = turn
        transients.setdefault("lastTeamRocketKoTurnSource", ["", ""])
        transients["lastTeamRocketKoTurnSource"][owner] = "observed"
    if any(_card_name_by_id(card_id).startswith("Ethan") for card_id in removed):
        transients["lastEthanKoTurn"][owner] = turn


def _attack_damage_log_identities(logs: list[Any] | None,
                                  attacker: int) -> list[list[tuple[int, int | None]]]:
    out: list[list[tuple[int, int | None]]] = [[], []]
    for log in logs or []:
        if int(_log_value(log, "type", -1)) != LOG_HP_CHANGE:
            continue
        owner = int(_log_value(log, "playerIndex", -1))
        if owner not in (0, 1) or owner == attacker:
            continue
        if bool(_log_value(log, "putDamageCounter", False)):
            continue
        value = int(_log_value(log, "value", 0) or 0)
        if value >= 0:
            continue
        card_id = _log_value(log, "cardId")
        if card_id is None:
            continue
        serial = _log_value(log, "serial")
        out[owner].append((int(card_id), int(serial) if serial is not None else None))
    return out


def _attack_damage_log_owners(logs: list[Any] | None,
                              attacker: int) -> list[bool]:
    out = [False, False]
    for log in logs or []:
        if int(_log_value(log, "type", -1)) != LOG_HP_CHANGE:
            continue
        owner = int(_log_value(log, "playerIndex", -1))
        if owner not in (0, 1) or owner == attacker:
            continue
        if bool(_log_value(log, "putDamageCounter", False)):
            continue
        value = int(_log_value(log, "value", 0) or 0)
        if value < 0:
            out[owner] = True
    return out


def _identity_in(identity: tuple[int, int | None],
                 identities: list[tuple[int, int | None]]) -> bool:
    cid, serial = identity
    for other_cid, other_serial in identities:
        if cid != other_cid:
            continue
        if serial is None or other_serial is None or serial == other_serial:
            return True
    return False


def _update_ko_transients(transients: dict[str, Any],
                          before: dict[str, Any] | None,
                          after: dict[str, Any] | None,
                          logs: list[Any] | None = None) -> None:
    if before is None or after is None:
        return
    turn = int(before.get("turn", -1))
    if turn < 0:
        return
    transients.setdefault("lastKoTurn", [-1, -1])
    transients.setdefault("lastAttackDamageKoTurn", [-1, -1])
    transients.setdefault("lastTeamRocketKoTurn", [-1, -1])
    transients.setdefault("lastEthanKoTurn", [-1, -1])
    transients.setdefault("prizeTakenTurn", [-1, -1])
    transients.setdefault("prizeTakenCount", [0, 0])
    transients.setdefault("pendingKoTurn", [-1, -1])
    transients.setdefault("pendingKoByOpponentTurn", [False, False])
    transients.setdefault("pendingKoRemoved", [[], []])
    transients.setdefault("pendingAttackDamageKoRemoved", [[], []])
    removed_by_owner: list[list[int]] = [[], []]
    attack_damage_removed_by_owner: list[list[int]] = [[], []]
    actor = int(before.get("yourIndex", -1))
    attack_damage_logs = _attack_damage_log_identities(logs, actor)
    attack_damage_owners = _attack_damage_log_owners(logs, actor)
    for owner in (0, 1):
        before_ids = _pokemon_identities_in_play(before["players"][owner])
        after_ids = _pokemon_identities_in_play(after["players"][owner])
        raw_removed = list((before_ids - after_ids).elements())
        attack_damage_fallback = (
            owner != actor and attack_damage_owners[owner] and
            not attack_damage_logs[owner] and len(raw_removed) == 1
        )
        if "discard" in before["players"][owner] and \
                "discard" in after["players"][owner]:
            discard_delta = (
                _discard_id_counts(after["players"][owner]) -
                _discard_id_counts(before["players"][owner])
            )
            for identity in raw_removed:
                card_id = identity[0]
                if discard_delta[card_id] <= 0:
                    continue
                removed_by_owner[owner].append(card_id)
                if _identity_in(identity, attack_damage_logs[owner]) or \
                        attack_damage_fallback:
                    attack_damage_removed_by_owner[owner].append(card_id)
                discard_delta[card_id] -= 1
        else:
            for identity in raw_removed:
                removed_by_owner[owner].append(identity[0])
                if _identity_in(identity, attack_damage_logs[owner]) or \
                        attack_damage_fallback:
                    attack_damage_removed_by_owner[owner].append(identity[0])
    for log in logs or []:
        if int(_log_value(log, "type", -1)) != LOG_MOVE:
            continue
        owner = int(_log_value(log, "playerIndex", -1))
        if owner not in (0, 1):
            continue
        if int(_log_value(log, "fromArea", -1)) not in (4, 5):
            continue
        if int(_log_value(log, "toArea", -1)) != 3:
            continue
        card_id = _log_value(log, "cardId")
        if card_id is not None:
            removed_by_owner[owner].append(int(card_id))
    for owner in (0, 1):
        if not removed_by_owner[owner]:
            continue
        transients["pendingKoTurn"][owner] = turn
        transients["pendingKoByOpponentTurn"][owner] = actor != owner
        pending_removed = list(transients["pendingKoRemoved"][owner])
        pending_removed.extend(removed_by_owner[owner])
        transients["pendingKoRemoved"][owner] = pending_removed
        pending_attack_damage_removed = list(
            transients["pendingAttackDamageKoRemoved"][owner])
        pending_attack_damage_removed.extend(attack_damage_removed_by_owner[owner])
        transients["pendingAttackDamageKoRemoved"][owner] = \
            pending_attack_damage_removed
        if actor != owner:
            _mark_ko_last_turn(
                transients,
                owner,
                turn,
                removed_by_owner[owner],
                attack_damage=bool(attack_damage_removed_by_owner[owner]),
            )
    for prize_taker in (0, 1):
        before_prizes = len(before["players"][prize_taker].get("prize") or [])
        after_prizes = len(after["players"][prize_taker].get("prize") or [])
        if after_prizes >= before_prizes:
            continue
        delta = before_prizes - after_prizes
        if transients["prizeTakenTurn"][prize_taker] == turn:
            transients["prizeTakenCount"][prize_taker] += delta
        else:
            transients["prizeTakenTurn"][prize_taker] = turn
            transients["prizeTakenCount"][prize_taker] = delta
        owner = 1 - prize_taker
        removed = list(transients["pendingKoRemoved"][owner])
        removed.extend(removed_by_owner[owner])
        attack_damage_removed = list(
            transients["pendingAttackDamageKoRemoved"][owner])
        attack_damage_removed.extend(attack_damage_removed_by_owner[owner])
        if transients["pendingKoTurn"][owner] == turn:
            ko_by_opponent = bool(transients["pendingKoByOpponentTurn"][owner])
        else:
            # Some cabt prize flows expose only the prize-count drop to the
            # bridge. Fall back to whose turn produced the prize drop: a player
            # losing their own Pokemon on their own turn does not enable
            # KO-last-opponent-turn cards such as Hassel.
            ko_by_opponent = actor != owner
        if ko_by_opponent:
            _mark_ko_last_turn(
                transients,
                owner,
                turn,
                removed,
                attack_damage=bool(attack_damage_removed),
            )
        transients["pendingKoTurn"][owner] = -1
        transients["pendingKoByOpponentTurn"][owner] = False
        transients["pendingKoRemoved"][owner] = []
        transients["pendingAttackDamageKoRemoved"][owner] = []


def _update_attack_damage_transients(transients: dict[str, Any],
                                     before: dict[str, Any] | None,
                                     after: dict[str, Any] | None,
                                     selected_main: tuple | None) -> None:
    if before is None or after is None or selected_main is None:
        return
    if selected_main[0] != "ATTACK":
        return
    turn = int(before.get("turn", -1))
    if turn < 0:
        return
    attacker = int(before["yourIndex"])
    kept = [
        eff for eff in transients.get("attackDamage", [])
        if int(eff.get("turn", -999)) >= turn - 1
    ]
    by_key = {
        (int(eff["owner"]), tuple(eff["identity"])): eff
        for eff in kept
    }
    for owner in (0, 1):
        before_cards = {
            ident: card
            for card in _iter_inplay_cards(before["players"][owner])
            for ident in [_card_identity(card)]
            if ident is not None
        }
        after_cards = {
            ident: card
            for card in _iter_inplay_cards(after["players"][owner])
            for ident in [_card_identity(card)]
            if ident is not None
        }
        for identity, before_card in before_cards.items():
            after_card = after_cards.get(identity)
            if after_card is None:
                continue
            amount = int(before_card.get("hp", 0)) - int(after_card.get("hp", 0))
            if amount <= 0:
                continue
            by_key[(owner, identity)] = {
                "owner": owner,
                "identity": identity,
                "turn": turn,
                "attacker": attacker,
                "amount": amount,
            }
    transients["attackDamage"] = list(by_key.values())


def _random_selection(select: dict[str, Any], rng: random.Random) -> list[int]:
    n = len(select.get("option") or [])
    if n == 0:
        return []
    k = rng.randint(int(select["minCount"]), min(int(select["maxCount"]), n))
    return rng.sample(range(n), k)


def _is_empty_zero_pending_select(select: Any) -> bool:
    if select is None:
        return False
    data = asdict(select) if not isinstance(select, dict) else select
    if int(data.get("context", MAIN_CONTEXT)) == MAIN_CONTEXT:
        return False
    return int(data.get("minCount", -1)) == 0 and \
        int(data.get("maxCount", -1)) == 0 and \
        not (data.get("option") or [])


def check_main_state(obs: dict[str, Any], deck0: list[int], deck1: list[int],
                     rng: random.Random, stats: ParityStats, label: str,
                     branch_limit: int | None = None,
                     strict_duplicates: bool = False,
                     strict_order: bool = False,
                     skip_deck_search_branches: bool = False,
                     transients: dict[str, Any] | None = None) -> None:
    current = obs["current"]
    select = obs["select"]
    det, hidden = _determinize(current, deck0, deck1, rng)
    cabt_options = canonical_options(current, select)
    native_state = eng.load_state(_state_for_native(current, hidden, transients),
                                  0, cabt_options)

    load_diffs = diff(canonical_state(current), eng.canonical(native_state))
    load_diffs = [
        d for d in load_diffs
        if not _is_prize_identity_path(d[0]) and
        not _is_bridge_only_path(d[0])
    ]
    if load_diffs:
        lines = "\n".join(
            f"  {path}: cabt={a!r} native={b!r}" for path, a, b in load_diffs[:12]
        )
        raise ParityFailure(f"{label}: load-state canonical mismatch\n{lines}")

    native_options = [tuple(o) for o in eng.legal_main(native_state)]
    compare_cabt_options = cabt_options
    compare_native_options = native_options
    if skip_deck_search_branches:
        compare_cabt_options = _without_deck_search_plays(compare_cabt_options)
        compare_native_options = _without_deck_search_plays(compare_native_options)
    _compare_options(f"{label}: root MAIN", compare_cabt_options, compare_native_options,
                     strict_duplicates, strict_order, current=current,
                     native_state=native_state, raw_native=native_options)
    native_by_desc = {_norm_desc(o): tuple(o) for o in native_options}
    stats.states_checked += 1

    root = None
    checked_here = 0
    try:
        root = search_begin(
            to_observation_class(obs),
            your_deck=det.your_deck,
            your_prize=det.your_prize,
            opponent_deck=det.opponent_deck,
            opponent_prize=det.opponent_prize,
            opponent_hand=det.opponent_hand,
            opponent_active=det.opponent_active,
        )

        for option_index, desc in enumerate(cabt_options):
            norm = _norm_desc(desc)
            stats.action_kinds[str(norm[0] if norm else "EMPTY")] += 1
            if branch_limit is not None and checked_here >= branch_limit:
                stats.branches_skipped_limit += 1
                continue
            if skip_deck_search_branches and _is_deck_search_play(norm):
                stats.branches_skipped_deck_search += 1
                continue
            if norm not in native_by_desc:
                stats.branches_skipped_missing_native += 1
                continue

            child = search_step(root.searchId, [option_index])
            child_logs = _new_logs(obs.get("logs", []), child.observation.logs)
            source_card = _source_card_for_action(current, norm)
            child_tape = _action_replay_tape(select, option_index, child_logs, norm,
                                             source_card)
            empty_resolved = None
            try:
                child_coin_logs = _coin_log_values(child_logs)
                if child_coin_logs:
                    stats.branches_replayed_coin += 1
                branched_native = eng.clone(native_state)
                native_action = native_by_desc[norm]
                if option_index < len(native_options) and \
                        _norm_desc(native_options[option_index]) == norm:
                    native_action = tuple(native_options[option_index])
                eng.apply(branched_native, native_action, child_tape)
                compare_obs = child.observation
                if _is_empty_zero_pending_select(child.observation.select):
                    try:
                        empty_resolved = search_step(child.searchId, [])
                        compare_obs = empty_resolved.observation
                    except ValueError as exc:
                        if "battle has ended" not in str(exc):
                            raise
                        compare_obs = SimpleNamespace(
                            current=child.observation.current,
                            select=None,
                        )
                source_note = f" source_card={source_card}" if source_card is not None else ""
                replay_note = (
                    f" coin_logs={child_coin_logs} replay_tape={child_tape}"
                    if child_coin_logs else
                    (f" replay_tape={child_tape}" if child_tape else "")
                )
                before_native_count = eng.canonical(native_state).get("turnActionCount")
                before_cabt_count = canonical_state(current).get("turnActionCount")
                _compare_next_decision(
                    f"{label}: action[{option_index}]={norm}{source_note}"
                    f"{replay_note}"
                    f" pre_count cabt={before_cabt_count} native={before_native_count}"
                    f" pre_active {_active_debug_summary(current)}",
                    compare_obs,
                    branched_native,
                    strict_duplicates,
                    strict_order,
                    ignore_deck_count_player=(
                        int(current["yourIndex"]) if _is_deck_search_play(norm) else None
                    ),
                    skip_deck_search_options=skip_deck_search_branches,
                )
                stats.branches_checked += 1
                checked_here += 1
                if compare_obs.select is None:
                    stats.next_contexts["terminal"] += 1
                else:
                    stats.next_contexts[int(compare_obs.select.context)] += 1
            finally:
                if empty_resolved is not None:
                    search_release(empty_resolved.searchId)
                search_release(child.searchId)
    finally:
        if root is not None:
            try:
                search_release(root.searchId)
            except Exception:
                pass
        search_end()


def run_random_branch_parity(deck0: list[int], deck1: list[int], *,
                             games: int = 3,
                             seed: int = 0,
                             max_steps: int = 400,
                             max_states: int | None = None,
                             branch_limit: int | None = None,
                             strict_duplicates: bool = False,
                             strict_order: bool = False,
                             skip_deck_search_branches: bool = False) -> ParityStats:
    stats = ParityStats()
    policy_rng = random.Random(seed)
    det_rng = random.Random(seed ^ 0x5EED_BA5E)

    for game_index in range(games):
        obs, sd = battle_start(deck0, deck1)
        if obs is None or sd.errorType != 0:
            raise RuntimeError(
                f"battle_start failed game={game_index} "
                f"errorPlayer={sd.errorPlayer} errorType={sd.errorType}"
            )
        stats.games += 1
        transients: dict[str, Any] = {
            "turn": -1,
            "player": -1,
            "fightingBuff": 0,
            "stadiumAbilityUsed": False,
            "supporterPlayed": False,
            "teamRocketSupporterPlayed": False,
            "ancientSupporterPlayed": False,
            "canariPlayed": False,
            "tarragonPlayed": False,
            "pendingTarragonTurn": -1,
            "pendingTarragonPlayer": -1,
            "noItemTurn": [-1, -1],
            "noSupporterTurn": [-1, -1],
            "noEvolveTurn": [-1, -1],
            "noStadiumTurn": [-1, -1],
            "teamReduceTurn": [-1, -1],
            "teamReduceAmount": [0, 0],
            "teamReduceType": [-1, -1],
            "activeExDamageBuffTurn": [-1, -1],
            "activeExDamageBuffAmount": [0, 0],
            "prizeBonusTurn": [-1, -1],
            "prizeBonusAmount": [0, 0],
            "prizeBonusKind": [0, 0],
            "lastKoTurn": [-1, -1],
            "lastKoTurnSource": ["", ""],
            "lastAttackDamageKoTurn": [-1, -1],
            "lastAttackDamageKoTurnSource": ["", ""],
            "lastTeamRocketKoTurn": [-1, -1],
            "lastTeamRocketKoTurnSource": ["", ""],
            "lastEthanKoTurn": [-1, -1],
            "lastAttackTurn": [-1, -1],
            "lastAttackId": [0, 0],
            "lastAncientAttackTurn": [-1, -1],
            "lastAncientAttackCard": [0, 0],
            "lastAncientAttackSerial": [0, 0],
            "poisonDamageCounters": [1, 1],
            "prizeTakenTurn": [-1, -1],
            "prizeTakenCount": [0, 0],
            "pendingKoTurn": [-1, -1],
            "pendingKoByOpponentTurn": [False, False],
            "pendingKoRemoved": [[], []],
            "pendingAttackDamageKoRemoved": [[], []],
            "inplayEffects": [],
            "attackDamage": [],
            "energyAttachOrder": {},
            "energyAttachCounter": 0,
            "toolAttachOrder": {},
            "toolAttachCounter": 0,
            "movedToActiveThisTurn": [[], []],
            "healedThisTurn": [[], []],
        }
        try:
            for step in range(max_steps):
                current = obs.get("current")
                select = obs.get("select")
                if current is None or select is None or current.get("result", -1) != -1:
                    break
                if transients["turn"] != int(current["turn"]) or \
                        transients["player"] != int(current["yourIndex"]):
                    last_ko = list(transients["lastKoTurn"])
                    last_ko_source = list(
                        transients.get("lastKoTurnSource", ["", ""]))
                    last_attack_damage_ko = list(
                        transients["lastAttackDamageKoTurn"])
                    last_attack_damage_ko_source = list(
                        transients.get("lastAttackDamageKoTurnSource", ["", ""]))
                    last_team_rocket_ko = list(transients["lastTeamRocketKoTurn"])
                    last_team_rocket_ko_source = list(
                        transients.get("lastTeamRocketKoTurnSource", ["", ""]))
                    last_ethan_ko = list(transients["lastEthanKoTurn"])
                    last_attack_turn = list(transients["lastAttackTurn"])
                    last_attack_id = list(transients["lastAttackId"])
                    last_ancient_attack_turn = list(
                        transients["lastAncientAttackTurn"])
                    last_ancient_attack_card = list(
                        transients["lastAncientAttackCard"])
                    last_ancient_attack_serial = list(
                        transients["lastAncientAttackSerial"])
                    poison_damage_counters = list(transients["poisonDamageCounters"])
                    prize_taken_turn = list(transients["prizeTakenTurn"])
                    prize_taken_count = list(transients["prizeTakenCount"])
                    pending_ko_turn = list(transients["pendingKoTurn"])
                    pending_ko_by_opponent = list(transients["pendingKoByOpponentTurn"])
                    pending_ko_removed = [
                        list(v) for v in transients["pendingKoRemoved"]
                    ]
                    pending_attack_damage_ko_removed = [
                        list(v) for v in transients["pendingAttackDamageKoRemoved"]
                    ]
                    no_item_turn = list(transients["noItemTurn"])
                    no_supporter_turn = list(transients["noSupporterTurn"])
                    no_evolve_turn = list(transients["noEvolveTurn"])
                    no_stadium_turn = list(transients["noStadiumTurn"])
                    team_reduce_turn = list(transients["teamReduceTurn"])
                    team_reduce_amount = list(transients["teamReduceAmount"])
                    team_reduce_type = list(transients["teamReduceType"])
                    active_ex_damage_buff_turn = list(
                        transients["activeExDamageBuffTurn"]
                    )
                    active_ex_damage_buff_amount = list(
                        transients["activeExDamageBuffAmount"]
                    )
                    prize_bonus_turn = list(transients["prizeBonusTurn"])
                    prize_bonus_amount = list(transients["prizeBonusAmount"])
                    prize_bonus_kind = list(transients["prizeBonusKind"])
                    inplay_effects = _prune_inplay_effects(
                        list(transients["inplayEffects"]),
                        int(current["turn"]),
                    )
                    attack_damage = [
                        eff for eff in transients["attackDamage"]
                        if int(eff.get("turn", -999)) >= int(current["turn"]) - 1
                    ]
                    energy_attach_order = dict(transients["energyAttachOrder"])
                    energy_attach_counter = int(transients["energyAttachCounter"])
                    tool_attach_order = dict(transients["toolAttachOrder"])
                    tool_attach_counter = int(transients["toolAttachCounter"])
                    moved_to_active = [[], []]
                    healed_this_turn = [[], []]
                    if transients["turn"] == int(current["turn"]):
                        moved_to_active = [
                            list(v) for v in transients.get(
                                "movedToActiveThisTurn", [[], []]
                            )
                        ]
                        healed_this_turn = [
                            list(v) for v in transients.get(
                                "healedThisTurn", [[], []]
                            )
                        ]
                    transients = {
                        "turn": int(current["turn"]),
                        "player": int(current["yourIndex"]),
                        "fightingBuff": 0,
                        "stadiumAbilityUsed": False,
                        "supporterPlayed": False,
                        "teamRocketSupporterPlayed": False,
                        "ancientSupporterPlayed": False,
                        "canariPlayed": False,
                        "tarragonPlayed": False,
                        "pendingTarragonTurn": -1,
                        "pendingTarragonPlayer": -1,
                        "noItemTurn": no_item_turn,
                        "noSupporterTurn": no_supporter_turn,
                        "noEvolveTurn": no_evolve_turn,
                        "noStadiumTurn": no_stadium_turn,
                        "teamReduceTurn": team_reduce_turn,
                        "teamReduceAmount": team_reduce_amount,
                        "teamReduceType": team_reduce_type,
                        "activeExDamageBuffTurn": active_ex_damage_buff_turn,
                        "activeExDamageBuffAmount": active_ex_damage_buff_amount,
                        "prizeBonusTurn": prize_bonus_turn,
                        "prizeBonusAmount": prize_bonus_amount,
                        "prizeBonusKind": prize_bonus_kind,
                        "lastKoTurn": last_ko,
                        "lastKoTurnSource": last_ko_source,
                        "lastAttackDamageKoTurn": last_attack_damage_ko,
                        "lastAttackDamageKoTurnSource":
                            last_attack_damage_ko_source,
                        "lastTeamRocketKoTurn": last_team_rocket_ko,
                        "lastTeamRocketKoTurnSource":
                            last_team_rocket_ko_source,
                        "lastEthanKoTurn": last_ethan_ko,
                        "lastAttackTurn": last_attack_turn,
                        "lastAttackId": last_attack_id,
                        "lastAncientAttackTurn": last_ancient_attack_turn,
                        "lastAncientAttackCard": last_ancient_attack_card,
                        "lastAncientAttackSerial": last_ancient_attack_serial,
                        "poisonDamageCounters": poison_damage_counters,
                        "prizeTakenTurn": prize_taken_turn,
                        "prizeTakenCount": prize_taken_count,
                        "pendingKoTurn": pending_ko_turn,
                        "pendingKoByOpponentTurn": pending_ko_by_opponent,
                        "pendingKoRemoved": pending_ko_removed,
                        "pendingAttackDamageKoRemoved":
                            pending_attack_damage_ko_removed,
                        "inplayEffects": inplay_effects,
                        "attackDamage": attack_damage,
                        "energyAttachOrder": energy_attach_order,
                        "energyAttachCounter": energy_attach_counter,
                        "toolAttachOrder": tool_attach_order,
                        "toolAttachCounter": tool_attach_counter,
                        "movedToActiveThisTurn": moved_to_active,
                        "healedThisTurn": healed_this_turn,
                    }

                if int(select["context"]) == MAIN_CONTEXT:
                    _sync_ko_gate_transients_from_main_options(
                        transients, current, select,
                    )
                    _sync_action_lock_transients_from_main_options(
                        transients, current, select,
                    )
                    _sync_prize_bonus_transients_from_current(transients, current)
                    check_main_state(
                        obs, deck0, deck1, det_rng, stats,
                        label=f"game={game_index} step={step} "
                              f"turn={current['turn']} p={current['yourIndex']}",
                        branch_limit=branch_limit,
                        strict_duplicates=strict_duplicates,
                        strict_order=strict_order,
                        skip_deck_search_branches=skip_deck_search_branches,
                        transients=transients,
                    )
                    if max_states is not None and stats.states_checked >= max_states:
                        break

                selection = _random_selection(select, policy_rng)
                selected_main = None
                if int(select["context"]) == MAIN_CONTEXT and len(selection) == 1:
                    options = canonical_options(current, select)
                    idx = int(selection[0])
                    if 0 <= idx < len(options):
                        selected_main = tuple(options[idx])
                    if selected_main == ("PLAY", 1141):
                        transients["fightingBuff"] += 30
                    _track_supporter_play_transients(transients, selected_main)
                    _track_stadium_play_transients(transients, selected_main)
                    _track_iron_defender_play(transients, current, selected_main)
                    _track_prize_bonus_play(transients, current, selected_main)
                    if selected_main == ("PLAY", 1211):
                        actor = int(current["yourIndex"])
                        turn = int(current["turn"])
                        if transients["activeExDamageBuffTurn"][actor] != turn:
                            transients["activeExDamageBuffAmount"][actor] = 0
                        transients["activeExDamageBuffTurn"][actor] = \
                            turn
                        transients["activeExDamageBuffAmount"][actor] += 40
                    if _is_stadium_ability_desc(selected_main):
                        transients["stadiumAbilityUsed"] = True
                    if selected_main == ("PLAY", 1238):
                        transients["pendingTarragonTurn"] = int(current["turn"])
                        transients["pendingTarragonPlayer"] = int(current["yourIndex"])

                next_obs = battle_select(selection)
                current_logs = obs.get("logs", []) if isinstance(obs, dict) else []
                next_logs = next_obs.get("logs", []) if isinstance(next_obs, dict) else []
                new_logs = _new_logs(current_logs, next_logs)
                next_current = next_obs.get("current") if isinstance(next_obs, dict) else None
                next_select = next_obs.get("select") if isinstance(next_obs, dict) else None
                _clear_inplay_effects_for_changed_active(
                    transients,
                    current,
                    next_current,
                    new_logs,
                )
                _track_tarragon_completion(
                    transients,
                    current,
                    next_current,
                    new_logs,
                    next_select,
                )
                _track_prevent_target_resolution(
                    transients,
                    current,
                    select,
                    selection,
                )
                _track_coin_prevention(
                    transients,
                    current,
                    selected_main,
                    new_logs,
                )
                _track_deterministic_damage_prevention(
                    transients, current, selected_main)
                _track_shadowy_side_kick_prevention(
                    transients, current, next_current, selected_main, new_logs)
                _track_self_damage_reduction(
                    transients, current, selected_main)
                _track_opp_attack_reduction(
                    transients, current, selected_main)
                _track_no_retreat(transients, current, selected_main)
                _track_attack_retreat_cost_more(
                    transients, current, selected_main)
                _track_attack_flip_fail(transients, current, selected_main)
                _track_no_weakness(transients, current, selected_main)
                _track_delayed_end_turn_effects(transients, current, selected_main)
                _track_if_damaged_reactive_effects(
                    transients, current, selected_main)
                _track_energy_attach_reactive_effects(
                    transients, current, selected_main)
                _track_named_attack_bonus(transients, current, selected_main)
                _track_poison_damage_counters(
                    transients, current, next_current, selected_main)
                _track_energy_attach_order(transients, new_logs)
                _track_tool_attach_order(transients, new_logs)
                _track_moved_to_active(transients, new_logs)
                _track_healed_this_turn(transients, current, next_current)
                if selected_main is not None and selected_main[0] == "ATTACK" and \
                        selected_main[1] in ITEM_LOCK_ATTACKS:
                    transients["noItemTurn"][1 - int(current["yourIndex"])] = \
                        int(current["turn"]) + 1
                if selected_main is not None and selected_main[0] == "ATTACK":
                    actor = int(current["yourIndex"])
                    attack_id = int(selected_main[1])
                    transients["lastAttackTurn"][actor] = int(current["turn"])
                    transients["lastAttackId"][actor] = attack_id
                    actor_state = (current.get("players") or [])[actor]
                    active = ((actor_state.get("active") or [None])[0]
                              if isinstance(actor_state, dict) else None)
                    active_id = int(active.get("id", 0)) \
                        if isinstance(active, dict) else 0
                    if active_id in ANCIENT_CARD_IDS:
                        transients["lastAncientAttackTurn"][actor] = \
                            int(current["turn"])
                        transients["lastAncientAttackCard"][actor] = active_id
                        transients["lastAncientAttackSerial"][actor] = \
                            int(active.get("serial", 0) or 0) \
                            if isinstance(active, dict) else 0
                    if attack_id in SUPPORTER_LOCK_ATTACKS:
                        transients["noSupporterTurn"][1 - actor] = int(current["turn"]) + 1
                    if attack_id in EVOLVE_LOCK_ATTACKS:
                        transients["noEvolveTurn"][1 - actor] = int(current["turn"]) + 1
                    if attack_id in STADIUM_LOCK_ATTACKS and \
                            _has_opponent_stadium(current, actor):
                        transients["noStadiumTurn"][1 - actor] = int(current["turn"]) + 1
                _update_ko_transients(
                    transients,
                    current,
                    next_current,
                    new_logs,
                )
                _update_attack_damage_transients(
                    transients,
                    current,
                    next_current,
                    selected_main,
                )
                obs = next_obs
                stats.battle_steps += 1
            else:
                raise RuntimeError(f"game={game_index} exceeded max_steps={max_steps}")
        finally:
            battle_finish()

        if max_states is not None and stats.states_checked >= max_states:
            break

    return stats


def run_random_branch_parity_pairs(pairs: list[DeckPair], *,
                                   games: int = 3,
                                   seed: int = 0,
                                   max_steps: int = 400,
                                   max_states: int | None = None,
                                   branch_limit: int | None = None,
                                   strict_duplicates: bool = False,
                                   strict_order: bool = False,
                                   skip_deck_search_branches: bool = False) -> ParityStats:
    stats = ParityStats()
    for pair_index, pair in enumerate(pairs):
        remaining_states = None if max_states is None else max_states - stats.states_checked
        if remaining_states is not None and remaining_states <= 0:
            break
        try:
            pair_stats = run_random_branch_parity(
                pair.deck0,
                pair.deck1,
                games=games,
                seed=seed + pair_index * 1_000_003,
                max_steps=max_steps,
                max_states=remaining_states,
                branch_limit=branch_limit,
                strict_duplicates=strict_duplicates,
                strict_order=strict_order,
                skip_deck_search_branches=skip_deck_search_branches,
            )
        except ParityFailure as exc:
            raise ParityFailure(f"{pair.name0} vs {pair.name1}: {exc}") from exc
        except RuntimeError as exc:
            raise RuntimeError(f"{pair.name0} vs {pair.name1}: {exc}") from exc
        stats.merge(pair_stats)
    return stats


def run_generated_random_deck_parity(deck_count: int, *,
                                     random_deck_seed: int = 0,
                                     cover_all: bool = True,
                                     sampling_method: str = "coverage",
                                     games: int = 3,
                                     seed: int = 0,
                                     max_steps: int = 400,
                                     max_states: int | None = None,
                                     branch_limit: int | None = None,
                                     strict_duplicates: bool = False,
                                     strict_order: bool = False,
                                     skip_deck_search_branches: bool = False) -> ParityStats:
    decks = generate_random_decks(
        count=deck_count,
        seed=random_deck_seed,
        cover_all=cover_all,
        sampling_method=sampling_method,
    )
    return run_random_branch_parity_pairs(
        pair_generated_decks(decks),
        games=games,
        seed=seed,
        max_steps=max_steps,
        max_states=max_states,
        branch_limit=branch_limit,
        strict_duplicates=strict_duplicates,
        strict_order=strict_order,
        skip_deck_search_branches=skip_deck_search_branches,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck0", default="mega_lucario",
                        help="deck name from validation/decks.py or path to 60 IDs")
    parser.add_argument("--deck1", default=None,
                        help="deck name/path; defaults to --deck0")
    parser.add_argument("--random-decks", type=int, default=0, metavar="N",
                        help="generate N decks with validation.decks.generate_random_decks and test them in pairs")
    parser.add_argument("--random-deck-seed", type=int, default=0,
                        help="seed for generated random decks")
    parser.add_argument("--random-deck-no-cover", action="store_true",
                        help="disable coverage-biased random deck generation")
    parser.add_argument("--random-deck-method", choices=("coverage", "competitive"),
                        default="coverage",
                        help="random deck sampling method")
    parser.add_argument("--games", type=int, default=3,
                        help="games per deck pair")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the random policy and hidden-zone determinization")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-states", type=int, default=None,
                        help="stop after checking this many MAIN states")
    parser.add_argument("--branch-limit", type=int, default=None,
                        help="limit checked branches per sampled MAIN state")
    parser.add_argument("--strict-duplicates", action="store_true",
                        help="compare duplicate semantic options as multiplicities")
    parser.add_argument("--strict-order", action="store_true",
                        help="compare legal/pending option order after semantic normalization")
    parser.add_argument("--skip-deck-search-branches", action="store_true",
                        help="skip result comparison for deck-search PLAY branches")
    args = parser.parse_args()

    if args.random_decks:
        from validation.random_decks import random_deck_coverage

        decks = generate_random_decks(
            count=args.random_decks,
            seed=args.random_deck_seed,
            cover_all=not args.random_deck_no_cover,
            sampling_method=args.random_deck_method,
        )
        pairs = pair_generated_decks(decks)
        coverage = random_deck_coverage(decks)
        print(
            f"random_decks={len(decks)} method={args.random_deck_method} pairs={len(pairs)} "
            f"coverage={coverage.covered}/{coverage.total} ({coverage.ratio:.1%})"
        )
        stats = run_random_branch_parity_pairs(
            pairs,
            games=args.games,
            seed=args.seed,
            max_steps=args.max_steps,
            max_states=args.max_states,
            branch_limit=args.branch_limit,
            strict_duplicates=args.strict_duplicates,
            strict_order=args.strict_order,
            skip_deck_search_branches=args.skip_deck_search_branches,
        )
    else:
        deck0 = load_deck(args.deck0)
        deck1 = load_deck(args.deck1 or args.deck0)
        stats = run_random_branch_parity(
            deck0,
            deck1,
            games=args.games,
            seed=args.seed,
            max_steps=args.max_steps,
            max_states=args.max_states,
            branch_limit=args.branch_limit,
            strict_duplicates=args.strict_duplicates,
            strict_order=args.strict_order,
            skip_deck_search_branches=args.skip_deck_search_branches,
        )
    print(stats.summary())


if __name__ == "__main__":
    main()
