"""Native backend for the public ``ptcg.cg.game`` compatibility API.

This module intentionally mirrors the small battle-loop surface from
``ptcg.cg.game`` while delegating state transitions to ``ptcg_engine``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import random
import sys
from collections import Counter

from ptcg.cg.native_payload import encode_native_search_begin

MASK64 = (1 << 64) - 1
_ENGINE_CACHE = None


@dataclass
class NativeStartData:
    """Start data shaped like ``cg.sim.StartData`` for normal Python callers."""

    battlePtr: int | None = None
    errorPlayer: int = -1
    errorType: int = 0


class NativeBattle:
    battle = None
    state = None
    setup = None
    obs = None
    last_logs: list[dict] = []
    deck0: list[int] | None = None
    deck1: list[int] | None = None
    seed: int = 1
    generation: int = 0


@dataclass
class _SetupPlayer:
    deck: list[int]
    hand: list[int] = field(default_factory=list)
    prize: list[int] = field(default_factory=list)
    active: int | None = None
    bench: list[int] = field(default_factory=list)
    bench_choices: list[int] = field(default_factory=list)
    mulligans: int = 0


@dataclass
class _SetupState:
    players: list[_SetupPlayer]
    rng: random.Random
    first_player: int = -1
    phase: str = "choose_first"
    select_player: int = 0
    active_order: list[int] = field(default_factory=list)
    active_done: set[int] = field(default_factory=set)
    bonus_pos: int = 0
    bench_pos: int = 0


def _engine():
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_dirs = [
        os.path.join(root, "engine"),
        os.path.join(root, "engine", "build"),
        os.path.join(root, "engine", "build", "Release"),
    ]
    for build_dir in build_dirs:
        if os.path.isdir(build_dir):
            while build_dir in sys.path:
                sys.path.remove(build_dir)
            sys.path.insert(0, build_dir)
    import ptcg_engine as E

    _ENGINE_CACHE = E
    return E


def _next_seed(seed: int) -> int:
    return (seed * 6364136223846793005 + 1442695040888963407) & MASK64


def _seed() -> int:
    raw = os.environ.get("PTCG_NATIVE_SEED", "1")
    try:
        return int(raw) & MASK64
    except ValueError:
        return 1


def _portable_search_payload() -> bool:
    return os.environ.get("PTCG_NATIVE_PORTABLE_SEARCH", "").lower() in {"1", "true", "yes"}


def _lazy_search_payload() -> bool:
    return os.environ.get("PTCG_NATIVE_LAZY_SEARCH", "").lower() in {"1", "true", "yes"}


def _cpp_battle_enabled() -> bool:
    return os.environ.get("PTCG_NATIVE_CPP_BATTLE", "").lower() in {"1", "true", "yes"}


def _current_generation() -> int:
    if NativeBattle.battle is not None:
        return int(NativeBattle.battle.generation)
    return int(NativeBattle.generation)


def _attach_search_begin_input(E, obs: dict, context: int, descriptors) -> None:
    main_options = descriptors if int(context) < 0 else None
    portable = _portable_search_payload()
    lazy = _lazy_search_payload() and not portable
    battle = NativeBattle.battle
    if battle is None:
        state = NativeBattle.state
        clone_state = None if lazy else E.clone(state)
        seed = NativeBattle.seed
        generation = NativeBattle.generation
        transients = E.search_transient_snapshot(state) if portable else None
    else:
        state = battle.state
        clone_state = None if lazy else battle.clone_state()
        seed = int(battle.seed)
        generation = int(battle.generation)
        transients = battle.transient_snapshot() if portable else None
    obs["search_begin_input"] = encode_native_search_begin(
        obs["current"],
        state=state if lazy else clone_state,
        context=int(context),
        descriptors=descriptors,
        main_options=main_options,
        transients=transients,
        seed=seed,
        portable=portable,
        live_generation=generation if lazy else None,
        live_generation_getter=_current_generation if lazy else None,
    )


def _observation(logs: list[dict] | None = None) -> dict:
    if NativeBattle.state is None:
        raise ValueError("battle has not started")
    E = _engine()
    if NativeBattle.battle is not None:
        obs, context, descriptors, _engine_logs = NativeBattle.battle.observation()
        NativeBattle.state = NativeBattle.battle.state
        NativeBattle.seed = int(NativeBattle.battle.seed)
        NativeBattle.generation = int(NativeBattle.battle.generation)
    else:
        obs, context, descriptors = E.cg_observation_with_view(NativeBattle.state)
    obs["logs"] = list(logs if logs is not None else NativeBattle.last_logs)
    _attach_search_begin_input(E, obs, int(context), descriptors)
    NativeBattle.obs = obs
    return obs


def _ids(cards) -> list[int]:
    return [int(c["id"]) for c in cards or [] if c is not None and c.get("id") is not None]


_CARD_LOOKUP_CACHE = None


def _card_lookup() -> dict[int, object]:
    global _CARD_LOOKUP_CACHE
    if _CARD_LOOKUP_CACHE is None:
        from ptcg.cg.api import _native_all_card_data

        _CARD_LOOKUP_CACHE = {int(card.cardId): card for card in _native_all_card_data()}
    return _CARD_LOOKUP_CACHE


def _card_info(card_id: int):
    return _card_lookup().get(int(card_id))


def _is_basic_pokemon(card_id: int) -> bool:
    from ptcg.cg.api import CardType

    card = _card_info(card_id)
    return (
        card is not None
        and int(card.cardType) == int(CardType.POKEMON)
        and bool(card.basic)
    )


def _card_dict(card_id: int, player: int, serial: int = 0) -> dict:
    return {"id": int(card_id), "serial": int(serial), "playerIndex": int(player)}


def _pokemon_dict(card_id: int, player: int, serial: int = 0) -> dict:
    card = _card_info(card_id)
    hp = int(card.hp) if card is not None else 0
    return {
        "id": int(card_id),
        "serial": int(serial),
        "playerIndex": int(player),
        "hp": hp,
        "maxHp": hp,
        "appearThisTurn": False,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def _deck_validation_error(deck0: list[int], deck1: list[int]) -> tuple[int, int]:
    from ptcg.cg.api import CardType, _native_all_card_data

    cards = {int(card.cardId): card for card in _native_all_card_data()}
    for player, deck in enumerate((deck0, deck1)):
        name_count: Counter[str] = Counter()
        ace_spec = False
        basic = False
        for raw_id in deck:
            card = cards.get(int(raw_id))
            if card is None:
                return player, 1
            if bool(card.aceSpec):
                if ace_spec:
                    return player, 4
                ace_spec = True
            if int(card.cardType) == int(CardType.POKEMON) and bool(card.basic):
                basic = True
            name_count[str(card.name)] += 1
            if name_count[str(card.name)] > 4 and int(card.cardType) != int(CardType.BASIC_ENERGY):
                return player, 2
        if not basic:
            return player, 3
    return -1, 0


def _active_id(player: dict) -> int | None:
    active = player.get("active") or []
    if not active or active[0] is None:
        return None
    return int(active[0]["id"])


def _active_hp(player: dict) -> int | None:
    active = player.get("active") or []
    if not active or active[0] is None:
        return None
    return int(active[0].get("hp", 0))


def _log(log_type: int, **fields) -> dict:
    return {"type": log_type, **{k: v for k, v in fields.items() if v is not None}}


def _card_log(card_id: int, player: int, **fields) -> dict:
    return {"cardId": int(card_id), "serial": 0, "playerIndex": int(player), **fields}


def _zone_delta(before: list[int], after: list[int]) -> tuple[list[int], list[int]]:
    b = Counter(before)
    a = Counter(after)
    added = list((a - b).elements())
    removed = list((b - a).elements())
    return added, removed


def _movement_logs(before: dict, after: dict) -> list[dict]:
    logs: list[dict] = []
    before_players = before.get("current", before).get("players", [])
    after_players = after.get("current", after).get("players", [])
    for pidx, (bp, ap) in enumerate(zip(before_players, after_players)):
        b_hand = _ids(bp.get("hand"))
        a_hand = _ids(ap.get("hand"))
        b_discard = _ids(bp.get("discard"))
        a_discard = _ids(ap.get("discard"))
        b_bench = _ids(bp.get("bench"))
        a_bench = _ids(ap.get("bench"))

        hand_added, hand_removed = _zone_delta(b_hand, a_hand)
        discard_added, _discard_removed = _zone_delta(b_discard, a_discard)
        bench_added, _bench_removed = _zone_delta(b_bench, a_bench)

        deck_draws = max(0, int(bp.get("deckCount", 0)) - int(ap.get("deckCount", 0)))
        hand_gain = max(0, int(ap.get("handCount", 0)) - int(bp.get("handCount", 0)))
        for cid in hand_added[: min(deck_draws, hand_gain, len(hand_added))]:
            logs.append(_log(4, **_card_log(cid, pidx)))

        for cid in hand_removed:
            if cid in discard_added:
                logs.append(_log(6, **_card_log(cid, pidx, fromArea=2, toArea=3)))
            elif cid in bench_added:
                logs.append(_log(6, **_card_log(cid, pidx, fromArea=2, toArea=5)))

        b_active = _active_id(bp)
        a_active = _active_id(ap)
        if b_active != a_active and b_active is not None and a_active is not None:
            logs.append(_log(
                8,
                playerIndex=pidx,
                cardIdActive=b_active,
                serialActive=0,
                cardIdBench=a_active,
                serialBench=0,
            ))

        b_hp = _active_hp(bp)
        a_hp = _active_hp(ap)
        if b_active == a_active and b_hp is not None and a_hp is not None and b_hp != a_hp:
            logs.append(_log(
                16,
                playerIndex=pidx,
                cardId=a_active,
                serial=0,
                value=int(a_hp - b_hp),
                putDamageCounter=False,
            ))

        for key, log_type in (
            ("poisoned", 17),
            ("burned", 18),
            ("asleep", 19),
            ("paralyzed", 20),
            ("confused", 21),
        ):
            if bool(bp.get(key, False)) != bool(ap.get(key, False)):
                logs.append(_log(log_type, playerIndex=pidx, isRecover=not bool(ap.get(key, False)),
                                 cardId=a_active, serial=0))

    b_result = int(before.get("current", before).get("result", -1))
    a_result = int(after.get("current", after).get("result", -1))
    if b_result < 0 <= a_result:
        logs.append(_log(23, result=a_result, reason=0))
    return logs


def _action_logs(before: dict, after: dict, descriptor: tuple | None,
                 pending_selection: bool) -> list[dict]:
    cur_before = before.get("current", before)
    cur_after = after.get("current", after)
    player = int(cur_before.get("yourIndex", 0))
    logs: list[dict] = []

    if descriptor:
        kind = descriptor[0]
        if kind == "PLAY":
            logs.append(_log(10, **_card_log(int(descriptor[1]), player)))
        elif kind == "ATTACH":
            logs.append(_log(
                11,
                playerIndex=player,
                cardId=int(descriptor[1]),
                serial=0,
                cardIdTarget=_active_id(cur_before["players"][player]),
                serialTarget=0,
            ))
        elif kind == "EVOLVE":
            logs.append(_log(
                12,
                playerIndex=player,
                cardId=int(descriptor[1]),
                serial=0,
                cardIdTarget=None,
                serialTarget=0,
            ))
        elif kind == "ATTACK":
            logs.append(_log(
                15,
                playerIndex=player,
                cardId=_active_id(cur_before["players"][player]),
                serial=0,
                attackId=int(descriptor[1]),
            ))
        elif kind == "END":
            logs.append(_log(3, playerIndex=player))

    if int(cur_before.get("yourIndex", 0)) != int(cur_after.get("yourIndex", 0)):
        logs.append(_log(2, playerIndex=int(cur_after.get("yourIndex", 0))))

    logs.extend(_movement_logs(before, after))
    return logs


def _official_setup_enabled() -> bool:
    return os.environ.get("PTCG_NATIVE_FAST_SETUP", "").lower() not in {"1", "true", "yes"}


def _has_basic(hand: list[int]) -> bool:
    return any(_is_basic_pokemon(card_id) for card_id in hand)


def _setup_card_options(setup: _SetupState, player: int, *, bench: bool) -> list[tuple[int, int]]:
    player_state = setup.players[player]
    room = max(0, 5 - len(player_state.bench_choices)) if bench else 1
    if room <= 0:
        return []
    return [
        (idx, card_id)
        for idx, card_id in enumerate(player_state.hand)
        if _is_basic_pokemon(card_id)
    ]


def _setup_descriptors(setup: _SetupState) -> tuple[int, list[tuple]]:
    if setup.phase == "choose_first":
        return 41, [("YES",), ("NO",)]
    if setup.phase == "active":
        player = setup.select_player
        return 1, [("SETUP_ACTIVE", card_id) for _idx, card_id in _setup_card_options(setup, player, bench=False)]
    if setup.phase == "bonus":
        max_bonus = setup.players[1 - setup.select_player].mulligans
        return 38, [("COUNT", n) for n in range(max_bonus + 1)]
    if setup.phase == "bench":
        player = setup.select_player
        return 2, [("CARD", "HAND", idx, card_id) for idx, card_id in _setup_card_options(setup, player, bench=True)]
    return 0, [("END",)]


def _setup_public_player(setup: _SetupState, player: int, viewer: int) -> dict:
    player_state = setup.players[player]
    active = [None] if player_state.active is not None else []
    hand = (
        [_card_dict(card_id, player, serial=idx) for idx, card_id in enumerate(player_state.hand)]
        if player == viewer else None
    )
    return {
        "active": active,
        "bench": [],
        "benchMax": 5,
        "deck": [],
        "deckCount": len(player_state.deck),
        "discard": [],
        "prize": [None for _ in player_state.prize],
        "handCount": len(player_state.hand),
        "hand": hand,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _setup_current(setup: _SetupState) -> dict:
    viewer = int(setup.select_player)
    return {
        "turn": 0,
        "turnActionCount": 0,
        "yourIndex": viewer,
        "firstPlayer": int(setup.first_player),
        "supporterPlayed": False,
        "stadiumPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": -1,
        "stadium": [],
        "looking": None,
        "players": [
            _setup_public_player(setup, 0, viewer),
            _setup_public_player(setup, 1, viewer),
        ],
    }


def _setup_select_data(setup: _SetupState) -> tuple[dict, list[tuple]]:
    context, descriptors = _setup_descriptors(setup)
    options: list[dict] = []
    min_count = 1
    max_count = 1
    select_type = 1
    if setup.phase == "choose_first":
        select_type = 9
        options = [{"type": 1}, {"type": 2}]
    elif setup.phase == "active":
        player = setup.select_player
        options = [
            {"type": 3, "area": 2, "index": idx, "playerIndex": player}
            for idx, _card_id in _setup_card_options(setup, player, bench=False)
        ]
    elif setup.phase == "bonus":
        select_type = 8
        max_bonus = setup.players[1 - setup.select_player].mulligans
        options = [{"type": 0, "number": n} for n in range(max_bonus + 1)]
    elif setup.phase == "bench":
        player = setup.select_player
        card_options = _setup_card_options(setup, player, bench=True)
        min_count = 0
        max_count = min(5 - len(setup.players[player].bench_choices), len(card_options))
        options = [
            {"type": 3, "area": 2, "index": idx, "playerIndex": player}
            for idx, _card_id in card_options
        ]
    select = {
        "context": context,
        "type": select_type,
        "contextCard": None,
        "minCount": min_count,
        "maxCount": max_count,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "effect": None,
        "deck": None,
        "option": options,
    }
    return select, descriptors


def _setup_obs(logs: list[dict] | None = None) -> dict:
    setup = NativeBattle.setup
    if setup is None:
        raise ValueError("setup is not active")
    select, descriptors = _setup_select_data(setup)
    obs = {
        "current": _setup_current(setup),
        "select": select,
        "logs": list(logs or []),
        "search_begin_input": None,
    }
    context, _ = _setup_descriptors(setup)
    obs["search_begin_input"] = encode_native_search_begin(
        obs["current"],
        context=int(context),
        descriptors=descriptors,
        seed=NativeBattle.seed,
        portable=_portable_search_payload(),
    )
    NativeBattle.obs = obs
    NativeBattle.last_logs = list(logs or [])
    return obs


def _setup_private_player(setup: _SetupState, player: int) -> dict:
    player_state = setup.players[player]
    return {
        "active": (
            [_pokemon_dict(player_state.active, player)]
            if player_state.active is not None else []
        ),
        "bench": [
            _pokemon_dict(card_id, player, serial=idx)
            for idx, card_id in enumerate(player_state.bench)
        ],
        "benchMax": 5,
        "deck": list(player_state.deck),
        "deckKnown": False,
        "deckKnownMask": [False for _ in player_state.deck],
        "deckCount": len(player_state.deck),
        "discard": [],
        "prize": [
            _card_dict(card_id, player, serial=idx)
            for idx, card_id in enumerate(player_state.prize)
        ],
        "prizesKnown": False,
        "prizesKnownMask": [False for _ in player_state.prize],
        "prizeFaceUp": [False for _ in player_state.prize],
        "handCount": len(player_state.hand),
        "hand": [
            _card_dict(card_id, player, serial=idx)
            for idx, card_id in enumerate(player_state.hand)
        ],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _setup_private_current(setup: _SetupState) -> dict:
    first = int(setup.first_player)
    return {
        "turn": 1,
        "turnActionCount": 1,
        "yourIndex": first,
        "firstPlayer": first,
        "supporterPlayed": False,
        "stadiumPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": -1,
        "stadium": [],
        "looking": None,
        "players": [
            _setup_private_player(setup, 0),
            _setup_private_player(setup, 1),
        ],
    }


def _setup_events_to_logs(events: list[tuple], viewer: int) -> list[dict]:
    logs: list[dict] = []
    for event in events:
        kind = event[0]
        if kind == "draw":
            _kind, player, card_id = event
            if int(player) == int(viewer):
                logs.append(_log(4, **_card_log(card_id, player)))
            else:
                logs.append(_log(5, playerIndex=player))
        elif kind == "has_basic":
            _kind, player, has_basic = event
            logs.append(_log(1, playerIndex=player, hasBasicPokemon=bool(has_basic)))
        elif kind == "shuffle":
            _kind, player = event
            logs.append(_log(0, playerIndex=player))
        elif kind == "move":
            _kind, player, card_id, from_area, to_area = event
            if int(player) == int(viewer) and int(card_id) > 0:
                logs.append(_log(6, **_card_log(card_id, player, fromArea=from_area, toArea=to_area)))
            else:
                logs.append(_log(7, playerIndex=player, fromArea=from_area, toArea=to_area))
        elif kind == "turn_start":
            _kind, player = event
            logs.append(_log(2, playerIndex=player))
    return logs


def _setup_draw(setup: _SetupState, player: int, count: int, events: list[tuple]) -> None:
    player_state = setup.players[player]
    for _ in range(int(count)):
        if not player_state.deck:
            break
        card_id = player_state.deck.pop()
        player_state.hand.append(card_id)
        events.append(("draw", player, card_id))


def _setup_mulligan_until_basic(setup: _SetupState, player: int, events: list[tuple]) -> None:
    player_state = setup.players[player]
    while not _has_basic(player_state.hand):
        for card_id in list(player_state.hand):
            events.append(("move", player, card_id, 2, 1))
        player_state.deck.extend(player_state.hand)
        player_state.hand.clear()
        setup.rng.shuffle(player_state.deck)
        player_state.mulligans += 1
        events.append(("shuffle", player))
        _setup_draw(setup, player, 7, events)
        events.append(("has_basic", player, _has_basic(player_state.hand)))


def _setup_initial_draw(setup: _SetupState, events: list[tuple]) -> None:
    for player_state in setup.players:
        setup.rng.shuffle(player_state.deck)
    for player in (0, 1):
        _setup_draw(setup, player, 7, events)
    for player in (0, 1):
        events.append(("has_basic", player, _has_basic(setup.players[player].hand)))


def _setup_deal_prizes(setup: _SetupState, events: list[tuple]) -> None:
    for player, player_state in enumerate(setup.players):
        if player_state.prize:
            continue
        for _ in range(6):
            if not player_state.deck:
                break
            card_id = player_state.deck.pop()
            player_state.prize.append(card_id)
            events.append(("move", player, card_id, 1, 6))


def _setup_advance_active(setup: _SetupState, events: list[tuple]) -> None:
    while True:
        for player in setup.active_order:
            if player not in setup.active_done and _has_basic(setup.players[player].hand):
                setup.phase = "active"
                setup.select_player = player
                return
        for player in setup.active_order:
            if player not in setup.active_done:
                _setup_mulligan_until_basic(setup, player, events)
                setup.phase = "active"
                setup.select_player = player
                return
        _setup_deal_prizes(setup, events)
        setup.phase = "bonus"
        setup.bonus_pos = 0
        _setup_advance_bonus_or_bench(setup, events)
        return


def _setup_advance_bonus_or_bench(setup: _SetupState, events: list[tuple]) -> None:
    while setup.bonus_pos < len(setup.active_order):
        player = setup.active_order[setup.bonus_pos]
        if setup.players[1 - player].mulligans > 0:
            setup.phase = "bonus"
            setup.select_player = player
            return
        setup.bonus_pos += 1
    setup.phase = "bench"
    setup.bench_pos = 0
    _setup_advance_bench(setup, events)


def _setup_advance_bench(setup: _SetupState, events: list[tuple]) -> None:
    while setup.bench_pos < len(setup.active_order):
        player = setup.active_order[setup.bench_pos]
        if _setup_card_options(setup, player, bench=True):
            setup.phase = "bench"
            setup.select_player = player
            return
        setup.bench_pos += 1
    _setup_complete(setup, events)


def _setup_complete(setup: _SetupState, events: list[tuple]) -> None:
    for player in setup.active_order:
        player_state = setup.players[player]
        selected = [
            (idx, player_state.hand[idx])
            for idx in player_state.bench_choices
            if 0 <= idx < len(player_state.hand)
        ]
        for idx, _card_id in sorted(selected, reverse=True):
            player_state.hand.pop(idx)
        for _idx, card_id in selected:
            player_state.bench.append(card_id)
            events.append(("move", player, card_id, 2, 5))

    first = setup.first_player
    events.append(("turn_start", first))
    _setup_draw(setup, first, 1, events)

    E = _engine()
    NativeBattle.state = E.load_state(_setup_private_current(setup), NativeBattle.seed, None)
    NativeBattle.setup = None
    obs, context, descriptors = E.cg_observation_with_view(NativeBattle.state)
    logs = _setup_events_to_logs(events, int(obs["current"]["yourIndex"]))
    NativeBattle.last_logs = logs
    obs["logs"] = logs
    _attach_search_begin_input(E, obs, int(context), descriptors)
    NativeBattle.obs = obs


def _setup_validate_selection(select_list: list[int], min_count: int, max_count: int, n_options: int) -> None:
    if not isinstance(select_list, list) or not all(isinstance(i, int) for i in select_list):
        raise ValueError("select_list is not list[int]")
    if not (min_count <= len(select_list) <= max_count):
        raise ValueError("Must be Observation.select.minCount <= len(select) <= Observation.select.maxCount.")
    if any(i < 0 or i >= n_options for i in select_list):
        raise IndexError()
    if len(set(select_list)) != len(select_list):
        raise ValueError("Duplicate select elements.")


def _setup_select(select_list: list[int]) -> dict:
    setup = NativeBattle.setup
    if setup is None:
        raise ValueError("setup is not active")
    select, _descriptors = _setup_select_data(setup)
    options = list(select["option"])
    _setup_validate_selection(select_list, int(select["minCount"]), int(select["maxCount"]), len(options))
    events: list[tuple] = []

    if setup.phase == "choose_first":
        setup.first_player = 0 if int(select_list[0]) == 0 else 1
        setup.active_order = [setup.first_player, 1 - setup.first_player]
        _setup_initial_draw(setup, events)
        _setup_advance_active(setup, events)
    elif setup.phase == "active":
        player = setup.select_player
        card_options = _setup_card_options(setup, player, bench=False)
        hand_index, card_id = card_options[int(select_list[0])]
        setup.players[player].hand.pop(hand_index)
        setup.players[player].active = card_id
        setup.active_done.add(player)
        events.append(("move", player, card_id, 2, 4))
        _setup_advance_active(setup, events)
    elif setup.phase == "bonus":
        player = setup.select_player
        number = int(options[int(select_list[0])]["number"])
        _setup_draw(setup, player, number, events)
        setup.bonus_pos += 1
        _setup_advance_bonus_or_bench(setup, events)
    elif setup.phase == "bench":
        player = setup.select_player
        card_options = _setup_card_options(setup, player, bench=True)
        setup.players[player].bench_choices = [card_options[int(i)][0] for i in select_list]
        setup.bench_pos += 1
        _setup_advance_bench(setup, events)
    else:
        raise IndexError()

    if NativeBattle.setup is None:
        return NativeBattle.obs
    return _setup_obs(_setup_events_to_logs(events, setup.select_player))


def _step_exact(state, select_list: list[int]) -> None:
    E = _engine()
    pending = E.pending_decision(state)
    if pending is not None:
        E.resolve(state, list(select_list))
        return None

    _ctx, descriptors = E.action_view(state)
    if len(select_list) != 1:
        raise ValueError("Must be Observation.select.minCount <= len(select) <= Observation.select.maxCount.")
    action = int(select_list[0])
    if action < 0 or action >= len(descriptors):
        raise IndexError()
    descriptor = tuple(descriptors[action])
    E.apply(state, descriptor)
    return descriptor


def battle_start(deck0: list[int], deck1: list[int]) -> tuple[dict | None, NativeStartData]:
    """Start a native battle using the ``ptcg.cg.game.battle_start`` shape."""

    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")

    E = _engine()
    NativeBattle.deck0 = list(deck0)
    NativeBattle.deck1 = list(deck1)
    NativeBattle.seed = _seed()
    NativeBattle.generation = 0
    error_player, error_type = _deck_validation_error(NativeBattle.deck0, NativeBattle.deck1)
    if error_type:
        NativeBattle.battle = None
        NativeBattle.state = None
        NativeBattle.setup = None
        return None, NativeStartData(
            battlePtr=None,
            errorPlayer=error_player,
            errorType=error_type,
        )
    try:
        if _official_setup_enabled() and not _cpp_battle_enabled():
            NativeBattle.battle = None
            NativeBattle.state = None
            NativeBattle.setup = _SetupState(
                players=[
                    _SetupPlayer(deck=list(NativeBattle.deck0)),
                    _SetupPlayer(deck=list(NativeBattle.deck1)),
                ],
                rng=random.Random(NativeBattle.seed or 1),
            )
            NativeBattle.last_logs = []
            obs = _setup_obs([])
            return obs, NativeStartData(battlePtr=1, errorPlayer=-1, errorType=0)
        if _cpp_battle_enabled():
            NativeBattle.battle = E.NativeCgBattle()
            obs, context, descriptors, engine_logs = NativeBattle.battle.start(
                NativeBattle.deck0,
                NativeBattle.deck1,
                NativeBattle.seed,
            )
            NativeBattle.state = NativeBattle.battle.state
            NativeBattle.seed = int(NativeBattle.battle.seed)
            NativeBattle.generation = int(NativeBattle.battle.generation)
            NativeBattle.last_logs = list(engine_logs)
        else:
            NativeBattle.battle = None
            NativeBattle.state = E.new_game(
                NativeBattle.deck0,
                NativeBattle.deck1,
                NativeBattle.seed,
            )
            obs, context, descriptors = E.cg_observation_with_view(NativeBattle.state)
            NativeBattle.last_logs = []
    except Exception:
        NativeBattle.battle = None
        NativeBattle.state = None
        NativeBattle.setup = None
        return None, NativeStartData(battlePtr=None, errorPlayer=-1, errorType=1)
    obs["logs"] = NativeBattle.last_logs
    _attach_search_begin_input(E, obs, int(context), descriptors)
    NativeBattle.obs = obs
    return obs, NativeStartData(battlePtr=1, errorPlayer=-1, errorType=0)


def battle_finish() -> None:
    """Finish the active native battle."""

    if NativeBattle.battle is not None:
        NativeBattle.battle.finish()
    NativeBattle.state = None
    NativeBattle.battle = None
    NativeBattle.setup = None
    NativeBattle.obs = None
    NativeBattle.deck0 = None
    NativeBattle.deck1 = None
    NativeBattle.generation += 1


def battle_select(select_list: list[int]) -> dict:
    """Apply one selected option and return the next CABT-shaped observation.

    The native interface exposes a single action index for the current decision
    and auto-resolves forced/multi-select sub-decisions. That keeps the common
    Kaggle loop working, but is not exact parity for callers that rely on
    handling every intermediate multi-select prompt themselves.
    """

    if not isinstance(select_list, list) or not all(isinstance(i, int) for i in select_list):
        raise ValueError("select_list is not list[int]")
    if NativeBattle.state is None:
        if NativeBattle.setup is not None:
            return _setup_select(select_list)
        raise ValueError("battle_ptr broken.")

    try:
        E = _engine()
        if NativeBattle.battle is None:
            obs, context, descriptors, engine_logs = E.cg_select_step(
                NativeBattle.state,
                select_list,
            )
            NativeBattle.seed = _next_seed(NativeBattle.seed)
            NativeBattle.generation += 1
        else:
            obs, context, descriptors, engine_logs = NativeBattle.battle.select(select_list)
            NativeBattle.state = NativeBattle.battle.state
            NativeBattle.seed = int(NativeBattle.battle.seed)
            NativeBattle.generation = int(NativeBattle.battle.generation)
        NativeBattle.last_logs = list(engine_logs)
        obs["logs"] = NativeBattle.last_logs
        _attach_search_begin_input(E, obs, int(context), descriptors)
        NativeBattle.obs = obs
        return obs
    except Exception as exc:
        raise IndexError() from exc


def visualize_data() -> str:
    """Return a simple native-state debug string."""

    if NativeBattle.state is None:
        return ""
    if NativeBattle.battle is not None:
        return str(NativeBattle.battle.canonical())
    E = _engine()
    return str(E.canonical(NativeBattle.state))
