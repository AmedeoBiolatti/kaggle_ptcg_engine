"""Native backend for the public ``cg.game`` compatibility API.

This module intentionally mirrors the small battle-loop surface from
``cg.game`` while delegating state transitions to ``ptcg_engine``.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from collections import Counter

from cg.native_payload import encode_native_search_begin

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
    obs = None
    last_logs: list[dict] = []
    deck0: list[int] | None = None
    deck1: list[int] | None = None
    seed: int = 1
    generation: int = 0


def _engine():
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dirs = [
        os.path.join(root, "engine", "build", "Release"),
        os.path.join(root, "engine", "build"),
    ]
    for build_dir in reversed(build_dirs):
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
    """Start a native battle using the ``cg.game.battle_start`` shape."""

    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")

    E = _engine()
    NativeBattle.deck0 = list(deck0)
    NativeBattle.deck1 = list(deck1)
    NativeBattle.seed = _seed()
    NativeBattle.generation = 0
    try:
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
