"""Opaque native payloads for ``Observation.search_begin_input``."""
from __future__ import annotations

import base64
import itertools
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


PREFIX = "native:"
_MAX_STATES = 1024
_COUNTER = itertools.count(1)


@dataclass
class _StateRef:
    state: object
    live_generation: int | None = None
    live_generation_getter: object | None = None


_STATES: OrderedDict[str, _StateRef] = OrderedDict()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _remember_state(
    token: str,
    state: object | None,
    *,
    live_generation: int | None = None,
    live_generation_getter: object | None = None,
) -> None:
    if state is None:
        return
    _STATES[token] = _StateRef(state, live_generation, live_generation_getter)
    _STATES.move_to_end(token)
    while len(_STATES) > _MAX_STATES:
        _STATES.popitem(last=False)


def encode_native_search_begin(
    current: dict,
    *,
    state: object | None = None,
    context: int = -1,
    descriptors: object | None = None,
    main_options: object | None = None,
    transients: object | None = None,
    seed: int | None = None,
    portable: bool = False,
    live_generation: int | None = None,
    live_generation_getter: object | None = None,
) -> str:
    token = format(next(_COUNTER), "x")
    _remember_state(
        token,
        state,
        live_generation=live_generation,
        live_generation_getter=live_generation_getter,
    )
    payload: dict[str, Any] = {
        "v": 1,
        "backend": "ptcg",
        "token": token,
        "context": int(context),
    }
    if live_generation is not None:
        payload["live_generation"] = int(live_generation)
    if portable:
        payload["current"] = _jsonable(current)
    if portable and descriptors is not None:
        payload["descriptors"] = _jsonable(descriptors)
    if main_options is not None:
        payload["main_options"] = _jsonable(main_options)
    if transients is not None:
        payload["transients"] = _jsonable(transients)
    if seed is not None:
        payload["seed"] = int(seed)
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_native_search_begin(value: str | None) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.startswith(PREFIX):
        return None
    token = value[len(PREFIX):]
    token += "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError("Invalid native search_begin_input.") from exc
    if not isinstance(payload, dict) or payload.get("backend") != "ptcg":
        raise ValueError("Invalid native search_begin_input.")
    return payload


def registered_native_state(payload: dict[str, Any] | None) -> object | None:
    if not payload:
        return None
    token = payload.get("token")
    if not isinstance(token, str):
        return None
    ref = _STATES.get(token)
    if ref is not None:
        _STATES.move_to_end(token)
        if ref.live_generation is not None:
            getter = ref.live_generation_getter
            if getter is None:
                return None
            try:
                if int(getter()) != int(ref.live_generation):
                    return None
            except Exception:
                return None
        return ref.state
    return None
