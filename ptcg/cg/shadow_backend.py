"""Run the reference cg engine and native engine in lockstep."""
from __future__ import annotations

import ctypes
import json
import os
from typing import Any

from validation.native_compare_core import NativeComparisonError, compare_observations


class _StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class _SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


class _ReferenceBattle:
    battle_ptr = None
    obs: dict[str, Any] | None = None


_REFERENCE_LIB = None


class ShadowBattle:
    cg_obs: dict[str, Any] | None = None
    native_obs: dict[str, Any] | None = None
    trace: list[list[int]] = []
    deck0: list[int] | None = None
    deck1: list[int] | None = None
    seed: int = 1


def _primary() -> str:
    return os.environ.get("PTCG_SHADOW_PRIMARY", "native").lower()


def _compare_logs() -> bool:
    return os.environ.get("PTCG_SHADOW_COMPARE_LOGS", "").lower() in {"1", "true", "yes"}


def _reference_lib():
    global _REFERENCE_LIB
    if _REFERENCE_LIB is not None:
        return _REFERENCE_LIB

    here = os.path.dirname(os.path.abspath(__file__))
    lib_name = "cg.dll" if os.name == "nt" else "libcg.so"
    explicit = os.environ.get("PTCG_REFERENCE_LIB", "")
    candidates = [
        explicit,
        os.path.join(here, lib_name),
        os.path.join(os.getcwd(), "cg", lib_name),
        os.path.join(os.getcwd(), lib_name),
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        lib = ctypes.cdll.LoadLibrary(path)
        lib.GameInitialize()
        lib.BattleStart.restype = _StartData
        lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.BattleFinish.argtypes = [ctypes.c_void_p]
        lib.GetBattleData.restype = _SerialData
        lib.GetBattleData.argtypes = [ctypes.c_void_p]
        lib.Select.restype = ctypes.c_int
        lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.VisualizeData.restype = ctypes.c_char_p
        lib.VisualizeData.argtypes = [ctypes.c_void_p]
        _REFERENCE_LIB = lib
        return lib
    raise FileNotFoundError(
        "reference cg shared library not found; expected cg.dll/libcg.so in "
        "PTCG_REFERENCE_LIB, ./cg/, or the current working directory"
    )


def _cg_start(deck0: list[int], deck1: list[int]):
    lib = _reference_lib()
    cards = list(deck0) + list(deck1)
    arg = (ctypes.c_int * len(cards))(*cards)
    start_data = lib.BattleStart(arg)
    _ReferenceBattle.battle_ptr = start_data.battlePtr
    if _ReferenceBattle.battle_ptr is None or _ReferenceBattle.battle_ptr == 0:
        return None, start_data
    return _cg_get_battle_data(), start_data


def _cg_get_battle_data() -> dict:
    lib = _reference_lib()
    sd = lib.GetBattleData(_ReferenceBattle.battle_ptr)
    _ReferenceBattle.obs = json.loads(sd.json.decode())
    _ReferenceBattle.obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
    return _ReferenceBattle.obs


def _cg_select(select_list: list[int]) -> dict:
    lib = _reference_lib()
    arg = (ctypes.c_int * len(select_list))(*select_list)
    err = lib.Select(_ReferenceBattle.battle_ptr, arg, len(select_list))
    if err != 0:
        if err == 30:
            raise ValueError("battle_ptr broken.")
        raise IndexError()
    return _cg_get_battle_data()


def _cg_finish() -> None:
    lib = _reference_lib()
    if _ReferenceBattle.battle_ptr is not None:
        lib.BattleFinish(_ReferenceBattle.battle_ptr)
    _ReferenceBattle.battle_ptr = None
    _ReferenceBattle.obs = None


def _native_debug():
    try:
        import ptcg_engine as E
        from .native_backend import NativeBattle

        if NativeBattle.state is None:
            return None
        return E.native_state_summary(NativeBattle.state)
    except Exception:
        return None


def _setup_in_progress(obs: dict[str, Any] | None) -> bool:
    if obs is None:
        return False
    current = obs.get("current") or {}
    return int(current.get("turn", 0)) <= 0


def _native_seed() -> int:
    raw = os.environ.get("PTCG_NATIVE_SEED", "1")
    try:
        return int(raw)
    except ValueError:
        return 1


def _bootstrap_native_from_cg(cg_obs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Rebuild native from the public cg observation after setup.

    Setup uses hidden randomness owned by the reference engine. Shadow mode
    therefore treats setup as a bootstrap phase, then asks native to emit the
    same public observation/select from the post-setup public state.
    """

    if cg_obs is None or _setup_in_progress(cg_obs):
        return None

    import ptcg_engine as E
    from . import native_backend

    state = E.load_state(cg_obs["current"], ShadowBattle.seed, None)
    obs, context, descriptors = E.cg_observation_with_view(state)
    obs["logs"] = list(cg_obs.get("logs") or [])

    native_backend.NativeBattle.battle = None
    native_backend.NativeBattle.state = state
    native_backend.NativeBattle.obs = obs
    native_backend.NativeBattle.last_logs = list(obs["logs"])
    native_backend.NativeBattle.deck0 = list(ShadowBattle.deck0 or [])
    native_backend.NativeBattle.deck1 = list(ShadowBattle.deck1 or [])
    native_backend.NativeBattle.seed = ShadowBattle.seed
    native_backend.NativeBattle.generation = 0
    native_backend._attach_search_begin_input(E, obs, int(context), descriptors)
    return obs


def _assert_match(label: str) -> None:
    if ShadowBattle.cg_obs is None or ShadowBattle.native_obs is None:
        if ShadowBattle.cg_obs is not ShadowBattle.native_obs:
            raise NativeComparisonError(
                compare_observations(label, ShadowBattle.cg_obs or {}, ShadowBattle.native_obs or {}),
                ShadowBattle.trace,
            )
        return
    result = compare_observations(
        label,
        ShadowBattle.cg_obs,
        ShadowBattle.native_obs,
        compare_logs=_compare_logs(),
    )
    if not result.ok:
        raise NativeComparisonError(result, ShadowBattle.trace)


def _primary_result(cg_result, native_result):
    return cg_result if _primary() == "cg" else native_result


def battle_start(deck0: list[int], deck1: list[int]):
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")

    ShadowBattle.trace = []
    ShadowBattle.deck0 = list(deck0)
    ShadowBattle.deck1 = list(deck1)
    ShadowBattle.seed = _native_seed()
    cg_obs, cg_start = _cg_start(deck0, deck1)
    native_obs = _bootstrap_native_from_cg(cg_obs)
    ShadowBattle.cg_obs = cg_obs
    ShadowBattle.native_obs = native_obs
    if native_obs is not None:
        _assert_match("battle_start")
    return cg_obs, cg_start


def battle_select(select_list: list[int]) -> dict:
    if not isinstance(select_list, list) or not all(isinstance(i, int) for i in select_list):
        raise ValueError("select_list is not list[int]")

    ShadowBattle.trace.append(list(select_list))
    from . import native_backend

    native_started = native_backend.NativeBattle.state is not None
    try:
        cg_obs = _cg_select(select_list)
    except Exception as cg_exc:
        if native_started:
            try:
                native_backend.battle_select(select_list)
            except Exception as native_exc:
                if type(cg_exc) is type(native_exc):
                    raise cg_exc
                raise NativeComparisonError(
                    compare_observations("battle_select exception", {}, {}),
                    ShadowBattle.trace,
                ) from native_exc
        raise

    try:
        if native_started:
            native_obs = native_backend.battle_select(select_list)
        else:
            native_obs = _bootstrap_native_from_cg(cg_obs)
    except Exception as native_exc:
        raise NativeComparisonError(
            compare_observations("battle_select native exception", cg_obs, {}),
            ShadowBattle.trace,
        ) from native_exc

    ShadowBattle.cg_obs = cg_obs
    ShadowBattle.native_obs = native_obs
    if native_obs is not None:
        _assert_match(f"battle_select step={len(ShadowBattle.trace)}")
        return _primary_result(cg_obs, native_obs)
    return cg_obs


def battle_finish() -> None:
    from . import native_backend

    try:
        _cg_finish()
    finally:
        native_backend.battle_finish()
        ShadowBattle.cg_obs = None
        ShadowBattle.native_obs = None
        ShadowBattle.trace = []
        ShadowBattle.deck0 = None
        ShadowBattle.deck1 = None


def visualize_data() -> str:
    if _primary() == "cg":
        return _reference_lib().VisualizeData(_ReferenceBattle.battle_ptr).decode()
    from .native_backend import visualize_data as native_visualize_data

    return native_visualize_data()
