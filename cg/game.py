import ctypes
import json
import os


def _use_native() -> bool:
    return os.environ.get("PTCG_BACKEND", os.environ.get("CG_BACKEND", "")).lower() == "native"


def _use_shadow() -> bool:
    backend = os.environ.get("PTCG_BACKEND", os.environ.get("CG_BACKEND", "")).lower()
    return backend == "shadow" or os.environ.get("PTCG_SHADOW", "").lower() in {"1", "true", "yes"}


def _native():
    from . import native_backend

    return native_backend


def _shadow():
    from . import shadow_backend

    return shadow_backend


def _sim():
    from .sim import Battle, lib

    return Battle, lib


def _get_battle_data() -> dict:
    """Retrieve the current state.

    Returns:
        dict: Current observation.
    """
    Battle, lib = _sim()
    sd = lib.GetBattleData(Battle.battle_ptr)
    Battle.obs = json.loads(sd.json.decode())
    Battle.obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
    return Battle.obs


def battle_start(deck0: list[int], deck1: list[int]):
    """Start the battle.

    Args:
        deck0: List of card IDs included in the first player’s deck.
        deck1: List of card IDs included in the second player’s deck.

    Returns:
        tuple: A tuple containing:
            - dict: First observation.
            - StartData: Battle start data.
    """
    if _use_shadow():
        return _shadow().battle_start(deck0, deck1)
    if _use_native():
        return _native().battle_start(deck0, deck1)
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")
    Battle, lib = _sim()
    cards = deck0 + deck1
    arg = (ctypes.c_int * len(cards))(*cards)
    start_data = lib.BattleStart(arg)
    Battle.battle_ptr = start_data.battlePtr
    if Battle.battle_ptr == None or Battle.battle_ptr == 0:
        return (None, start_data)
    else:
        return (_get_battle_data(), start_data)


def battle_finish():
    """End the battle and free the memory used during it."""
    if _use_shadow():
        return _shadow().battle_finish()
    if _use_native():
        return _native().battle_finish()
    Battle, lib = _sim()
    lib.BattleFinish(Battle.battle_ptr)


def battle_select(select_list: list[int]) -> dict:
    """Select option.

    Args:
        select_list:

    Returns:
        dict: Next observation.
    """
    if _use_shadow():
        return _shadow().battle_select(select_list)
    if _use_native():
        return _native().battle_select(select_list)
    if not isinstance(select_list, list) or not all(isinstance(i, int) for i in select_list):
        raise ValueError("select_list is not list[int]")
    Battle, lib = _sim()
    arg = (ctypes.c_int * len(select_list))(*select_list)
    err = lib.Select(Battle.battle_ptr, arg, len(select_list))
    if err != 0:
        if err == 30:
            raise ValueError("battle_ptr broken.")
        else:
            raise IndexError()
    return _get_battle_data()


def visualize_data() -> str:
    """Retrieve the data to be used by the visualizer.

    Returns:
        str: The data to be used by the visualizer.
    """
    if _use_shadow():
        return _shadow().visualize_data()
    if _use_native():
        return _native().visualize_data()
    Battle, lib = _sim()
    return lib.VisualizeData(Battle.battle_ptr).decode()
