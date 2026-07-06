from __future__ import annotations

import ctypes
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_DIR = os.path.join(ROOT, "engine")
# Only extensions built from this checkout are eligible.
BUILD_DIR = os.path.join(ROOT, "engine", "build", "Release")
BUILD_DIR_NINJA = os.path.join(ROOT, "engine", "build")
for path in (ENGINE_DIR, BUILD_DIR_NINJA, BUILD_DIR, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from validation.decks import MEGA_LUCARIO  # noqa: E402


def _card(card_id: int, serial: int = 0, player: int = 0) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": player}


def _pokemon(
    card_id: int,
    player: int,
    *,
    serial: int = 0,
    energy_cards: list[int] | None = None,
    tools: list[int] | None = None,
    pre_evolution: list[int] | None = None,
) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "playerIndex": player,
        "hp": 100,
        "maxHp": 100,
        "appearThisTurn": False,
        "energies": [],
        "energyCards": [
            {"id": card_id, "serial": idx, "playerIndex": player}
            for idx, card_id in enumerate(energy_cards or [])
        ],
        "tools": [
            {"id": card_id, "serial": idx, "playerIndex": player}
            for idx, card_id in enumerate(tools or [])
        ],
        "preEvolution": [
            {"id": card_id, "serial": idx, "playerIndex": player}
            for idx, card_id in enumerate(pre_evolution or [])
        ],
    }


def _player(player: int, *, active: dict | None = None, hand: list[int] | None = None) -> dict:
    hand_cards = [_card(card_id, idx, player) for idx, card_id in enumerate(hand or [])]
    return {
        "active": [active] if active is not None else [],
        "bench": [],
        "benchMax": 5,
        "deckCount": 0,
        "discard": [],
        "prize": [],
        "handCount": len(hand_cards),
        "hand": hand_cards,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _state_for_serializer() -> dict:
    return {
        "turn": 2,
        "turnActionCount": 1,
        "yourIndex": 0,
        "firstPlayer": 0,
        "supporterPlayed": False,
        "stadiumPlayed": False,
        "energyAttached": False,
        "retreated": False,
        "result": -1,
        "stadium": [_card(1267, 0, 0)],
        "players": [
            _player(
                0,
                active=_pokemon(
                    673,
                    0,
                    serial=10,
                    energy_cards=[6, 10],
                    tools=[1101],
                    pre_evolution=[672],
                ),
                hand=[1142, 6, 1102],
            ),
            _player(1, active=_pokemon(676, 1, serial=20)),
        ],
    }


def _packed_card(value: int) -> int | None:
    return None if value <= 0 else value


def _pov_for_side(current: dict, side: int) -> int:
    return 0 if side == int(current["yourIndex"]) else 1


def _actual_player_from_pov(current: dict, pov: int) -> int | None:
    if pov == 0:
        return int(current["yourIndex"])
    if pov == 1:
        return 1 - int(current["yourIndex"])
    return None


def _reference_cg_dll_path() -> str | None:
    explicit = os.environ.get("PTCG_REFERENCE_LIB")
    if explicit and os.path.exists(explicit):
        return explicit
    names = ("cg.dll", "libcg.so")
    workspace = os.path.dirname(ROOT)
    candidates = [
        os.path.join(ROOT, "cg"),
        os.path.join(workspace, "cg"),
        os.path.join(workspace, "data", "sample_submission", "cg"),
    ]
    for path in candidates:
        for name in names:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
    return None


def _reference_cg_dll_exists() -> bool:
    return _reference_cg_dll_path() is not None


def _reference_card_metadata() -> tuple[dict[int, dict], dict[int, dict]]:
    lib_path = _reference_cg_dll_path()
    if lib_path is None:
        raise unittest.SkipTest("reference cg shared library is not present")
    from ptcg.cg import shadow_backend

    old_reference = os.environ.get("PTCG_REFERENCE_LIB")
    os.environ["PTCG_REFERENCE_LIB"] = lib_path
    try:
        lib = shadow_backend._reference_lib()
    finally:
        if old_reference is None:
            os.environ.pop("PTCG_REFERENCE_LIB", None)
        else:
            os.environ["PTCG_REFERENCE_LIB"] = old_reference
    lib.AllCard.restype = ctypes.c_char_p
    lib.AllAttack.restype = ctypes.c_char_p
    cards = {int(card["cardId"]): card for card in json.loads(lib.AllCard().decode())}
    attacks = {int(attack["attackId"]): attack for attack in json.loads(lib.AllAttack().decode())}
    return cards, attacks


class NativeCgCompatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_backend = os.environ.get("PTCG_BACKEND")
        self.old_portable_search = os.environ.get("PTCG_NATIVE_PORTABLE_SEARCH")
        self.old_lazy_search = os.environ.get("PTCG_NATIVE_LAZY_SEARCH")
        self.old_cpp_battle = os.environ.get("PTCG_NATIVE_CPP_BATTLE")
        self.old_fast_setup = os.environ.get("PTCG_NATIVE_FAST_SETUP")
        os.environ["PTCG_BACKEND"] = "native"
        os.environ["PTCG_NATIVE_PORTABLE_SEARCH"] = "1"
        os.environ["PTCG_NATIVE_FAST_SETUP"] = "1"
        os.environ.pop("PTCG_NATIVE_LAZY_SEARCH", None)
        os.environ.pop("PTCG_NATIVE_CPP_BATTLE", None)

    def tearDown(self) -> None:
        if self.old_backend is None:
            os.environ.pop("PTCG_BACKEND", None)
        else:
            os.environ["PTCG_BACKEND"] = self.old_backend
        if self.old_portable_search is None:
            os.environ.pop("PTCG_NATIVE_PORTABLE_SEARCH", None)
        else:
            os.environ["PTCG_NATIVE_PORTABLE_SEARCH"] = self.old_portable_search
        if self.old_lazy_search is None:
            os.environ.pop("PTCG_NATIVE_LAZY_SEARCH", None)
        else:
            os.environ["PTCG_NATIVE_LAZY_SEARCH"] = self.old_lazy_search
        if self.old_cpp_battle is None:
            os.environ.pop("PTCG_NATIVE_CPP_BATTLE", None)
        else:
            os.environ["PTCG_NATIVE_CPP_BATTLE"] = self.old_cpp_battle
        if self.old_fast_setup is None:
            os.environ.pop("PTCG_NATIVE_FAST_SETUP", None)
        else:
            os.environ["PTCG_NATIVE_FAST_SETUP"] = self.old_fast_setup

    def test_native_battle_start_defaults_to_official_setup_flow(self) -> None:
        from ptcg.cg.game import battle_finish, battle_select, battle_start

        os.environ.pop("PTCG_NATIVE_FAST_SETUP", None)
        obs, start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        self.assertIsNotNone(obs)
        self.assertEqual(start.errorPlayer, -1)
        self.assertEqual(start.errorType, 0)
        self.assertEqual(obs["current"]["turn"], 0)
        self.assertEqual(obs["current"]["firstPlayer"], -1)
        self.assertEqual(obs["select"]["context"], 41)
        self.assertEqual(obs["select"]["type"], 9)
        self.assertEqual([o["type"] for o in obs["select"]["option"]], [1, 2])

        obs = battle_select([0])
        self.assertEqual(obs["current"]["turn"], 0)
        self.assertEqual(obs["current"]["firstPlayer"], 0)
        self.assertEqual(obs["select"]["context"], 1)
        self.assertEqual(obs["select"]["type"], 1)
        self.assertGreaterEqual(len(obs["select"]["option"]), 1)
        battle_finish()

    def test_battle_select_exposes_pending_prompt(self) -> None:
        from ptcg.cg.game import battle_finish, battle_select, battle_start
        from ptcg.cg.native_payload import decode_native_search_begin

        obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        self.assertEqual(obs["select"]["context"], 0)
        self.assertTrue(obs["search_begin_input"].startswith("native:"))
        payload = decode_native_search_begin(obs["search_begin_input"])
        self.assertIsNotNone(payload)
        self.assertEqual(payload["current"]["yourIndex"], obs["current"]["yourIndex"])
        self.assertEqual(payload["context"], -1)

        obs = battle_select([2])
        self.assertEqual(obs["select"]["context"], 7)
        self.assertEqual(obs["select"]["type"], 1)
        self.assertGreater(len(obs["select"]["option"]), 1)
        self.assertTrue(obs["search_begin_input"].startswith("native:"))
        self.assertTrue(obs["logs"])
        self.assertEqual(obs["logs"][0]["type"], 10)
        self.assertIn("cardId", obs["logs"][0])

        obs = battle_select([0])
        self.assertEqual(obs["select"]["context"], 0)
        self.assertTrue(isinstance(obs["logs"], list))
        battle_finish()

    def test_search_begin_restores_pending_prompt_from_native_payload(self) -> None:
        from ptcg.cg.api import search_begin, search_step, to_observation_class
        from ptcg.cg.game import battle_select, battle_start

        obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        pending = battle_select([2])
        root = self._native_search_from_observation(
            to_observation_class(pending),
            search_begin,
        )
        self.assertEqual(root.observation.select.context, 7)
        self.assertEqual(root.observation.select.type, 1)

        child = search_step(root.searchId, [0])
        self.assertEqual(child.observation.select.context, 0)

    def test_lazy_search_begin_restores_latest_pending_prompt(self) -> None:
        from ptcg.cg.api import search_begin, search_step, to_observation_class
        from ptcg.cg.game import battle_select, battle_start
        from ptcg.cg.native_payload import decode_native_search_begin

        os.environ.pop("PTCG_NATIVE_PORTABLE_SEARCH", None)
        os.environ["PTCG_NATIVE_LAZY_SEARCH"] = "1"

        _obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        pending = battle_select([2])
        payload = decode_native_search_begin(pending["search_begin_input"])
        self.assertIsNotNone(payload)
        self.assertIn("live_generation", payload)
        self.assertNotIn("current", payload)

        root = self._native_search_from_observation(
            to_observation_class(pending),
            search_begin,
        )
        self.assertEqual(root.observation.select.context, 7)
        child = search_step(root.searchId, [0])
        self.assertEqual(child.observation.select.context, 0)

    def test_lazy_search_begin_rejects_stale_pending_prompt(self) -> None:
        from ptcg.cg.api import search_begin, to_observation_class
        from ptcg.cg.game import battle_select, battle_start

        os.environ.pop("PTCG_NATIVE_PORTABLE_SEARCH", None)
        os.environ["PTCG_NATIVE_LAZY_SEARCH"] = "1"

        _obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        pending = battle_select([2])
        battle_select([0])

        with self.assertRaisesRegex(ValueError, "Native search_begin_input is stale"):
            self._native_search_from_observation(
                to_observation_class(pending),
                search_begin,
            )

    def test_cpp_battle_object_backend_smoke(self) -> None:
        from ptcg.cg.game import battle_finish, battle_select, battle_start

        os.environ["PTCG_NATIVE_CPP_BATTLE"] = "1"

        obs, start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        self.assertEqual(start.errorType, 0)
        self.assertEqual(obs["select"]["context"], 0)
        obs = battle_select([0])
        self.assertIsInstance(obs["logs"], list)
        self.assertIsNotNone(obs["current"])
        battle_finish()

    def test_search_begin_rejects_stale_pending_native_payload(self) -> None:
        from ptcg.cg.api import search_begin, to_observation_class
        from ptcg.cg.game import battle_select, battle_start
        from ptcg.cg.native_payload import decode_native_search_begin, encode_native_search_begin

        _obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        pending = battle_select([2])
        payload = decode_native_search_begin(pending["search_begin_input"])
        pending["search_begin_input"] = encode_native_search_begin(
            payload["current"],
            context=payload["context"],
            descriptors=payload["descriptors"],
            seed=payload.get("seed"),
        )

        with self.assertRaisesRegex(ValueError, "Native search_begin_input is stale"):
            self._native_search_from_observation(
                to_observation_class(pending),
                search_begin,
            )

    def test_search_begin_restores_pending_prompt_from_portable_snapshot(self) -> None:
        from ptcg.cg.api import search_begin, search_step, to_observation_class
        from ptcg.cg.game import battle_select, battle_start
        from ptcg.cg.native_payload import decode_native_search_begin, encode_native_search_begin

        _obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        pending = battle_select([2])
        payload = decode_native_search_begin(pending["search_begin_input"])
        pending["search_begin_input"] = encode_native_search_begin(
            payload["current"],
            context=payload["context"],
            descriptors=payload["descriptors"],
            transients=payload["transients"],
            seed=payload.get("seed"),
        )

        root = self._native_search_from_observation(
            to_observation_class(pending),
            search_begin,
        )
        self.assertEqual(root.observation.select.context, 7)
        child = search_step(root.searchId, [0])
        self.assertEqual(child.observation.select.context, 0)

    def test_search_step_exposes_pending_prompt_with_known_deck(self) -> None:
        from ptcg.cg.api import search_begin, search_step, to_observation_class
        from ptcg.cg.game import battle_start

        root = self._native_search_root(battle_start, search_begin, to_observation_class)

        me = root.observation.current.yourIndex
        hand = root.observation.current.players[me].hand or []
        search_option = next(
            i
            for i, option in enumerate(root.observation.select.option)
            if int(option.type) == 7 and hand[int(option.index)].id == 1142
        )
        child = search_step(root.searchId, [search_option])
        self.assertEqual(child.observation.select.context, 7)
        self.assertEqual(child.observation.select.type, 1)
        self.assertGreater(len(child.observation.select.option), 1)
        self.assertTrue(child.observation.logs)
        self.assertEqual(child.observation.logs[0].type, 10)

    def test_search_begin_preserves_true_hidden_zones_at_main_root(self) -> None:
        from ptcg.cg.api import search_begin, to_observation_class
        from ptcg.cg.game import battle_start

        obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        root = self._native_search_from_live_hidden(
            to_observation_class(obs),
            search_begin,
            validate=True,
        )
        self._assert_search_hidden_matches_live(root.searchId)

    def test_search_begin_preserves_true_hidden_zones_at_pending_deck_prompt(self) -> None:
        from ptcg.cg.api import search_begin, to_observation_class
        from ptcg.cg.game import battle_select, battle_start

        _obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        pending = battle_select([2])
        root = self._native_search_from_live_hidden(
            to_observation_class(pending),
            search_begin,
            validate=True,
        )
        self.assertEqual(root.observation.select.context, 7)
        self._assert_search_hidden_matches_live(root.searchId)

    def test_search_release_and_select_validation_match_public_errors(self) -> None:
        from ptcg.cg.api import search_begin, search_release, search_step, to_observation_class
        from ptcg.cg.game import battle_start

        root = self._native_search_root(battle_start, search_begin, to_observation_class)

        with self.assertRaisesRegex(
            ValueError,
            "Must be Observation.select.minCount <= len\\(select\\) <= Observation.select.maxCount.",
        ):
            search_step(root.searchId, [])
        with self.assertRaisesRegex(
            ValueError,
            "Must be 0 <= select elements < len\\(Observation.select.option\\).",
        ):
            search_step(root.searchId, [999])

        search_release(root.searchId)
        with self.assertRaisesRegex(ValueError, "Released item."):
            search_step(root.searchId, [0])

    def test_native_pending_select_schema_matches_released_api_shapes(self) -> None:
        import ptcg_engine as E

        state = E.load_state(_state_for_serializer())
        snapshot = E.search_transient_snapshot(state)

        def set_pending(context: int, options: list[tuple], min_count: int = 1, max_count: int = 1):
            snapshot["pending"] = {
                "context": context,
                "minCount": min_count,
                "maxCount": max_count,
                "options": options,
            }
            E.restore_search_transients(state, snapshot)
            return E.cg_observation(state)["select"]

        select = set_pending(30, [("ENERGY", "ACTIVE", 0, 6)])
        self.assertEqual(select["type"], 4)
        self.assertEqual(select["option"][0]["type"], 6)
        self.assertEqual(select["option"][0]["area"], 4)
        self.assertEqual(select["option"][0]["index"], 0)
        self.assertEqual(select["option"][0]["playerIndex"], 0)
        self.assertEqual(select["option"][0]["energyIndex"], 0)
        self.assertEqual(select["option"][0]["count"], 1)

        select = set_pending(29, [
            ("CARD", "STADIUM", -1, 1267),
            ("CARD", "ACTIVE", 100000, 1101),
            ("ENERGY", "ACTIVE", 200001, 10),
        ])
        self.assertEqual(select["type"], 3)
        self.assertEqual(select["option"][0]["type"], 3)
        self.assertEqual(select["option"][0]["area"], 7)
        self.assertEqual(select["option"][0]["index"], 0)
        self.assertEqual(select["option"][1]["type"], 4)
        self.assertEqual(select["option"][1]["toolIndex"], 0)
        self.assertEqual(select["option"][2]["type"], 5)
        self.assertEqual(select["option"][2]["energyIndex"], 1)
        self.assertNotIn("count", select["option"][2])

        self.assertEqual(set_pending(43, [("YES",), ("NO",)])["type"], 9)
        self.assertEqual(set_pending(39, [("NUMBER", 2)])["type"], 8)
        special = set_pending(47, [("SPECIAL_CONDITION", 0)])
        self.assertEqual(special["type"], 10)
        self.assertEqual(special["option"][0]["type"], 16)
        self.assertEqual(special["option"][0]["specialConditionType"], 0)

    def test_native_observation_uses_attached_card_ids_and_zone_order(self) -> None:
        import ptcg_engine as E

        obs = E.cg_observation(E.load_state(_state_for_serializer()))
        active = obs["current"]["players"][0]["active"][0]
        self.assertEqual([card["id"] for card in active["energyCards"]], [6, 10])
        self.assertEqual([card["id"] for card in active["tools"]], [1101])
        self.assertEqual([card["id"] for card in active["preEvolution"]], [672])
        self.assertEqual([card["id"] for card in obs["current"]["players"][0]["hand"]], [1142, 6, 1102])

    def test_rl_state_ids_reconstruct_current_observation_projection(self) -> None:
        import ptcg_engine as E

        state = E.load_state(_state_for_serializer())
        obs = E.cg_observation(state)
        current = obs["current"]
        packed = E.rl_state_ids(state)
        in_play = packed["in_play"]
        zones = packed["zones"]
        counts = packed["player_counts"]
        status = packed["player_status"]
        global_state = packed["global"]

        self.assertEqual(int(global_state[0]), current["turn"])
        self.assertEqual(int(global_state[1]), current["turnActionCount"])
        self.assertEqual(int(global_state[2]), current["yourIndex"])
        self.assertEqual(int(global_state[3]), current["firstPlayer"])
        self.assertEqual(int(global_state[4]), current["result"])
        self.assertEqual(bool(global_state[5]), current["supporterPlayed"])
        self.assertEqual(bool(global_state[6]), current["stadiumPlayed"])
        self.assertEqual(bool(global_state[7]), current["energyAttached"])
        self.assertEqual(bool(global_state[8]), current["retreated"])
        self.assertEqual(int(global_state[11]), current["stadium"][0]["id"])
        self.assertEqual(_actual_player_from_pov(current, int(global_state[12])), current["stadium"][0]["playerIndex"])

        for side, player in enumerate(current["players"]):
            pov = _pov_for_side(current, side)
            self.assertEqual(int(counts[pov, 0]), player["handCount"])
            self.assertEqual(int(counts[pov, 1]), player["deckCount"])
            self.assertEqual(int(counts[pov, 2]), len(player["prize"]))
            self.assertEqual(int(counts[pov, 3]), len(player["discard"]))
            self.assertEqual(int(counts[pov, 4]), len(player["bench"]))
            self.assertEqual(int(global_state[19 + pov]), player["benchMax"])
            for idx, key in enumerate(["poisoned", "burned", "asleep", "paralyzed", "confused"]):
                self.assertEqual(bool(status[pov, idx]), player[key])

            active_present = bool(player["active"])
            active = player["active"][0] if active_present else None
            active_row = in_play[pov, 0]
            self.assertEqual(bool(active_row[0]), active_present)
            if active_present and active is None:
                self.assertEqual(int(active_row[1]), 0)
                self.assertEqual(int(active_row[2]), -1)
            elif active is not None:
                self.assertEqual(int(active_row[1]), 1)
                self.assertEqual(int(active_row[2]), active["id"])
                self.assertEqual(int(active_row[5]), active["hp"])
                self.assertEqual(int(active_row[6]), active["maxHp"])
                self.assertEqual(int(active_row[8]), active["serial"])
                self.assertEqual(bool(int(active_row[7]) & 1), active["appearThisTurn"])
                self.assertEqual(
                    [int(v) for v in active_row[32 : 32 + len(active["energies"])]],
                    active["energies"],
                )
                self.assertEqual(
                    [int(v) for v in active_row[16 : 16 + len(active["energyCards"])]],
                    [card["id"] for card in active["energyCards"]],
                )
                self.assertEqual(
                    [int(v) for v in active_row[48 : 48 + len(active["tools"])]],
                    [card["id"] for card in active["tools"]],
                )
                self.assertEqual(
                    [int(v) for v in active_row[52 : 52 + len(active["preEvolution"])]],
                    [card["id"] for card in active["preEvolution"]],
                )

            for bench_idx, pokemon in enumerate(player["bench"]):
                row = in_play[pov, 1 + bench_idx]
                self.assertEqual(int(row[0]), 1)
                self.assertEqual(int(row[2]), pokemon["id"])
                self.assertEqual(int(row[8]), pokemon["serial"])

            hand_zone = zones[pov, int(packed["zone_hand"])]
            if player["hand"] is None:
                self.assertTrue(all(int(v) <= 0 for v in hand_zone[: player["handCount"]]))
            else:
                self.assertEqual(
                    [int(v) for v in hand_zone[: player["handCount"]]],
                    [card["id"] for card in player["hand"]],
                )
            self.assertEqual(
                [int(v) for v in zones[pov, int(packed["zone_discard"])][: len(player["discard"])]],
                [card["id"] for card in player["discard"]],
            )
            self.assertEqual(
                [_packed_card(int(v)) for v in zones[pov, int(packed["zone_prizes"])][: len(player["prize"])]],
                [None if card is None else card["id"] for card in player["prize"]],
            )

    def test_rl_action_ids_reconstruct_select_projection(self) -> None:
        import ptcg_engine as E

        state = E.new_game(MEGA_LUCARIO, MEGA_LUCARIO, 1)
        obs = E.cg_observation(state)
        current = obs["current"]
        select = obs["select"]
        packed = E.rl_action_ids(state)
        meta = packed["meta"]
        options = packed["options"]
        deck = packed["deck"]
        mask = packed["mask"]

        self.assertEqual(int(meta[0]), select["context"])
        self.assertEqual(int(meta[1]), select["type"])
        self.assertEqual(int(meta[2]), select["minCount"])
        self.assertEqual(int(meta[3]), select["maxCount"])
        self.assertEqual(int(meta[4]), len(select["option"]))
        self.assertEqual(int(mask.sum()), len(select["option"]))

        if select["deck"] is None:
            self.assertEqual(int(meta[5]), 0)
        else:
            self.assertEqual(int(meta[5]), len(select["deck"]))
            self.assertEqual(
                [_packed_card(int(v)) for v in deck[: len(select["deck"])]],
                [None if card is None else card["id"] for card in select["deck"]],
            )

        for action_index, option in enumerate(select["option"]):
            row = options[action_index]
            self.assertEqual(int(row[0]), 1)
            self.assertEqual(int(row[1]), option["type"])
            self.assertEqual(int(row[17]), action_index)
            if "area" in option:
                self.assertEqual(int(row[3]), option["area"])
            if "index" in option:
                self.assertEqual(int(row[4]), option["index"])
            if "playerIndex" in option:
                self.assertEqual(_actual_player_from_pov(current, int(row[5])), option["playerIndex"])
            if "inPlayArea" in option:
                self.assertEqual(int(row[6]), option["inPlayArea"])
            if "inPlayIndex" in option:
                self.assertEqual(int(row[7]), option["inPlayIndex"])
            if "attackId" in option:
                self.assertEqual(int(row[10]), option["attackId"])
            if "number" in option:
                self.assertEqual(int(row[11]), option["number"])
            if "serial" in option:
                self.assertEqual(int(row[12]), option["serial"])
            if "energyIndex" in option:
                self.assertEqual(int(row[13]), option["energyIndex"])
            if "count" in option:
                self.assertEqual(int(row[14]), option["count"])
            if "toolIndex" in option:
                self.assertEqual(int(row[15]), option["toolIndex"])
            if "specialConditionType" in option:
                self.assertEqual(int(row[16]), option["specialConditionType"])

        vector = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 4, 1)
        batch = vector.action_ids()
        self.assertEqual(batch["meta"].shape, (4, E.ACTION_META_WIDTH))
        self.assertEqual(batch["options"].shape, (4, E.RL_MAX_ACTIONS, E.ACTION_OPTION_WIDTH))
        self.assertEqual(batch["deck"].shape, (4, E.STATE_ZONE_SLOTS))
        self.assertEqual(batch["mask"].shape, (4, E.RL_MAX_ACTIONS))

    def test_native_battle_start_deck_validation_matches_released_errors(self) -> None:
        from ptcg.cg.api import CardType, _native_all_card_data
        from ptcg.cg.game import battle_start

        cards = list(_native_all_card_data())
        basic_id = next(
            card.cardId
            for card in cards
            if int(card.cardType) == int(CardType.POKEMON) and card.basic
        )
        ace_id = next(card.cardId for card in cards if card.aceSpec)

        invalid = list(MEGA_LUCARIO)
        invalid[0] = 999999
        obs, start = battle_start(invalid, MEGA_LUCARIO)
        self.assertIsNone(obs)
        self.assertEqual((start.errorPlayer, start.errorType), (0, 1))

        no_basic = [6] * 60
        obs, start = battle_start(no_basic, MEGA_LUCARIO)
        self.assertIsNone(obs)
        self.assertEqual((start.errorPlayer, start.errorType), (0, 3))

        too_many = [basic_id] * 5 + [6] * 55
        obs, start = battle_start(too_many, MEGA_LUCARIO)
        self.assertIsNone(obs)
        self.assertEqual((start.errorPlayer, start.errorType), (0, 2))

        duplicate_ace = [basic_id] * 4 + [ace_id] * 2 + [6] * 54
        obs, start = battle_start(duplicate_ace, MEGA_LUCARIO)
        self.assertIsNone(obs)
        self.assertEqual((start.errorPlayer, start.errorType), (0, 4))

    def test_native_card_and_attack_metadata_match_public_api(self) -> None:
        import ptcg.cg.api as api

        official_cards, official_attacks = _reference_card_metadata()

        api._CARD_DB_CACHE = None
        api._CARD_TEXT_CACHE = None
        api._OFFICIAL_CARD_META_CACHE = None
        native_cards = {card.cardId: card for card in api._native_all_card_data()}
        native_attacks = {attack.attackId: attack for attack in api._native_all_attack()}

        self.assertEqual(set(official_cards), set(native_cards))
        self.assertEqual(set(official_attacks), set(native_attacks))
        card_fields = [
            "cardId",
            "name",
            "cardType",
            "retreatCost",
            "hp",
            "weakness",
            "resistance",
            "energyType",
            "basic",
            "stage1",
            "stage2",
            "ex",
            "megaEx",
            "tera",
            "aceSpec",
            "evolvesFrom",
            "attacks",
        ]
        for card_id, official in official_cards.items():
            native = native_cards[card_id]
            native_plain = {
                "cardId": native.cardId,
                "name": native.name,
                "cardType": int(native.cardType),
                "retreatCost": native.retreatCost,
                "hp": native.hp,
                "weakness": None if native.weakness is None else int(native.weakness),
                "resistance": None if native.resistance is None else int(native.resistance),
                "energyType": int(native.energyType),
                "basic": native.basic,
                "stage1": native.stage1,
                "stage2": native.stage2,
                "ex": native.ex,
                "megaEx": native.megaEx,
                "tera": native.tera,
                "aceSpec": native.aceSpec,
                "evolvesFrom": native.evolvesFrom,
                "attacks": list(native.attacks),
            }
            for field in card_fields:
                self.assertEqual(native_plain[field], official[field], (card_id, field))
            self.assertEqual(
                [(skill.name, skill.text) for skill in native.skills],
                [(skill["name"], skill["text"]) for skill in official["skills"]],
                card_id,
            )
        for attack_id, official in official_attacks.items():
            native = native_attacks[attack_id]
            self.assertEqual(native.name, official["name"], attack_id)
            self.assertEqual(native.text, official["text"], attack_id)
            self.assertEqual(native.damage, official["damage"], attack_id)
            self.assertEqual([int(e) for e in native.energies], official["energies"], attack_id)

    def test_shadow_backend_bootstraps_through_setup(self) -> None:
        from ptcg.cg.game import battle_finish, battle_start
        from ptcg.cg.game import battle_select

        if not _reference_cg_dll_exists():
            raise unittest.SkipTest("reference cg shared library is not present in this checkout")

        old_backend = os.environ.get("PTCG_BACKEND")
        old_primary = os.environ.get("PTCG_SHADOW_PRIMARY")
        old_reference = os.environ.get("PTCG_REFERENCE_LIB")
        reference_lib = _reference_cg_dll_path()
        os.environ["PTCG_BACKEND"] = "shadow"
        os.environ["PTCG_SHADOW_PRIMARY"] = "cg"
        if reference_lib is not None:
            os.environ["PTCG_REFERENCE_LIB"] = reference_lib
        try:
            obs, start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
            self.assertEqual(start.errorType, 0)
            self.assertEqual(obs["current"]["turn"], 0)
            for _ in range(8):
                if int(obs["current"]["turn"]) >= 1:
                    break
                select = obs["select"]
                chosen = [0] if int(select.get("minCount", 1)) else []
                obs = battle_select(chosen)
            self.assertGreaterEqual(int(obs["current"]["turn"]), 1)
        finally:
            try:
                battle_finish()
            except Exception:
                pass
            if old_backend is None:
                os.environ.pop("PTCG_BACKEND", None)
            else:
                os.environ["PTCG_BACKEND"] = old_backend
            if old_primary is None:
                os.environ.pop("PTCG_SHADOW_PRIMARY", None)
            else:
                os.environ["PTCG_SHADOW_PRIMARY"] = old_primary
            if old_reference is None:
                os.environ.pop("PTCG_REFERENCE_LIB", None)
            else:
                os.environ["PTCG_REFERENCE_LIB"] = old_reference

    def _native_search_root(self, battle_start, search_begin, to_observation_class):
        obs, _start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
        return self._native_search_from_observation(to_observation_class(obs), search_begin)

    def _native_search_from_observation(self, ob, search_begin):
        me = ob.current.yourIndex
        opp = 1 - me
        return search_begin(
            ob,
            MEGA_LUCARIO,
            MEGA_LUCARIO[: len(ob.current.players[me].prize)],
            MEGA_LUCARIO,
            MEGA_LUCARIO[: len(ob.current.players[opp].prize)],
            MEGA_LUCARIO[: ob.current.players[opp].handCount],
            [],
        )

    def _native_search_from_live_hidden(self, ob, search_begin, *, validate: bool = False):
        import ptcg_engine as E
        from ptcg.cg.native_backend import NativeBattle

        old_validate = os.environ.get("PTCG_NATIVE_VALIDATE_SEARCH")
        if validate:
            os.environ["PTCG_NATIVE_VALIDATE_SEARCH"] = "1"
        try:
            zones = E.native_state_summary(NativeBattle.state)["hidden"]["players"]
            me = ob.current.yourIndex
            opp = 1 - me
            return search_begin(
                ob,
                list(zones[me]["deck"]),
                list(zones[me]["prizes"]),
                list(zones[opp]["deck"]),
                list(zones[opp]["prizes"]),
                list(zones[opp]["hand"]),
                [],
            )
        finally:
            if validate:
                if old_validate is None:
                    os.environ.pop("PTCG_NATIVE_VALIDATE_SEARCH", None)
                else:
                    os.environ["PTCG_NATIVE_VALIDATE_SEARCH"] = old_validate

    def _assert_search_hidden_matches_live(self, search_id: int) -> None:
        import ptcg_engine as E
        import ptcg.cg.api as api
        from ptcg.cg.native_backend import NativeBattle

        live = E.native_state_summary(NativeBattle.state)["hidden"]["players"]
        search = api._native_search_state_summary(search_id)["hidden"]["players"]
        for player in (0, 1):
            self.assertEqual(list(search[player]["deck"]), list(live[player]["deck"]))
            self.assertEqual(list(search[player]["prizes"]), list(live[player]["prizes"]))
            self.assertEqual(
                sorted(search[player]["hand"]),
                sorted(live[player]["hand"]),
            )


if __name__ == "__main__":
    unittest.main()
