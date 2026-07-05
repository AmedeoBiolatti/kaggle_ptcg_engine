from __future__ import annotations

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


def _reference_cg_dll_path() -> str | None:
    explicit = os.environ.get("PTCG_REFERENCE_LIB")
    if explicit and os.path.exists(explicit):
        return explicit
    names = ("cg.dll", "libcg.so")
    candidates = [os.path.join(ROOT, "cg")]
    for path in candidates:
        for name in names:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
    return None


def _reference_cg_dll_exists() -> bool:
    return _reference_cg_dll_path() is not None


class NativeCgCompatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_backend = os.environ.get("PTCG_BACKEND")
        self.old_portable_search = os.environ.get("PTCG_NATIVE_PORTABLE_SEARCH")
        self.old_lazy_search = os.environ.get("PTCG_NATIVE_LAZY_SEARCH")
        self.old_cpp_battle = os.environ.get("PTCG_NATIVE_CPP_BATTLE")
        os.environ["PTCG_BACKEND"] = "native"
        os.environ["PTCG_NATIVE_PORTABLE_SEARCH"] = "1"
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

        search_option = 4
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

    def test_native_card_and_attack_metadata_match_public_api(self) -> None:
        import ptcg.cg.api as api

        if not _reference_cg_dll_exists():
            raise unittest.SkipTest("reference cg.dll is not present in this checkout")

        os.environ.pop("PTCG_BACKEND", None)
        official_cards = {card.cardId: card for card in api.all_card_data()}
        official_attacks = {attack.attackId: attack for attack in api.all_attack()}

        os.environ["PTCG_BACKEND"] = "native"
        api._CARD_DB_CACHE = None
        api._CARD_TEXT_CACHE = None
        api._OFFICIAL_CARD_META_CACHE = None
        native_cards = {card.cardId: card for card in api.all_card_data()}
        native_attacks = {attack.attackId: attack for attack in api.all_attack()}

        self.assertEqual(set(official_cards), set(native_cards))
        self.assertEqual(set(official_attacks), set(native_attacks))
        for card_id, official in official_cards.items():
            native = native_cards[card_id]
            self.assertEqual(native.name, official.name)
            self.assertEqual(
                [(skill.name, skill.text) for skill in native.skills],
                [(skill.name, skill.text) for skill in official.skills],
            )
        for attack_id, official in official_attacks.items():
            native = native_attacks[attack_id]
            self.assertEqual(native.name, official.name)
            self.assertEqual(native.text, official.text)

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
