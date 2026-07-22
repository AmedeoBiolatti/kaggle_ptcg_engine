from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(ROOT, "engine", "build"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmarks.bench_engines import (  # noqa: E402
    _compute_gae,
    _ppo_indices,
    bench_native_ppo_jax,
    bench_native_ppo_torch,
)
from validation.decks import MEGA_LUCARIO  # noqa: E402


class PpoBenchmarkTest(unittest.TestCase):
    def test_gae_resets_across_done(self) -> None:
        rewards = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        dones = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        values = np.zeros((2, 2), dtype=np.float32)
        bootstrap = np.array([10.0, 10.0], dtype=np.float32)

        advantages, returns = _compute_gae(
            rewards,
            dones,
            values,
            bootstrap,
            gamma=0.99,
            lam=0.95,
        )

        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)
        self.assertAlmostEqual(float(returns[0, 0]), 1.0, places=5)
        self.assertGreater(float(returns[0, 1]), float(returns[0, 0]))
        self.assertAlmostEqual(float(returns[1, 0]), 10.9, places=5)
        self.assertAlmostEqual(float(returns[1, 1]), 10.9, places=5)

    def test_ppo_indices_shape_and_bounds(self) -> None:
        indices = _ppo_indices(total=12, minibatch_size=4, epochs=3, seed=7)
        self.assertEqual(indices.shape, (9, 4))
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 12))
        with self.assertRaises(ValueError):
            _ppo_indices(total=3, minibatch_size=4, epochs=1, seed=1)

    def test_vector_envs_expose_unified_id_step_api(self) -> None:
        try:
            import ptcg_engine as E
        except ImportError as exc:
            raise unittest.SkipTest(f"native extension missing: {exc}") from exc

        def assert_id_batch(state: dict, action: dict, n: int) -> None:
            self.assertEqual(state["in_play"].shape, (n, 2, E.STATE_INPLAY_SLOTS, E.STATE_INPLAY_WIDTH))
            self.assertEqual(state["zones"].shape, (n, 2, E.STATE_ZONE_COUNT, E.STATE_ZONE_SLOTS))
            self.assertEqual(state["global"].shape, (n, E.STATE_GLOBAL_WIDTH))
            self.assertEqual(state["select_options"].shape, (n, E.RL_MAX_ACTIONS, E.STATE_SELECT_OPTION_WIDTH))
            self.assertEqual(action["meta"].shape, (n, E.ACTION_META_WIDTH))
            self.assertEqual(action["options"].shape, (n, E.RL_MAX_ACTIONS, E.ACTION_OPTION_WIDTH))
            self.assertEqual(action["deck"].shape, (n, E.STATE_ZONE_SLOTS))
            self.assertEqual(action["mask"].shape, (n, E.RL_MAX_ACTIONS))
            self.assertTrue(np.all(action["mask"].sum(axis=1) > 0))

        for env_cls, has_episode_len in ((E.VectorEnv, False), (E.PpoBatchEnv, True)):
            env = env_cls(MEGA_LUCARIO, MEGA_LUCARIO, 4, 1)
            self.assertTrue(hasattr(env, "observe_ids"))
            self.assertTrue(hasattr(env, "step"))
            self.assertFalse(hasattr(env, "step_ids"))
            state, action, player, result = env.observe_ids()
            assert_id_batch(state, action, env.size())
            self.assertEqual(player.shape, (env.size(),))
            self.assertEqual(result.shape, (env.size(),))

            chosen = np.array(
                [np.flatnonzero(action["mask"][i])[0] for i in range(env.size())],
                dtype=np.int32,
            )
            stepped = env.step(chosen)
            if has_episode_len:
                state2, reward, done, action2, player2, result2, episode_len = stepped
                self.assertEqual(episode_len.shape, (env.size(),))
            else:
                state2, reward, done, action2, player2, result2 = stepped
            assert_id_batch(state2, action2, env.size())
            self.assertEqual(reward.shape, (env.size(),))
            self.assertEqual(done.shape, (env.size(),))
            self.assertEqual(player2.shape, (env.size(),))
            self.assertEqual(result2.shape, (env.size(),))

    def test_vector_env_explicit_thread_counts_are_deterministic(self) -> None:
        try:
            import ptcg_engine as E
        except ImportError as exc:
            raise unittest.SkipTest(f"native extension missing: {exc}") from exc

        serial = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 256, 17, 1)
        parallel = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 256, 17, 4)
        for _ in range(8):
            serial_step = serial.step(np.zeros(256, dtype=np.int32))
            parallel_step = parallel.step(np.zeros(256, dtype=np.int32))
            (
                serial_state,
                serial_reward,
                serial_done,
                serial_action,
                serial_player,
                serial_result,
            ) = serial_step
            (
                parallel_state,
                parallel_reward,
                parallel_done,
                parallel_action,
                parallel_player,
                parallel_result,
            ) = parallel_step
            for key in serial_state:
                if isinstance(serial_state[key], np.ndarray):
                    np.testing.assert_array_equal(serial_state[key], parallel_state[key])
            for key in serial_action:
                if isinstance(serial_action[key], np.ndarray):
                    np.testing.assert_array_equal(serial_action[key], parallel_action[key])
            np.testing.assert_array_equal(serial_reward, parallel_reward)
            np.testing.assert_array_equal(serial_done, parallel_done)
            np.testing.assert_array_equal(serial_player, parallel_player)
            np.testing.assert_array_equal(serial_result, parallel_result)

    def test_vector_env_int16_ids_are_lossless_and_half_sized(self) -> None:
        try:
            import ptcg_engine as E
        except ImportError as exc:
            raise unittest.SkipTest(f"native extension missing: {exc}") from exc

        full_env = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 16, 23, 1)
        compact_env = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 16, 23, 1)
        full_state, full_action, full_player, full_result = full_env.observe_ids()
        compact_state, compact_action, compact_player, compact_result = (
            compact_env.observe_ids16()
        )

        for key in ("in_play", "zones", "player_counts", "player_status", "global",
                    "select_meta", "select_deck"):
            self.assertEqual(compact_state[key].dtype, np.int16)
            np.testing.assert_array_equal(compact_state[key], full_state[key])
            self.assertEqual(compact_state[key].nbytes * 2, full_state[key].nbytes)
        for key in ("meta", "deck"):
            self.assertEqual(compact_action[key].dtype, np.int16)
            np.testing.assert_array_equal(compact_action[key], full_action[key])

        compact_options = compact_action["options"].astype(np.int32)
        reconstructed_refs = compact_options[..., 18] + (
            compact_options[..., 19] << int(compact_action["raw_ref_shift"])
        )
        np.testing.assert_array_equal(reconstructed_refs, full_action["options"][..., 18])
        for column in list(range(18)) + list(range(20, E.ACTION_OPTION_WIDTH)):
            np.testing.assert_array_equal(
                compact_options[..., column], full_action["options"][..., column]
            )
        np.testing.assert_array_equal(compact_action["mask"], full_action["mask"])
        np.testing.assert_array_equal(compact_player, full_player)
        np.testing.assert_array_equal(compact_result, full_result)

        actions = np.zeros(16, dtype=np.int32)
        full_step = full_env.step(actions)
        compact_step = compact_env.step16(actions)
        for key in ("in_play", "zones", "player_counts", "player_status", "global"):
            np.testing.assert_array_equal(compact_step[0][key], full_step[0][key])
        np.testing.assert_array_equal(compact_step[1], full_step[1])
        np.testing.assert_array_equal(compact_step[2], full_step[2])
        np.testing.assert_array_equal(compact_step[4], full_step[4])
        np.testing.assert_array_equal(compact_step[5], full_step[5])

        into_env = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 16, 23, 1)
        into_state, into_action, into_player, into_result = (
            into_env.observe_ids16()
        )
        reward = np.empty(16, dtype=np.float32)
        done = np.empty(16, dtype=np.uint8)
        state_ptr = into_state["in_play"].__array_interface__["data"][0]
        option_ptr = into_action["options"].__array_interface__["data"][0]
        returned = into_env.step16_into(
            actions, into_state, into_action, reward, done,
            into_player, into_result
        )
        self.assertIsNone(returned)
        self.assertEqual(
            state_ptr, into_state["in_play"].__array_interface__["data"][0]
        )
        self.assertEqual(
            option_ptr, into_action["options"].__array_interface__["data"][0]
        )
        for key in ("in_play", "zones", "player_counts", "player_status", "global"):
            np.testing.assert_array_equal(into_state[key], compact_step[0][key])
        np.testing.assert_array_equal(reward, compact_step[1])
        np.testing.assert_array_equal(done, compact_step[2])
        np.testing.assert_array_equal(into_action["mask"], compact_step[3]["mask"])
        np.testing.assert_array_equal(into_player, compact_step[4])
        np.testing.assert_array_equal(into_result, compact_step[5])

    def test_float_feature_env_methods_are_deprecated(self) -> None:
        try:
            import ptcg_engine as E
            import warnings
        except ImportError as exc:
            raise unittest.SkipTest(f"native extension missing: {exc}") from exc

        for env_cls in (E.VectorEnv, E.PpoBatchEnv):
            env = env_cls(MEGA_LUCARIO, MEGA_LUCARIO, 2, 1)
            state, action, _player, _result = env.observe_ids()
            chosen = np.array(
                [np.flatnonzero(action["mask"][i])[0] for i in range(env.size())],
                dtype=np.int32,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                env.step(chosen)
            self.assertEqual(caught, [])

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                env.observe_features()
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                env.step_features(chosen)
            self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

        env = E.PpoBatchEnv(MEGA_LUCARIO, MEGA_LUCARIO, 2, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env.action_features()
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_torch_masked_samples_are_legal_and_update_changes_params(self) -> None:
        try:
            import ptcg_engine as E
            import torch
            import torch.nn.functional as F
            from ptcg.nn import PtcgPolicyValueNet
        except ImportError as exc:
            raise unittest.SkipTest(f"optional native torch dependencies missing: {exc}") from exc

        env = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 4, 1)
        if not hasattr(env, "state_ids") or not hasattr(env, "action_ids"):
            raise unittest.SkipTest("VectorEnv id APIs are not available in this build")
        state = env.state_ids()
        action = env.action_ids()
        model = PtcgPolicyValueNet(card_dim=8, hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        out = model(state, action)
        sampled = torch.distributions.Categorical(logits=out["logits"]).sample()
        mask = torch.as_tensor(action["mask"], dtype=torch.bool)
        self.assertTrue(bool(mask.gather(1, sampled[:, None]).all()))

        before = [param.detach().clone() for param in model.parameters()]
        logp = F.log_softmax(out["logits"], dim=-1).gather(1, sampled[:, None]).squeeze(1)
        loss = -logp.mean() + 0.5 * out["value"].pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        changed = any(
            not torch.equal(previous, current.detach())
            for previous, current in zip(before, model.parameters(), strict=True)
        )
        self.assertTrue(changed)

    def test_torch_ppo_tiny_smoke(self) -> None:
        try:
            import ptcg_engine as E
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"optional native torch dependencies missing: {exc}") from exc
        if not hasattr(E.VectorEnv, "observe_ids_into"):
            raise unittest.SkipTest("VectorEnv.observe_ids_into is not available in this build")

        steps, games, warm_seconds, metrics = bench_native_ppo_torch(
            MEGA_LUCARIO,
            MEGA_LUCARIO,
            seed=1,
            batch_size=4,
            rollout_steps=3,
            minibatch_size=4,
            epochs=1,
            gamma=0.99,
            lam=0.95,
            clip=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            lr=3e-4,
            card_dim=8,
            hidden_dim=8,
            device="cpu",
            matmul_precision=None,
            compile_model=True,
            mixed_precision="bf16",
        )
        self.assertEqual(steps, 12)
        self.assertEqual(games, 4)
        self.assertGreater(warm_seconds, 0.0)
        for key in ("rollout/s", "update/s", "total/s", "policy_loss", "value_loss", "entropy", "total_loss"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))

    def test_jax_ppo_tiny_smoke(self) -> None:
        try:
            import jax  # noqa: F401
            import ptcg_engine as E
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"optional native jax dependencies missing: {exc}") from exc
        if not hasattr(E.VectorEnv, "observe_ids_into"):
            raise unittest.SkipTest("VectorEnv.observe_ids_into is not available in this build")

        steps, games, warm_seconds, metrics = bench_native_ppo_jax(
            MEGA_LUCARIO,
            MEGA_LUCARIO,
            seed=1,
            batch_size=4,
            rollout_steps=3,
            minibatch_size=4,
            epochs=1,
            gamma=0.99,
            lam=0.95,
            clip=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            lr=3e-4,
            card_dim=8,
            hidden_dim=8,
            matmul_precision="default",
            mixed_precision="bf16",
        )
        self.assertEqual(steps, 12)
        self.assertEqual(games, 4)
        self.assertGreater(warm_seconds, 0.0)
        for key in ("rollout/s", "update/s", "total/s", "policy_loss", "value_loss", "entropy", "total_loss"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]))


if __name__ == "__main__":
    unittest.main()
