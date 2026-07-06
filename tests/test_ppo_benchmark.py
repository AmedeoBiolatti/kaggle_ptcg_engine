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
