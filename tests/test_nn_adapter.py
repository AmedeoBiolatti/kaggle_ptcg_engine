from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(ROOT, "engine", "build"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from validation.decks import MEGA_LUCARIO  # noqa: E402


class NnAdapterTest(unittest.TestCase):
    def test_policy_value_net_shapes(self) -> None:
        try:
            import ptcg_engine as E
            import torch
            from ptcg.nn import PtcgPolicyValueNet
        except ImportError as exc:
            raise unittest.SkipTest(f"optional torch adapter dependency missing: {exc}") from exc

        state = E.new_game(MEGA_LUCARIO, MEGA_LUCARIO, 1)
        model = PtcgPolicyValueNet(card_dim=32, hidden_dim=64)
        out = model(E.rl_state_ids(state), E.rl_action_ids(state))
        self.assertEqual(tuple(out["logits"].shape), (E.RL_MAX_ACTIONS,))
        self.assertEqual(tuple(out["action_embedding"].shape), (E.RL_MAX_ACTIONS, 32))
        self.assertEqual(tuple(out["state_embedding"].shape), (32,))
        self.assertEqual(tuple(out["mask"].shape), (E.RL_MAX_ACTIONS,))
        self.assertEqual(out["value"].ndim, 0)

        env = E.VectorEnv(MEGA_LUCARIO, MEGA_LUCARIO, 4, 1)
        if not hasattr(env, "state_ids"):
            raise unittest.SkipTest("VectorEnv.state_ids is not available in this build")
        state_batch = env.state_ids()
        self.assertEqual(tuple(state_batch["in_play"].shape), (4, 2, E.STATE_INPLAY_SLOTS, E.STATE_INPLAY_WIDTH))
        self.assertEqual(tuple(state_batch["zones"].shape), (4, 2, E.STATE_ZONE_COUNT, E.STATE_ZONE_SLOTS))
        self.assertEqual(tuple(state_batch["player_counts"].shape), (4, 2, 5))
        self.assertEqual(tuple(state_batch["player_status"].shape), (4, 2, 5))
        self.assertEqual(tuple(state_batch["global"].shape), (4, E.STATE_GLOBAL_WIDTH))
        action_batch = env.action_ids()
        batched = model(state_batch, action_batch)
        self.assertEqual(tuple(batched["logits"].shape), (4, E.RL_MAX_ACTIONS))
        self.assertEqual(tuple(batched["value"].shape), (4,))

        if not hasattr(env, "observe_ids_into"):
            raise unittest.SkipTest("VectorEnv observe_ids_into APIs are not available in this build")
        state_into = {
            key: np.empty_like(value)
            for key, value in state_batch.items()
            if hasattr(value, "shape")
        }
        action_into = {
            key: np.empty_like(value)
            for key, value in action_batch.items()
            if hasattr(value, "shape")
        }
        player = np.empty((4,), dtype=np.int32)
        result = np.empty((4,), dtype=np.int32)
        env.observe_ids_into(state_into, action_into, player, result)
        for key in state_into:
            np.testing.assert_array_equal(state_into[key], state_batch[key])
        for key in action_into:
            np.testing.assert_array_equal(action_into[key], action_batch[key])

        torch_state = {key: torch.from_numpy(value) for key, value in state_into.items()}
        torch_action = {key: torch.from_numpy(value) for key, value in action_into.items()}
        viewed = model(torch_state, torch_action)
        self.assertEqual(tuple(viewed["logits"].shape), (4, E.RL_MAX_ACTIONS))
        self.assertEqual(tuple(viewed["value"].shape), (4,))
        tensor_viewed = model.forward_tensors(
            torch_state["in_play"],
            torch_state["zones"],
            torch_state["player_counts"],
            torch_state["player_status"],
            torch_state["global"],
            torch_action["options"],
            torch_action["mask"],
        )
        self.assertEqual(tuple(tensor_viewed["logits"].shape), (4, E.RL_MAX_ACTIONS))
        self.assertEqual(tuple(tensor_viewed["value"].shape), (4,))
        torch.testing.assert_close(tensor_viewed["logits"], viewed["logits"])
        torch.testing.assert_close(tensor_viewed["value"], viewed["value"])


if __name__ == "__main__":
    unittest.main()
