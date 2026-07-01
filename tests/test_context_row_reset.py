"""Per-row episode restart mechanics for batched inference.

Covers the buffer primitives behind `PolicyRunner.reset_rows`: a per-agent
`is_bos` vector and row-scoped context wipes that leave surviving rows intact.
"""
import unittest

import numpy as np

from causal_gpt_rl.inference.context.buffer import ContextBuffer


def _make_buffer(num_agents=3, context_length=4, state_size=2, action_size=1):
    return ContextBuffer(
        num_agents=num_agents,
        context_length=context_length,
        state_size=state_size,
        action_size=action_size,
    )


class TestPerRowIsBos(unittest.TestCase):
    def test_scalar_is_bos_unchanged(self):
        """Scalar calls stay byte-identical to the pre-vector behavior."""
        buf = _make_buffer()
        state = np.arange(6, dtype=np.float32).reshape(3, 2)
        action = np.ones((3, 1), dtype=np.float32)

        buf.update_data(state, action, is_bos=1.0)

        # BOS seeds the visible slot (-2) for every row and marks is_bos=1.
        np.testing.assert_array_equal(buf.states[:, -2], state)
        np.testing.assert_array_equal(buf.is_bos[:, -2, 0], np.ones(3))
        np.testing.assert_array_equal(buf.masks[:, -2], np.ones(3))

    def test_vector_is_bos_seeds_only_flagged_rows(self):
        buf = _make_buffer()
        state = np.arange(6, dtype=np.float32).reshape(3, 2)
        action = np.zeros((3, 1), dtype=np.float32)

        # Row 1 starts a fresh episode; rows 0 and 2 continue.
        buf.update_data(state, action, is_bos=np.array([0.0, 1.0, 0.0]))

        np.testing.assert_array_equal(buf.is_bos[:, -2, 0], np.array([0.0, 1.0, 0.0]))
        # Only the BOS row duplicates its observation into slot -2.
        np.testing.assert_array_equal(buf.states[1, -2], state[1])
        # Non-BOS rows leave slot -2 as the rolled-in (zero) history.
        np.testing.assert_array_equal(buf.states[0, -2], np.zeros(2))
        np.testing.assert_array_equal(buf.states[2, -2], np.zeros(2))


class TestResetContextRows(unittest.TestCase):
    def test_wipes_only_masked_rows_and_drops_cache(self):
        buf = _make_buffer()
        # Fill some history for all rows.
        buf.update_data(np.ones((3, 2), np.float32), np.zeros((3, 1), np.float32), 1.0)
        buf.update_data(np.full((3, 2), 2.0, np.float32), np.ones((3, 1), np.float32))
        buf.set_past_key_values(("stub-cache",))

        survivor_states = buf.states[0].copy()
        survivor_masks = buf.masks[0].copy()

        buf.reset_context_rows(np.array([False, True, False]))

        # Survivor row untouched.
        np.testing.assert_array_equal(buf.states[0], survivor_states)
        np.testing.assert_array_equal(buf.masks[0], survivor_masks)
        # Reset row wiped: zeroed states, all-BOS, fully masked out.
        np.testing.assert_array_equal(buf.states[1], np.zeros_like(buf.states[1]))
        np.testing.assert_array_equal(buf.masks[1], np.zeros_like(buf.masks[1]))
        np.testing.assert_array_equal(buf.is_bos[1], np.ones_like(buf.is_bos[1]))
        # Shared cache is invalidated (recomputed from the buffer next step).
        self.assertIsNone(buf.get_past_key_values())

    def test_noop_mask_leaves_everything(self):
        buf = _make_buffer()
        buf.update_data(np.ones((3, 2), np.float32), np.zeros((3, 1), np.float32), 1.0)
        before = buf.states.copy()

        buf.reset_context_rows(np.zeros(3, dtype=bool))

        np.testing.assert_array_equal(buf.states, before)


if __name__ == "__main__":
    unittest.main()
