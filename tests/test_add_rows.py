"""Grow-only batch resize: `ContextBuffer.add_rows` / `PolicyRunner.add_rows`.

Adds agent rows to a live runner without disturbing existing ones. The buffer
contract mirrors `reset_context_rows` (new rows seeded as a fresh BOS episode,
existing rows byte-untouched, shared cache invalidated); the model recompute
path is the same warm-start as `reset_rows`, so these tests cover the buffer
bookkeeping plus an end-to-end shape/loop smoke test on a tiny CPU model.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from causal_gpt_rl.inference.context.buffer import ContextBuffer
from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_CFG = ModelConfig(d_model=32, num_heads=4)
_ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def _make_buffer(num_agents=2, context_length=4, state_size=2, action_size=1):
    return ContextBuffer(
        num_agents=num_agents,
        context_length=context_length,
        state_size=state_size,
        action_size=action_size,
    )


def _model() -> AutoregressiveModel:
    return AutoregressiveModel(
        _CFG,
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=extract_data_specs_from_space(_ACTION_SPACE),
        device=torch.device("cpu"),
    )


def _runner(num_envs=2, *, context_length=8) -> PolicyRunner:
    return PolicyRunner(
        model=_model(),
        action_schedule=[("continuous", 2, None, None)],
        state_size=2,
        context_length=context_length,
        num_envs=num_envs,
    )


# --------------------------------------------------------------------------- #
# Buffer-level contract (pure numpy).
# --------------------------------------------------------------------------- #

def test_add_rows_appends_seeded_rows():
    buf = _make_buffer(num_agents=2)
    # Give existing rows some history.
    buf.update_data(np.ones((2, 2), np.float32), np.zeros((2, 1), np.float32), 1.0)
    buf.update_data(np.full((2, 2), 2.0, np.float32), np.ones((2, 1), np.float32))

    init = np.array([[7.0, 8.0]], dtype=np.float32)  # one new row
    buf.add_rows(init)

    assert buf.num_agents == 3
    for arr in (buf.states, buf.actions, buf.is_bos, buf.masks):
        assert arr.shape[0] == 3
    # New row seeded exactly like a fresh reset+BOS: obs at visible slot (-2),
    # is_bos=1 there, masked in; the trailing staged slot (-1) also holds obs.
    np.testing.assert_array_equal(buf.states[2, -2], init[0])
    np.testing.assert_array_equal(buf.states[2, -1], init[0])
    assert buf.is_bos[2, -2, 0] == 1.0
    assert buf.masks[2, -2] == 1.0
    np.testing.assert_array_equal(buf.actions[2], np.zeros_like(buf.actions[2]))


def test_add_rows_leaves_existing_rows_untouched_and_drops_cache():
    buf = _make_buffer(num_agents=2)
    buf.update_data(np.ones((2, 2), np.float32), np.zeros((2, 1), np.float32), 1.0)
    buf.update_data(np.full((2, 2), 2.0, np.float32), np.ones((2, 1), np.float32))
    buf.set_past_key_values(("stub-cache",))

    before_states = buf.states.copy()
    before_actions = buf.actions.copy()

    buf.add_rows(np.array([[7.0, 8.0]], dtype=np.float32))

    # Existing rows byte-identical.
    np.testing.assert_array_equal(buf.states[:2], before_states)
    np.testing.assert_array_equal(buf.actions[:2], before_actions)
    # Shared cache invalidated (rebuilt at the new batch size next step).
    assert buf.get_past_key_values() is None


def test_add_rows_can_append_multiple():
    buf = _make_buffer(num_agents=1)
    buf.update_data(np.ones((1, 2), np.float32), np.zeros((1, 1), np.float32), 1.0)
    buf.add_rows(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32))
    assert buf.num_agents == 4


def test_add_rows_rejects_bad_shape_and_empty():
    buf = _make_buffer(num_agents=2)
    with pytest.raises(ValueError):
        buf.add_rows(np.zeros((1, 3), dtype=np.float32))  # wrong state_size
    with pytest.raises(ValueError):
        buf.add_rows(np.zeros((0, 2), dtype=np.float32))  # no rows


# --------------------------------------------------------------------------- #
# Runner-level (tiny CPU model) — end-to-end shape / loop.
# --------------------------------------------------------------------------- #

def test_runner_add_rows_grows_and_continues():
    runner = _runner(num_envs=2)
    runner.reset(np.zeros((2, 2), dtype=np.float32))
    for _ in range(3):
        runner.act()
        runner.observe(np.zeros((2, 2), dtype=np.float32))

    runner.add_rows(np.array([[0.5, -0.5]], dtype=np.float32))
    assert runner.num_envs == 3

    action = runner.act()
    assert np.asarray(action).shape == (3, 2)
    # Loop keeps working at the new batch size.
    runner.observe(np.zeros((3, 2), dtype=np.float32))
    action = runner.act()
    assert np.asarray(action).shape == (3, 2)


def test_runner_add_rows_preserves_existing_buffer_rows():
    runner = _runner(num_envs=2)
    runner.reset(np.zeros((2, 2), dtype=np.float32))
    for step in range(3):
        runner.act()
        runner.observe(np.full((2, 2), float(step), dtype=np.float32))

    before = runner.buffer.states[:2].copy()
    runner.add_rows(np.array([[9.0, 9.0]], dtype=np.float32))
    # add_rows only appends: the two original rows' buffer content is unchanged.
    np.testing.assert_array_equal(runner.buffer.states[:2], before)


def test_runner_add_rows_before_reset_raises():
    runner = _runner(num_envs=2)
    with pytest.raises(RuntimeError):
        runner.add_rows(np.zeros((1, 2), dtype=np.float32))
