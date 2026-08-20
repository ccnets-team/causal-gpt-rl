"""The cached path stages two tokens instead of carrying a full window.

`predict_incremental_cached` reads one token per step once the cache holds
history, so the rolling window it is sliced from is staging, not context — the
context is in the KV cache. Two slots is the floor: `update_data` puts a new
observation in the trailing slot with no action beside it, and only the next
roll pairs it into the `(state, action)` token the model consumes.

The control here is the same runner with the pre-change buffer rebuilt at
`context_length + 1`. Every comparison against it has to come out at exactly
zero: this is a storage change, not an inference change, and a tolerance would
hide the one thing worth pinning.

The windowed path reads all of the window and keeps it, which is why the mode
cannot move once the buffer is sized from it.
"""
import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch

from causal_gpt_rl.inference.context.buffer import ContextBuffer
from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def _model(network="Llama") -> AutoregressiveModel:
    # `use_eos=True` is what the trainer emits, so it is the shape every bundle
    # has: the termination head exists and `act_with_info` carries a real
    # probability. Defaulting it off here would measure parity on a model no
    # deployment runs.
    torch.manual_seed(0)
    return AutoregressiveModel(
        ModelConfig(d_model=32, num_heads=4, network_name=network, use_eos=True),
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=extract_data_specs_from_space(_ACTION_SPACE),
        device=torch.device("cpu"),
    )


def _runner(num_envs=3, *, ctx=32, kv=1000, bos="retain", windowed=False,
            network="Llama"):
    return PolicyRunner(
        model=_model(network),
        action_schedule=[("continuous", 2, None, None)],
        state_size=2,
        context_length=ctx,
        num_envs=num_envs,
        use_windowed=windowed,
        kv_cache_max_len=kv,
        bos_cache_mode=bos,
    )


def _with_full_window(runner):
    """Rebuild the buffer the way it was sized before this change."""
    runner.buffer = ContextBuffer(
        num_agents=runner.num_envs,
        context_length=runner.context_length,
        state_size=runner.state_size,
        action_size=runner.action_size,
        kv_cache_max_len=None if runner.use_windowed else runner.kv_cache_max_len,
    )
    return runner


def _rollout(runner, seed, *, scenario="plain", steps=20):
    """Every step's action and the auxiliary output that rides along with it."""
    rng = np.random.default_rng(seed)
    runner.reset(rng.standard_normal((runner.num_envs, 2)).astype(np.float32))
    steps_out = []
    for step in range(steps):
        action, info = runner.act_with_info()
        steps_out.append((np.asarray(action), info))
        if scenario == "reset_rows" and step == 5:
            done = np.zeros(runner.num_envs, dtype=bool)
            done[1] = True
            runner.reset_rows(done)
        if scenario == "add_rows" and step == 5:
            runner.add_rows(rng.standard_normal((2, 2)).astype(np.float32))
        runner.observe(
            rng.standard_normal((runner.num_envs, 2)).astype(np.float32)
        )
    return steps_out


def test_cached_runner_allocates_two_slots():
    """The cached buffer holds the pending observation and one whole token."""
    runner = _runner()
    assert runner.buffer._internal_len == 2
    # The bundle's trained window is unchanged: provenance and the KV default
    # are both keyed off it.
    assert runner.context_length == 32


def test_windowed_runner_keeps_the_whole_window():
    """Full-window inference reads the window, so it still gets one."""
    runner = _runner(windowed=True)
    assert runner.buffer._internal_len == runner.context_length + 1


@pytest.mark.parametrize("network", ["Llama", "GPT2"])
@pytest.mark.parametrize(
    "scenario,ctx,kv,bos",
    [
        ("plain", 32, 1000, "retain"),
        ("plain", 32, 1000, "discard"),
        ("plain", 6, 64, "retain"),      # retention past the trained window
        ("plain", 32, 8, "retain"),      # retention under it
        ("reset_rows", 32, 1000, "retain"),
        ("add_rows", 32, 1000, "retain"),
    ],
)
def test_two_slot_window_matches_the_full_window(scenario, ctx, kv, bos, network):
    """Storage only: the whole forward must be bit-identical to the old buffer.

    Both the action and `act_with_info`'s auxiliary output, because the two come
    off the same forward and a staging change must not move either.

    Both backbone families run. Position ids come from the cache either way, so
    the learned-absolute-position backbone should be as unaffected as the rotary
    one — but 0.17.0 records the two differing on partial restarts, so it is
    measured rather than assumed.
    """
    kw = dict(ctx=ctx, kv=kv, bos=bos, network=network)
    staged = _rollout(_runner(**kw), 7, scenario=scenario)
    full = _rollout(_with_full_window(_runner(**kw)), 7, scenario=scenario)
    for step, ((a_act, a_info), (b_act, b_info)) in enumerate(zip(staged, full)):
        np.testing.assert_array_equal(a_act, b_act, err_msg=f"action, step {step}")
        assert a_info.keys() == b_info.keys(), f"info keys, step {step}"
        for key in a_info:
            a_val, b_val = a_info[key], b_info[key]
            assert a_val is not None, f"{key} is None; the head under test is absent"
            np.testing.assert_array_equal(
                np.asarray(a_val), np.asarray(b_val), err_msg=f"{key}, step {step}"
            )


def test_assigning_use_windowed_warns_and_leaves_the_mode_alone():
    """A live mode switch is refused, and says so, without raising."""
    runner = _runner()
    with pytest.warns(UserWarning, match="fixed at construction"):
        runner.use_windowed = True
    assert runner.use_windowed is False
    assert runner.buffer._internal_len == 2


def test_assigning_the_same_mode_is_silent():
    """Re-asserting the mode the runner is already in changes nothing."""
    runner = _runner()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        runner.use_windowed = False
    assert runner.use_windowed is False


def test_add_rows_refuses_a_cache_it_cannot_grow():
    """No rolling window means no rebuild, so the fallback is gone.

    Only a cache exposing neither `layers` nor `key_cache` reaches this; every
    supported DynamicCache grows in place.
    """
    runner = _runner(num_envs=2)
    _rollout(runner, 0, steps=3)
    runner.buffer.cache.past_key_values = object()   # an ungrowable layout

    with pytest.raises(RuntimeError, match="cannot grow the shared KV cache"):
        runner.add_rows(np.zeros((1, 2), dtype=np.float32))


def test_a_refused_add_rows_leaves_the_batch_alone():
    """Refusing before the batch is widened is what makes the error survivable.

    Raising after the widening would leave a wider batch with its cache dropped,
    and a caller that caught the error would `act()` straight into the silent
    truncation the refusal exists to prevent.
    """
    runner = _runner(num_envs=2)
    _rollout(runner, 0, steps=3)
    owned = runner.buffer.cache._valid_len.copy()
    sentinel = object()
    runner.buffer.cache.past_key_values = sentinel

    with pytest.raises(RuntimeError):
        runner.add_rows(np.zeros((1, 2), dtype=np.float32))

    # Nothing widened, and the cache was not dropped: `add_agent_rows` resets it
    # on a failed grow, so its survival is what says the grow was never tried.
    assert runner.num_envs == 2
    assert runner.buffer.states.shape[0] == 2
    assert runner.buffer.cache.num_agents == 2
    assert runner._is_reset is True
    assert runner.buffer.cache.past_key_values is sentinel
    np.testing.assert_array_equal(runner.buffer.cache._valid_len, owned)
