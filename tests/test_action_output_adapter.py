"""P5 tests — action output adapter at the runner boundary.

Covers the serving side of action_container: a flat per-head model action is
unflattened back into the declared Dict / Tuple container, while a non-container
action space (Box / Discrete / MultiDiscrete) keeps the byte-identical flat
per-head path. Mirrors tests/test_state_input_adapter.py on the output side.

The action heads stay in declared order (no continuous-first permutation; output
== the next input, AR self-feedback — L2 §2), so the adapter is a plain
gym.spaces.unflatten of the declared-order flat.
"""
import tempfile

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.adapters import (
    ActionOutputAdapter,
    make_action_output_adapter,
)
from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_CFG = ModelConfig(d_model=32, num_heads=4)


class _Norm:
    """Minimal duck-typed state normalizer (mean/var state_dict)."""

    def __init__(self, n: int):
        self.n = n

    def state_dict(self):
        return {"mean": torch.zeros(self.n), "var": torch.ones(self.n)}


def _tuple_box_discrete() -> gym.spaces.Tuple:
    return gym.spaces.Tuple(
        (gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32), gym.spaces.Discrete(3))
    )


def _dict_box_discrete() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "move": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "mode": gym.spaces.Discrete(3),
        }
    )


def _container_action_model(
    action_space: gym.spaces.Space, state_size: int = 2
) -> AutoregressiveModel:
    # The trainer's path: typed action specs come straight from the declared
    # space (same walk order gym.spaces.flatten/unflatten use), so the per-head
    # model layout and the adapter's unflatten agree by construction.
    # Typed action specs straight from the declared space (the trainer's path) —
    # bounds are float32 numpy arrays. This exercises the realistic export path
    # and regression-guards the input/output adapter buffer-aliasing fix (a
    # float32 numpy bound shared across the feedback + output adapters used to
    # make safetensors refuse the bundle).
    action_specs = extract_data_specs_from_space(action_space)
    state_specs = [
        SpaceSpec(
            type="continuous",
            size=state_size,
            dtype=torch.float32,
            low=[-1.0] * state_size,
            high=[1.0] * state_size,
        )
    ]
    return AutoregressiveModel(
        _CFG, state_specs=state_specs, action_specs=action_specs,
        device=torch.device("cpu"),
    )


# --------------------------------------------------------------------------- #
# make_action_output_adapter — engage only for Dict / Tuple containers
# --------------------------------------------------------------------------- #

def test_make_adapter_none_for_non_container():
    assert make_action_output_adapter(None) is None
    assert make_action_output_adapter(
        gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    ) is None
    assert make_action_output_adapter(gym.spaces.Discrete(4)) is None
    assert make_action_output_adapter(gym.spaces.MultiDiscrete([2, 3])) is None


def test_make_adapter_built_for_containers():
    assert isinstance(make_action_output_adapter(_tuple_box_discrete()), ActionOutputAdapter)
    assert isinstance(make_action_output_adapter(_dict_box_discrete()), ActionOutputAdapter)


# --------------------------------------------------------------------------- #
# ActionOutputAdapter — unflatten gym-flat -> declared container
# --------------------------------------------------------------------------- #

def test_adapter_tuple_unflattens_in_declared_order():
    space = _tuple_box_discrete()
    adapter = make_action_output_adapter(space)
    assert adapter.flatdim == 5  # 2 continuous + 3 one-hot
    # gym-flat convention: [box(2) | one_hot(class=2)], declared (positional) order.
    env_flat = np.array([0.5, -0.5, 0.0, 0.0, 1.0], dtype=np.float32)
    out = adapter(env_flat)
    assert isinstance(out, tuple) and len(out) == 2
    assert np.allclose(out[0], [0.5, -0.5])
    assert int(out[1]) == 2


def test_adapter_dict_respects_gym_key_sorting_and_start():
    # gymnasium sorts Dict keys ("mode" < "move"), so the declared/flatten order
    # is mode, move — NOT insertion order. And Discrete(start=1) must map the
    # one-hot class index to start + index; gym.spaces.unflatten does both.
    space = gym.spaces.Dict(
        {
            "move": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "mode": gym.spaces.Discrete(3, start=1),
        }
    )
    adapter = make_action_output_adapter(space)
    assert adapter.flatdim == 5
    # Sorted-key flat: mode one_hot(3) (class 1) then move box(2).
    env_flat = np.array([0.0, 1.0, 0.0, 0.2, 0.3], dtype=np.float32)
    out = adapter(env_flat)
    assert int(out["mode"]) == 2  # start=1 applied: 1 + class 1
    assert np.allclose(out["move"], [0.2, 0.3])


def test_adapter_flatdim_mismatch_is_loud():
    adapter = make_action_output_adapter(_tuple_box_discrete())  # flatdim 5
    try:
        adapter(np.zeros(4, dtype=np.float32))
    except ValueError as exc:
        assert "flatdim" in str(exc)
        return
    raise AssertionError("expected ValueError on flat width mismatch")


def test_runner_refuses_action_space_disagreeing_with_specs():
    # The runner refuses a bundle whose declared action space and action_size
    # disagree (mirrors the input adapter's flatdim guard).
    model = AutoregressiveModel(
        _CFG,
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                                low=[-1.0, -1.0], high=[1.0, 1.0], squash="tanh")],
        device=torch.device("cpu"),
    )
    try:
        PolicyRunner(
            model=model,
            action_schedule=[("continuous", 2, [-1.0, -1.0], [1.0, 1.0])],  # size 2
            state_size=2,
            context_length=8,
            action_space=_tuple_box_discrete(),  # flatdim 5 != action_size 2
        )
    except ValueError as exc:
        assert "flatdim" in str(exc)
        return
    raise AssertionError("expected ValueError on action flatdim/action_size mismatch")


def test_bounded_box_action_exports_from_extracted_specs():
    # Regression for the input/output adapter buffer-aliasing fix: a bounded-tanh
    # continuous action with float32 numpy bounds (straight from a gym Box, the
    # trainer's path) must export. Before the fix, the feedback and output
    # adapters shared that numpy buffer and safetensors refused to save it.
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    action_specs = extract_data_specs_from_space(action_space)
    model = AutoregressiveModel(
        _CFG,
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=action_specs,
        device=torch.device("cpu"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        bundle.export_bundle(  # must not raise a safetensors shared-storage error
            tmp, model=model, model_config=_CFG,
            state_specs=model.state_specs, action_specs=model.action_specs,
            context_length=8, action_space=action_space, state_normalizer=_Norm(2),
        )
        runner = bundle.load_runner(tmp)
    # A bare Box action carries no container adapter — flat continuous output.
    assert runner._output_adapter is None
    runner.reset(np.zeros(2, dtype=np.float32))
    assert np.asarray(runner.act()).shape == (2,)


# --------------------------------------------------------------------------- #
# End-to-end — container-action bundle load + run (synthetic stand-in for the
# trainer hybrid-action fixture; swap in the real fixture once it lands).
# --------------------------------------------------------------------------- #

def _export_and_load(action_space, tmp, **load_kw):
    model = _container_action_model(action_space)
    bundle.export_bundle(
        tmp,
        model=model,
        model_config=_CFG,
        state_specs=model.state_specs,
        action_specs=model.action_specs,
        context_length=8,
        action_space=action_space,
        state_normalizer=_Norm(2),
    )
    return bundle.load_runner(tmp, **load_kw)


def test_end_to_end_tuple_action_bundle():
    action_space = _tuple_box_discrete()
    with tempfile.TemporaryDirectory() as tmp:
        runner = _export_and_load(action_space, tmp)

    assert runner._output_adapter is not None
    assert isinstance(runner.action_space, gym.spaces.Tuple)

    runner.reset(np.zeros(2, dtype=np.float32))
    action = runner.act()
    assert isinstance(action, tuple) and len(action) == 2
    box_val = np.asarray(action[0], dtype=np.float32)
    assert box_val.shape == (2,)
    assert np.all(box_val >= -1.0) and np.all(box_val <= 1.0)  # clipped to bounds
    assert 0 <= int(action[1]) < 3

    # A second step keeps working (autoregressive feedback path).
    action2 = runner.act(np.ones(2, dtype=np.float32))
    assert isinstance(action2, tuple) and len(action2) == 2


def test_end_to_end_dict_action_bundle():
    action_space = _dict_box_discrete()
    with tempfile.TemporaryDirectory() as tmp:
        runner = _export_and_load(action_space, tmp)

    assert isinstance(runner.action_space, gym.spaces.Dict)
    runner.reset(np.zeros(2, dtype=np.float32))
    action = runner.act()
    assert set(action.keys()) == {"move", "mode"}
    assert np.asarray(action["move"]).shape == (2,)
    assert 0 <= int(action["mode"]) < 3


def test_container_output_is_per_env_list_for_multi_env():
    action_space = _tuple_box_discrete()
    with tempfile.TemporaryDirectory() as tmp:
        runner = _export_and_load(action_space, tmp, num_envs=3)

    runner.reset(np.zeros((3, 2), dtype=np.float32))
    actions = runner.act()
    assert isinstance(actions, list) and len(actions) == 3
    for a in actions:
        assert isinstance(a, tuple) and len(a) == 2
        assert np.asarray(a[0]).shape == (2,)
        assert 0 <= int(a[1]) < 3
