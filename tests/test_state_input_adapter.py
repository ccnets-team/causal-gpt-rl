"""P4 tests — structured-state input adapter at the runner boundary.

Covers the serving side of hybrid_state: a structured env observation
(Dict / Tuple / Discrete) is flattened continuous-first into the model's
canonical flat state, while a plain Box / no space keeps the byte-identical
raw-passthrough path. Mirrors the trainer fixture shape Dict{Box(3)+Discrete(4)}.
"""
import tempfile

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.adapters import (
    StateInputAdapter,
    make_state_input_adapter,
)
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


def _dict_box_discrete() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "cont": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "disc": gym.spaces.Discrete(4),
        }
    )


# --------------------------------------------------------------------------- #
# make_state_input_adapter — engage only for structured spaces
# --------------------------------------------------------------------------- #

def test_make_adapter_none_for_box_and_none():
    assert make_state_input_adapter(None) is None
    box = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
    assert make_state_input_adapter(box) is None


def test_make_adapter_built_for_structured_spaces():
    for space in (
        _dict_box_discrete(),
        gym.spaces.Discrete(4),
        gym.spaces.MultiDiscrete([2, 3]),
        gym.spaces.Tuple(
            (gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
             gym.spaces.Discrete(3))
        ),
    ):
        assert isinstance(make_state_input_adapter(space), StateInputAdapter)


# --------------------------------------------------------------------------- #
# StateInputAdapter — flatten continuous-first
# --------------------------------------------------------------------------- #

def test_adapter_dict_box_discrete_is_continuous_first():
    space = _dict_box_discrete()
    adapter = make_state_input_adapter(space)
    assert adapter.flatdim == 7  # 3 continuous + 4 one-hot
    assert adapter.cf.n_cont == 3

    obs = {"cont": np.array([0.1, 0.2, 0.3], dtype=np.float32), "disc": 2}
    out = adapter(obs)
    assert out.dtype == np.float32
    assert out.shape == (7,)
    # Continuous block lands in the front n_cont slots; one-hot tail follows.
    assert np.allclose(out[:3], [0.1, 0.2, 0.3])
    assert np.allclose(out[3:], [0.0, 0.0, 1.0, 0.0])  # one-hot of class 2


def test_adapter_reorders_discrete_declared_before_continuous():
    # Keys sort so the discrete leaf is *declared* first; the adapter must still
    # emit continuous-first, i.e. a non-identity permutation.
    space = gym.spaces.Dict(
        {
            "a_disc": gym.spaces.Discrete(4),
            "z_cont": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        }
    )
    adapter = make_state_input_adapter(space)
    assert adapter.flatdim == 7
    assert adapter.cf.n_cont == 3
    assert adapter.cf.flat_perm != list(range(7))  # real reordering happened

    obs = {"a_disc": 1, "z_cont": np.array([0.5, 0.6, 0.7], dtype=np.float32)}
    out = adapter(obs)
    assert np.allclose(out[:3], [0.5, 0.6, 0.7])      # continuous moved to front
    assert np.allclose(out[3:], [0.0, 1.0, 0.0, 0.0])  # one-hot of class 1


def test_adapter_flatdim_mismatch_with_state_size_is_loud():
    # The runner refuses a bundle whose declared space and state_size disagree.
    model = AutoregressiveModel(
        _CFG,
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                                low=[-1.0, -1.0], high=[1.0, 1.0], squash="tanh")],
        device=torch.device("cpu"),
    )
    from causal_gpt_rl.inference.runner import PolicyRunner

    try:
        PolicyRunner(
            model=model,
            action_schedule=[("continuous", 2, [-1.0, -1.0], [1.0, 1.0])],
            state_size=2,
            context_length=8,
            obs_space=_dict_box_discrete(),  # flatdim 7 != state_size 2
        )
    except ValueError as exc:
        assert "flatdim" in str(exc)
        return
    raise AssertionError("expected ValueError on flatdim/state_size mismatch")


# --------------------------------------------------------------------------- #
# End-to-end — structured bundle load + run (synthetic stand-in for the
# trainer fixture; swap in the real fixture once it lands).
# --------------------------------------------------------------------------- #

def _structured_model(obs_space: gym.spaces.Space) -> AutoregressiveModel:
    # The trainer's path: typed state specs come straight from the declared space.
    state_specs = extract_data_specs_from_space(obs_space)
    action_specs = [
        SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                  low=[-1.0, -1.0], high=[1.0, 1.0], squash="tanh")
    ]
    return AutoregressiveModel(
        _CFG, state_specs=state_specs, action_specs=action_specs,
        device=torch.device("cpu"),
    )


def test_end_to_end_dict_box_discrete_bundle():
    obs_space = _dict_box_discrete()
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    model = _structured_model(obs_space)

    with tempfile.TemporaryDirectory() as tmp:
        bundle.export_bundle(
            tmp,
            model=model,
            model_config=_CFG,
            state_specs=model.state_specs,
            action_specs=model.action_specs,
            context_length=8,
            obs_space=obs_space,
            action_space=action_space,
            state_normalizer=_Norm(7),
        )
        runner = bundle.load_runner(tmp)

    assert runner._input_adapter is not None
    assert isinstance(runner.obs_space, gym.spaces.Dict)

    # A structured observation flows reset -> act without a flat-shape error.
    obs = {"cont": np.array([0.1, -0.2, 0.3], dtype=np.float32), "disc": 2}
    runner.reset(obs)
    action = runner.act()
    assert np.asarray(action).shape == (2,)

    # A second structured step keeps working (autoregressive feedback path).
    obs2 = {"cont": np.array([0.0, 0.0, 0.0], dtype=np.float32), "disc": 0}
    action2 = runner.act(obs2)
    assert np.asarray(action2).shape == (2,)
