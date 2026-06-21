"""MultiBinary action/obs support — serving side (spaces round-trip + decode).

Self-contained (no `.local` fixture): builds tiny bundles in a temp dir, loads
them, and checks `runner._decode(model_action)` against an oracle computed
INDEPENDENTLY of the runner. MultiBinary = independent Bernoulli per element:
the model emits raw logits, the env decode thresholds them at 0 (== prob 0.5),
and `gym.spaces.flatten(MultiBinary)` is the {0,1} n-vector itself (no one-hot).
"""
import numpy as np
import gymnasium as gym
import pytest
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.spaces import (
    deserialize_space,
    extract_data_specs_from_space,
    serialize_space,
)
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_CFG = ModelConfig(d_model=32, num_heads=4, context_length=8)


class _Norm:
    """Minimal duck-typed state normalizer (mean/var state_dict)."""

    def __init__(self, n: int):
        self.n = n

    def state_dict(self):
        return {"mean": torch.zeros(self.n), "var": torch.ones(self.n)}


def _state_specs(n: int):
    return [
        SpaceSpec(type="continuous", size=n, dtype=torch.float32,
                  low=[-1.0] * n, high=[1.0] * n)
    ]


def _build_and_load(tmp_path, action_space, *, state_size: int = 2):
    action_specs = extract_data_specs_from_space(action_space)
    state_specs = _state_specs(state_size)
    model = AutoregressiveModel(
        _CFG, state_specs=state_specs, action_specs=action_specs,
        device=torch.device("cpu"),
    )
    bundle.export_bundle(
        tmp_path, model=model, model_config=_CFG,
        state_specs=model.state_specs, action_specs=model.action_specs,
        context_length=8, action_space=action_space, state_normalizer=_Norm(state_size),
    )
    return bundle.load_runner(tmp_path)


# --- spaces round-trip -----------------------------------------------------

@pytest.mark.parametrize("space", [
    gym.spaces.MultiBinary(5),
    gym.spaces.Tuple((gym.spaces.Box(-1.0, 1.0, (2,), np.float32), gym.spaces.MultiBinary(3))),
    gym.spaces.Dict({"flags": gym.spaces.MultiBinary(4), "move": gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}),
])
def test_serialize_round_trip_flatten_invariant(space):
    restored = deserialize_space(serialize_space(space))
    x = space.sample()
    a = np.asarray(gym.spaces.flatten(space, x), dtype=np.float32)
    b = np.asarray(gym.spaces.flatten(restored, x), dtype=np.float32)
    np.testing.assert_array_equal(a, b)


def test_extract_specs_multibinary():
    specs = extract_data_specs_from_space(gym.spaces.MultiBinary(4))
    assert len(specs) == 1
    assert specs[0].type == "multi_binary"
    assert specs[0].size == 4

    specs = extract_data_specs_from_space(
        gym.spaces.Tuple((gym.spaces.Box(-1.0, 1.0, (2,), np.float32), gym.spaces.MultiBinary(3)))
    )
    assert [s.type for s in specs] == ["continuous", "multi_binary"]
    assert [s.size for s in specs] == [2, 3]


def test_multibinary_is_supported_not_out_of_scope():
    # Used to raise a self-describing "out of scope" error; now serializes.
    assert serialize_space(gym.spaces.MultiBinary(2)) == {"type": "MultiBinary", "n": 2}


# --- bare MultiBinary decode ----------------------------------------------

def test_bare_multibinary_decode(tmp_path):
    runner = _build_and_load(tmp_path, gym.spaces.MultiBinary(4))
    assert runner._output_adapter is None  # bare = no container adapter
    logits = np.array([[0.5, -0.3, 2.0, -1.0]], dtype=np.float32)
    env_action, buffer_action = runner._decode(logits)
    np.testing.assert_array_equal(env_action, np.array([1, 0, 1, 0], dtype=np.int8))
    # AR feedback is the {0,1} float vector, size n (no one-hot blowup).
    np.testing.assert_array_equal(buffer_action[0], np.array([1, 0, 1, 0], dtype=np.float32))


def test_bare_multibinary_end_to_end_sample(tmp_path):
    runner = _build_and_load(tmp_path, gym.spaces.MultiBinary(4))
    runner.reset(np.zeros(runner.state_size, dtype=np.float32))
    action = runner.act()
    assert gym.spaces.MultiBinary(4).contains(np.asarray(action, dtype=np.int8))


# --- container with a MultiBinary leaf ------------------------------------

def _independent_oracle(action_space, model_action):
    """expected container, hand-computed (NOT via the runner)."""
    box, mb = action_space.spaces
    cont = np.clip(model_action[:2], box.low, box.high).astype(np.float32)
    flags = (model_action[2:2 + int(mb.n)] > 0.0).astype(np.int8)
    return (cont, flags)


def _build_model(action_space, state_size: int = 2):
    return AutoregressiveModel(
        _CFG, state_specs=_state_specs(state_size),
        action_specs=extract_data_specs_from_space(action_space),
        device=torch.device("cpu"),
    )


def test_multibinary_sampling_is_independent_bernoulli():
    model = _build_model(gym.spaces.MultiBinary(3))
    assert not model._is_discrete  # independent Bernoulli, not a single n-way choice
    # out = [mean head (3 logits), value head (1)]; std_scale=0 => threshold at 0.
    out = [torch.tensor([[2.0, -1.0, 3.0]]), torch.zeros(1, 1)]
    sampled = model.sample_action_from_heads(out, std_scale=0.0)
    # Two bits on. A categorical mis-decode would emit a single one-hot ([0,0,1]).
    torch.testing.assert_close(sampled, torch.tensor([[1.0, 0.0, 1.0]]))


def test_multibinary_sampling_stochastic_values_are_binary():
    model = _build_model(gym.spaces.MultiBinary(5))
    out = [torch.zeros(1, 5), torch.zeros(1, 1)]  # logits 0 => p=0.5
    sampled = model.sample_action_from_heads(out, std_scale=1.0)
    assert sampled.shape == (1, 5)
    assert set(np.unique(sampled.numpy()).tolist()).issubset({0.0, 1.0})


def test_container_multibinary_leaf_decode(tmp_path):
    action_space = gym.spaces.Tuple(
        (gym.spaces.Box(-1.0, 1.0, (2,), np.float32), gym.spaces.MultiBinary(3))
    )
    runner = _build_and_load(tmp_path, action_space)
    assert runner._output_adapter is not None
    for logits in ([0.3, -0.7, 0.5, -0.1, 2.0], [1.5, -2.0, -0.5, 0.2, -3.0]):
        arr = np.asarray(logits, dtype=np.float32).reshape(1, -1)
        container, _ = runner._decode(arr)
        got = np.asarray(gym.spaces.flatten(action_space, container), dtype=np.float32)
        exp = np.asarray(
            gym.spaces.flatten(action_space, _independent_oracle(action_space, arr[0])),
            dtype=np.float32,
        )
        np.testing.assert_allclose(got, exp, atol=1e-6)
