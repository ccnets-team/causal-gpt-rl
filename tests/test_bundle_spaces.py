"""P2 bundle tests — serialized Gymnasium spaces + hybrid_state capability gate.

Covers the serving side of the L2 contract: export serializes the declared
spaces into the config, stamps `hybrid_state` for discrete state, and load
either attaches the deserialized spaces (continuous) or refuses loudly
(hybrid_state, until the input adapter ships).
"""
import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.spaces import deserialize_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_CFG = ModelConfig(d_model=32, num_heads=4)


class _Norm:
    """Minimal duck-typed state normalizer (mean/std state_dict)."""

    def __init__(self, n: int):
        self.n = n

    def state_dict(self):
        # `var` (not `std`) — this is what StateNormalizer.from_state_dict reads
        # when the sidecar is loaded back on load_runner.
        return {"mean": torch.zeros(self.n), "var": torch.ones(self.n)}


def _continuous_model(state_size: int = 2, action_size: int = 2) -> AutoregressiveModel:
    return AutoregressiveModel(
        ModelConfig(d_model=32, num_heads=4),
        state_specs=[
            SpaceSpec(
                type="continuous",
                size=state_size,
                dtype=torch.float32,
                low=[-1.0] * state_size,
                high=[1.0] * state_size,
            )
        ],
        action_specs=[
            SpaceSpec(
                type="continuous",
                size=action_size,
                dtype=torch.float32,
                low=[-1.0] * action_size,
                high=[1.0] * action_size,
                squash="tanh",
            )
        ],
        device=torch.device("cpu"),
    )


def _discrete_state_model(n: int = 4, action_size: int = 2) -> AutoregressiveModel:
    return AutoregressiveModel(
        ModelConfig(d_model=32, num_heads=4),
        state_specs=[
            SpaceSpec(
                type="discrete",
                size=n,
                dtype=np.int64,
                low=np.full((n,), -np.inf),
                high=np.full((n,), np.inf),
            )
        ],
        action_specs=[
            SpaceSpec(
                type="continuous",
                size=action_size,
                dtype=torch.float32,
                low=[-1.0] * action_size,
                high=[1.0] * action_size,
                squash="tanh",
            )
        ],
        device=torch.device("cpu"),
    )


def _read_config(tmpdir: str) -> dict:
    return json.loads((Path(tmpdir) / "config.json").read_text())


def test_export_serializes_spaces_into_config():
    model = _continuous_model(2, 2)
    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
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
            state_normalizer=_Norm(2),
        )
        cfg = _read_config(tmp)

    assert cfg["state_container"] is not None
    assert cfg["action_container"] is not None
    # Round-trips to a flatten-equivalent space.
    obs_space.seed(0)
    x = obs_space.sample()
    restored = deserialize_space(cfg["state_container"])
    assert np.allclose(gym.spaces.flatten(obs_space, x), gym.spaces.flatten(restored, x))
    # Pure-continuous state is NOT gated.
    assert "hybrid_state" not in cfg["requires_capabilities"]
    # JSON is standard (no Infinity tokens).
    json.dumps(cfg, allow_nan=False)


def test_export_omits_spaces_when_absent():
    model = _continuous_model(2, 2)
    with tempfile.TemporaryDirectory() as tmp:
        bundle.export_bundle(
            tmp,
            model=model,
            model_config=_CFG,
            state_specs=model.state_specs,
            action_specs=model.action_specs,
            context_length=8,
            state_normalizer=_Norm(2),
        )
        cfg = _read_config(tmp)

    assert cfg["state_container"] is None
    assert cfg["action_container"] is None


def test_export_stamps_hybrid_state_for_discrete_state():
    model = _discrete_state_model(4, 2)
    with tempfile.TemporaryDirectory() as tmp:
        bundle.export_bundle(
            tmp,
            model=model,
            model_config=_CFG,
            state_specs=model.state_specs,
            action_specs=model.action_specs,
            context_length=8,
            obs_space=gym.spaces.Discrete(4),
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            state_normalizer=_Norm(4),
        )
        cfg = _read_config(tmp)

    assert "hybrid_state" in cfg["requires_capabilities"]


def test_load_refuses_hybrid_state_until_adapter_ships():
    # Forced capability stamp on a continuous model -> load must reject loudly
    # because the runtime does not yet advertise hybrid_state.
    model = _continuous_model(2, 2)
    with tempfile.TemporaryDirectory() as tmp:
        bundle.export_bundle(
            tmp,
            model=model,
            model_config=_CFG,
            state_specs=model.state_specs,
            action_specs=model.action_specs,
            context_length=8,
            state_normalizer=_Norm(2),
            requires_capabilities=["hybrid_state"],
        )
        try:
            bundle.load_runner(tmp)
        except ValueError as exc:
            assert "hybrid_state" in str(exc)
            return
    raise AssertionError("expected ValueError refusing the hybrid_state bundle")


def test_load_attaches_deserialized_spaces():
    model = _continuous_model(2, 2)
    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
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
            state_normalizer=_Norm(2),
        )
        runner = bundle.load_runner(tmp)

    assert runner.obs_space is not None
    assert runner.action_space is not None
    obs_space.seed(1)
    x = obs_space.sample()
    assert np.allclose(
        gym.spaces.flatten(obs_space, x),
        gym.spaces.flatten(runner.obs_space, x),
    )


def test_load_old_bundle_has_none_spaces():
    model = _continuous_model(2, 2)
    with tempfile.TemporaryDirectory() as tmp:
        bundle.export_bundle(
            tmp,
            model=model,
            model_config=_CFG,
            state_specs=model.state_specs,
            action_specs=model.action_specs,
            context_length=8,
            state_normalizer=_Norm(2),
        )
        runner = bundle.load_runner(tmp)

    assert runner.obs_space is None
    assert runner.action_space is None
