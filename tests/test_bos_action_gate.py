"""BOS action gate — neutralize the absent previous-action channel at episode start.

Covers the opt-in `use_bos_action_gate` capability:
- default-off keeps the input path byte-identical (no params, no state_dict keys);
- on (gate-to-zero, emb frozen at 0) makes bos=1 ignore the action column while
  bos=0 still uses it, across single- and multi-head action spaces;
- the learned-prior escalation (`use_bos_action_prior`) unfreezes the emb;
- old checkpoints load into a gate-on model via strict=False (only gate keys
  missing);
- the flag persists through ModelConfig and a full bundle round-trip.

See .local/docs/dev/model-output/bos-gate-to-zero.md.
"""
import tempfile

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_DEV = torch.device("cpu")
_STATE = [SpaceSpec(type="continuous", size=3, dtype=torch.float32,
                    low=[-1.0, -1.0, -1.0], high=[1.0, 1.0, 1.0])]
_ACTION = [SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                     low=[-1.0, -1.0], high=[1.0, 1.0], squash="tanh")]


class _Norm:
    def __init__(self, n): self.n = n
    def state_dict(self): return {"mean": torch.zeros(self.n), "var": torch.ones(self.n)}


def _model(*, gate=False, prior=False, action_specs=None):
    cfg = ModelConfig(d_model=32, num_heads=4, use_bos_action_gate=gate)
    if prior:
        # trainer-only toggle; not a ModelConfig field on purpose
        cfg.use_bos_action_prior = True
    torch.manual_seed(0)
    return AutoregressiveModel(
        cfg, state_specs=_STATE, action_specs=action_specs or _ACTION, device=_DEV,
    )


# --------------------------------------------------------------------------- #
# default-off: byte-identical, no params
# --------------------------------------------------------------------------- #

def test_default_off_has_no_gate_state():
    m = _model(gate=False)
    assert m.use_bos_action_gate is False
    assert not hasattr(m, "bos_action_gate_emb")
    assert not any("bos_action_gate_emb" in k for k in m.state_dict())


# --------------------------------------------------------------------------- #
# gate-to-zero behavior (emb frozen at 0)
# --------------------------------------------------------------------------- #

def test_gate_emb_is_zeros_and_frozen_by_default():
    m = _model(gate=True)
    assert m._input_action_positions == [1]
    assert m._input_bos_index == 2
    for p in m.bos_action_gate_emb.parameters():
        assert p.requires_grad is False
        assert torch.count_nonzero(p) == 0


def test_bos1_ignores_action_and_bos0_uses_it():
    m = _model(gate=True).eval()
    B, T = 2, 4
    s = torch.randn(B, T, 3)
    a = torch.randn(B, T, 2)
    z = torch.zeros_like(a)
    bos1 = torch.ones(B, T, 1)
    bos0 = torch.zeros(B, T, 1)
    with torch.no_grad():
        # bos=1: action value is neutralized -> equal regardless of action input
        assert torch.allclose(
            m.adapt_input(torch.cat([s, a, bos1], -1)),
            m.adapt_input(torch.cat([s, z, bos1], -1)),
            atol=1e-6,
        )
        # bos=0: action value is used -> differs
        assert not torch.allclose(
            m.adapt_input(torch.cat([s, a, bos0], -1)),
            m.adapt_input(torch.cat([s, z, bos0], -1)),
            atol=1e-6,
        )


def test_multi_head_action_all_columns_gated():
    action = [
        SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                  low=[-1.0, -1.0], high=[1.0, 1.0], squash="tanh"),
        SpaceSpec(type="discrete", size=4, dtype=torch.float32),
    ]
    m = _model(gate=True, action_specs=action).eval()
    assert len(m._input_action_positions) == 2
    B, T = 2, 3
    s = torch.randn(B, T, 3)
    a = torch.randn(B, T, 6)  # 2 continuous + 4 one-hot width
    bos1 = torch.ones(B, T, 1)
    with torch.no_grad():
        assert torch.allclose(
            m.adapt_input(torch.cat([s, a, bos1], -1)),
            m.adapt_input(torch.cat([s, torch.zeros_like(a), bos1], -1)),
            atol=1e-6,
        )


# --------------------------------------------------------------------------- #
# learned-prior escalation
# --------------------------------------------------------------------------- #

def test_prior_mode_unfreezes_emb():
    m = _model(gate=True, prior=True)
    for p in m.bos_action_gate_emb.parameters():
        assert p.requires_grad is True


# --------------------------------------------------------------------------- #
# compatibility: old checkpoint -> gate-on model
# --------------------------------------------------------------------------- #

def test_old_checkpoint_loads_with_strict_false():
    old_sd = _model(gate=False).state_dict()
    m_on = _model(gate=True)
    missing, unexpected = m_on.load_state_dict(old_sd, strict=False)
    assert not unexpected
    assert missing and all(k.startswith("bos_action_gate_emb") for k in missing)


# --------------------------------------------------------------------------- #
# persistence: config + full bundle round-trip
# --------------------------------------------------------------------------- #

def test_config_round_trip_persists_gate_but_not_prior():
    d = ModelConfig(d_model=32, num_heads=4, use_bos_action_gate=True).to_dict()
    assert d["use_bos_action_gate"] is True
    assert ModelConfig.from_dict(d).use_bos_action_gate is True
    # trainer-only toggle must not leak into the persisted capability set
    assert "use_bos_action_prior" not in d


def test_bundle_round_trip_persists_flag_and_emb():
    cfg = ModelConfig(d_model=32, num_heads=4, use_bos_action_gate=True)
    torch.manual_seed(1)
    m = AutoregressiveModel(cfg, state_specs=_STATE, action_specs=_ACTION, device=_DEV)
    with torch.no_grad():  # simulate a trained (nonzero) prior emb
        m.bos_action_gate_emb["1"].copy_(torch.tensor([[[0.5, -0.25]]]))

    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        bundle.export_bundle(
            tmp, model=m, model_config=cfg,
            state_specs=m.state_specs, action_specs=m.action_specs,
            context_length=8, obs_space=obs_space, action_space=act_space,
            state_normalizer=_Norm(3),
        )
        runner = bundle.load_runner(tmp)

    lm = runner.model
    assert lm.use_bos_action_gate is True
    assert torch.allclose(lm.bos_action_gate_emb["1"], m.bos_action_gate_emb["1"])

    runner.reset(np.array([0.1, -0.2, 0.3], dtype=np.float32))
    assert np.asarray(runner.act()).shape == (2,)
