"""Tests for the `bos_cache_mode` serving convention.

`bos_cache_mode` decides whether the episode-start bos token's KV survives in
the persisted cache ("retain") or is dropped after the first act ("discard").
It is a runtime serving convention — no weights / architecture / I/O schema
change — resolved as: explicit arg > bundle `serving.bos_cache_mode` > "discard".

The discriminating signal is the persisted KV-cache length right after each
act():
  * discard: [0, 1, 2, 3, ...]  (bos KV dropped after act#0, then bos=0 grows)
  * retain:  [1, 2, 3, 4, ...]  (bos KV kept at act#0 → always +1)
See .local/docs/dev/model-output/bos-kv-cache-discard.md for the mechanism.
"""
import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_CFG = ModelConfig(d_model=32, num_heads=4)
_ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


class _Norm:
    """Minimal duck-typed state normalizer (mean/var state_dict)."""

    def __init__(self, n: int):
        self.n = n

    def state_dict(self):
        return {"mean": torch.zeros(self.n), "var": torch.ones(self.n)}


def _model() -> AutoregressiveModel:
    # Action specs via extract_data_specs_from_space so the bounded-continuous
    # head carries the squash the output adapter requires (matches the trainer
    # path); manual state specs stay on the input side.
    return AutoregressiveModel(
        _CFG,
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=extract_data_specs_from_space(_ACTION_SPACE),
        device=torch.device("cpu"),
    )


def _runner(bos_cache_mode=None, *, context_length: int = 8) -> PolicyRunner:
    return PolicyRunner(
        model=_model(),
        action_schedule=[("continuous", 2, None, None)],
        state_size=2,
        context_length=context_length,
        bos_cache_mode=bos_cache_mode,
    )


def _cache_lengths(runner: PolicyRunner, steps: int = 4) -> list[int]:
    """Persisted KV-cache length observed right after each act()."""
    runner.reset(np.zeros(2, dtype=np.float32))
    lengths = []
    for _ in range(steps):
        runner.act()
        lengths.append(runner.buffer.get_kv_cache_length())
        runner.observe(np.zeros(2, dtype=np.float32))
    return lengths


# --------------------------------------------------------------------------- #
# Runner-level behavior (no bundle).
# --------------------------------------------------------------------------- #

def test_default_is_discard():
    assert _runner().bos_cache_mode == "discard"


def test_discard_drops_bos_kv_after_first_act():
    # act#0 builds a 1-token (bos=1) cache, then discards it -> length 0.
    assert _cache_lengths(_runner("discard")) == [0, 1, 2, 3]


def test_retain_keeps_bos_kv_after_first_act():
    # act#0's bos=1 token stays; every subsequent bos=0 token appends -> +1.
    assert _cache_lengths(_runner("retain")) == [1, 2, 3, 4]


def test_none_matches_discard_byte_for_byte():
    assert _cache_lengths(_runner(None)) == _cache_lengths(_runner("discard"))


def test_retain_is_discard_plus_one_every_step():
    discard = _cache_lengths(_runner("discard"), steps=6)
    retain = _cache_lengths(_runner("retain"), steps=6)
    assert retain == [d + 1 for d in discard]


@pytest.mark.parametrize("bad", ["keep", "Retain", "", "0", "drop"])
def test_invalid_mode_raises(bad):
    with pytest.raises(ValueError, match="bos_cache_mode"):
        _runner(bad)


# --------------------------------------------------------------------------- #
# Bundle read path (serving.bos_cache_mode) + explicit override.
# --------------------------------------------------------------------------- #

def _export(tmp_path) -> None:
    model = _model()
    bundle.export_bundle(
        str(tmp_path), model=model, model_config=_CFG,
        state_specs=model.state_specs, action_specs=model.action_specs,
        context_length=8, action_space=_ACTION_SPACE, state_normalizer=_Norm(2),
    )


def _set_serving(tmp_path, serving: dict) -> None:
    cfg_path = tmp_path / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["serving"] = serving
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")


def test_absent_serving_defaults_to_discard(tmp_path):
    _export(tmp_path)  # exporter writes no `serving` block
    runner = bundle.load_runner(str(tmp_path))
    assert runner.bos_cache_mode == "discard"


def test_load_runner_reads_serving_retain(tmp_path):
    _export(tmp_path)
    _set_serving(tmp_path, {"bos_cache_mode": "retain"})
    runner = bundle.load_runner(str(tmp_path))
    assert runner.bos_cache_mode == "retain"


def test_explicit_arg_overrides_bundle(tmp_path):
    _export(tmp_path)
    _set_serving(tmp_path, {"bos_cache_mode": "retain"})
    # Caller-supplied mode wins over the bundle declaration.
    runner = bundle.load_runner(str(tmp_path), bos_cache_mode="discard")
    assert runner.bos_cache_mode == "discard"


# --------------------------------------------------------------------------- #
# Build-time selection (export_bundle bakes serving.bos_cache_mode).
# --------------------------------------------------------------------------- #

def test_export_omits_serving_by_default(tmp_path):
    _export(tmp_path)  # bos_cache_mode not passed
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "serving" not in cfg  # byte-identical to legacy bundles
    assert bundle.load_runner(str(tmp_path)).bos_cache_mode == "discard"


@pytest.mark.parametrize("mode", ["discard", "retain"])
def test_export_bakes_chosen_mode(tmp_path, mode):
    model = _model()
    bundle.export_bundle(
        str(tmp_path), model=model, model_config=_CFG,
        state_specs=model.state_specs, action_specs=model.action_specs,
        context_length=8, action_space=_ACTION_SPACE, state_normalizer=_Norm(2),
        bos_cache_mode=mode,
    )
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["serving"]["bos_cache_mode"] == mode
    assert bundle.load_runner(str(tmp_path)).bos_cache_mode == mode


def test_export_rejects_invalid_mode(tmp_path):
    model = _model()
    with pytest.raises(ValueError, match="bos_cache_mode"):
        bundle.export_bundle(
            str(tmp_path), model=model, model_config=_CFG,
            state_specs=model.state_specs, action_specs=model.action_specs,
            context_length=8, action_space=_ACTION_SPACE,
            state_normalizer=_Norm(2), bos_cache_mode="keep",
        )
