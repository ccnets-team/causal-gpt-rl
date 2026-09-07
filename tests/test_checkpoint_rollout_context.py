"""Checkpoint loading must preserve the declared rollout action coordinate."""
import numpy as np
import pytest
import torch

from causal_gpt_rl.inference.bundle import load_runner
from causal_gpt_rl.inference.checkpoint import load_inference_checkpoint
from causal_gpt_rl.inference.runner import PolicyRunner
from test_action_normalization import components
from test_pre_tanh_rollout_context import DIRECT, RAW, make_bundle


def runner_from_checkpoint(path, *, windowed=False):
    # Validation must use the checkpoint's loaded normalization state.
    model, _, _, specs, _ = components(enabled=False)
    return PolicyRunner.from_checkpoint(
        path, model=model, action_specs=specs, state_size=3,
        context_length=4, use_windowed=windowed,
    )


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("saturated", [False, True])
def test_direct_checkpoint_matches_bundle_feedback(tmp_path, windowed, saturated):
    model = make_bundle(tmp_path / "bundle", saturated=saturated)
    metadata = {"action_coordinate": DIRECT}
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "rollout_context": metadata}, checkpoint)
    assert load_inference_checkpoint(checkpoint)["rollout_context"] == metadata
    runner = runner_from_checkpoint(checkpoint, windowed=windowed)
    reference = load_runner(tmp_path / "bundle", use_windowed=windowed)
    assert runner.rollout_action_context_coordinate == DIRECT
    rng = np.random.default_rng(87)
    for r in (runner, reference):
        r.reset(np.zeros(3, np.float32))
    for _ in range(8):
        np.testing.assert_array_equal(runner.act(), reference.act())
        np.testing.assert_array_equal(runner._last_buffer_action, reference._last_buffer_action)
        if saturated:
            np.testing.assert_array_equal(runner._last_buffer_action, [[40., -20.]])
        observation = rng.normal(size=3).astype(np.float32)
        for r in (runner, reference):
            r.observe(observation)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_checkpoint_environment_coordinate_remains_compatible(tmp_path, enabled, explicit):
    model, _, _, specs, _ = components(enabled=enabled)
    payload = {"model_state": model.state_dict()}
    if explicit:
        payload["rollout_context"] = {"action_coordinate": RAW}
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    runner = runner_from_checkpoint(checkpoint)
    reference = PolicyRunner(
        model=model, action_schedule=PolicyRunner._resolve_action_specs(specs),
        state_size=3, context_length=4,
    )
    assert runner.rollout_action_context_coordinate == RAW
    for r in (runner, reference):
        r.reset(np.zeros(3, np.float32))
    for step in range(4):
        np.testing.assert_array_equal(runner.act(), reference.act())
        np.testing.assert_array_equal(runner._last_buffer_action, reference._last_buffer_action)
        for r in (runner, reference):
            r.observe(np.full(3, step + 1, np.float32))


@pytest.mark.parametrize("metadata", [
    None, {}, [], DIRECT, {"action_coordinate": None}, {"action_coordinate": "unknown_v1"},
])
def test_checkpoint_rejects_invalid_rollout_metadata(tmp_path, metadata):
    model, *_ = components()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "rollout_context": metadata}, checkpoint)
    with pytest.raises(ValueError, match="[Rr]ollout"):
        runner_from_checkpoint(checkpoint)


@pytest.mark.parametrize("missing_state", [False, True])
def test_direct_checkpoint_requires_loaded_action_normalization(tmp_path, missing_state):
    model, *_ = components(enabled=False)
    weights = model.state_dict()
    if missing_state:
        weights = {k: v for k, v in weights.items() if not k.startswith("action_normalization_")}
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": weights, "rollout_context": {"action_coordinate": DIRECT}}, checkpoint)
    with pytest.raises(ValueError, match="requires enabled action normalization"):
        runner_from_checkpoint(checkpoint)
