"""Direct-z feedback preserves a selected action across the environment boundary."""
import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from causal_gpt_rl.export import export_onnx
from causal_gpt_rl.export.onnx import _WindowedPolicy
from causal_gpt_rl.inference.bundle import export_bundle, load_runner
from causal_gpt_rl.model.utils.action_normalization import (
    ENVIRONMENT_ACTION_COORDINATE as RAW,
    PRE_TANH_ACTION_COORDINATE as DIRECT,
    PRE_TANH_ROLLOUT_CAPABILITY as CAPABILITY,
)
from test_action_normalization import components


def make_bundle(path, *, direct=True, hybrid=False, saturated=False, enabled=True):
    model, config, states, specs, space = components(hybrid=hybrid, enabled=enabled)
    if saturated:
        with torch.no_grad():
            for i in model.mean_action_indices:
                if model.flat_output_specs[i].type == "continuous":
                    layer = model.output_head[i].layers[-1]
                    layer.weight.zero_()
                    layer.bias.copy_(torch.tensor([40., -20.]))
    export_bundle(
        path, model=model, model_config=config, state_specs=states,
        action_specs=specs, action_space=space, context_length=4,
        rollout_action_context_coordinate=DIRECT if direct else RAW,
    )
    return model


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("hybrid", [False, True])
def test_saturation_and_returned_action_ownership(tmp_path, windowed, hybrid):
    model = make_bundle(tmp_path, hybrid=hybrid, saturated=True)
    runner = load_runner(tmp_path, use_windowed=windowed)
    runner.reset(np.zeros(3, np.float32))
    action = runner.act()
    saved = runner._last_buffer_action.copy()
    idx = model._normalized_action_indices.numpy()
    np.testing.assert_array_equal(saved[:, idx], [[40., -20.]])
    flat = gym.spaces.flatten(runner.action_space, action)
    np.testing.assert_array_equal(flat[idx], [1., 2.])
    # Environment saturation cannot recover this selected z through atanh.
    encoded = runner.encode_action_context(flat[None])
    assert np.max(np.abs(encoded[:, idx] - saved[:, idx])) > 1
    if hybrid:
        action[1][:] = 99
    else:
        action[:] = 99
    np.testing.assert_array_equal(runner._last_buffer_action, saved)
    runner.observe(np.ones(3, np.float32))
    np.testing.assert_array_equal(runner.buffer.get_context()[1][:, -1], saved)
    runner.reset(np.zeros(3, np.float32))
    assert runner._last_buffer_action is None
    assert not runner.buffer.actions.any()
    assert runner.buffer.get_kv_cache_length() == 0


def test_external_prefix_and_generated_positions_use_one_coordinate(tmp_path):
    model = make_bundle(tmp_path, hybrid=True)
    runner = load_runner(tmp_path, use_windowed=True)
    raw = torch.zeros(1, 4, model.action_size)
    idx = model._normalized_action_indices
    raw[..., idx] = torch.tensor([[[-3., 2.], [1., 10.], [-2., 7.], [0., 5.]]])
    raw[..., 0] = 1  # existing categorical one-hot representation
    raw[:, 0, 0] = 0  # absent categorical BOS sentinel, as in a real runner
    bos = torch.zeros(1, 4, 1)
    bos[:, 0] = 1
    encoded = runner.encode_action_context(raw, is_bos=bos)
    assert np.isfinite(encoded).all()
    np.testing.assert_array_equal(encoded[:, 0], 0)
    np.testing.assert_array_equal(encoded[:, 1:, 0], 1)
    tokens = torch.cat((torch.randn(1, 4, 3), raw, bos), -1)
    direct_tokens = tokens.clone()
    direct_tokens[..., 3:-1] = torch.from_numpy(encoded)
    # For externally sourced actions, raw and explicitly encoded calls agree.
    torch.testing.assert_close(
        model.adapt_input(tokens),
        model.adapt_input(direct_tokens, action_context_coordinate=DIRECT),
        atol=0, rtol=0,
    )
    # Generated positions retain z beyond the recoverable environment range.
    direct_tokens[:, -1, 3 + idx] = torch.tensor([40., -20.])
    model.input_proj = torch.nn.Identity()
    adapted = model.adapt_input(direct_tokens, action_context_coordinate=DIRECT)
    torch.testing.assert_close(adapted[:, -1, 3 + idx], torch.tensor([[40., -20.]]))


def test_external_prefix_cache_continues_with_generated_z(tmp_path):
    from causal_gpt_rl.model.utils.kv_cache import build_kv_cache

    model = make_bundle(tmp_path).eval()
    states = torch.randn(1, 4, 3)
    raw = torch.tensor([[[-1., 6.], [0., 7.], [-2., 4.], [1., 10.]]])
    bos = torch.zeros(1, 4, 1)
    bos[:, 0] = 1
    actions = model.actions_to_rollout_context(raw, action_context_coordinate=DIRECT, is_bos=bos)
    cache = build_kv_cache(model.backbone, max_len=8)
    with torch.no_grad():
        _, cache = model.infer_cached(
            torch.cat((states[:, :3], actions[:, :3], bos[:, :3]), -1),
            past_key_values=cache, padding_mask=torch.ones(1, 3, dtype=torch.bool),
            action_context_coordinate=DIRECT,
        )
        for _ in range(2):
            expected, info = model._predict_with_window(
                states, actions, bos, torch.ones(states.shape[:2], dtype=torch.bool),
                return_info=True, action_context_coordinate=DIRECT,
            )
            actual, cache, cached_info = model._predict_incremental_cached(
                states[:, -1:], actions[:, -1:], bos[:, -1:],
                past_key_values=cache, return_info=True, action_context_coordinate=DIRECT,
            )
            torch.testing.assert_close(torch.cat(actual, -1), torch.cat(expected, -1)[:, -1:], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(cached_info["_context_action"], info["_context_action"][:, -1:], atol=1e-6, rtol=1e-5)
            states = torch.cat((states, torch.randn(1, 1, 3)), 1)
            actions = torch.cat((actions, cached_info["_context_action"]), 1)
            bos = torch.cat((bos, torch.zeros(1, 1, 1)), 1)


@pytest.mark.parametrize("windowed", [False, True])
def test_added_rows_preserve_surviving_direct_context(tmp_path, windowed):
    make_bundle(tmp_path)
    runner = load_runner(tmp_path, use_windowed=windowed, bos_cache_mode="retain")
    reference = load_runner(tmp_path, use_windowed=windowed, bos_cache_mode="retain")
    for r in (runner, reference):
        r.reset(np.zeros(3, np.float32))
        for step in range(3):
            r.act()
            r.observe(np.full(3, step + 1, np.float32))
    previous = runner.buffer.actions.copy()
    pending = runner._last_buffer_action.copy()
    lengths = runner.buffer.get_kv_valid_lengths().copy()
    runner.add_rows(np.full((1, 3), 7, np.float32))
    np.testing.assert_array_equal(runner.buffer.actions[:1], previous)
    np.testing.assert_array_equal(runner._last_buffer_action[:1], pending)
    np.testing.assert_array_equal(runner.buffer.actions[1], 0)
    np.testing.assert_array_equal(runner.buffer.get_kv_valid_lengths()[:1], lengths)
    for step in range(2):
        np.testing.assert_allclose(runner.act()[0], reference.act(), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(runner._last_buffer_action[:1], reference._last_buffer_action, atol=1e-6, rtol=1e-5)
        runner.observe(np.full((2, 3), step + 4, np.float32))
        reference.observe(np.full(3, step + 4, np.float32))


def test_direct_wrapper_keeps_state_normalization_and_bos(tmp_path):
    make_bundle(tmp_path)
    runner = load_runner(tmp_path, use_windowed=True)
    runner.model.set_state_normalization(mean=[1., 2., 3.], std=[2., 3., 4.])
    wrapper = _WindowedPolicy(runner).eval()
    captured = []
    handle = runner.model.input_proj.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    states = torch.ones(1, 4, 3) * 5
    actions = torch.ones(1, 4, 2) * 40
    bos = torch.zeros(1, 4, 1)
    bos[:, 0] = 1
    wrapper(states, actions, bos, torch.ones(1, 4))
    handle.remove()
    torch.testing.assert_close(captured[0][..., :3], (states - torch.tensor([1., 2., 3.])) / torch.tensor([2., 3., 4.]))
    torch.testing.assert_close(captured[0][:, 1:, 3:5], actions[:, 1:])
    assert not captured[0][:, 0, 3:5].any()


@pytest.mark.parametrize("hybrid", [False, True])
def test_same_sample_produces_both_representations_once(tmp_path, hybrid, monkeypatch):
    model = make_bundle(tmp_path, hybrid=hybrid)
    heads = [torch.randn(2, 3, s.size) for s in model.flat_output_specs]
    randn = torch.randn_like
    gaussian_calls = []
    def draw(mean):
        value = randn(mean)
        gaussian_calls.append(value.clone())
        return value
    monkeypatch.setattr(torch, "randn_like", draw)
    torch.manual_seed(79)
    env_heads, feedback = model._action_and_context_from_heads(heads, std_scale=0.7)
    final_rng = torch.random.get_rng_state()
    assert len(gaussian_calls) == 1
    # Legacy sampling consumes exactly the same draws for this head layout.
    torch.manual_seed(79)
    sampled = model.sample_action_from_heads(heads, std_scale=0.7)
    assert torch.equal(final_rng, torch.random.get_rng_state())
    torch.testing.assert_close(torch.cat(env_heads, -1), sampled)
    idx = model._normalized_action_indices
    expected = torch.tensor([-3., 2.]) + (torch.tanh(feedback[..., idx] * torch.tensor([0.4, 1.7]) + torch.tensor([0.7, -1.2])) + 1) * torch.tensor([2., 4.])
    torch.testing.assert_close(torch.cat(env_heads, -1)[..., idx], expected)
    before = torch.random.get_rng_state()
    model._action_and_context_from_heads(heads, std_scale=0)
    assert torch.equal(before, torch.random.get_rng_state())


def test_coordinate_is_fixed_and_requires_normalization(tmp_path):
    make_bundle(tmp_path)
    runner = load_runner(tmp_path)
    with pytest.raises(AttributeError):
        runner.rollout_action_context_coordinate = RAW
    runner.reset(np.zeros(3, np.float32))
    runner.act()
    with pytest.raises(AttributeError):
        runner.rollout_action_context_coordinate = RAW
    with pytest.raises(ValueError, match="requires enabled"):
        make_bundle(tmp_path / "disabled", enabled=False)
    assert not (tmp_path / "disabled").exists()
    legacy, *_ = components(enabled=False)
    with pytest.raises(ValueError, match="requires enabled"):
        legacy.adapt_input(torch.zeros(1, 1, legacy.total_input_dim), action_context_coordinate=DIRECT)


@pytest.mark.parametrize("damage", ["missing", "unknown", "raw", "no_capability", "null", "no_coordinate", "disabled"])
def test_inconsistent_metadata_fails_fast(tmp_path, damage):
    make_bundle(tmp_path)
    config_file = tmp_path / "config.json"
    config = json.loads(config_file.read_text())
    if damage == "missing":
        del config["rollout_context"]
    elif damage == "unknown":
        config["rollout_context"]["action_coordinate"] = "pre_tanh_v0"
    elif damage == "raw":
        config["rollout_context"]["action_coordinate"] = RAW
    elif damage == "no_capability":
        config["requires_capabilities"].remove(CAPABILITY)
    elif damage == "null":
        config["rollout_context"] = None
    elif damage == "no_coordinate":
        config["rollout_context"] = {}
    else:
        from safetensors.torch import load_file, save_file
        path = tmp_path / "model.safetensors"
        weights = {k: v.clone() for k, v in load_file(str(path)).items()}
        weights["action_normalization_enabled"].zero_()
        save_file(weights, str(path))
        config["requires_capabilities"].remove("action_normalization")
        del config["action_normalization"]
    config_file.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_runner(tmp_path)


def test_old_runtime_gate_and_legacy_defaults(tmp_path, monkeypatch):
    from causal_gpt_rl.inference import bundle as module
    make_bundle(tmp_path / "direct")
    make_bundle(tmp_path / "raw", direct=False)
    runner = load_runner(tmp_path / "raw")
    assert runner.rollout_action_context_coordinate == RAW
    config = json.loads((tmp_path / "raw/config.json").read_text())
    assert "rollout_context" not in config and CAPABILITY not in config["requires_capabilities"]
    monkeypatch.setattr(module, "_SUPPORTED_CAPABILITIES", module._SUPPORTED_CAPABILITIES - {CAPABILITY})
    with pytest.raises(ValueError, match=CAPABILITY):
        load_runner(tmp_path / "direct")


@pytest.mark.parametrize("bos_mode", ["retain", "discard"])
def test_cached_windowed_prefix_and_partial_reset_parity(tmp_path, bos_mode):
    make_bundle(tmp_path)
    cached = load_runner(tmp_path, num_envs=2, bos_cache_mode=bos_mode)
    windowed = load_runner(tmp_path, num_envs=2, use_windowed=True, bos_cache_mode=bos_mode)
    rng = np.random.default_rng(14)
    obs = rng.normal(size=(2, 3)).astype(np.float32)
    for r in (cached, windowed):
        r.reset(obs)
    for step in range(4):
        np.testing.assert_allclose(cached.act(), windowed.act(), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(cached._last_buffer_action, windowed._last_buffer_action, atol=1e-6, rtol=1e-5)
        if bos_mode == "discard":
            # Existing bos_cache_mode governs KV only; align the reference
            # window's visible tokens explicitly, as the ONNX caller does.
            windowed.buffer.masks[windowed.buffer.is_bos[..., 0] != 0] = 0
        if step == 1:
            for r in (cached, windowed):
                r.reset_rows([False, True])
        obs = rng.normal(size=(2, 3)).astype(np.float32)
        for r in (cached, windowed):
            r.observe(obs)


@pytest.mark.parametrize("hybrid", [False, True])
def test_direct_onnx_contract_and_rollout_parity(tmp_path, hybrid):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from examples.unity.evaluate_onnx import _run_onnx_with_context
    make_bundle(tmp_path / "bundle", hybrid=hybrid)
    result = export_onnx(tmp_path / "bundle", tmp_path / "direct.onnx", batch_size=2)
    assert result.max_abs_error < 1e-4
    graph = onnx.load(str(result.output_path))
    assert [o.name for o in graph.graph.output] == ["action", "context_action"]
    metadata = {p.key: p.value for p in graph.metadata_props}
    assert metadata["causal_gpt_rl.rollout_action_context_coordinate"] == DIRECT
    session = ort.InferenceSession(str(result.output_path), providers=["CPUExecutionProvider"])
    runner = load_runner(tmp_path / "bundle", num_envs=2, use_windowed=True)
    wrapper = _WindowedPolicy(runner).eval()
    rng = np.random.default_rng(61)
    runner.reset(rng.normal(size=(2, 3)).astype(np.float32))
    for step in range(6):
        states, actions, bos, mask, _ = runner.buffer.get_context()
        if step == 2:
            # Includes original z far beyond the environment round-trip range.
            actions[..., runner.model._normalized_action_indices.numpy()] = 40
        feeds = dict(zip(("states", "actions", "is_bos", "mask"), (states, actions, bos, mask)))
        actual = _run_onnx_with_context(session, feeds, batch=2)
        references = wrapper(*(torch.from_numpy(a) for a in feeds.values()))
        for a, ref in zip(actual, references):
            assert np.max(np.abs(a - ref.detach().numpy())) < 1e-4
        if step != 2:
            env = runner.act()
            expected_feedback = actual[1]
            np.testing.assert_allclose(runner._last_buffer_action, expected_feedback, atol=1e-4, rtol=0)
            decoded, _ = runner._decode(actual[0])
            for a, b in zip(env, decoded):
                np.testing.assert_allclose(gym.spaces.flatten(runner.action_space, a), gym.spaces.flatten(runner.action_space, b), atol=1e-4, rtol=0)
        else:
            runner.act()
        runner.observe(rng.normal(size=(2, 3)).astype(np.float32))
