"""Serving contract: bounds -> atanh -> standardize, exactly once."""
import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from causal_gpt_rl.export import export_onnx
from causal_gpt_rl.export.onnx import _WindowedPolicy
from causal_gpt_rl.inference.bundle import export_bundle, load_runner, SUPPORTED_CAPABILITIES
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model import AutoregressiveModel, ModelConfig, SpaceSpec
from causal_gpt_rl.model.utils.action_normalization import ACTION_NORMALIZATION_COORDINATE
from causal_gpt_rl.model.utils.kv_cache import build_kv_cache


def components(*, hybrid=False, enabled=True, gate=False):
    torch.manual_seed(123)
    box = gym.spaces.Box(np.array([-3., 2.], np.float32), np.array([1., 10.], np.float32))
    space = gym.spaces.Tuple((gym.spaces.Discrete(3), box, gym.spaces.MultiBinary(2), gym.spaces.MultiDiscrete([2, 3]))) if hybrid else box
    specs = extract_data_specs_from_space(space)
    states = [SpaceSpec(type="continuous", size=3)]
    config = ModelConfig(context_length=4, d_model=16, num_layers=1, num_heads=2,
                         intermediate_size=32, max_position_embeddings=32, use_bos_action_gate=gate)
    model = AutoregressiveModel(config, state_specs=states, action_specs=specs, device="cpu").eval()
    model.set_state_normalization(mean=[0., 0., 0.], std=[1., 1., 1.])
    if enabled:
        mean, std = torch.zeros(model.action_size), torch.ones(model.action_size)
        mean[model._normalized_action_indices] = torch.tensor([0.7, -1.2])
        std[model._normalized_action_indices] = torch.tensor([0.4, 1.7])
        model.set_action_normalization_from_state_dict({"mean": mean, "var": std.square()})
    return model, config, states, specs, space


def bundle(path, **kwargs):
    model, config, states, specs, space = components(**kwargs)
    export_bundle(path, model=model, model_config=config, state_specs=states,
                  action_specs=specs, action_space=space, context_length=4)
    return model


@pytest.mark.parametrize("hybrid", [False, True])
def test_round_trip_bounds_and_input_columns(hybrid):
    model, _, _, _, _ = components(hybrid=hybrid)
    section = model._normalized_action_indices
    raw = torch.tensor([[[-3., 2.], [1., 10.], [-2., 7.]]])
    x = (2 * (raw.double() - torch.tensor([-3., 2.])) / torch.tensor([4., 8.]) - 1).clamp(-1 + 1e-6, 1 - 1e-6)
    expected = (torch.atanh(x) - torch.tensor([0.7, -1.2])) / torch.tensor([0.4, 1.7])
    z = model._normalize_action_head(raw, section)
    assert torch.isfinite(z).all()
    # Float32 clamp rounds the exact-bound 1e-6; compare the interior in float64.
    torch.testing.assert_close(z[:, 2].double(), expected[:, 2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model._denormalize_action_head(z, section), raw, atol=5e-6, rtol=0)
    model.input_proj = torch.nn.Identity()
    tokens = torch.randn(1, 3, model.total_input_dim)
    tokens[..., 3 + section] = raw
    tokens[..., -1] = 0
    actual = model.adapt_input(tokens)
    torch.testing.assert_close(actual[..., 3 + section], z)
    other = [i for i in range(model.total_input_dim) if i not in (3 + section).tolist()]
    torch.testing.assert_close(actual[..., other], tokens[..., other], rtol=0, atol=0)


@pytest.mark.parametrize("gate", [False, True])
def test_bos_remains_zero_in_model_coordinates(gate):
    model, *_ = components(gate=gate)
    model.input_proj = torch.nn.Identity()
    tokens = torch.randn(2, 3, model.total_input_dim)
    tokens[..., -1] = 1
    actual = model.adapt_input(tokens)
    assert torch.equal(actual[..., 3:5], torch.zeros(2, 3, 2))


@pytest.mark.parametrize("hybrid", [False, True])
def test_mean_and_seeded_sample_reference(hybrid):
    model, *_ = components(hybrid=hybrid)
    heads = [torch.randn(2, 3, spec.size) for spec in model.flat_output_specs]
    mean = torch.cat([heads[i] for i in model.mean_action_indices if model.flat_output_specs[i].type == "continuous"], -1)
    log_std = torch.cat([heads[i] for i in model.log_std_action_indices], -1).clamp(-20, 2)
    def reference(z):
        return torch.tensor([-3., 2.]) + (torch.tanh(z * torch.tensor([0.4, 1.7]) + torch.tensor([0.7, -1.2])) + 1) * torch.tensor([2., 4.])
    deterministic = torch.cat(model._extract_mean_action(heads), -1)
    torch.testing.assert_close(deterministic[..., model._normalized_action_indices], reference(mean))
    torch.manual_seed(77)
    expected = reference(mean + 0.6 * log_std.exp() * torch.randn_like(mean))
    torch.manual_seed(77)
    sampled = model.sample_action_from_heads(heads, std_scale=0.6)
    torch.testing.assert_close(sampled[..., model._normalized_action_indices], expected)


@pytest.mark.parametrize("change", [
    {"std": [0., 1.]}, {"std": [-1., 1.]}, {"std": [float("inf"), 1.]},
    {"mean": [float("nan"), 0.]}, {"mean": [0.]}, {"std": [1.]},
    {"var": [-1., 1.]}, {"var": [0., 1.]}, {"enabled": 0.5},
])
def test_invalid_statistics_are_rejected_atomically(change):
    model, *_ = components()
    before = model.action_normalization_mean.clone()
    args = dict(mean=[0., 0.], std=[1., 1.])
    if "var" in change:
        args.pop("std")
    args.update(change)
    with pytest.raises(ValueError):
        model.set_action_normalization(**args)
    assert torch.equal(model.action_normalization_mean, before)


def test_discrete_statistics_must_be_identity():
    model, *_ = components(hybrid=True)
    with pytest.raises(ValueError, match="identity"):
        model.set_action_normalization(mean=torch.ones(model.action_size), std=torch.ones(model.action_size))


@pytest.mark.parametrize("hybrid", [False, True])
def test_bundle_and_clipped_feedback(tmp_path, hybrid):
    original = bundle(tmp_path, hybrid=hybrid)
    config = json.loads((tmp_path / "config.json").read_text())
    assert "action_normalization" in SUPPORTED_CAPABILITIES
    assert "action_normalization" in config["requires_capabilities"]
    assert config["action_normalization"] == {"embedded": True, "coordinate": ACTION_NORMALIZATION_COORDINATE}
    runner = load_runner(tmp_path, use_windowed=True)
    assert runner.model.has_embedded_action_normalizer()
    assert torch.equal(original.action_normalization_mean, runner.model.action_normalization_mean)
    raw = np.ones((1, runner.action_size), np.float32)
    raw[:, original._normalized_action_indices.numpy()] = [-100., 100.]
    env, feedback = runner._decode(raw)
    env_flat = gym.spaces.flatten(runner.action_space, env)
    np.testing.assert_array_equal(feedback[0], env_flat)
    np.testing.assert_array_equal(feedback[0, original._normalized_action_indices.numpy()], [-3., 10.])
    runner.reset(np.zeros(3, np.float32))
    # Force an out-of-bounds model result to exercise final decode/storage.
    runner.model._extract_mean_action = lambda _: list(torch.split(
        torch.from_numpy(raw).unsqueeze(1), [s.size for s in original.action_specs], dim=-1,
    ))
    emitted = runner.act()
    np.testing.assert_array_equal(runner._last_buffer_action[0], gym.spaces.flatten(runner.action_space, emitted))
    runner.observe(np.ones(3, np.float32))
    _, history, _, _, _ = runner.buffer.get_context()
    np.testing.assert_array_equal(history[:, -1], feedback)


@pytest.mark.parametrize("damage", ["mean", "std", "all_stats", "zero_std", "nan_std", "flag", "coordinate", "old_coordinate", "capability", "metadata", "disabled"])
def test_corrupt_bundle_fails_fast(tmp_path, damage):
    bundle(tmp_path)
    config_path, weight_path = tmp_path / "config.json", tmp_path / "model.safetensors"
    config = json.loads(config_path.read_text())
    # Detach from safetensors' mapped file before overwriting it on Windows.
    weights = {k: v.clone() for k, v in load_file(str(weight_path)).items()}
    if damage in ("mean", "std"):
        del weights["action_normalization_" + damage]
    elif damage == "all_stats":
        weights = {k: v for k, v in weights.items() if not k.startswith("action_normalization_")}
    elif damage in ("zero_std", "nan_std"):
        weights["action_normalization_std"][0] = 0 if damage == "zero_std" else float("nan")
    elif damage == "flag":
        weights["action_normalization_enabled"][0] = float("nan")
    elif damage == "coordinate":
        del config["action_normalization"]["coordinate"]
    elif damage == "old_coordinate":
        config["action_normalization"]["coordinate"] = "bounds_then_standardize_v1"
    elif damage == "capability":
        config["requires_capabilities"] = []
    elif damage == "metadata":
        del config["action_normalization"]
    elif damage == "disabled":
        weights["action_normalization_enabled"].zero_()
    save_file(weights, str(weight_path))
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="[Aa]ction normalization|action_normalization"):
        load_runner(tmp_path)


def test_legacy_bundle_and_adapter_golden_parity(tmp_path):
    model = bundle(tmp_path, enabled=False)
    tokens = torch.randn(2, 4, model.total_input_dim)
    expected = model.input_proj(torch.cat([a(h) for a, h in zip(model.input_adapter, model._split_input_heads(tokens))], -1))
    torch.testing.assert_close(model.adapt_input(tokens), expected, rtol=0, atol=0)
    heads = model(tokens)
    expected_actions = torch.cat([model.output_adapter[i](heads[i]) for i in model.mean_action_indices], -1)
    assert torch.equal(torch.cat(model._extract_mean_action(heads), -1), expected_actions)
    weight_path = tmp_path / "model.safetensors"
    legacy = {k: v.clone() for k, v in load_file(str(weight_path)).items() if not k.startswith("action_normalization_")}
    save_file(legacy, str(weight_path))
    loaded = load_runner(tmp_path).model
    assert not loaded.has_embedded_action_normalizer()
    for a, b in zip(loaded(tokens), heads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.set_action_normalization(mean=[2., 3.], std=[1., 1.])
    model.load_state_dict(legacy, strict=True)
    assert not model.has_embedded_action_normalizer()
    config = json.loads((tmp_path / "config.json").read_text())
    assert "action_normalization" not in config
    assert "action_normalization" not in config["requires_capabilities"]


def test_windowed_incremental_and_prefix_precompute_parity(tmp_path):
    model = bundle(tmp_path)
    states = torch.randn(2, 4, 3)
    actions = torch.tensor([[[-3., 2.], [-1., 6.], [1., 10.], [-2., 7.]]]).expand(2, -1, -1)
    bos = torch.zeros(2, 4, 1)
    bos[:, 0] = 1
    mask = torch.ones(2, 4, dtype=torch.bool)
    window = model.predict_with_window(states, actions, bos, mask)[0]
    cache = None
    for t in range(4):
        actual, cache = model.predict_incremental_cached(states[:, t:t+1], actions[:, t:t+1], bos[:, t:t+1], past_key_values=cache, cache_max_len=8)
        torch.testing.assert_close(actual[0], window[:, t:t+1], atol=1e-6, rtol=1e-5)
    _, cache = model.infer_cached(torch.cat([states[:, :3], actions[:, :3], bos[:, :3]], -1), past_key_values=build_kv_cache(model.backbone, 8))
    actual, _ = model.predict_incremental_cached(states[:, 3:], actions[:, 3:], bos[:, 3:], past_key_values=cache)
    torch.testing.assert_close(actual[0], window[:, -1:], atol=1e-6, rtol=1e-5)
    wrapper = _WindowedPolicy(load_runner(tmp_path, num_envs=2, use_windowed=True)).eval()
    torch.testing.assert_close(wrapper(states, actions, bos, mask), window[:, -1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("hybrid", [False, True])
def test_onnx_normalized_boundary_parity(tmp_path, hybrid):
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    model = bundle(tmp_path / "bundle", hybrid=hybrid)
    result = export_onnx(tmp_path / "bundle", tmp_path / "policy.onnx", batch_size=2)
    assert result.max_abs_error < 1e-4
    runner = load_runner(tmp_path / "bundle", num_envs=2, use_windowed=True)
    wrapper = _WindowedPolicy(runner).eval()
    states = torch.randn(2, 4, 3)
    actions = torch.zeros(2, 4, model.action_size)
    actions[..., model._normalized_action_indices] = torch.tensor([[[-3., 2.], [1., 10.], [-2., 7.], [0., 5.]]])
    bos = torch.zeros(2, 4, 1)
    bos[:, 0] = 1
    mask = torch.ones(2, 4)
    sample = (states, actions, bos, mask)
    session = ort.InferenceSession(str(result.output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {k: v.numpy() for k, v in zip(("states", "actions", "is_bos", "mask"), sample)})[0]
    expected = wrapper(*sample).detach().numpy()
    assert np.max(np.abs(actual - expected)) < 1e-4


@pytest.mark.parametrize("hybrid", [False, True])
def test_runner_matches_export_wrapper_over_rollout(tmp_path, hybrid):
    bundle(tmp_path, hybrid=hybrid)
    runner = load_runner(tmp_path, num_envs=2, use_windowed=True)
    wrapper = _WindowedPolicy(runner).eval()
    rng = np.random.default_rng(101)
    runner.reset(rng.normal(size=(2, 3)).astype(np.float32))
    for step in range(7):
        states, actions, bos, mask, _ = runner.buffer.get_context()
        sample = tuple(torch.as_tensor(a, dtype=torch.float32) for a in (states, actions, bos, mask))
        expected = wrapper(*sample).detach().numpy()
        env = runner.act()
        env_flat = np.stack([gym.spaces.flatten(runner.action_space, a) for a in env])
        np.testing.assert_allclose(env_flat, runner._gym_flatten_action(expected), atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(env_flat, runner._last_buffer_action)
        runner.observe(rng.normal(size=(2, 3)).astype(np.float32))
