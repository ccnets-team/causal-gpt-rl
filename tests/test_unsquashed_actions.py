"""squash=None preserves raw continuous coordinates throughout serving."""
import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from causal_gpt_rl.export.onnx import _WindowedPolicy
from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model import AutoregressiveModel, ModelConfig, SpaceSpec
from causal_gpt_rl.model.utils.output_adapter import OutputToInputAdapter


def components(space=None, *, specs=None, enabled=True):
    if space is None:
        space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
    specs = specs if specs is not None else extract_data_specs_from_space(space)
    config = ModelConfig(context_length=4, d_model=16, num_heads=2, num_layers=1,
                         intermediate_size=32, max_position_embeddings=32,
                         use_unsquashed_actions=enabled)
    states = [SpaceSpec(type="continuous", size=3)]
    model = AutoregressiveModel(config, state_specs=states, action_specs=specs, device="cpu").eval()
    model.set_state_normalization(mean=[0., 0., 0.], std=[1., 1., 1.])
    return model, config, states, specs, space


def save_bundle(path, parts):
    model, config, states, specs, space = parts
    bundle.export_bundle(path, model=model, model_config=config, state_specs=states,
                         action_specs=specs, action_space=space, context_length=4,
                         write_state_normalizer_sidecar=False)


@pytest.mark.parametrize("bounds", [(None, None), ([-np.inf] * 2, [np.inf] * 2),
                                    ([-1e6] * 2, [1e6] * 2)])
def test_no_squash_adapter_is_identity_even_with_bound_metadata(bounds):
    adapter = OutputToInputAdapter("continuous", 2, None, *bounds, use_unsquashed_actions=True)
    value = torch.tensor([[[-40., 25.]]], requires_grad=True)
    assert adapter(value) is value
    adapter(value).sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


@pytest.mark.parametrize("boundless_specs", [False, True])
def test_raw_input_mean_and_sample_share_the_same_coordinate(boundless_specs):
    specs = [SpaceSpec(type="continuous", size=2)] if boundless_specs else None
    model, *_ = components(specs=specs)
    model.input_proj = torch.nn.Identity()
    tokens = torch.tensor([[[1., 2., 3., -40., 25., 0.]]])
    torch.testing.assert_close(model.adapt_input(tokens), tokens, rtol=0, atol=0)
    heads = [torch.zeros(1, 1, s.size) for s in model.flat_output_specs]
    heads[model.mean_action_indices[0]] = tokens[..., 3:5]
    heads[model.log_std_action_indices[0]].fill_(-0.7)
    torch.testing.assert_close(torch.cat(model._extract_mean_action(heads), -1), tokens[..., 3:5])
    torch.testing.assert_close(model.sample_action_from_heads(heads, std_scale=0), tokens[..., 3:5])
    torch.manual_seed(71)
    expected = tokens[..., 3:5] + 0.6 * np.exp(-0.7) * torch.randn(1, 1, 2)
    torch.manual_seed(71)
    torch.testing.assert_close(model.sample_action_from_heads(heads, std_scale=0.6), expected)
    assert torch.isfinite(model._action_scale).all() and torch.isfinite(model._action_bias).all()


@pytest.mark.parametrize("categorical", [False, True])
def test_mixed_heads_preserve_raw_values_and_existing_tanh_sampling(categorical):
    spaces = [gym.spaces.Box(-3., 5., (1,), np.float32),
              gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)]
    if categorical:
        spaces.append(gym.spaces.Discrete(3))
    model, *_ = components(gym.spaces.Tuple(tuple(spaces)))
    heads = [torch.zeros(2, 3, s.size) for s in model.flat_output_specs]
    for index in model.mean_action_indices:
        heads[index].fill_(2.5)
    for index in model.log_std_action_indices:
        heads[index].fill_(-1.)
    mean = torch.full((2, 3, 3), 2.5)
    for std_scale in (0., 0.7):
        torch.manual_seed(19)
        z = mean + std_scale * np.exp(-1.) * torch.randn_like(mean)
        expected = torch.cat((4 * torch.tanh(z[..., :1]) + 1, z[..., 1:]), -1)
        torch.manual_seed(19)
        actual = model.sample_action_from_heads(heads, std_scale=std_scale)
        torch.testing.assert_close(actual[..., :3], expected)
        if categorical:
            torch.testing.assert_close(actual[..., 3:].sum(-1), torch.ones(2, 3))
    means = torch.cat(model._extract_mean_action(heads), -1)
    torch.testing.assert_close(means[..., :1], 4 * torch.tanh(mean[..., :1]) + 1)
    torch.testing.assert_close(means[..., 1:3], mean[..., 1:])
    z = torch.tensor([[[0.3, -40., 25.]]], requires_grad=True)
    result = model._squash_continuous_action(z)
    result.sum().backward()
    assert torch.isfinite(z.grad).all()
    torch.testing.assert_close(z.grad[..., 1:], torch.ones_like(z.grad[..., 1:]))


@pytest.mark.parametrize("finite_bounds", [False, True])
def test_raw_head_cannot_enable_bounds_atanh_normalization(finite_bounds):
    specs = [SpaceSpec(type="continuous", size=2, low=[-10.] * 2, high=[10.] * 2)] if finite_bounds else None
    model, *_ = components(specs=specs)
    with pytest.raises(ValueError, match="Unsquashed continuous"):
        model.set_action_normalization(mean=[0., 0.], std=[1., 1.])
    assert not model.has_embedded_action_normalizer()


@pytest.mark.parametrize("windowed", [False, True])
def test_bundle_checkpoint_and_onnx_wrapper_keep_large_raw_actions(tmp_path, windowed):
    parts = components()
    model, config, states, specs, space = parts
    expected = np.array([40., -25.], np.float32)
    with torch.no_grad():
        head = model.output_head[model.mean_action_indices[0]].layers[-1]
        head.weight.zero_()
        head.bias.copy_(torch.from_numpy(expected))
    save_bundle(tmp_path / "bundle", parts)
    payload = json.loads((tmp_path / "bundle/config.json").read_text())
    assert bundle.UNSQUASHED_ACTION_CAPABILITY in payload["requires_capabilities"]
    assert payload["action_specs"][0]["low"] == [None, None]
    assert payload["action_specs"][0]["high"] == [None, None]
    assert payload["action_specs"][0]["squash"] is None
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    restored, *_ = components()
    runners = [bundle.load_runner(tmp_path / "bundle", use_windowed=windowed),
               PolicyRunner.from_checkpoint(checkpoint, model=restored, action_specs=specs,
                                            state_size=3, context_length=4, use_windowed=windowed)]
    for runner in runners:
        runner.reset(np.zeros(3, np.float32))
        for step in range(6):
            np.testing.assert_array_equal(runner.act(), expected)
            np.testing.assert_array_equal(runner._last_buffer_action, expected[None])
            runner.observe(np.full(3, step, np.float32))
    wrapper = _WindowedPolicy(runners[0])
    action = wrapper(torch.zeros(1, 4, 3), torch.full((1, 4, 2), 40.),
                     torch.zeros(1, 4, 1), torch.ones(1, 4, dtype=torch.bool))
    np.testing.assert_array_equal(action.detach().numpy(), expected[None])


def test_raw_bundle_requires_consistent_capability_and_older_runtimes_refuse(tmp_path, monkeypatch):
    save_bundle(tmp_path, components())
    config_path = tmp_path / "config.json"
    payload = json.loads(config_path.read_text())
    with monkeypatch.context() as patch:
        patch.setattr(bundle, "_SUPPORTED_CAPABILITIES",
                      bundle.SUPPORTED_CAPABILITIES - {bundle.UNSQUASHED_ACTION_CAPABILITY})
        with pytest.raises(ValueError, match="does not support"):
            bundle.load_runner(tmp_path)
    payload["requires_capabilities"].remove(bundle.UNSQUASHED_ACTION_CAPABILITY)
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="capability metadata"):
        bundle.load_runner(tmp_path)


@pytest.mark.parametrize("enabled", [False, True])
def test_bounded_bundle_needs_no_new_weights_or_capability(tmp_path, enabled):
    parts = components(gym.spaces.Box(-3., 5., (2,), np.float32), enabled=enabled)
    save_bundle(tmp_path, parts)
    payload = json.loads((tmp_path / "config.json").read_text())
    assert bundle.UNSQUASHED_ACTION_CAPABILITY not in payload["requires_capabilities"]
    model = parts[0]
    assert "_continuous_squash_mask" not in model.state_dict()
    clone, *_ = components(parts[-1], enabled=enabled)
    clone.load_state_dict(model.state_dict(), strict=True)
    z = torch.tensor([[[-3., 4.]]])
    torch.testing.assert_close(model._squash_continuous_action(z), 4 * torch.tanh(z) + 1,
                               rtol=0, atol=0)


def test_option_defaults_off_and_preserves_legacy_none_validation():
    assert not ModelConfig().use_unsquashed_actions
    assert not ModelConfig.from_dict({}).use_unsquashed_actions
    with pytest.raises(AssertionError, match="If squash is None"):
        components(enabled=False)
    with pytest.raises(AssertionError, match="If squash is None"):
        OutputToInputAdapter("continuous", 2, None, [-1e6] * 2, [1e6] * 2)
    # This pre-existing adapter-only path remains a passthrough with the flag off.
    value = torch.full((1, 1, 2), 25.)
    assert OutputToInputAdapter("continuous", 2)(value) is value


def test_enabling_option_does_not_change_tanh_heads_or_state_dict():
    space = gym.spaces.Box(-3., 5., (2,), np.float32)
    torch.manual_seed(33)
    legacy, *_ = components(space, enabled=False)
    torch.manual_seed(33)
    opted_in, *_ = components(space, enabled=True)
    assert legacy.state_dict().keys() == opted_in.state_dict().keys()
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, opted_in.state_dict()[key], rtol=0, atol=0)
    tokens = torch.zeros(2, 4, legacy.total_input_dim)
    tokens[..., 3:5] = 2.5
    torch.testing.assert_close(legacy.adapt_input(tokens), opted_in.adapt_input(tokens), rtol=0, atol=0)
    heads = legacy(tokens)
    for std_scale in (0., 0.7):
        torch.manual_seed(14)
        expected = legacy.sample_action_from_heads(heads, std_scale=std_scale)
        torch.manual_seed(14)
        actual = opted_in.sample_action_from_heads(heads, std_scale=std_scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_bundle_without_option_still_loads(tmp_path):
    save_bundle(tmp_path, components(gym.spaces.Box(-1., 1., (2,), np.float32), enabled=False))
    path = tmp_path / "config.json"
    payload = json.loads(path.read_text())
    payload["model_config"].pop("use_unsquashed_actions")
    path.write_text(json.dumps(payload))
    assert not bundle.load_runner(tmp_path).model.use_unsquashed_actions


def test_export_rejects_dropping_the_enabled_option(tmp_path):
    model, config, states, specs, space = components()
    config.use_unsquashed_actions = False
    with pytest.raises(ValueError, match="config must match"):
        bundle.export_bundle(tmp_path, model=model, model_config=config, state_specs=states,
                             action_specs=specs, context_length=4)


def test_enabled_raw_path_never_evaluates_tanh_or_atanh(monkeypatch):
    model, *_ = components()
    model.input_proj = torch.nn.Identity()
    tokens = torch.tensor([[[1., 2., 3., -40., 25., 0.]]])
    heads = [torch.zeros(1, 1, s.size) for s in model.flat_output_specs]
    heads[model.mean_action_indices[0]] = tokens[..., 3:5]
    def forbidden(*args, **kwargs):
        raise AssertionError("Raw heads must not evaluate tanh/atanh")
    monkeypatch.setattr(torch, "tanh", forbidden)
    monkeypatch.setattr(torch, "atanh", forbidden)
    torch.testing.assert_close(model.adapt_input(tokens), tokens)
    torch.testing.assert_close(torch.cat(model._extract_mean_action(heads), -1), tokens[..., 3:5])
    torch.testing.assert_close(model.sample_action_from_heads(heads, std_scale=0), tokens[..., 3:5])


def test_raw_onnx_export_preserves_mean_without_squash(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from causal_gpt_rl.export import export_onnx
    parts = components()
    with torch.no_grad():
        head = parts[0].output_head[parts[0].mean_action_indices[0]].layers[-1]
        head.weight.zero_()
        head.bias.copy_(torch.tensor([40., -25.]))
    save_bundle(tmp_path / "bundle", parts)
    output = tmp_path / "raw.onnx"
    result = export_onnx(tmp_path / "bundle", output, batch_size=2)
    assert result.max_abs_error is not None and result.max_abs_error < 1e-4
    graph = onnx.load(str(output))
    assert not {"Tanh", "Atanh"}.intersection(node.op_type for node in graph.graph.node)
    properties = {p.key: p.value for p in graph.metadata_props}
    assert bundle.UNSQUASHED_ACTION_CAPABILITY in json.loads(properties["causal_gpt_rl.requires_capabilities"])
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    inputs = {"states": np.zeros((2, 4, 3), np.float32),
              "actions": np.full((2, 4, 2), 40., np.float32),
              "is_bos": np.zeros((2, 4, 1), np.float32),
              "mask": np.ones((2, 4), np.float32)}
    actual = session.run(None, {i.name: inputs[i.name] for i in session.get_inputs()})[0]
    np.testing.assert_array_equal(actual, np.array([[40., -25.]] * 2, np.float32))


def test_export_rejects_changing_raw_head_specs(tmp_path):
    model, config, states, specs, space = components()
    different = extract_data_specs_from_space(gym.spaces.Box(-1., 1., (2,), np.float32))
    with pytest.raises(ValueError, match="specs must match"):
        bundle.export_bundle(tmp_path, model=model, model_config=config, state_specs=states,
                             action_specs=different, context_length=4)


def test_bounded_bundle_cannot_claim_raw_capability(tmp_path):
    model, config, states, specs, space = components(gym.spaces.Box(-1., 1., (2,), np.float32))
    with pytest.raises(ValueError, match="requires an enabled raw"):
        bundle.export_bundle(tmp_path, model=model, model_config=config, state_specs=states,
                             action_specs=specs, context_length=4,
                             requires_capabilities=[bundle.UNSQUASHED_ACTION_CAPABILITY])
