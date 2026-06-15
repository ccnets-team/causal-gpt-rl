import json
from pathlib import Path

import pytest
import torch

from causal_gpt_rl.inference.bundle import export_bundle, load_runner
from causal_gpt_rl.inference.state_normalizer import StateNormalizer
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec


def _components():
    state_specs = [
        SpaceSpec(
            type="continuous",
            size=3,
            dtype=torch.float32,
            low=[-10, -10, -10],
            high=[10, 10, 10],
        )
    ]
    action_specs = [
        SpaceSpec(
            type="continuous",
            size=2,
            dtype=torch.float32,
            low=[-1, -1],
            high=[1, 1],
            squash="tanh",
        )
    ]
    config = ModelConfig(
        context_length=4,
        d_model=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    normalizer = StateNormalizer(num_features=3)
    with torch.no_grad():
        normalizer.mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
        normalizer.var.copy_(torch.tensor([4.0, 9.0, 16.0]))
    model = AutoregressiveModel(
        config,
        state_specs=state_specs,
        action_specs=action_specs,
        device="cpu",
    )
    return model, config, state_specs, action_specs, normalizer


def test_bundle_loads_with_embedded_state_normalizer_without_sidecar(tmp_path: Path):
    model, config, state_specs, action_specs, normalizer = _components()
    bundle_dir = tmp_path / "bundle"

    export_bundle(
        bundle_dir,
        model=model,
        model_config=config,
        state_specs=state_specs,
        action_specs=action_specs,
        context_length=4,
        state_normalizer=normalizer,
        write_state_normalizer_sidecar=False,
    )

    assert not (bundle_dir / "state_normalizer.safetensors").exists()
    payload = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    assert payload["state_normalization"]["embedded"] is True
    assert payload["state_normalization"]["legacy_sidecar"] is False
    assert payload["bundle_format_version"] == 2

    runner = load_runner(bundle_dir)

    assert runner.state_normalizer is None
    assert runner.model.has_embedded_state_normalizer()
    sample = torch.tensor([[[1.0, 5.0, 11.0]]])
    expected = torch.tensor([[[0.0, 1.0, 2.0]]])
    assert torch.allclose(
        runner.model.normalize_states_for_inference(sample),
        expected,
        atol=1e-6,
    )


def test_legacy_sidecar_normalizer_syncs_into_model(tmp_path: Path):
    safetensors = pytest.importorskip("safetensors.torch")
    model, config, state_specs, action_specs, normalizer = _components()
    bundle_dir = tmp_path / "bundle"

    export_bundle(
        bundle_dir,
        model=model,
        model_config=config,
        state_specs=state_specs,
        action_specs=action_specs,
        context_length=4,
        state_normalizer=normalizer,
        write_state_normalizer_sidecar=True,
    )

    payload = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    assert payload["bundle_format_version"] == 1
    assert payload["state_normalization"]["legacy_sidecar"] is True

    model_path = bundle_dir / "model.safetensors"
    model_state = safetensors.load_file(str(model_path), device="cpu")
    model_state = {
        key: value.detach().clone()
        for key, value in model_state.items()
        if not key.startswith("state_normalization_")
    }
    model_path.unlink()
    torch.save(model_state, bundle_dir / "model.pt")

    runner = load_runner(bundle_dir)

    assert runner.state_normalizer is None
    assert runner.model.has_embedded_state_normalizer()
    assert torch.allclose(
        runner.model.state_normalization_mean.cpu(),
        normalizer.mean.cpu(),
    )
