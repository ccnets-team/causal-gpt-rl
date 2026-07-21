from pathlib import Path

import pytest
import torch

from causal_gpt_rl.export import export_onnx
from causal_gpt_rl.inference.bundle import export_bundle
from causal_gpt_rl.inference.state_normalizer import StateNormalizer
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec


def _make_bundle(path: Path) -> Path:
    state_specs = [
        SpaceSpec(
            type="continuous",
            size=3,
            dtype=torch.float32,
            low=[-10.0] * 3,
            high=[10.0] * 3,
        )
    ]
    action_specs = [
        SpaceSpec(
            type="continuous",
            size=2,
            dtype=torch.float32,
            low=[-1.0] * 2,
            high=[1.0] * 2,
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
    model = AutoregressiveModel(
        config,
        state_specs=state_specs,
        action_specs=action_specs,
        device="cpu",
    )
    normalizer = StateNormalizer(num_features=3)
    return export_bundle(
        path,
        model=model,
        model_config=config,
        state_specs=state_specs,
        action_specs=action_specs,
        context_length=4,
        state_normalizer=normalizer,
        write_state_normalizer_sidecar=False,
    )


def test_export_onnx_fixed_batch_and_verification(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    bundle = _make_bundle(tmp_path / "bundle")
    output = tmp_path / "policy.onnx"

    result = export_onnx(bundle, output, batch_size=2)

    assert result.output_path == output
    assert result.batch_size == 2
    assert result.context_length == 4
    assert result.state_size == 3
    assert result.action_size == 2
    assert result.export_backend in {"dynamo", "legacy"}
    assert result.max_abs_error is not None
    assert result.max_abs_error < 1e-4
    assert output.is_file()
    assert not Path(str(output) + ".data").exists()

    graph = onnx.load(str(output))
    inputs = {value.name: value for value in graph.graph.input}
    outputs = {value.name: value for value in graph.graph.output}

    def shape(value):
        return [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    assert shape(inputs["states"]) == [2, 4, 3]
    assert shape(inputs["actions"]) == [2, 4, 2]
    assert shape(inputs["is_bos"]) == [2, 4, 1]
    assert shape(inputs["mask"]) == [2, 4]
    assert shape(outputs["action"]) == [2, 2]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_export_rejects_non_positive_batch_before_loading_bundle(
    tmp_path: Path, batch_size: int
):
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        export_onnx(tmp_path / "missing", tmp_path / "out.onnx", batch_size=batch_size)
