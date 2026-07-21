"""Export an inference bundle as a fixed-batch, self-contained ONNX policy.

The input is a complete Causal GPT-RL bundle, not a standalone safetensors
file. ``config.json`` supplies the architecture, spaces, normalization contract,
and context length; ``model.safetensors`` supplies the learned parameters.
Legacy bundles may additionally carry ``state_normalizer.safetensors``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from causal_gpt_rl.inference import load_runner


@dataclass(frozen=True)
class OnnxExportResult:
    """Summary returned after a successful export."""

    output_path: Path
    batch_size: int
    context_length: int
    state_size: int
    action_size: int
    export_backend: str
    max_abs_error: float | None


class _WindowedPolicy(nn.Module):
    """Raw observation window to newest model-space action."""

    def __init__(self, runner: Any):
        super().__init__()
        self.model = runner.model
        self.normalizer = runner.state_normalizer

    def forward(self, states, actions, is_bos, mask):
        if self.normalizer is not None:
            states = self.normalizer(states)
        elif hasattr(self.model, "normalize_states_for_inference"):
            states = self.model.normalize_states_for_inference(states)
        tokens = torch.cat([states, actions, is_bos], dim=-1)
        embedded = self.model.adapt_input(tokens)
        hidden = self.model.backbone(embedded, padding_mask=mask.to(torch.bool))
        outputs = self.model.project_output_heads(hidden)
        action = self.model._extract_mean_action(outputs)
        if isinstance(action, (list, tuple)):
            action = torch.cat(list(action), dim=-1)
        return action[:, -1]


def _patch_transformers_causal_mask(attention: str) -> None:
    """Use an ONNX-exportable causal/padding mask implementation."""
    try:
        from transformers import masking_utils
    except ImportError:  # Older Transformers already builds a plain mask.
        return

    def no_mask(*args, **kwargs):
        return None

    def plain_mask(*args, **kwargs):
        embedded = kwargs.get("input_embeds")
        if embedded is None:
            embedded = args[1]
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) > 2:
            attention_mask = args[2]
        sequence = embedded.shape[1]
        causal = torch.tril(
            torch.ones(sequence, sequence, dtype=torch.bool, device=embedded.device)
        )
        valid = causal.unsqueeze(0)
        if attention_mask is not None:
            valid = valid & attention_mask.to(torch.bool).unsqueeze(1)
        zero = torch.zeros((), dtype=embedded.dtype, device=embedded.device)
        negative = torch.full(
            (), torch.finfo(embedded.dtype).min,
            dtype=embedded.dtype, device=embedded.device,
        )
        return torch.where(valid.unsqueeze(1), zero, negative)

    builder = no_mask if attention == "sdpa" else plain_mask
    masking_utils._ignore_causal_mask_sdpa = lambda *args, **kwargs: True
    masking_utils.create_causal_mask = builder
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("transformers.models.") and hasattr(
            module, "create_causal_mask"
        ):
            module.create_causal_mask = builder


def _sample_from_space(space: Any, rng: np.random.Generator, flat_size: int):
    """Create a finite structured observation matching the declared space."""
    import gymnasium as gym

    if isinstance(space, gym.spaces.Dict):
        return {key: _sample_from_space(value, rng, flat_size) for key, value in space.spaces.items()}
    if isinstance(space, gym.spaces.Tuple):
        return tuple(_sample_from_space(value, rng, flat_size) for value in space.spaces)
    if isinstance(space, gym.spaces.Discrete):
        return int(rng.integers(space.n))
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.asarray([rng.integers(int(n)) for n in space.nvec], dtype=space.dtype)
    if isinstance(space, gym.spaces.MultiBinary):
        return rng.integers(0, 2, size=space.shape, dtype=np.int8)
    shape = tuple(space.shape) if space is not None else (flat_size,)
    return rng.normal(0.0, 1.0, size=shape).astype(np.float32)


def _sample_context(runner: Any, batch_size: int) -> tuple[torch.Tensor, ...]:
    """Populate the real runner buffer so exported shapes and ordering are exact."""
    rng = np.random.default_rng(0)
    space = getattr(runner, "obs_space", None)

    def observation():
        if space is None:
            return rng.normal(0.0, 1.0, size=(runner.state_size,)).astype(np.float32)
        return _sample_from_space(space, rng, runner.state_size)

    def batch():
        samples = [observation() for _ in range(batch_size)]
        if batch_size == 1:
            return samples[0]
        return samples

    # PolicyRunner accepts a sequence of structured per-env observations. Keep
    # that contract instead of stacking Dict/Tuple leaves ourselves.
    runner.reset(batch())
    for _ in range(runner.context_length + 1):
        runner.act(batch())
    states, actions, is_bos, mask, _ = runner.buffer.get_context()
    return tuple(
        torch.as_tensor(value, dtype=torch.float32)
        for value in (states, actions, is_bos, mask)
    )


def _require_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "ONNX export dependencies are missing. Install "
            "`causal-gpt-rl[onnx]`."
        ) from exc
    return onnx


def export_onnx(
    bundle_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 1,
    opset: int = 18,
    attention: str = "eager",
    verify: bool = True,
) -> OnnxExportResult:
    """Convert a policy bundle to one fixed-batch ONNX file.

    ``batch_size`` is the number of agent contexts controlled by this policy in
    one call, not necessarily the total number of agents in the environment.
    For example, SoccerTwos uses 16 when the policy controls one team in eight
    parallel 2-vs-2 fields.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if attention not in {"eager", "sdpa"}:
        raise ValueError("attention must be 'eager' or 'sdpa'")
    onnx = _require_onnx()

    bundle_path = Path(bundle_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner = load_runner(
        bundle_path,
        device="cpu",
        num_envs=batch_size,
        kv_cache_max_len=None,
        use_windowed=True,
    )
    context_length = int(runner.context_length)
    _patch_transformers_causal_mask(attention)
    network_config = runner.model.backbone.net.config
    network_config._attn_implementation = attention

    sample = _sample_context(runner, batch_size)
    policy = _WindowedPolicy(runner).eval()
    with torch.no_grad():
        reference = policy(*sample)

    export_kwargs = {
        "input_names": ["states", "actions", "is_bos", "mask"],
        "output_names": ["action"],
        "opset_version": opset,
    }
    export_backend = "dynamo"
    try:
        with torch.no_grad():
            torch.onnx.export(
                policy, sample, str(output_path), dynamo=True, **export_kwargs
            )
    except Exception as exc:
        export_backend = "legacy"
        warnings.warn(
            f"Dynamo ONNX export failed ({type(exc).__name__}: {exc}); "
            "retrying with the legacy exporter.",
            RuntimeWarning,
            stacklevel=2,
        )
        with torch.no_grad():
            torch.onnx.export(
                policy, sample, str(output_path), dynamo=False, **export_kwargs
            )

    # Dynamo commonly externalizes weights. Fold them back into one portable
    # artifact and remove its temporary sidecar.
    onnx.save(
        onnx.load(str(output_path), load_external_data=True),
        str(output_path),
        save_as_external_data=False,
    )
    sidecar = Path(str(output_path) + ".data")
    if sidecar.exists():
        sidecar.unlink()
    onnx.checker.check_model(str(output_path))

    max_abs_error = None
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX Runtime verification requires `causal-gpt-rl[onnx]`."
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(output_path), options, providers=["CPUExecutionProvider"]
        )
        feeds = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                ("states", "actions", "is_bos", "mask"), sample
            )
        }
        actual = session.run(["action"], feeds)[0]
        max_abs_error = float(np.max(np.abs(actual - reference.cpu().numpy())))
        if not np.isfinite(max_abs_error) or max_abs_error >= 1e-4:
            raise RuntimeError(
                f"ONNX verification failed: max_abs_error={max_abs_error:.6g}"
            )

    return OnnxExportResult(
        output_path=output_path,
        batch_size=batch_size,
        context_length=context_length,
        state_size=int(runner.state_size),
        action_size=int(runner.action_size),
        export_backend=export_backend,
        max_abs_error=max_abs_error,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Causal GPT-RL bundle as a self-contained ONNX policy."
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = export_onnx(
        args.bundle,
        args.out,
        batch_size=args.batch_size,
        opset=args.opset,
        attention=args.attention,
        verify=not args.no_verify,
    )
    error = (
        "not run" if result.max_abs_error is None else f"{result.max_abs_error:.3e}"
    )
    print(f"wrote: {result.output_path}")
    print(
        f"contract: batch={result.batch_size} context={result.context_length} "
        f"state={result.state_size} action={result.action_size}"
    )
    print(f"export backend: {result.export_backend}")
    print(f"verification max_abs_error: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
