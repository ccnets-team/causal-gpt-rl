"""Schema and config types for the autoregressive model.

Two-layer spec design:

- `SpaceSpec` — env-facing slim spec. What the env produces and what the
  bundle stores: type / size / dtype / low / high / squash. JSON-safe
  serialization helpers (`to_dict`/`from_dict`) live here.

- `DataSpec` — model-facing super-set. Adds `role` / `sub_role` /
  `init_type` for the model's I/O routing and weight init. Built from
  `SpaceSpec` lists by `model.spec_factory.build_model_specs`.

`ModelConfig` is the pure architecture hyperparam dataclass — JSON-safe,
no specs / device. The model receives state/action SpaceSpecs separately
at construction and decorates them internally.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

import numpy as np
import torch


SPACE_TYPE_ALIASES = {
    "box": "continuous",
    "continuous": "continuous",
    "discrete": "discrete",
    "multi_discrete": "multi_discrete",
}

VALID_SPACE_TYPES = set(SPACE_TYPE_ALIASES.values())

VALID_SQUASH_TYPES = {
    None,
    "tanh",
    "sigmoid",
}

VALID_TASK_TYPES = {
    "binary_classification",
    "multi_class_classification",
    "multi_label_classification",
    "regression",
    "ordinal_regression",
    "compositional_regression",
}

VALID_TYPES = VALID_SPACE_TYPES.union(VALID_TASK_TYPES)


_DTYPE_TO_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}
_STR_TO_TORCH_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}


def _serialize_dtype(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return dtype
    if isinstance(dtype, torch.dtype):
        return _DTYPE_TO_STR.get(dtype, str(dtype))
    if isinstance(dtype, np.dtype):
        return dtype.name
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return np.dtype(dtype).name
    return str(dtype)


def _deserialize_dtype(value: Any) -> Any:
    if value is None or not isinstance(value, str):
        return value
    if value in _STR_TO_TORCH_DTYPE:
        return _STR_TO_TORCH_DTYPE[value]
    try:
        return np.dtype(value)
    except TypeError:
        return value


def _serialize_bounds(arr: Any) -> Any:
    """JSON-safe bounds: ±inf → null, finite → float. Sign info is implicit
    in which slot (low vs high) the array is written to.
    """
    if arr is None:
        return None
    flat = np.asarray(arr).reshape(-1).tolist()
    return [None if not np.isfinite(v) else float(v) for v in flat]


def _deserialize_bounds(value: Any, *, sign: int) -> Any:
    """Restore a bounds array. Accepts `null` (preferred) and legacy float
    `Infinity` for unbounded entries; also accepts a bare scalar (e.g.
    `-1.0`) for handwritten / single-value configs. `sign=-1` for `low`,
    `sign=+1` for `high`.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    # Normalize to a flat Python list so scalars, lists, and tuples all flow
    # through the same null/inf substitution path — matches _serialize_bounds.
    flat = np.asarray(value, dtype=object).reshape(-1).tolist()
    filled = [sign * np.inf if v is None else float(v) for v in flat]
    return np.asarray(filled, dtype=np.float64)


@dataclass(frozen=True, kw_only=True)
class SpaceSpec:
    """Env-facing spec — JSON-safe slim form (what bundles store).

    `type` is restricted to space types ("continuous" / "discrete" /
    "multi_discrete") plus task-classification types for non-RL targets.
    """

    type: str
    size: int
    dtype: Optional[Any] = None
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    squash: Optional[str] = None

    def __post_init__(self):
        if self.type not in VALID_TYPES:
            raise ValueError(
                f"Invalid type '{self.type}'. "
                f"Valid options are {sorted(VALID_TYPES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain dict — preserves numpy arrays / torch.dtype as-is."""
        return asdict(self)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dict — numpy arrays → list, torch.dtype → str.

        Continuous bounds use `null` for unbounded entries (±inf) so the
        emitted JSON is valid per the standard (no `Infinity` tokens).
        """
        return {
            "type": self.type,
            "size": int(self.size),
            "dtype": _serialize_dtype(self.dtype),
            "low": _serialize_bounds(self.low),
            "high": _serialize_bounds(self.high),
            "squash": self.squash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpaceSpec":
        own_field_names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in own_field_names}
        kwargs["dtype"] = _deserialize_dtype(kwargs.get("dtype"))
        # Accept both v2 (null) and legacy v1 (float Infinity) encodings.
        kwargs["low"] = _deserialize_bounds(kwargs.get("low"), sign=-1)
        kwargs["high"] = _deserialize_bounds(kwargs.get("high"), sign=+1)
        return cls(**kwargs)


@dataclass(frozen=True, kw_only=True)
class DataSpec(SpaceSpec):
    """Model-facing super-set — adds I/O routing and init metadata.

    `role` ∈ {"state", "action", "value", "bos_indicator", ...}
    `sub_role` ∈ {"mean", "log_std", ...} (action heads)
    `init_type` controls per-head weight initialization.
    """

    init_type: Optional[Any] = None
    role: Optional[str] = None
    sub_role: Optional[str] = None


def flatten_specs(spec):
    if isinstance(spec, SpaceSpec):
        return [spec]
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(flatten_specs(s))
        return out
    if isinstance(spec, dict):
        out = []
        for s in spec.values():
            out.extend(flatten_specs(s))
        return out
    raise TypeError(type(spec))


def nested_to_flat_list_heads(nested_heads):
    """
    Flattens a nested head structure into a flat list of tensors.

    Args:
        nested_heads: nested list/tuple structure of Tensors

    Returns:
        list[Tensor]: flat list of tensors (preorder traversal)
    """
    flat = []

    def _flatten(x):
        if torch.is_tensor(x):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for xi in x:
                _flatten(xi)
        else:
            raise AssertionError(
                "nested_heads must contain only Tensors or list/tuple of Tensors", type(x)
            )

    _flatten(nested_heads)
    return flat


def ensure_tensor_heads(x: Any) -> torch.Tensor:
    """
    Ensures input is converted into a single flat Tensor by concatenation.

    Args:
        x: Tensor or nested(list/tuple) of Tensors

    Returns:
        Tensor: concatenated tensor along the last dimension

    Raises:
        AssertionError: if tensors have incompatible device/dtype/shape
    """
    if torch.is_tensor(x):
        return x

    list_heads = nested_to_flat_list_heads(x)
    assert len(list_heads) > 0, "No tensors found in input"

    ref = list_heads[0]
    assert torch.is_tensor(ref), "Input must contain tensors"

    ref_device = ref.device
    ref_dtype = ref.dtype
    ref_shape_prefix = ref.shape[:-1]

    for i, t in enumerate(list_heads):
        assert torch.is_tensor(t), f"Element {i} is not a Tensor"
        assert t.device == ref_device, (
            f"Device mismatch at index {i}: {t.device} vs {ref_device}"
        )
        assert t.dtype == ref_dtype, (
            f"Dtype mismatch at index {i}: {t.dtype} vs {ref_dtype}"
        )
        assert t.shape[:-1] == ref_shape_prefix, (
            f"Shape prefix mismatch at index {i}: "
            f"{t.shape[:-1]} vs {ref_shape_prefix}"
        )

    return torch.cat(list_heads, dim=-1)


_GPT2_NAMES = {"gpt2", "gpt", "gpt-2"}


def _resolve_max_position_embeddings(network_name: str, context_length: int) -> int:
    """Backbone-aware default sizing for the position window.

    GPT-2 (absolute pos): wpe is a trained parameter sized max_pos × d_model;
    over-spec wastes memory and yields untrained positions. Keep it tight.

    Llama (RoPE): max_pos only sizes the cos/sin buffer (not trained), so a
    larger margin gives cached evaluation room past the training window.
    """
    if network_name.lower() in _GPT2_NAMES:
        return max(8, int(context_length) * 2)
    return max(32, int(context_length) * 8)


@dataclass
class ModelConfig:
    """Architecture hyperparams — JSON-safe. Specs/device live outside.

    `intermediate_size`, `max_position_embeddings`, and `context_length`
    accept `None` at construction; `__post_init__` resolves them to concrete
    ints so exported `config.json` always carries the full architecture spec.
    `rope_theta` is used by Llama/RoPE backbones and kept here so bundles can
    reproduce non-default positional encoding choices.
    """

    network_name: str = "Llama"  # e.g. "Llama", "GPT-2".
    num_layers: int = 4
    d_model: int = 256
    num_heads: int = 8
    dropout: float = 0.02
    rope_theta: float = 1e3
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    context_length: Optional[int] = None

    def __post_init__(self) -> None:
        self.rope_theta = float(self.rope_theta)

        if self.context_length is None:
            self.context_length = 32
        else:
            self.context_length = int(self.context_length)

        if self.intermediate_size is None:
            self.intermediate_size = int(self.d_model) * 4
        else:
            self.intermediate_size = int(self.intermediate_size)

        if self.max_position_embeddings is None:
            self.max_position_embeddings = _resolve_max_position_embeddings(
                self.network_name, self.context_length
            )
        else:
            self.max_position_embeddings = int(self.max_position_embeddings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        if "train_context_length" in d and "context_length" not in d:
            d = {**d, "context_length": d["train_context_length"]}
        own_field_names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in own_field_names}
        return cls(**kwargs)


__all__ = [
    "SPACE_TYPE_ALIASES",
    "VALID_SPACE_TYPES",
    "VALID_SQUASH_TYPES",
    "VALID_TASK_TYPES",
    "VALID_TYPES",
    "SpaceSpec",
    "DataSpec",
    "ModelConfig",
    "flatten_specs",
    "nested_to_flat_list_heads",
    "ensure_tensor_heads",
]
