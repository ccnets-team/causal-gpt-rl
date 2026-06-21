"""Public bundle save/load for a single-directory inference layout.

Layout (HF-style, per-file):

    <bundle_dir>/
    model.safetensors             # weights (state_dict); embeds state normalization (v2)
    config.json                   # ModelConfig + state/action specs + context_length
    state_normalizer.safetensors  # legacy StateNormalizer sidecar (v1 only)

Format versions:
  * v1 — state normalization shipped as a `state_normalizer.safetensors`
    sidecar. Still emitted when `write_state_normalizer_sidecar=True` so the
    bundle stays readable by older (<=0.2.x) loaders.
  * v2 — no sidecar; normalization statistics are embedded in the model
    `state_dict`. Older loaders reject this with a clear "unsupported version"
    error instead of failing on the missing sidecar.

`config.json` schema:

    {
      "bundle_format_version": 1 | 2,   # 2 when the sidecar is omitted
      "package_version": "<causal-gpt-rl version that exported this>",  # optional, added 0.1.0+
      "model_config":    { ... ModelConfig.to_dict() ... },
      "state_specs":     [ ... SpaceSpec.to_json_dict() ... ],
      "action_specs":    [ ... SpaceSpec.to_json_dict() ... ],
      "context_length":  <int>,
      "requires_capabilities": [ "hybrid_action", ... ],  # forward-compat gate
      "state_container":  <serialized gym space> | null,  # input adapter schema, added 0.6.0+
      "action_container": <serialized gym space> | null,  # output adapter schema, added 0.6.0+
      "state_normalization": {          # added 0.3.0+
        "embedded":      <bool>,        # stats live in model.safetensors
        "legacy_sidecar": <bool>        # state_normalizer.safetensors written
      },
      "env_id":          "<gymnasium env id>"  # optional, added 0.2.0+
    }

Release-candidate v1 details:
  * `SpaceSpec.low`/`high`: unbounded entries are encoded as JSON `null`
    instead of the non-standard `Infinity` float token.
  * `model_config` carries explicit derived fields where available so the
    bundle describes the architecture without implicit warnings.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import torch

if TYPE_CHECKING:
    from gymnasium.spaces import Space

try:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
except Exception:  # pragma: no cover - depends on optional package availability.
    load_safetensors = None
    save_safetensors = None

from .. import __version__ as _PACKAGE_VERSION
from ..model.autoregressive_model import AutoregressiveModel
from ..model.schema import ModelConfig, SpaceSpec
from .runner import PolicyRunner
from .spaces import deserialize_space, serialize_space
from .state_normalizer import StateNormalizer

BUNDLE_FORMAT_VERSION = 2
_LEGACY_SIDECAR_BUNDLE_VERSION = 1
_SUPPORTED_BUNDLE_VERSIONS = (1, 2)

# Fine-grained forward-compat gate, orthogonal to `bundle_format_version`.
# `bundle_format_version` tracks the on-disk file/serialization layout;
# `requires_capabilities` declares optional model/runtime FEATURES a bundle
# depends on (e.g. "hybrid_action", or a future "dict_space"). A bundle lists
# what it needs; a runtime that does not advertise a required capability refuses
# to load it (loud) instead of silently mis-decoding actions. Additive features
# that degrade safely on an unaware runtime (e.g. EOS via strict=False weight
# load) must NOT be listed here — only features an unaware runtime would
# mis-handle. Extend this set as such features ship.
#
# "hybrid_action": the bundle's action schedule mixes more than one spec type
# (e.g. continuous + discrete). A runtime older than per-head decoding assumes a
# single uniform family and cannot decode it.
#
# "hybrid_state": the bundle's state mixes discrete/structured leaves, so its
# observations must be flattened (gym.flatten) and reordered continuous-first
# before the model. Advertised since the input adapter shipped (P4): `load_runner`
# wires `make_state_input_adapter` onto the runner, so this runtime can serve such
# bundles. Older runtimes that lack the adapter still refuse them loudly.
#
# "action_container": the bundle's action_space is a Dict/Tuple, so the emitted
# action must be unflattened back into that declared container. Advertised since
# the output adapter shipped (P5): `load_runner` wires `make_action_output_adapter`
# onto the runner, which restores the container via gym.spaces.unflatten. Older
# runtimes that only decode per head (flat) still refuse such bundles loudly.
_SUPPORTED_CAPABILITIES: frozenset[str] = frozenset(
    {"hybrid_action", "hybrid_state", "action_container"}
)

# Capabilities this runtime RECOGNIZES but does not yet implement. A bundle
# requiring one is refused by `load_runner` with the reason below rather than a
# bare token, so the product boundary self-describes. An entry graduates into
# `_SUPPORTED_CAPABILITIES` once its adapter ships (as "action_container" did
# when P5 landed). Currently empty. (MultiBinary action/obs is now supported via
# the "multi_binary" head type; it needs no gate — an older runtime fails loud at
# SpaceSpec construction on the unknown type rather than mis-decoding silently.)
_DEFERRED_CAPABILITY_REASONS: dict[str, str] = {}

_MODEL_FILENAME = "model.safetensors"
_LEGACY_MODEL_FILENAME = "model.pt"
_CONFIG_FILENAME = "config.json"
_NORMALIZER_FILENAME = "state_normalizer.safetensors"
_LEGACY_NORMALIZER_FILENAME = "state_normalizer.pt"


def _require_safetensors() -> None:
    if save_safetensors is None or load_safetensors is None:
        raise ImportError(
            "safetensors is required for inference bundles. "
            "Install it with `pip install safetensors`."
        )


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _ensure_tensor_state_dict(state: Mapping[str, object], *, source: Path) -> dict:
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected tensor state_dict in {source}, got {type(state)!r}")

    non_tensor_keys = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        preview = ", ".join(str(key) for key in non_tensor_keys[:5])
        raise TypeError(
            f"Expected only tensors in {source}; non-tensor keys: {preview}"
        )
    return dict(state)


def convert_legacy_bundle_to_safetensors(
    bundle_dir: str | Path,
    *,
    remove_legacy: bool = False,
) -> Path:
    """Convert legacy `model.pt` bundle weights to safetensors in-place.

    This is intended for old public bundles produced before safetensors export.
    It expects `model.pt` and `state_normalizer.pt` to contain plain tensor
    state_dicts, not full training checkpoints.
    """
    _require_safetensors()

    bundle_dir = Path(bundle_dir)
    legacy_model_path = bundle_dir / _LEGACY_MODEL_FILENAME
    model_path = bundle_dir / _MODEL_FILENAME
    if not legacy_model_path.is_file():
        if model_path.is_file():
            return bundle_dir
        raise FileNotFoundError(f"Legacy bundle weights not found: {legacy_model_path}")

    model_state = _ensure_tensor_state_dict(
        torch.load(legacy_model_path, map_location="cpu"),
        source=legacy_model_path,
    )
    save_safetensors(model_state, model_path)

    legacy_normalizer_path = bundle_dir / _LEGACY_NORMALIZER_FILENAME
    normalizer_path = bundle_dir / _NORMALIZER_FILENAME
    if not legacy_normalizer_path.is_file():
        raise FileNotFoundError(
            f"Legacy bundle state normalizer not found: {legacy_normalizer_path}"
        )
    normalizer_state = _ensure_tensor_state_dict(
        torch.load(legacy_normalizer_path, map_location="cpu"),
        source=legacy_normalizer_path,
    )
    save_safetensors(normalizer_state, normalizer_path)

    if remove_legacy:
        _remove_if_exists(legacy_model_path)
        _remove_if_exists(legacy_normalizer_path)

    return bundle_dir


def export_bundle(
    bundle_dir: str | Path,
    *,
    model: AutoregressiveModel,
    model_config: ModelConfig,
    state_specs: Iterable[SpaceSpec],
    action_specs: Iterable[SpaceSpec],
    context_length: int,
    obs_space: "Optional[Space]" = None,
    action_space: "Optional[Space]" = None,
    state_normalizer: Optional[StateNormalizer] = None,
    env_id: Optional[str] = None,
    requires_capabilities: Optional[Iterable[str]] = None,
    write_state_normalizer_sidecar: bool = True,
) -> Path:
    """Write the bundle to `bundle_dir`. Creates the directory if needed."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    state_specs = list(state_specs)
    action_specs = list(action_specs)

    if state_normalizer is not None and hasattr(
        model, "set_state_normalization_from_state_dict"
    ):
        model.set_state_normalization_from_state_dict(state_normalizer.state_dict())

    write_sidecar = bool(
        state_normalizer is not None and write_state_normalizer_sidecar
    )
    # v1 keeps the sidecar so older loaders can still read it; v2 drops the
    # sidecar and relies on normalization embedded in the model state_dict.
    bundle_format_version = (
        _LEGACY_SIDECAR_BUNDLE_VERSION if write_sidecar else BUNDLE_FORMAT_VERSION
    )

    # A schedule mixing more than one action spec type cannot be decoded by a
    # runtime that assumes a single uniform family, so gate it behind the
    # "hybrid_action" capability — older runtimes then refuse loudly instead of
    # erroring out cryptically mid-load. Uniform-type bundles stay ungated and
    # load on any runtime.
    action_types = [s.type for s in action_specs]
    capabilities = set(requires_capabilities or [])
    if len(set(action_types)) > 1:
        capabilities.add("hybrid_action")
    # Discrete/one-hot state needs the input adapter (gym.flatten + continuous-
    # first permutation); a runtime that feeds raw state would mis-shape it, so
    # gate it behind "hybrid_state". Pure-continuous state stays ungated — its
    # flatten is a plain concat and the permutation is identity.
    if any(s.type != "continuous" for s in state_specs):
        capabilities.add("hybrid_state")

    # Serialized declared Gymnasium spaces — the lossless schema gym.spaces
    # .unflatten needs. The model stores only flat head sizes; structure (Dict
    # key order, Tuple order) lives here so the input/output adapters can
    # flatten/unflatten. None when the exporter did not supply the space
    # (positional, structure-less I/O — today's continuous bundles).
    state_container = serialize_space(obs_space) if obs_space is not None else None
    action_container = (
        serialize_space(action_space) if action_space is not None else None
    )

    # A Dict/Tuple action_space carries declared container structure the runner's
    # per-head decode does NOT restore — it returns a flat per-head action.
    # Restoring the container is the P5 output adapter; until it ships, gate such
    # bundles behind "action_container" so a runtime without the adapter refuses
    # them loudly rather than silently emitting the wrong shape. Box / Discrete /
    # MultiDiscrete actions already decode to their declared env form, so they are
    # never gated. (Mirrors the input side: hybrid_state gates structured obs.)
    if action_container is not None and action_container["type"] in ("Dict", "Tuple"):
        capabilities.add("action_container")

    config_payload = {
        "bundle_format_version": bundle_format_version,
        "package_version": _PACKAGE_VERSION,
        "model_config": model_config.to_dict(),
        "state_specs": [s.to_json_dict() for s in state_specs],
        "action_specs": [s.to_json_dict() for s in action_specs],
        "context_length": int(context_length),
        "requires_capabilities": sorted(capabilities),
        # `action_container` is the reserved slot the output adapter unflattens
        # into; the parallel `state_container` is the input adapter's source
        # schema. A bundle populating these also gates itself via
        # `requires_capabilities`, so old runtimes refuse rather than mis-shape.
        "state_container": state_container,
        "action_container": action_container,
        "state_normalization": {
            "embedded": bool(
                getattr(model, "has_embedded_state_normalizer", lambda: False)()
            ),
            "legacy_sidecar": write_sidecar,
        },
    }
    if env_id:
        config_payload["env_id"] = str(env_id)
    (bundle_dir / _CONFIG_FILENAME).write_text(
        json.dumps(config_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    _require_safetensors()
    save_safetensors(model.state_dict(), bundle_dir / _MODEL_FILENAME)
    _remove_if_exists(bundle_dir / _LEGACY_MODEL_FILENAME)

    has_embedded_normalizer = bool(
        getattr(model, "has_embedded_state_normalizer", lambda: False)()
    )
    if state_normalizer is None and not has_embedded_normalizer:
        raise ValueError(
            "state_normalizer or embedded model state normalization is required "
            "for public inference bundles."
        )

    if write_sidecar:
        save_safetensors(
            state_normalizer.state_dict(),
            bundle_dir / _NORMALIZER_FILENAME,
        )
        _remove_if_exists(bundle_dir / _LEGACY_NORMALIZER_FILENAME)
    else:
        _remove_if_exists(bundle_dir / _NORMALIZER_FILENAME)
        _remove_if_exists(bundle_dir / _LEGACY_NORMALIZER_FILENAME)

    return bundle_dir


def load_runner(
    bundle_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    num_envs: int = 1,
    kv_cache_max_len: Optional[int] = None,
    use_windowed: bool = False,
) -> PolicyRunner:
    """Load a bundle and return a ready-to-run `PolicyRunner`.

    When `kv_cache_max_len` is omitted, PolicyRunner uses
    `4 * context_length` from the bundle config as the cached inference cap.
    """
    bundle_dir = Path(bundle_dir)
    config_path = bundle_dir / _CONFIG_FILENAME
    model_path = bundle_dir / _MODEL_FILENAME
    legacy_model_path = bundle_dir / _LEGACY_MODEL_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Bundle config not found: {config_path}")
    if not model_path.is_file() and not legacy_model_path.is_file():
        raise FileNotFoundError(
            f"Bundle weights not found: {model_path} or {legacy_model_path}"
        )

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))

    version = int(config_payload.get("bundle_format_version", 0))
    if version not in _SUPPORTED_BUNDLE_VERSIONS:
        raise ValueError(
            f"Unsupported bundle_format_version={version}; "
            f"this build supports {_SUPPORTED_BUNDLE_VERSIONS}."
        )

    # Forward-compat gate: refuse (loudly) a bundle that depends on a feature
    # this runtime does not implement, rather than silently mis-decoding. A
    # missing/empty list means "needs only baseline capabilities".
    required_caps = config_payload.get("requires_capabilities") or []
    missing_caps = sorted(set(required_caps) - _SUPPORTED_CAPABILITIES)
    if missing_caps:
        details = [
            f"  - {cap}: {_DEFERRED_CAPABILITY_REASONS[cap]}"
            if cap in _DEFERRED_CAPABILITY_REASONS
            else f"  - {cap}"
            for cap in missing_caps
        ]
        raise ValueError(
            f"Bundle requires capabilities this causal-gpt-rl "
            f"{_PACKAGE_VERSION} build does not support:\n"
            + "\n".join(details)
            + "\nUpgrade causal-gpt-rl to a build that advertises them."
        )

    # Loader contract for the `action_container` slot: absent/None → positional
    # per-head output (e.g. Box/Discrete actions); a Dict/Tuple value means the
    # action is unflattened into that declared container by the output adapter
    # (P5), wired onto the runner below from `action_space`. The exporter stamps
    # "action_container" alongside such a value, gating it so a runtime without
    # the adapter refuses loudly (handled above). The runner core itself stays
    # container-agnostic; structuring lives in the adapter layer.

    model_config = ModelConfig.from_dict(config_payload["model_config"])
    state_specs = [SpaceSpec.from_dict(d) for d in config_payload["state_specs"]]
    action_specs = [SpaceSpec.from_dict(d) for d in config_payload["action_specs"]]
    context_length = int(config_payload["context_length"])

    torch_device = torch.device(device)
    model = AutoregressiveModel(
        model_config,
        state_specs=state_specs,
        action_specs=action_specs,
        device=torch_device,
    )
    if model_path.is_file():
        _require_safetensors()
        model_state = load_safetensors(str(model_path), device=str(torch_device))
    else:
        model_state = torch.load(legacy_model_path, map_location=torch_device)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    normalizer: Optional[StateNormalizer] = None
    normalizer_path = bundle_dir / _NORMALIZER_FILENAME
    legacy_normalizer_path = bundle_dir / _LEGACY_NORMALIZER_FILENAME
    if normalizer_path.is_file():
        _require_safetensors()
        normalizer = StateNormalizer.from_state_dict(
            load_safetensors(str(normalizer_path), device=str(torch_device))
        )
        normalizer.to(torch_device)
    elif legacy_normalizer_path.is_file():
        normalizer = StateNormalizer.from_state_dict(
            torch.load(legacy_normalizer_path, map_location=torch_device)
        )
        normalizer.to(torch_device)
    if (
        normalizer is not None
        and not model.has_embedded_state_normalizer()
        and hasattr(model, "set_state_normalization_from_state_dict")
    ):
        model.set_state_normalization_from_state_dict(normalizer.state_dict())
        normalizer = None

    action_schedule = PolicyRunner._resolve_action_specs(action_specs)
    state_size = int(sum(s.size for s in state_specs))

    # Declared Gymnasium spaces, deserialized from their lossless schema. None
    # for older bundles that did not carry them (positional, structure-less I/O).
    # `obs_space` drives the input adapter (P4): a structured space makes the
    # runner flatten observations continuous-first; a Box / None keeps the raw
    # passthrough. `action_space` drives the output adapter (P5): a Dict/Tuple
    # space makes the runner unflatten the emitted action into that container;
    # a non-container / None keeps the flat per-head action.
    state_container = config_payload.get("state_container")
    action_container = config_payload.get("action_container")
    obs_space = (
        deserialize_space(state_container) if state_container is not None else None
    )
    action_space = (
        deserialize_space(action_container) if action_container is not None else None
    )

    runner = PolicyRunner(
        model=model,
        action_schedule=action_schedule,
        state_size=state_size,
        context_length=context_length,
        state_normalizer=normalizer,
        num_envs=num_envs,
        kv_cache_max_len=kv_cache_max_len,
        use_windowed=use_windowed,
        obs_space=obs_space,
        action_space=action_space,
    )

    return runner


def _resolve_hub_bundle_dir(snapshot_path: Path, subfolder: str) -> Path:
    return snapshot_path / subfolder if subfolder else snapshot_path


def load_runner_from_hub(
    repo_id: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    subfolder: str = "",
    cache_dir: Optional[str | Path] = None,
    token: Optional[str | bool] = None,
    local_files_only: bool = False,
    device: str | torch.device = "cpu",
    num_envs: int = 1,
    kv_cache_max_len: Optional[int] = None,
    use_windowed: bool = False,
) -> PolicyRunner:
    """Download an inference bundle from Hugging Face Hub and load a runner.

    The Hub repository should contain the standard bundle layout at the repo
    root, or under `subfolder` for per-environment repositories such as
    `ccnets/causal-gpt-rl` with `subfolder="ant-v5"`.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "huggingface_hub is required to load bundles from the Hub. "
            "Install it with `pip install causal-gpt-rl[hub]` or "
            "`pip install huggingface_hub`."
        ) from exc

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            token=token,
            local_files_only=local_files_only,
        )
    )
    bundle_dir = _resolve_hub_bundle_dir(snapshot_path, subfolder)
    return load_runner(
        bundle_dir,
        device=device,
        num_envs=num_envs,
        kv_cache_max_len=kv_cache_max_len,
        use_windowed=use_windowed,
    )


__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "convert_legacy_bundle_to_safetensors",
    "export_bundle",
    "load_runner",
    "load_runner_from_hub",
]
