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
from typing import Iterable, Mapping, Optional

import torch

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
from .state_normalizer import StateNormalizer

BUNDLE_FORMAT_VERSION = 2
_LEGACY_SIDECAR_BUNDLE_VERSION = 1
_SUPPORTED_BUNDLE_VERSIONS = (1, 2)

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
    state_normalizer: Optional[StateNormalizer] = None,
    env_id: Optional[str] = None,
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

    config_payload = {
        "bundle_format_version": bundle_format_version,
        "package_version": _PACKAGE_VERSION,
        "model_config": model_config.to_dict(),
        "state_specs": [s.to_json_dict() for s in state_specs],
        "action_specs": [s.to_json_dict() for s in action_specs],
        "context_length": int(context_length),
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

    action_mode, head_sizes, action_size, lows, highs = PolicyRunner._resolve_action_specs(
        action_specs
    )
    state_size = int(sum(s.size for s in state_specs))

    return PolicyRunner(
        model=model,
        action_mode=action_mode,
        action_head_sizes=head_sizes,
        state_size=state_size,
        action_size=action_size,
        context_length=context_length,
        state_normalizer=normalizer,
        num_envs=num_envs,
        kv_cache_max_len=kv_cache_max_len,
        use_windowed=use_windowed,
        action_low=lows,
        action_high=highs,
    )


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
