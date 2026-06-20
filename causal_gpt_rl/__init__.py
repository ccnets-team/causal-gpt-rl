"""causal-gpt-rl: public inference runtime for the Causal-GPT-RL policy.

This package exposes the trained autoregressive policy as a runnable artifact.
It contains the model definition (`causal_gpt_rl.model`) and the stable
inference surface (`causal_gpt_rl.inference`): bundle loading, evaluation
helpers, and `PolicyRunner`.

Training-time code (optimizer, dataset pipeline, online normalizer updates) is
intentionally not part of this package.
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("causal-gpt-rl")
except PackageNotFoundError:
    # Keep this fallback in sync with `pyproject.toml`'s `version` field
    # so editable/dev checkouts without `pip install` still report correctly.
    __version__ = "0.5.0"

from .inference import (
    BUNDLE_FORMAT_VERSION,
    PolicyRunner,
    convert_legacy_bundle_to_safetensors,
    export_bundle,
    load_runner,
    load_runner_from_hub,
    run_episodes,
)
from .model import (
    AutoregressiveModel,
    DataSpec,
    ModelConfig,
    PolicyModel,
    SpaceSpec,
    build_model_specs,
)

__all__ = [
    "__version__",
    "AutoregressiveModel",
    "BUNDLE_FORMAT_VERSION",
    "DataSpec",
    "ModelConfig",
    "PolicyModel",
    "PolicyRunner",
    "SpaceSpec",
    "build_model_specs",
    "convert_legacy_bundle_to_safetensors",
    "export_bundle",
    "load_runner",
    "load_runner_from_hub",
    "run_episodes",
]
