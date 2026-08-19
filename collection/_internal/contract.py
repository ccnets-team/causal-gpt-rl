"""Validation for the recorded-episode contract used by ``collection/``.

The contract is intentionally small: typed-vector observations, actions,
rewards, and episode boundaries. Source-specific encoding stays with the user;
these helpers only verify that the recorded values match the spaces they
declare before Minari writes anything.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = (
    "observations",
    "actions",
    "rewards",
    "terminations",
    "truncations",
)
ACTION_KINDS = {"continuous", "discrete", "hybrid"}

# Minari 0.5.3 accepts nested namespace segments, but the dataset name itself
# must end in ``-v<integer>``. Validate here because Minari can create the output
# directory before its own parser reports a malformed id.
_DATASET_ID_RE = re.compile(
    r"(?:(?P<namespace>[-_\w][-_\w/]*[-_\w]+)/)?"
    r"(?P<dataset>[-_\w]+)-v(?P<version>\d+)$"
)


class ContractError(ValueError):
    """A recorded episode or its declared spaces violate the input contract."""


def validate_dataset_id(dataset_id: str) -> None:
    """Require the Minari ``[namespace/]name-vN`` identifier form."""
    if not isinstance(dataset_id, str) or _DATASET_ID_RE.fullmatch(dataset_id) is None:
        raise ContractError(
            f"Malformed dataset id {dataset_id!r}; expected "
            "'[namespace/]name-v<integer>', for example "
            "'mujoco/humanoid/simple-v0'."
        )


def load_spec(raw_dir: str | Path) -> dict[str, Any]:
    """Load ``spec.json`` or return the backward-compatible continuous default."""
    path = Path(raw_dir) / "spec.json"
    if not path.is_file():
        return {"action_kind": "continuous"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a positive integer") from exc
    if integer <= 0 or integer != value:
        raise ContractError(f"{name} must be a positive integer")
    return integer


def _positive_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ContractError(f"{name} must be a non-empty list of positive integers")
    return [_positive_int(item, f"{name}[{i}]") for i, item in enumerate(value)]


def _bounds(value: Any, size: int, name: str, default: float) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=np.float32)
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a number or {size}-element list") from exc
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float32)
    elif array.shape == (size,):
        array = array.astype(np.float32, copy=False)
    else:
        raise ContractError(f"{name} must be a number or {size}-element list")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    return array


def normalize_spec(
    spec: Mapping[str, Any], *, observation_width: int, action_width: int
) -> dict[str, Any]:
    """Validate and fill the small space declaration understood by the packager."""
    normalized = dict(spec)
    action_kind = normalized.get("action_kind", "continuous")
    if action_kind not in ACTION_KINDS:
        raise ContractError(
            f"action_kind must be one of {sorted(ACTION_KINDS)}, got {action_kind!r}"
        )
    normalized["action_kind"] = action_kind

    channels_value = normalized.get("obs_channels")
    obs_channels = (
        [observation_width]
        if channels_value in (None, [])
        else _positive_int_list(channels_value, "obs_channels")
    )
    if sum(obs_channels) != observation_width:
        raise ContractError(
            f"obs_channels {obs_channels} (sum={sum(obs_channels)}) != "
            f"stored observation width {observation_width}"
        )
    normalized["obs_channels"] = obs_channels

    if action_kind == "continuous":
        act_dim = _positive_int(normalized.get("act_dim", action_width), "act_dim")
        expected_width = act_dim
        continuous_size = act_dim
    elif action_kind == "discrete":
        branches = _positive_int_list(normalized.get("branches"), "branches")
        normalized["branches"] = branches
        expected_width = len(branches)
        continuous_size = 0
    else:
        continuous_size = _positive_int(
            normalized.get("continuous_size"), "continuous_size"
        )
        branches = _positive_int_list(normalized.get("branches"), "branches")
        normalized["continuous_size"] = continuous_size
        normalized["branches"] = branches
        expected_width = continuous_size + len(branches)

    if action_width != expected_width:
        raise ContractError(
            f"stored action width {action_width} != declared width {expected_width} "
            f"for action_kind={action_kind!r}"
        )

    if continuous_size:
        low = _bounds(normalized.get("act_low"), continuous_size, "act_low", -1.0)
        high = _bounds(normalized.get("act_high"), continuous_size, "act_high", 1.0)
        if np.any(low >= high):
            raise ContractError("every act_low value must be smaller than act_high")
        normalized["act_low"] = low
        normalized["act_high"] = high
        normalized["act_dim"] = continuous_size
    elif "act_low" in normalized or "act_high" in normalized:
        raise ContractError("act_low/act_high apply only to continuous actions")

    return normalized


def _numeric_2d(array: Any, name: str, source: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2:
        raise ContractError(f"{source}: {name} must be a 2-D array, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
        raise ContractError(f"{source}: {name} must be numeric, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ContractError(f"{source}: {name} contains NaN or infinity")
    return value


def _numeric_1d(array: Any, name: str, source: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 1:
        raise ContractError(f"{source}: {name} must be a 1-D array, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
        raise ContractError(f"{source}: {name} must be numeric, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ContractError(f"{source}: {name} contains NaN or infinity")
    return value


def validate_episode(
    arrays: Mapping[str, Any], normalized_spec: Mapping[str, Any], *, source: str
) -> None:
    """Validate one episode against an already normalized space declaration."""
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ContractError(f"{source}: missing arrays: {', '.join(missing)}")

    observations = _numeric_2d(arrays["observations"], "observations", source)
    actions = _numeric_2d(arrays["actions"], "actions", source)
    rewards = _numeric_1d(arrays["rewards"], "rewards", source)
    terminations = np.asarray(arrays["terminations"])
    truncations = np.asarray(arrays["truncations"])
    for name, flags in (("terminations", terminations), ("truncations", truncations)):
        if flags.ndim != 1 or not np.issubdtype(flags.dtype, np.bool_):
            raise ContractError(f"{source}: {name} must be a 1-D bool array")

    transition_count = len(rewards)
    if transition_count < 1:
        raise ContractError(f"{source}: an episode must contain at least one transition")
    lengths = {
        "actions": len(actions),
        "rewards": transition_count,
        "terminations": len(terminations),
        "truncations": len(truncations),
    }
    if any(length != transition_count for length in lengths.values()):
        raise ContractError(f"{source}: transition-array lengths differ: {lengths}")
    if len(observations) != transition_count + 1:
        raise ContractError(
            f"{source}: observations length {len(observations)} != T+1 "
            f"({transition_count + 1})"
        )

    expected_obs_width = sum(normalized_spec["obs_channels"])
    if observations.shape[1] != expected_obs_width:
        raise ContractError(
            f"{source}: observation width {observations.shape[1]} != "
            f"declared width {expected_obs_width}"
        )

    action_kind = normalized_spec["action_kind"]
    if action_kind == "continuous":
        expected_action_width = normalized_spec["act_dim"]
        continuous = actions
        discrete = None
    elif action_kind == "discrete":
        expected_action_width = len(normalized_spec["branches"])
        continuous = None
        discrete = actions
    else:
        continuous_size = normalized_spec["continuous_size"]
        expected_action_width = continuous_size + len(normalized_spec["branches"])
        continuous = actions[:, :continuous_size]
        discrete = actions[:, continuous_size:]
    if actions.shape[1] != expected_action_width:
        raise ContractError(
            f"{source}: action width {actions.shape[1]} != declared width "
            f"{expected_action_width}"
        )

    if continuous is not None:
        low = np.asarray(normalized_spec["act_low"])
        high = np.asarray(normalized_spec["act_high"])
        if np.any(continuous < low) or np.any(continuous > high):
            raise ContractError(
                f"{source}: continuous action lies outside declared "
                f"[{low.tolist()}, {high.tolist()}] bounds"
            )

    if discrete is not None:
        if action_kind == "hybrid":
            # One array carries both parts of a hybrid action, so the indices
            # can only be stored as integral floats beside the continuous
            # values; the packager casts them back with `astype(np.int64)`.
            if np.any(discrete != np.rint(discrete)):
                raise ContractError(
                    f"{source}: discrete action indices must be whole numbers"
                )
        elif not np.issubdtype(discrete.dtype, np.integer):
            raise ContractError(f"{source}: discrete action indices must use an integer dtype")
        branches = np.asarray(normalized_spec["branches"], dtype=np.int64)
        if np.any(discrete < 0) or np.any(discrete >= branches):
            raise ContractError(
                f"{source}: discrete action index lies outside branches "
                f"{branches.tolist()}"
            )

    if np.any(terminations[:-1]) or np.any(truncations[:-1]):
        raise ContractError(f"{source}: only the final transition may end the episode")
    if not (bool(terminations[-1]) or bool(truncations[-1])):
        raise ContractError(
            f"{source}: the final transition must be terminated or truncated"
        )


def validate_raw_directory(
    raw_dir: str | Path, spec: Mapping[str, Any] | None = None
) -> tuple[list[Path], dict[str, Any]]:
    """Preflight every episode and return its files plus normalized declaration."""
    directory = Path(raw_dir)
    files = sorted(directory.glob("ep_*.npz"))
    if not files:
        raise ContractError(f"No ep_*.npz files found in {directory}")
    raw_spec = dict(spec) if spec is not None else load_spec(directory)

    try:
        with np.load(files[0], allow_pickle=False) as first:
            missing = [name for name in REQUIRED_ARRAYS if name not in first]
            if missing:
                raise ContractError(
                    f"{files[0].name}: missing arrays: {', '.join(missing)}"
                )
            observations = np.asarray(first["observations"])
            actions = np.asarray(first["actions"])
            if observations.ndim != 2 or actions.ndim != 2:
                raise ContractError(
                    f"{files[0].name}: observations and actions must be 2-D arrays"
                )
            normalized = normalize_spec(
                raw_spec,
                observation_width=observations.shape[1],
                action_width=actions.shape[1],
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Cannot read {files[0]}: {exc}") from exc

    for path in files:
        try:
            with np.load(path, allow_pickle=False) as episode:
                validate_episode(episode, normalized, source=path.name)
        except (OSError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"Cannot read {path}: {exc}") from exc
    return files, normalized
