"""The collection boundary rejects data that contradicts its declared spaces."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from collection._contract import (
    ContractError,
    validate_dataset_id,
    validate_raw_directory,
)


ROOT = Path(__file__).resolve().parents[1]


def _episode(*, obs_width=3, action_width=1, transitions=4):
    terminations = np.zeros(transitions, dtype=bool)
    truncations = np.zeros(transitions, dtype=bool)
    terminations[-1] = True
    return {
        "observations": np.zeros((transitions + 1, obs_width), dtype=np.float32),
        "actions": np.zeros((transitions, action_width), dtype=np.float32),
        "rewards": np.zeros(transitions, dtype=np.float32),
        "terminations": terminations,
        "truncations": truncations,
    }


def _save(path, arrays):
    np.savez(path, **arrays)


@pytest.mark.parametrize(
    "dataset_id",
    ["simple-v0", "review/badinput-v0", "mujoco/humanoid/simple-v12"],
)
def test_dataset_id_accepts_minari_version_suffix(dataset_id):
    validate_dataset_id(dataset_id)


@pytest.mark.parametrize(
    "dataset_id",
    ["review/badinput/v0", "missing-version", "bad-vx", ""],
)
def test_dataset_id_rejects_ambiguous_or_missing_version(dataset_id):
    with pytest.raises(ContractError, match="Malformed dataset id"):
        validate_dataset_id(dataset_id)


def test_preflight_accepts_consistent_continuous_episodes(tmp_path):
    _save(tmp_path / "ep_000000.npz", _episode())
    _save(tmp_path / "ep_000001.npz", _episode())

    files, spec = validate_raw_directory(tmp_path)

    assert [path.name for path in files] == ["ep_000000.npz", "ep_000001.npz"]
    assert spec["obs_channels"] == [3]
    assert spec["act_dim"] == 1
    np.testing.assert_array_equal(spec["act_low"], [-1.0])
    np.testing.assert_array_equal(spec["act_high"], [1.0])


@pytest.mark.parametrize("field", ["observations", "actions", "rewards"])
def test_preflight_rejects_nan_or_infinity(tmp_path, field):
    arrays = _episode()
    arrays[field].flat[0] = np.nan if field != "actions" else np.inf
    _save(tmp_path / "ep_000000.npz", arrays)

    with pytest.raises(ContractError, match="NaN or infinity"):
        validate_raw_directory(tmp_path)


def test_preflight_rejects_action_width_change_between_files(tmp_path):
    _save(tmp_path / "ep_000000.npz", _episode(action_width=1))
    _save(tmp_path / "ep_000001.npz", _episode(action_width=7))

    with pytest.raises(ContractError, match="action width 7 != declared width 1"):
        validate_raw_directory(tmp_path)


def test_preflight_rejects_continuous_action_outside_declared_bounds(tmp_path):
    arrays = _episode()
    arrays["actions"][0, 0] = 5.0
    _save(tmp_path / "ep_000000.npz", arrays)

    with pytest.raises(ContractError, match="outside declared"):
        validate_raw_directory(tmp_path)


def test_preflight_uses_explicit_per_dimension_action_bounds(tmp_path):
    arrays = _episode(action_width=2)
    arrays["actions"][:, 0] = -2.0
    arrays["actions"][:, 1] = 20.0
    _save(tmp_path / "ep_000000.npz", arrays)
    (tmp_path / "spec.json").write_text(
        json.dumps(
            {
                "action_kind": "continuous",
                "act_dim": 2,
                "act_low": [-2.0, 10.0],
                "act_high": [2.0, 30.0],
            }
        ),
        encoding="utf-8",
    )

    _, spec = validate_raw_directory(tmp_path)

    np.testing.assert_array_equal(spec["act_low"], [-2.0, 10.0])
    np.testing.assert_array_equal(spec["act_high"], [2.0, 30.0])


def test_preflight_rejects_non_integer_or_out_of_range_discrete_actions(tmp_path):
    arrays = _episode()
    arrays["actions"] = np.full((4, 1), 1.5, dtype=np.float32)
    _save(tmp_path / "ep_000000.npz", arrays)
    (tmp_path / "spec.json").write_text(
        json.dumps({"action_kind": "discrete", "branches": [2]}),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="integer dtype"):
        validate_raw_directory(tmp_path)

    arrays["actions"] = np.full((4, 1), 2, dtype=np.int64)
    _save(tmp_path / "ep_000000.npz", arrays)
    with pytest.raises(ContractError, match="outside branches"):
        validate_raw_directory(tmp_path)


def test_preflight_rejects_misaligned_lengths_and_open_episode(tmp_path):
    arrays = _episode()
    arrays["truncations"] = np.zeros(3, dtype=bool)
    _save(tmp_path / "ep_000000.npz", arrays)
    with pytest.raises(ContractError, match="lengths differ"):
        validate_raw_directory(tmp_path)

    arrays = _episode()
    arrays["terminations"][-1] = False
    _save(tmp_path / "ep_000000.npz", arrays)
    with pytest.raises(ContractError, match="final transition"):
        validate_raw_directory(tmp_path)

    arrays = _episode()
    arrays["truncations"][-1] = True
    _save(tmp_path / "ep_000000.npz", arrays)
    validate_raw_directory(tmp_path)


def test_cli_round_trip_writes_only_after_preflight(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    arrays = _episode(action_width=2)
    _save(raw / "ep_000000.npz", arrays)
    datasets = tmp_path / "datasets"
    env = dict(os.environ, MINARI_DATASETS_PATH=str(datasets))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "collection" / "build_minari.py"),
            "--raw",
            str(raw),
            "--dataset-id",
            "review/collection-smoke-v0",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "created and verified" in result.stdout
    assert (datasets / "review" / "collection-smoke-v0" / "data").is_dir()


def test_cli_bad_input_leaves_no_dataset_directory(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    arrays = _episode()
    arrays["actions"][0, 0] = 5.0
    _save(raw / "ep_000000.npz", arrays)
    datasets = tmp_path / "datasets"
    env = dict(os.environ, MINARI_DATASETS_PATH=str(datasets))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "collection" / "build_minari.py"),
            "--raw",
            str(raw),
            "--dataset-id",
            "review/rejected-v0",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "outside declared" in result.stderr
    assert not (datasets / "review" / "rejected-v0").exists()
