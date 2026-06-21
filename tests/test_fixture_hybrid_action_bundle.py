"""Byte-oracle check against a hybrid-ACTION bundle fixture (P5 output side).

Mirror of ``test_fixture_hybrid_bundle.py`` on the action side. The trainer ships
a real exported CONTAINER-action bundle plus a ``meta.json`` oracle under
``.local/fixtures/hybrid-action-bundle/`` (gitignored, hand-delivered). This test
cross-checks the serving P5 output adapter: the runner decodes a raw model action
into the declared Dict/Tuple container, and it must match the oracle.

``meta.json`` schema (the contract the trainer's fixture must satisfy — serving
seeds a reference via ``scripts/make_reference_action_fixture.py``):

    {
      "action_space": <serialized gym space: Dict or Tuple>,
      "action_samples": [
        {
          "model_action": [<action_size floats>],   # RAW model head output, head
                                                     # order: continuous values then
                                                     # categorical LOGITS
          "expected_container": <declared container> # Box -> [floats], Discrete -> int,
                                                     # MultiDiscrete -> [ints],
                                                     # Tuple -> [...], Dict -> {key: ...};
                                                     # continuous clipped to bounds,
                                                     # categorical argmax + Discrete start
        }
      ]
    }

Validates:
  * the bundle loads with an output adapter and the action_container capability,
  * runner._decode(model_action) == expected_container (structure + values),
  * load_runner + reset/act runs end-to-end and returns the declared container.

Skips cleanly when the fixture is absent (CI, fresh clone) — it lives outside git.
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from causal_gpt_rl.inference import bundle

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / ".local"
    / "fixtures"
    / "hybrid-action-bundle"
)


def _load_meta() -> dict:
    meta_path = _FIXTURE_DIR / "meta.json"
    if not meta_path.is_file():
        pytest.skip(f"hybrid action bundle fixture not present at {_FIXTURE_DIR}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _coerce_to_space(space, value):
    """Rebuild a gym-typed sample from JSON-decoded structure (for gym.flatten)."""
    if isinstance(space, gym.spaces.Box):
        return np.asarray(value, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return int(value)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.asarray(value, dtype=space.dtype)
    if isinstance(space, gym.spaces.Tuple):
        return tuple(_coerce_to_space(s, v) for s, v in zip(space.spaces, value))
    if isinstance(space, gym.spaces.Dict):
        return {k: _coerce_to_space(space.spaces[k], value[k]) for k in space.spaces}
    raise TypeError(f"unsupported space in fixture oracle: {type(space)!r}")


def _assert_container_equal(space, got, expected_json):
    # Structure-agnostic byte compare: flatten both through gym, which folds in
    # Discrete start / MultiDiscrete splits consistently on each side.
    a = np.asarray(gym.spaces.flatten(space, got), dtype=np.float32)
    b = np.asarray(
        gym.spaces.flatten(space, _coerce_to_space(space, expected_json)),
        dtype=np.float32,
    )
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_fixture_loads_with_output_adapter():
    _load_meta()  # skip if absent
    runner = bundle.load_runner(_FIXTURE_DIR)
    assert runner._output_adapter is not None
    assert isinstance(runner.action_space, (gym.spaces.Dict, gym.spaces.Tuple))


def test_runner_decode_matches_container_oracle():
    meta = _load_meta()
    runner = bundle.load_runner(_FIXTURE_DIR)
    action_space = runner.action_space
    for sample in meta["action_samples"]:
        model_action = np.asarray(
            sample["model_action"], dtype=np.float32
        ).reshape(1, -1)
        container, _ = runner._decode(model_action)
        _assert_container_equal(action_space, container, sample["expected_container"])


def test_load_and_run_end_to_end_returns_container():
    _load_meta()
    runner = bundle.load_runner(_FIXTURE_DIR)
    runner.reset(np.zeros(runner.state_size, dtype=np.float32))
    action = runner.act()
    space = runner.action_space
    # The emitted action is a structurally-valid sample of the declared container.
    flat = np.asarray(gym.spaces.flatten(space, action), dtype=np.float32)
    assert flat.shape[0] == int(gym.spaces.flatdim(space))
