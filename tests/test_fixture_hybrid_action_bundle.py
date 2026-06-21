"""Byte-oracle check against hybrid-ACTION bundle fixtures (P5 output side).

Mirror of ``test_fixture_hybrid_bundle.py`` on the action side. The trainer ships
a real exported CONTAINER-action bundle plus a ``meta.json`` oracle under
``.local/fixtures/<name>/`` (gitignored, hand-delivered). This test discovers any
such fixture and cross-checks the serving P5 output adapter: the runner decodes a
per-head flat model action into the declared Dict/Tuple container, and it must
match the oracle.

Fixtures are discovered by glob (not a hardcoded name) so a trainer-delivered
bundle (e.g. ``hybrid-bundle-dict-action-box2-disc3``) and serving's own
reference (``hybrid-action-bundle`` from ``scripts/make_reference_action_fixture.py``)
are both validated.

``meta.json`` schema (the contract the trainer's fixture satisfies):

    {
      "action_space": <serialized gym space: Dict or Tuple>,
      "action_samples": [
        {
          "model_action": [<action_size floats>],   # per-head flat the model emits,
                                                     # declared/gym.flatten order
          "expected_container": <declared container> # Box -> [floats], Discrete -> int,
                                                     # MultiDiscrete -> [ints],
                                                     # Tuple -> [...], Dict -> {key: ...}
        }
      ]
    }

Validation is ``runner._decode(model_action) == expected_container`` — the real
serving path. For in-bounds (tanh-squashed) continuous outputs this equals the
trainer's stated ``gym.spaces.unflatten(action_space, model_action)`` contract;
the runner additionally clips continuous heads to their bounds.

Skips cleanly when no fixture is present (CI, fresh clone).
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from causal_gpt_rl.inference import bundle

_FIXTURES_BASE = Path(__file__).resolve().parents[1] / ".local" / "fixtures"


def _discover_action_fixtures() -> list[Path]:
    """Fixture dirs whose meta.json carries a Dict/Tuple action oracle."""
    found: list[Path] = []
    if not _FIXTURES_BASE.is_dir():
        return found
    for meta_path in sorted(_FIXTURES_BASE.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        action_space = meta.get("action_space") or {}
        if "action_samples" in meta and action_space.get("type") in ("Dict", "Tuple"):
            found.append(meta_path.parent)
    return found


_ACTION_FIXTURES = _discover_action_fixtures()

pytestmark = pytest.mark.skipif(
    not _ACTION_FIXTURES,
    reason=f"no container-action fixture present under {_FIXTURES_BASE}",
)


def _coerce_to_space(space, value):
    """Rebuild a gym-typed sample from JSON-decoded structure (for gym.flatten)."""
    if isinstance(space, gym.spaces.Box):
        return np.asarray(value, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return int(value)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return np.asarray(value, dtype=space.dtype)
    if isinstance(space, gym.spaces.MultiBinary):
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


def _meta(fixture_dir: Path) -> dict:
    return json.loads((fixture_dir / "meta.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture_dir", _ACTION_FIXTURES, ids=lambda p: p.name)
def test_fixture_loads_with_output_adapter(fixture_dir):
    runner = bundle.load_runner(fixture_dir)
    assert runner._output_adapter is not None
    assert isinstance(runner.action_space, (gym.spaces.Dict, gym.spaces.Tuple))


@pytest.mark.parametrize("fixture_dir", _ACTION_FIXTURES, ids=lambda p: p.name)
def test_runner_decode_matches_container_oracle(fixture_dir):
    meta = _meta(fixture_dir)
    runner = bundle.load_runner(fixture_dir)
    for sample in meta["action_samples"]:
        model_action = np.asarray(
            sample["model_action"], dtype=np.float32
        ).reshape(1, -1)
        container, _ = runner._decode(model_action)
        _assert_container_equal(
            runner.action_space, container, sample["expected_container"]
        )


@pytest.mark.parametrize("fixture_dir", _ACTION_FIXTURES, ids=lambda p: p.name)
def test_load_and_run_end_to_end_returns_container(fixture_dir):
    runner = bundle.load_runner(fixture_dir)
    runner.reset(np.zeros(runner.state_size, dtype=np.float32))
    action = runner.act()
    space = runner.action_space
    # The emitted action is a structurally-valid sample of the declared container.
    flat = np.asarray(gym.spaces.flatten(space, action), dtype=np.float32)
    assert flat.shape[0] == int(gym.spaces.flatdim(space))
