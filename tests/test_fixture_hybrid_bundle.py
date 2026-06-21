"""Byte-oracle check against the trainer's delivered hybrid bundle fixture.

The trainer ships a real exported bundle plus a `meta.json` oracle under
`.local/fixtures/` (gitignored, hand-delivered). This test cross-checks the
serving P4 input adapter (and the L2 output primitive) against that oracle:

  * adapter(raw_obs) == expected_flat        (gym.flatten + continuous-first)
  * normalize_once(expected_flat) == expected_norm  (no double-normalize: v2
    embeds normalization, applied once inside the runtime, not in the adapter)
  * load_runner + reset/act runs end-to-end on the real bundle
  * unflatten_from_model(model_action) == expected_unflat  (P5 primitive)

Skips cleanly when the fixture is absent (CI, fresh clone) — the fixture lives
outside git on purpose.
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.adapters import StateInputAdapter
from causal_gpt_rl.inference.spaces import deserialize_space

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / ".local"
    / "fixtures"
    / "hybrid-bundle-dict-box3-disc4"
)


def _load_meta() -> dict:
    meta_path = _FIXTURE_DIR / "meta.json"
    if not meta_path.is_file():
        pytest.skip(f"hybrid bundle fixture not present at {_FIXTURE_DIR}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _to_observation(raw_obs: dict) -> dict:
    # JSON gives lists/ints; gym.flatten wants array-likes for Box, ints for Discrete.
    return {
        "pos": np.asarray(raw_obs["pos"], dtype=np.float32),
        "kind": int(raw_obs["kind"]),
    }


def test_adapter_matches_expected_flat():
    meta = _load_meta()
    obs_space = deserialize_space(meta["obs_space"])
    adapter = StateInputAdapter(obs_space)

    assert adapter.flatdim == meta["canonical"]["flatdim"]
    assert adapter.cf.n_cont == meta["canonical"]["n_cont"]

    for sample in meta["samples"]:
        out = adapter(_to_observation(sample["raw_obs"]))
        np.testing.assert_allclose(
            out, np.asarray(sample["expected_flat"], dtype=np.float32), atol=1e-6
        )


def test_normalize_once_matches_expected_norm():
    # Guards the double-normalize hazard: applying the bundle's embedded
    # normalization exactly once to expected_flat must reproduce expected_norm.
    meta = _load_meta()
    runner = bundle.load_runner(_FIXTURE_DIR)
    model = runner.model
    assert model.has_embedded_state_normalizer()

    for sample in meta["samples"]:
        flat = torch.tensor(
            sample["expected_flat"], dtype=torch.float32
        ).view(1, 1, -1)
        normed = model.normalize_states_for_inference(flat).view(-1).cpu().numpy()
        np.testing.assert_allclose(
            normed, np.asarray(sample["expected_norm"], dtype=np.float32), atol=1e-5
        )


def test_load_and_run_end_to_end():
    meta = _load_meta()
    runner = bundle.load_runner(_FIXTURE_DIR)
    assert runner._input_adapter is not None

    first = _to_observation(meta["samples"][0]["raw_obs"])
    runner.reset(first)
    action = runner.act()
    assert np.asarray(action).shape == tuple(meta["action_space"]["shape"])

    second = _to_observation(meta["samples"][1]["raw_obs"])
    action2 = runner.act(second)
    assert np.asarray(action2).shape == tuple(meta["action_space"]["shape"])


def test_unflatten_matches_expected_unflat():
    # P5 output primitive against the oracle. Per the corrected L2 contract §2,
    # action output is DECLARED order (output == next input, AR self-feedback) —
    # restore the container with plain gym.spaces.unflatten, NO continuous-first
    # permutation. (Here action is a plain Box, so it's identity anyway; the real
    # per-head container structuring lands with P5.)
    meta = _load_meta()
    action_space = deserialize_space(meta["action_space"])

    for sample in meta["action_samples"]:
        restored = gym.spaces.unflatten(
            action_space, np.asarray(sample["model_action"], dtype=np.float32)
        )
        np.testing.assert_allclose(
            np.asarray(restored, dtype=np.float32),
            np.asarray(sample["expected_unflat"], dtype=np.float32),
            atol=1e-6,
        )
