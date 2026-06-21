"""Generate a REFERENCE hybrid-action bundle fixture for the P5 byte-verify test.

Serving-side self-test + an exact template for the trainer's real fixture
(consumed by ``tests/test_fixture_hybrid_action_bundle.py``). Produces a
container-action bundle ``Tuple(Box(2), Discrete(3))`` under
``.local/fixtures/hybrid-action-bundle/`` with a ``meta.json`` oracle computed
INDEPENDENTLY of the runner (continuous clipped to bounds, categorical argmax +
Discrete start). The runner must reproduce that oracle.

The trainer's real fixture — same dir, same schema, but exported from a *trained*
model — supersedes this reference. Because the oracle is derived directly from
``model_action`` (not the model's forward pass), the random-init weights here are
irrelevant: the test drives ``runner._decode(model_action)`` and bypasses the model.

This also doubles as an integration check for the (D) buffer-alias fix: it exports
a bounded-tanh continuous action from float32 numpy specs, which safetensors
refused to save before that fix.

Run: ``python scripts/make_reference_action_fixture.py``

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.spaces import (
    extract_data_specs_from_space,
    serialize_space,
)
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_OUT = (
    Path(__file__).resolve().parents[1] / ".local" / "fixtures" / "hybrid-action-bundle"
)
_CFG = ModelConfig(d_model=32, num_heads=4)


class _Norm:
    """Minimal duck-typed state normalizer (mean/var state_dict)."""

    def __init__(self, n: int):
        self.n = n

    def state_dict(self):
        return {"mean": torch.zeros(self.n), "var": torch.ones(self.n)}


def _independent_oracle(action_space: gym.spaces.Tuple, model_action: np.ndarray):
    """expected_container, hand-computed (NOT via the runner).

    Continuous head clipped to its Box bounds; categorical head = argmax of its
    logits + the Discrete start. This is the env-facing decode contract serving
    must match.
    """
    box, disc = action_space.spaces
    cont = np.clip(model_action[:2], box.low, box.high).astype(np.float32)
    logits = model_action[2 : 2 + int(disc.n)]
    cls = int(np.argmax(logits)) + int(disc.start)
    return [cont.tolist(), cls]


def main() -> None:
    action_space = gym.spaces.Tuple(
        (gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32), gym.spaces.Discrete(3))
    )
    state_size = 2
    action_specs = extract_data_specs_from_space(action_space)
    state_specs = [
        SpaceSpec(type="continuous", size=state_size, dtype=torch.float32,
                  low=[-1.0] * state_size, high=[1.0] * state_size)
    ]
    model = AutoregressiveModel(
        _CFG, state_specs=state_specs, action_specs=action_specs,
        device=torch.device("cpu"),
    )

    _OUT.mkdir(parents=True, exist_ok=True)
    bundle.export_bundle(
        _OUT, model=model, model_config=_CFG,
        state_specs=model.state_specs, action_specs=model.action_specs,
        context_length=8, action_space=action_space, state_normalizer=_Norm(state_size),
    )

    # Raw model head outputs: [cont(2) values | discrete(3) logits].
    raw_actions = [
        [0.30, -0.70, 0.1, 0.9, 0.2],   # in-bounds, argmax class 1
        [1.50, -2.00, 0.5, 0.1, 0.9],   # continuous clipped to [-1, 1], argmax class 2
        [0.00, 0.00, 0.9, 0.1, 0.1],    # argmax class 0
    ]
    samples = []
    for ra in raw_actions:
        arr = np.asarray(ra, dtype=np.float32)
        samples.append({
            "model_action": arr.tolist(),
            "expected_container": _independent_oracle(action_space, arr),
        })

    meta = {
        "fixture": "hybrid-action-bundle",
        "note": (
            "REFERENCE/self-test fixture produced by serving "
            "(scripts/make_reference_action_fixture.py). model_action = RAW model "
            "head output (continuous values then categorical LOGITS, head order). "
            "expected_container = declared container with continuous clipped to "
            "bounds and categorical argmax + Discrete start, computed INDEPENDENTLY "
            "of the runner. The trainer's real fixture (trained model) supersedes "
            "this; same dir, same schema."
        ),
        "action_space": serialize_space(action_space),
        "action_samples": samples,
    }
    (_OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote reference hybrid-action fixture to {_OUT}")


if __name__ == "__main__":
    main()
