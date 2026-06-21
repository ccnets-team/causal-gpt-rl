"""Adapters bridging structured Gymnasium I/O and the model's flat tensors.

The model core speaks a single flat vector per step: state in continuous-first
*canonical* order, action as a flat per-head concatenation. Customers, however,
declare structured Gymnasium spaces (Dict / Tuple / Discrete / MultiDiscrete).
These adapters sit at the :class:`PolicyRunner` boundary and translate between
the two without leaking structure into the model core.

  * input  (P4): structured env observation -> canonical flat model state, via
                 the L2 :func:`flatten_to_model` (``gym.flatten`` then the
                 continuous-first permutation). Normalization happens downstream
                 in the runner; the embedded stats are themselves canonical with
                 the one-hot tail baked to identity, so block *order* is all this
                 layer must get right (flatten -> normalize, decision 6).
  * output (P5): model action -> declared container.  [ships with P5]

A plain Box obs space needs no structuring — ``gym.flatten`` of a 1-D Box is a
raveling concat and the continuous-first permutation is identity — so
:func:`make_state_input_adapter` returns ``None`` for it and the runner keeps its
byte-identical raw-passthrough path.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from .spaces import (
    ContinuousFirst,
    derive_continuous_first,
    extract_data_specs_from_space,
    flatten_to_model,
)


class StateInputAdapter:
    """Structured env observation -> canonical flat model state vector.

    Holds the declared :class:`gym.spaces.Space` and the
    :class:`ContinuousFirst` reordering derived from it once at construction, so
    each step is a single ``gym.flatten`` + permutation (no per-step rederive).
    """

    def __init__(self, obs_space: gym.spaces.Space):
        self.obs_space = obs_space
        self.specs = extract_data_specs_from_space(obs_space)
        self.cf: ContinuousFirst = derive_continuous_first(self.specs)
        # Flat width the model expects == sum of spec sizes == permutation length.
        self.flatdim = len(self.cf.flat_perm)

    def __call__(self, observation) -> np.ndarray:
        """One structured observation -> ``(flatdim,)`` float32 canonical vector."""
        return flatten_to_model(self.obs_space, observation, self.cf)


def make_state_input_adapter(
    obs_space: Optional[gym.spaces.Space],
) -> Optional[StateInputAdapter]:
    """Build the input adapter, or ``None`` when raw passthrough is correct.

    Returns ``None`` for a missing space or a plain Box (pure-continuous,
    structure-less state): the runner then feeds the raw flat vector unchanged,
    byte-identical to the legacy continuous path. Any structured container
    (Dict / Tuple / Discrete / MultiDiscrete) gets a real adapter.
    """
    if obs_space is None or isinstance(obs_space, gym.spaces.Box):
        return None
    return StateInputAdapter(obs_space)


__all__ = ["StateInputAdapter", "make_state_input_adapter"]
