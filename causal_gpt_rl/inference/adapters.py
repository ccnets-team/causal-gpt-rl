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
  * output (P5): flat per-head model action -> the declared ``Dict`` / ``Tuple``
                 container, via ``gym.spaces.unflatten``. Actions are NOT
                 reordered continuous-first: the action heads stay in declared
                 order (output == the next input, AR self-feedback; L2 §2), so a
                 plain unflatten of the declared-order flat is exactly right.

A plain Box obs space needs no structuring — ``gym.flatten`` of a 1-D Box is a
raveling concat and the continuous-first permutation is identity — so
:func:`make_state_input_adapter` returns ``None`` for it and the runner keeps its
byte-identical raw-passthrough path. Symmetrically, a non-container action space
(Box / Discrete / MultiDiscrete) needs no restructuring — its flat per-head env
form already *is* the declared action — so :func:`make_action_output_adapter`
returns ``None`` and only ``Dict`` / ``Tuple`` actions get an output adapter.

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


class ActionOutputAdapter:
    """Flat per-head model action -> the declared Gymnasium container.

    Holds the declared (``Dict`` / ``Tuple``) ``action_space``. Each call takes a
    single env's action in ``gym.spaces.flatten`` convention — continuous leaves
    as (clipped) raw values, categorical leaves one-hot, concatenated in declared
    (head) order — and returns the structured sample via ``gym.spaces.unflatten``.

    No continuous-first permutation: action heads stay in declared order (output
    == the next input, AR self-feedback; L2 contract §2), so a plain unflatten of
    the declared-order flat is exactly right. ``gym.spaces.unflatten`` also applies
    each leaf's semantics for free — Discrete ``start`` offsets, MultiDiscrete
    splits, leaf shapes/dtypes — which a hand-rolled per-head mapping would have
    to re-derive.
    """

    def __init__(self, action_space: gym.spaces.Space):
        self.action_space = action_space
        # == sum of per-head sizes the model emits (Box dims + one-hot widths).
        self.flatdim = int(gym.spaces.flatdim(action_space))

    def __call__(self, env_flat):
        """One env's gym-flat action vector -> the declared container sample."""
        flat = np.asarray(env_flat, dtype=np.float32).reshape(-1)
        if flat.shape[0] != self.flatdim:
            raise ValueError(
                f"action flat width {flat.shape[0]} != action_space flatdim "
                f"{self.flatdim}; the bundle's declared space and action specs "
                f"disagree."
            )
        return gym.spaces.unflatten(self.action_space, flat)


def make_action_output_adapter(
    action_space: Optional[gym.spaces.Space],
) -> Optional[ActionOutputAdapter]:
    """Build the output adapter, or ``None`` when the flat action is the output.

    Returns ``None`` for a missing space or any non-container leaf
    (Box / Discrete / MultiDiscrete): the flat per-head env action already *is*
    the declared action, so the runner emits it unchanged (byte-identical legacy
    path). Only ``Dict`` / ``Tuple`` carry structure to restore.
    """
    if isinstance(action_space, (gym.spaces.Dict, gym.spaces.Tuple)):
        return ActionOutputAdapter(action_space)
    return None


__all__ = [
    "StateInputAdapter",
    "make_state_input_adapter",
    "ActionOutputAdapter",
    "make_action_output_adapter",
]
