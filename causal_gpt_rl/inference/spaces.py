"""Gym space to SpaceSpec extraction for PolicyRunner construction.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from typing import Optional

import gymnasium as gym
import numpy as np

from ..model.schema import SpaceSpec


def _get_squash_from_box_space(low: np.ndarray, high: np.ndarray) -> Optional[str]:
    """
    Pick a squash for continuous Box spaces. Returns None when bounds are
    not finite, otherwise tanh.
    """
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)

    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        return None

    return "tanh"


def extract_data_specs_from_space(space: gym.spaces.Space) -> list[SpaceSpec]:
    """
    Recursively extract SpaceSpec objects from a Gym space.
    Each leaf space produces exactly one SpaceSpec.
    """
    specs = []

    if isinstance(space, gym.spaces.Box):
        size = int(space.shape[-1])
        low = np.atleast_1d(space.low if space.low.ndim <= 1 else space.low[-1])
        high = np.atleast_1d(space.high if space.high.ndim <= 1 else space.high[-1])

        squash = _get_squash_from_box_space(low, high)

        specs.append(
            SpaceSpec(
                type="continuous",
                squash=squash,
                size=size,
                dtype=space.dtype,
                low=low,
                high=high,
            )
        )

    elif isinstance(space, gym.spaces.Discrete):
        specs.append(
            SpaceSpec(
                type="discrete",
                size=space.n,
                dtype=np.int64,
                low=np.full((space.n,), -np.inf),
                high=np.full((space.n,), np.inf),
            )
        )

    elif isinstance(space, gym.spaces.MultiDiscrete):
        for n_i in space.nvec:
            n_i = int(n_i)
            specs.append(
                SpaceSpec(
                    type="multi_discrete",
                    size=n_i,
                    dtype=np.int64,
                    low=np.full((n_i,), -np.inf, dtype=np.float64),
                    high=np.full((n_i,), np.inf, dtype=np.float64),
                )
            )

    elif isinstance(space, gym.spaces.Tuple):
        for subspace in space.spaces:
            specs.extend(extract_data_specs_from_space(subspace))

    elif isinstance(space, gym.spaces.Dict):
        for key in space.spaces:
            specs.extend(extract_data_specs_from_space(space.spaces[key]))

    else:
        raise ValueError(f"Unsupported space type: {type(space)}")

    return specs


__all__ = ["extract_data_specs_from_space"]
