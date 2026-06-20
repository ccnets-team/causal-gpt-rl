"""Gym space to SpaceSpec extraction for PolicyRunner construction.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from ..model.schema import SpaceSpec, continuous_first_order


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


# ---------------------------------------------------------------------------
# L2 API — gymnasium space (de)serialization + continuous-first permutation
#
# FROZEN CONTRACT. Single owner = serving (this package); the trainer repo
# imports these. The signatures and the direction conventions below are
# authoritative — see the cross-repo note
#   .local/docs/dev/model-output/instruction/l2-api-contract.md
# for the worked example both repos verify against.
#
# Scope: vector spaces only — Box (1-D), Discrete, MultiDiscrete and their
# Tuple/Dict nesting (mirrors `extract_data_specs_from_space`). Image Box
# (n-D) and Sequence are out of scope.
# ---------------------------------------------------------------------------

_INF_TOKENS = {"inf": np.inf, "-inf": -np.inf, "nan": np.nan}


def _encode_bounds(arr) -> list:
    """Box bound array -> JSON-safe flat list (±inf / nan -> string tokens)."""
    out: list = []
    for v in np.asarray(arr, dtype=np.float64).ravel().tolist():
        if v == np.inf:
            out.append("inf")
        elif v == -np.inf:
            out.append("-inf")
        elif v != v:  # NaN
            out.append("nan")
        else:
            out.append(float(v))
    return out


def _decode_bounds(values, shape) -> np.ndarray:
    flat = [_INF_TOKENS[v] if isinstance(v, str) else float(v) for v in values]
    return np.asarray(flat, dtype=np.float64).reshape(shape)


def serialize_space(space: gym.spaces.Space) -> dict:
    """Serialize a gymnasium space to a JSON-safe dict (lossless schema).

    `gym.spaces.flatten` is lossy (values only); the *structure* lives in the
    `space`. This captures that structure — Box shape/bounds/dtype, Discrete
    n/start, MultiDiscrete nvec, and crucially **Dict key order / Tuple order**
    — so `deserialize_space` rebuilds a space that flattens/unflattens
    identically (the schema `gym.spaces.unflatten` requires).

    Dict is emitted as an ordered list of `[key, subspace]` pairs so key order
    survives the JSON round-trip regardless of dict-ordering quirks.

    Invariant: for any sample `x`,
        flatten(space, x) == flatten(deserialize_space(serialize_space(space)), x)
    """
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": _encode_bounds(space.low),
            "high": _encode_bounds(space.high),
        }
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n), "start": int(space.start)}
    if isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": [int(n) for n in np.asarray(space.nvec).ravel().tolist()],
            "dtype": str(space.dtype),
        }
    if isinstance(space, gym.spaces.Tuple):
        return {"type": "Tuple", "spaces": [serialize_space(s) for s in space.spaces]}
    if isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": [
                [key, serialize_space(sub)] for key, sub in space.spaces.items()
            ],
        }
    raise ValueError(f"Unsupported space type for serialization: {type(space)}")


def deserialize_space(payload: dict) -> gym.spaces.Space:
    """Inverse of :func:`serialize_space` (preserves Dict / Tuple order)."""
    kind = payload["type"]
    if kind == "Box":
        shape = tuple(payload["shape"])
        return gym.spaces.Box(
            low=_decode_bounds(payload["low"], shape),
            high=_decode_bounds(payload["high"], shape),
            shape=shape,
            dtype=np.dtype(payload["dtype"]),
        )
    if kind == "Discrete":
        return gym.spaces.Discrete(
            n=int(payload["n"]), start=int(payload.get("start", 0))
        )
    if kind == "MultiDiscrete":
        return gym.spaces.MultiDiscrete(
            np.asarray(payload["nvec"], dtype=np.int64),
            dtype=np.dtype(payload.get("dtype", "int64")),
        )
    if kind == "Tuple":
        return gym.spaces.Tuple([deserialize_space(s) for s in payload["spaces"]])
    if kind == "Dict":
        return gym.spaces.Dict(
            OrderedDict(
                (key, deserialize_space(sub)) for key, sub in payload["spaces"]
            )
        )
    raise ValueError(f"Unknown serialized space type: {kind!r}")


@dataclass(frozen=True)
class ContinuousFirst:
    """Deterministic continuous-first reordering derived from a spec list.

    Conventions (authoritative — both repos must agree byte-for-byte):
      * **declared order** — the order `extract_data_specs_from_space` /
        `gym.spaces.flatten` walk the space (Dict = key order).
      * **canonical order** — every `continuous` block first, then the discrete
        (one-hot) blocks; order *within* each group stays declared order
        (stable; no arbitrary sort).

    Fields:
      block_perm     canonical_specs[i] == specs[block_perm[i]]       (block level)
      flat_perm      canonical_flat[i]  == declared_flat[flat_perm[i]]    (scalar)
      inv_flat_perm  declared_flat[i]   == canonical_flat[inv_flat_perm[i]]
      n_cont         # continuous scalar dims in canonical_flat; equals the
                     normalize split index ([:n_cont] normalize, [n_cont:] pass).
    """

    block_perm: list
    flat_perm: list
    inv_flat_perm: list
    n_cont: int


def derive_continuous_first(specs: list[SpaceSpec]) -> ContinuousFirst:
    """Compute the :class:`ContinuousFirst` reordering for `specs`.

    Pure function of the spec list (deterministic, nothing stored in the bundle
    — both repos recompute). `continuous` specs form the front block;
    `discrete` / `multi_discrete` (one-hot) specs form the tail block.

    Worked example (see l2-api-contract.md):
        specs types/sizes = [cont(3), disc(4), cont(2), disc(3)]
        -> block_perm    = [0, 2, 1, 3]
           flat_perm     = [0,1,2, 7,8, 3,4,5,6, 9,10,11]
           inv_flat_perm = [0,1,2, 5,6,7,8, 3,4, 9,10,11]
           n_cont        = 5
    """
    sizes = [int(s.size) for s in specs]
    starts: list = []
    acc = 0
    for sz in sizes:
        starts.append(acc)
        acc += sz

    # Same ordering the model uses for its state heads (single source of truth)
    # so the head split and this permutation agree by construction.
    block_perm = continuous_first_order(specs)

    flat_perm: list = []
    for b in block_perm:
        flat_perm.extend(range(starts[b], starts[b] + sizes[b]))

    inv_flat_perm = [0] * len(flat_perm)
    for canon_idx, decl_idx in enumerate(flat_perm):
        inv_flat_perm[decl_idx] = canon_idx

    n_cont = sum(int(s.size) for s in specs if s.type == "continuous")
    return ContinuousFirst(
        block_perm=block_perm,
        flat_perm=flat_perm,
        inv_flat_perm=inv_flat_perm,
        n_cont=n_cont,
    )


def flatten_to_model(
    space: gym.spaces.Space, x, cf: ContinuousFirst
) -> np.ndarray:
    """Structured sample `x` -> model's flat input (continuous-first, float32).

    `gym.spaces.flatten(space, x)` (declared order) then `cf.flat_perm`. `cf`
    must be `derive_continuous_first(extract_data_specs_from_space(space))` for
    this same `space` (derive once, reuse).
    """
    declared = np.asarray(gym.spaces.flatten(space, x), dtype=np.float32)
    if declared.shape[-1] != len(cf.flat_perm):
        raise ValueError(
            f"flatdim {declared.shape[-1]} != permutation length "
            f"{len(cf.flat_perm)} — cf was derived from a different space."
        )
    return declared[np.asarray(cf.flat_perm, dtype=np.intp)]


def unflatten_from_model(
    space: gym.spaces.Space, flat: np.ndarray, cf: ContinuousFirst
):
    """Model's flat output (canonical order) -> customer's declared container.

    `cf.inv_flat_perm` (canonical -> declared) then `gym.spaces.unflatten`.
    Returns the customer's declared structure (Box / Discrete / Tuple / Dict).
    """
    canonical = np.asarray(flat)
    if canonical.shape[-1] != len(cf.inv_flat_perm):
        raise ValueError(
            f"flat length {canonical.shape[-1]} != permutation length "
            f"{len(cf.inv_flat_perm)} — cf was derived from a different space."
        )
    declared = canonical[np.asarray(cf.inv_flat_perm, dtype=np.intp)]
    return gym.spaces.unflatten(space, declared)


__all__ = [
    "extract_data_specs_from_space",
    "serialize_space",
    "deserialize_space",
    "ContinuousFirst",
    "derive_continuous_first",
    "flatten_to_model",
    "unflatten_from_model",
]
