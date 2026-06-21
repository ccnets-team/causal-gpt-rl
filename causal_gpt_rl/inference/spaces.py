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


# Vector spaces this package handles end-to-end (flatten/unflatten + a model
# head type). Mirrors the branches in `extract_data_specs_from_space` and
# `serialize_space`.
_SUPPORTED_SPACE_TYPES = (
    "Box (1-D)", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict"
)

# Known gymnasium leaf spaces deliberately out of scope (no model head), each
# with the reason it has none — surfaced in the error so the boundary explains
# itself instead of leaving a bare `type(...)` repr. ASCII-only: these strings
# reach raised errors, which may print on a non-UTF-8 console (e.g. Windows).
_OUT_OF_SCOPE_SPACE_HINTS = {
    "Text": "variable-length token sequences",
    "Sequence": "variable-length sequences",
    "Graph": "graph-structured leaves",
}
# `image (n-D Box)` is not a distinct gymnasium class, so it cannot be matched by
# `type(...).__name__`; it is only named in the out-of-scope summary.
_OUT_OF_SCOPE_SUMMARY = ", ".join(list(_OUT_OF_SCOPE_SPACE_HINTS) + ["image (n-D Box)"])


def _unsupported_space_error(space: gym.spaces.Space, *, action: str) -> ValueError:
    """Build a self-describing error for a space outside the supported set.

    Names the offending type (with why it is out of scope, when known), then the
    full supported set, so a customer hitting an unsupported space learns what to
    do instead of seeing a bare type repr.
    """
    name = type(space).__name__
    hint = _OUT_OF_SCOPE_SPACE_HINTS.get(name)
    reason = f" ({hint})" if hint else ""
    return ValueError(
        f"Unsupported space type to {action}: {name}{reason}. "
        f"Supported: {', '.join(_SUPPORTED_SPACE_TYPES)}. "
        f"Out of scope: {_OUT_OF_SCOPE_SUMMARY}."
    )


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

    elif isinstance(space, gym.spaces.MultiBinary):
        # One head of `n` independent Bernoulli leaves. `gym.spaces.flatten`
        # leaves a MultiBinary as its `n`-vector of {0,1} (no one-hot blowup,
        # unlike Discrete/MultiDiscrete), so the head size is `n`. Bounds are
        # [0, 1] but unused downstream (dropped for non-continuous action heads;
        # the decode thresholds logits rather than clipping).
        n = int(np.prod(space.shape))
        specs.append(
            SpaceSpec(
                type="multi_binary",
                size=n,
                dtype=np.int8,
                low=np.zeros((n,), dtype=np.float64),
                high=np.ones((n,), dtype=np.float64),
            )
        )

    elif isinstance(space, gym.spaces.Tuple):
        for subspace in space.spaces:
            specs.extend(extract_data_specs_from_space(subspace))

    elif isinstance(space, gym.spaces.Dict):
        for key in space.spaces:
            specs.extend(extract_data_specs_from_space(space.spaces[key]))

    else:
        raise _unsupported_space_error(space, action="extract specs from")

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
# Scope: vector spaces only — Box (1-D), Discrete, MultiDiscrete, MultiBinary
# and their Tuple/Dict nesting (mirrors `extract_data_specs_from_space`). Image
# Box (n-D), Text, Sequence and Graph are out of scope.
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
        payload = {
            "type": "MultiDiscrete",
            "nvec": [int(n) for n in np.asarray(space.nvec).ravel().tolist()],
            "dtype": str(space.dtype),
        }
        # Per-dimension `start` offsets, like Discrete's. Emit only when any is
        # non-zero so existing all-zero MultiDiscrete configs stay byte-identical
        # and old loaders (which ignore the key) round-trip unchanged.
        start = getattr(space, "start", None)
        if start is not None:
            start = [int(s) for s in np.asarray(start).ravel().tolist()]
            if any(start):
                payload["start"] = start
        return payload
    if isinstance(space, gym.spaces.MultiBinary):
        # Scope: 1-D MultiBinary(n). `n` is the leaf count; flatten/unflatten is
        # the identity {0,1} n-vector, so n alone round-trips losslessly.
        return {"type": "MultiBinary", "n": int(np.prod(space.shape))}
    if isinstance(space, gym.spaces.Tuple):
        return {"type": "Tuple", "spaces": [serialize_space(s) for s in space.spaces]}
    if isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": [
                [key, serialize_space(sub)] for key, sub in space.spaces.items()
            ],
        }
    raise _unsupported_space_error(space, action="serialize")


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
        nvec = np.asarray(payload["nvec"], dtype=np.int64)
        dtype = np.dtype(payload.get("dtype", "int64"))
        start = payload.get("start")  # absent on pre-start bundles -> default 0
        if start is not None:
            return gym.spaces.MultiDiscrete(
                nvec, dtype=dtype, start=np.asarray(start, dtype=np.int64)
            )
        return gym.spaces.MultiDiscrete(nvec, dtype=dtype)
    if kind == "MultiBinary":
        return gym.spaces.MultiBinary(int(payload["n"]))
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
    """Canonical flat (continuous-first) -> customer's declared container.

    The pure inverse of `flatten_to_model`: `cf.inv_flat_perm` (canonical ->
    declared) then `gym.spaces.unflatten`. Use it to undo a continuous-first
    encoding (state, or anything the model emits in canonical order).

    NOT the action-output path. The model's action heads stay in *declared*
    order (output == the next input, AR self-feedback — see l2-api-contract.md
    §2), so a declared per-head action is restored with plain
    `gym.spaces.unflatten(space, x)`, no permutation. A non-identity `cf` here
    would mis-permute it.
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
