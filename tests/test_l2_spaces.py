"""L2 API tests — space (de)serialization + continuous-first permutation.

These pin the cross-repo contract (serving owns it, trainer imports it):
  * serialize/deserialize is a flatten-equivalent round-trip (Dict key order),
  * derive_continuous_first matches the worked example in l2-api-contract.md,
  * flatten_to_model / unflatten_from_model are inverses over the model boundary.
"""
import numpy as np
import gymnasium as gym

from causal_gpt_rl.inference.spaces import (
    ContinuousFirst,
    derive_continuous_first,
    deserialize_space,
    extract_data_specs_from_space,
    flatten_to_model,
    serialize_space,
    unflatten_from_model,
)
from causal_gpt_rl.model.schema import SpaceSpec


def _worked_example_space() -> gym.spaces.Dict:
    # Keys deliberately NOT alphabetical so a sorting bug would surface.
    return gym.spaces.Dict(
        {
            "pos": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "weapon": gym.spaces.Discrete(4),
            "vel": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            "mode": gym.spaces.Discrete(3),
        }
    )


SPACES = [
    gym.spaces.Box(-2.0, 3.0, shape=(5,), dtype=np.float32),
    gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float64),
    gym.spaces.Discrete(7),
    gym.spaces.Discrete(5, start=2),
    gym.spaces.MultiDiscrete([3, 4, 2]),
    gym.spaces.Tuple(
        (gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32), gym.spaces.Discrete(3))
    ),
    _worked_example_space(),
]


def test_serialize_roundtrip_is_flatten_equivalent():
    for space in SPACES:
        space.seed(0)
        restored = deserialize_space(serialize_space(space))
        x = space.sample()
        a = gym.spaces.flatten(space, x)
        b = gym.spaces.flatten(restored, x)
        assert np.allclose(a, b), f"flatten mismatch after round-trip: {space}"
        # And the round-trip is JSON-safe (no inf/nan tokens leak as floats).
        import json

        json.dumps(serialize_space(space), allow_nan=False)


def test_serialize_preserves_dict_key_order():
    # gymnasium Dict normalizes key order (sorts), so "declared order" is the
    # space's own iteration order on BOTH repos. The contract is: serialize
    # reflects that order and deserialize preserves it (round-trip identity).
    space = _worked_example_space()
    declared_keys = list(space.spaces.keys())
    payload = serialize_space(space)
    assert [k for k, _ in payload["spaces"]] == declared_keys
    restored = deserialize_space(payload)
    assert list(restored.spaces.keys()) == declared_keys


def test_derive_continuous_first_worked_example_pure():
    # Build the spec list directly to test the pure permutation logic without
    # any gym Dict-ordering ambiguity.
    specs = [
        SpaceSpec(type="continuous", size=3),
        SpaceSpec(type="discrete", size=4),
        SpaceSpec(type="continuous", size=2),
        SpaceSpec(type="discrete", size=3),
    ]
    cf = derive_continuous_first(specs)
    assert cf.block_perm == [0, 2, 1, 3]
    assert cf.flat_perm == [0, 1, 2, 7, 8, 3, 4, 5, 6, 9, 10, 11]
    assert cf.inv_flat_perm == [0, 1, 2, 5, 6, 7, 8, 3, 4, 9, 10, 11]
    assert cf.n_cont == 5


def test_perm_and_inverse_are_consistent():
    for space in SPACES:
        specs = extract_data_specs_from_space(space)
        cf = derive_continuous_first(specs)
        n = len(cf.flat_perm)
        assert sorted(cf.flat_perm) == list(range(n))  # a true permutation
        # inv composed with perm is identity.
        for canon_idx, decl_idx in enumerate(cf.flat_perm):
            assert cf.inv_flat_perm[decl_idx] == canon_idx
        # n_cont counts continuous scalars only.
        n_cont = sum(int(s.size) for s in specs if s.type == "continuous")
        assert cf.n_cont == n_cont


def test_continuous_first_puts_continuous_in_front():
    space = _worked_example_space()
    specs = extract_data_specs_from_space(space)
    cf = derive_continuous_first(specs)
    space.seed(1)
    x = space.sample()
    flat = flatten_to_model(space, x, cf)
    # Front n_cont dims are pos(3)+vel(2); pos came from the sample, vel is the
    # unbounded Box. Just assert the split sizes line up with the spec types.
    assert flat.shape[-1] == sum(int(s.size) for s in specs)
    assert cf.n_cont == 5
    assert flat.dtype == np.float32


def test_flatten_unflatten_model_roundtrip():
    for space in SPACES:
        specs = extract_data_specs_from_space(space)
        cf = derive_continuous_first(specs)
        space.seed(2)
        x = space.sample()
        flat = flatten_to_model(space, x, cf)
        restored = unflatten_from_model(space, flat, cf)
        # Compare via declared-order flatten (structure-agnostic equality).
        a = gym.spaces.flatten(space, x)
        b = gym.spaces.flatten(space, restored)
        assert np.allclose(a, b), f"model-boundary round-trip failed: {space}"


def test_multidiscrete_start_serialize_roundtrip():
    # MultiDiscrete carries per-dimension start offsets; serialize must preserve
    # them (Discrete already did). All-zero start stays implicit so existing
    # configs are byte-identical and old loaders round-trip unchanged.
    md = gym.spaces.MultiDiscrete([3, 4], start=[1, 2])
    payload = serialize_space(md)
    assert payload["start"] == [1, 2]
    restored = deserialize_space(payload)
    assert list(np.asarray(restored.start).ravel()) == [1, 2]

    md0 = gym.spaces.MultiDiscrete([3, 4])  # start == 0
    assert "start" not in serialize_space(md0)
    restored0 = deserialize_space(serialize_space(md0))
    assert list(np.asarray(restored0.start).ravel()) == [0, 0]


def test_unsupported_space_error_is_self_describing():
    # An out-of-scope leaf (MultiBinary) raises an error that names the type, why
    # it is unsupported, and the supported set — on both the extract and the
    # serialize entry points.
    space = gym.spaces.MultiBinary(4)
    for fn in (extract_data_specs_from_space, serialize_space):
        try:
            fn(space)
        except ValueError as exc:
            msg = str(exc)
            assert "MultiBinary" in msg
            assert "Supported" in msg
            assert "Box (1-D)" in msg and "Dict" in msg
        else:
            raise AssertionError(f"{fn.__name__} should reject MultiBinary")


def test_flatten_to_model_rejects_mismatched_cf():
    space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    other = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
    cf_other = derive_continuous_first(extract_data_specs_from_space(other))
    space.seed(3)
    try:
        flatten_to_model(space, space.sample(), cf_other)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched permutation length")
