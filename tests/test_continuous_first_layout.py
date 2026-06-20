"""P3 tests — continuous-first canonical state-head layout.

Pins the lynchpin: the model's state-head order (build_model_specs) and the
inference adapter's permutation (derive_continuous_first) come from the SAME
shared `continuous_first_order`, so they agree by construction. Identity for
all-continuous state keeps existing bundles byte-unchanged.
"""
from causal_gpt_rl.inference.spaces import derive_continuous_first
from causal_gpt_rl.model.schema import SpaceSpec, continuous_first_order
from causal_gpt_rl.model.spec_factory import build_model_specs


def _mixed_state():
    return [
        SpaceSpec(type="continuous", size=3),
        SpaceSpec(type="discrete", size=4),
        SpaceSpec(type="continuous", size=2),
        SpaceSpec(type="discrete", size=3),
    ]


_ACTION = [SpaceSpec(type="continuous", size=2, squash="tanh")]


def _state_heads(state_specs):
    input_specs, _ = build_model_specs(state_specs, _ACTION)
    return [s for s in input_specs if s.role == "state"]


def test_continuous_first_order_logic():
    assert continuous_first_order(_mixed_state()) == [0, 2, 1, 3]
    # identity for all-continuous
    assert continuous_first_order(
        [SpaceSpec(type="continuous", size=2), SpaceSpec(type="continuous", size=5)]
    ) == [0, 1]


def test_build_model_specs_reorders_state_continuous_first():
    heads = _state_heads(_mixed_state())
    assert [s.type for s in heads] == [
        "continuous",
        "continuous",
        "discrete",
        "discrete",
    ]
    assert [s.size for s in heads] == [3, 2, 4, 3]


def test_build_model_specs_identity_for_continuous_state():
    # Regression: pure-continuous state head order is unchanged (declared order),
    # so existing continuous bundles keep a byte-identical layout.
    heads = _state_heads(
        [SpaceSpec(type="continuous", size=2), SpaceSpec(type="continuous", size=3)]
    )
    assert [s.size for s in heads] == [2, 3]


def test_model_head_order_matches_adapter_permutation():
    specs = _mixed_state()
    heads = _state_heads(specs)
    cf = derive_continuous_first(specs)
    # The model's state-head sizes follow the same block order the adapter uses.
    assert [s.size for s in heads] == [specs[i].size for i in cf.block_perm]
    # n_cont is the contiguous front continuous width.
    front = 0
    for s in heads:
        if s.type == "continuous":
            front += s.size
        else:
            break
    assert cf.n_cont == front == 5
