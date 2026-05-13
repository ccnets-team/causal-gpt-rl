"""Decorate env-side SpaceSpecs into model-side DataSpec lists.

Given the env's `state_specs` / `action_specs` (slim SpaceSpec form,
either from `extract_data_specs_from_space` or a saved bundle), produce
the full `(input_specs, output_specs)` DataSpec lists that the model
constructs its I/O heads, adapters, and routing from.

Layout (architecture convention):
- `input_specs`  = state heads + action-mean heads + is_bos indicator
- `output_specs` = action-mean heads + action-log_std heads (continuous only) + value head

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import torch

from .schema import DataSpec, SpaceSpec


def _to_data_spec(
    spec: SpaceSpec,
    *,
    role: str,
    sub_role: str | None = None,
    init_type: str | None = None,
    type_override: str | None = None,
    drop_bounds: bool = False,
) -> DataSpec:
    base = asdict(spec)
    base["dtype"] = torch.float32
    base["role"] = role
    base["sub_role"] = sub_role
    base["init_type"] = init_type
    if type_override is not None:
        base["type"] = type_override
    if drop_bounds:
        base["low"] = None
        base["high"] = None
        base["squash"] = None
    return DataSpec(**base)


def build_model_specs(
    state_specs: Iterable[SpaceSpec],
    action_specs: Iterable[SpaceSpec],
) -> tuple[list[DataSpec], list[DataSpec]]:
    """Decorate env-side SpaceSpecs and return (input_specs, output_specs)."""
    state_specs = list(state_specs)
    action_specs = list(action_specs)

    decorated_state_specs = [
        _to_data_spec(s, role="state", init_type="xavier_uniform")
        for s in state_specs
    ]

    mean_action_specs = [
        _to_data_spec(
            s,
            role="action",
            sub_role="mean",
            init_type="normal",
            drop_bounds=(s.type != "continuous"),
        )
        for s in action_specs
    ]

    log_std_action_specs = [
        _to_data_spec(
            s,
            role="action",
            sub_role="log_std",
            init_type="log_std",
            type_override="continuous",
        )
        for s in action_specs
        if s.type == "continuous"
    ]

    value_spec = DataSpec(
        type="continuous",
        size=1,
        dtype=torch.float32,
        init_type="xavier_uniform",
        role="value",
        sub_role=None,
    )

    is_bos_spec = DataSpec(
        type="continuous",
        size=1,
        dtype=torch.float32,
        init_type="xavier_uniform",
        role="bos_indicator",
        sub_role=None,
    )

    input_specs = list(decorated_state_specs) + list(mean_action_specs) + [is_bos_spec]
    output_specs = list(mean_action_specs) + list(log_std_action_specs) + [value_spec]
    return input_specs, output_specs


__all__ = ["build_model_specs"]
