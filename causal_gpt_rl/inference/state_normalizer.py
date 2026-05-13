"""Inference-only state normalizer.

Applies (x - mean) / std with statistics loaded from a training
checkpoint. No online updates, no freeze flags, no dataset/return-range
or sum-reward logic — those are training-only concerns.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateNormalizer(nn.Module):
    """Read-only state normalizer for inference.

    Mirrors the forward contract of the training-time RunningMeanStd
    (project's normalization stack), but exposes only `mean`/`var`
    buffers — `count`/`decay_factor` are training bookkeeping and are
    intentionally dropped.
    """

    EPS = 1e-8

    def __init__(self, num_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_features = int(num_features)
        self.input_dtype = dtype
        self.output_dtype = torch.float32
        self.register_buffer("mean", torch.zeros(self.num_features, dtype=dtype))
        self.register_buffer("var", torch.ones(self.num_features, dtype=dtype))

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var) + self.EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.input_dtype)
        feat = x.size(-1)
        # Slice from the tail so callers may pass a sub-feature view that
        # matches the trailing portion of the registered feature space.
        mean = self.mean[-feat:].view(1, 1, -1)
        std = self.std[-feat:].view(1, 1, -1)
        if x.device != mean.device:
            mean = mean.to(x.device)
            std = std.to(x.device)
        return ((x - mean) / std).to(dtype=self.output_dtype)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        num_features: int | None = None,
    ) -> "StateNormalizer":
        """Build from a RunningMeanStd-style state_dict.

        Reads only `mean` and `var`; any other keys (e.g. `count`,
        `decay_factor`) are ignored.
        """
        mean = state_dict["mean"]
        var = state_dict["var"]
        if num_features is None:
            num_features = int(mean.numel())
        norm = cls(num_features=num_features, dtype=mean.dtype)
        with torch.no_grad():
            norm.mean.copy_(mean)
            norm.var.copy_(var)
        return norm
