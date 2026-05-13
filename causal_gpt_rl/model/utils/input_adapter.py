"""Input head adapter — inverse-tanh-squash for bounded continuous action inputs.

Mirror of `output_adapter.OutputToInputAdapter`: when the policy output applies
tanh squashing to bounded continuous actions, autoregressive feedback must
undo that squash before the action re-enters the model as input.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

import torch
import torch.nn as nn


class InputHeadAdapter(nn.Module):
    def __init__(self, spec, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.use_atanh = False

        low = getattr(spec, "low", None)
        high = getattr(spec, "high", None)
        is_bounded_continuous_action = (
            spec.role == "action"
            and spec.sub_role == "mean"
            and spec.type == "continuous"
            and getattr(spec, "squash", None) == "tanh"
            and low is not None
            and high is not None
        )

        if is_bounded_continuous_action:
            low_t = torch.as_tensor(low, dtype=torch.float32).view(1, 1, -1)
            high_t = torch.as_tensor(high, dtype=torch.float32).view(1, 1, -1)
            self.use_atanh = bool(
                torch.isfinite(low_t).all()
                and torch.isfinite(high_t).all()
                and (high_t > low_t).all()
            )
            if self.use_atanh:
                self.register_buffer("low", low_t)
                self.register_buffer("high", high_t)
                return

        self.register_buffer("low", None)
        self.register_buffer("high", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_atanh:
            return x

        low = self.low.to(device=x.device, dtype=x.dtype)
        high = self.high.to(device=x.device, dtype=x.dtype)
        scaled = 2.0 * (x - low) / (high - low) - 1.0
        scaled = scaled.clamp(min=-1.0 + self.eps, max=1.0 - self.eps)
        return torch.atanh(scaled)
