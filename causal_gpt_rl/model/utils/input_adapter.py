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
            # `.clone()` so the buffer owns its storage. torch.as_tensor on a
            # float32 numpy `low`/`high` returns a VIEW sharing that array; the
            # same spec feeds both this input-feedback adapter and the action
            # output adapter, so without the clone their low/high buffers alias
            # and safetensors refuses to save the bundle (shared storage).
            low_t = torch.as_tensor(low, dtype=torch.float32).view(1, 1, -1).clone()
            high_t = torch.as_tensor(high, dtype=torch.float32).view(1, 1, -1).clone()
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
