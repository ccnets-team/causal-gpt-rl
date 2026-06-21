"""Output-to-input adapter for autoregressive feedback.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

# tensor_sequence.py
import torch
from typing import Optional


class OutputToInputAdapter(torch.nn.Module):
    def __init__(self, type: str, output_size: int, squash: Optional[str] = None, low: Optional[float] = None, high: Optional[float] = None):
        super().__init__()
        self._type = type
        self.output_size = output_size
        assert squash is not None or (low is None and high is None), \
            "If squash is None, low and high must be None"
        self.squash = squash
        # Register low/high as buffers so they follow device/dtype. `.clone()` so
        # each buffer owns its storage: torch.as_tensor on a float32 numpy bound
        # returns a VIEW sharing that array, and the same spec also feeds the
        # input-feedback adapter — without the clone the two low/high buffers
        # alias and safetensors refuses to save the bundle (shared storage).
        if low is not None:
            low_t = torch.as_tensor(low, dtype=torch.float32).view(1, 1, -1).clone()
            self.register_buffer("low", low_t)
        else:
            self.register_buffer("low", None)

        if high is not None:
            high_t = torch.as_tensor(high, dtype=torch.float32).view(1, 1, -1).clone()
            self.register_buffer("high", high_t)
        else:
            self.register_buffer("high", None)
            
    def forward(self, output: torch.Tensor) -> torch.Tensor:
        assert output.ndim == 3, f"expected (B,T,D), got {tuple(output.shape)}"
        B, T, D = output.shape
        dtype = output.dtype
        
        # discrete, continuous, multi_discrete
        if self._type == "continuous":
            if self.squash == "tanh":
                output = torch.tanh(output)
                scale = (self.high - self.low) / 2.0
                bias = (self.high + self.low) / 2.0
                output = output * scale + bias
            return output
        
        if self._type in ("discrete", "multi_discrete"):
            idx = torch.argmax(output, dim=-1)
            return idx.unsqueeze(-1).long()

        if self._type == "multi_binary":
            # Independent Bernoulli action head: threshold raw logits at 0
            # (== prob 0.5). Keeps the n-vector shape so the {0,1} feedback
            # matches the runner's env decode (`gym.flatten(MultiBinary)`).
            return (output > 0.0).to(dtype)

        if self._type == "binary_classification":
            return (output > 0.5).to(dtype)

        if self._type == "multi_label_classification":
            return (output > 0.5).to(dtype)

        if self._type == "multi_class_classification":
            idx = torch.argmax(output, dim=-1)
            return idx.unsqueeze(-1).long()

        if self._type == "ordinal_regression":
            base = output if D == 1 else output[..., :1]
            idx = torch.round(base).clamp(0, self.output_size - 1).long()
            return idx.unsqueeze(-1)

        # regression / continuous
        return output
