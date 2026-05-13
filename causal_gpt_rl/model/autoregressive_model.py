"""
Causal Autoregressive Reinforcement Learning implementation in PyTorch.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import ModelConfig, SpaceSpec, flatten_specs, nested_to_flat_list_heads
from .spec_factory import build_model_specs
from .autoregressive_backbone import GPTBackbone
from .utils.kv_cache import (
    cache_has_history,
    prepare_cache_warm_start_inputs,
    prepare_kv_cache_inputs,
    _truncate_kv_cache_keep_latest,
)
from .utils.input_adapter import InputHeadAdapter
from .utils.layers import TransformLayer
from .utils.output_adapter import OutputToInputAdapter

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def _resolve_action_value_routing(flat_output_specs):
    """Build index/spec routing for RL action sampling from output specs."""
    mean_action_indices = []
    mean_action_types = []
    mean_action_specs = []
    log_std_action_indices = []
    value_indices = []
    for i, spec in enumerate(flat_output_specs):
        if spec.role == "action" and spec.sub_role == "mean":
            mean_action_indices.append(i)
            mean_action_types.append(spec.type)
            mean_action_specs.append(spec)
        elif spec.role == "action" and spec.sub_role == "log_std":
            log_std_action_indices.append(i)
        elif spec.role == "value":
            value_indices.append(i)
    assert len(value_indices) == 1, "There should be exactly one value head in the model."
    return {
        "mean_action_indices": mean_action_indices,
        "mean_action_types": mean_action_types,
        "mean_action_specs": mean_action_specs,
        "log_std_action_indices": log_std_action_indices,
        "value_index": value_indices[0],
    }


class AutoregressiveModel(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        state_specs: list[SpaceSpec],
        action_specs: list[SpaceSpec],
        device,
    ):
        super().__init__()
        self.network_name = model_config.network_name
        self.state_specs = list(state_specs)
        self.action_specs = list(action_specs)
        input_specs, output_specs = build_model_specs(state_specs, action_specs)
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.flat_input_specs = flatten_specs(input_specs)
        self.flat_output_specs = flatten_specs(output_specs)  # for autoreg, input_specs includes output_specs
        self.device = device

        # I/O wiring
        self.total_input_dim = sum(spec.size for spec in self.flat_input_specs)
        self.input_head_sizes = [spec.size for spec in self.flat_input_specs]
        self.input_adapter = nn.ModuleList(
            [InputHeadAdapter(spec) for spec in self.flat_input_specs]
        )
        self.input_proj = TransformLayer(
            self.total_input_dim,
            model_config.d_model,
            input_act_fn="none",
            output_act_fn="none",
        )
        self.output_head = nn.ModuleList([
            TransformLayer(model_config.d_model, spec.size, input_act_fn="relu", output_act_fn="none")
            for spec in self.flat_output_specs
        ])
        self.output_adapter = nn.ModuleList(
            [OutputToInputAdapter(spec.type, spec.size, spec.squash, spec.low, spec.high)
             for spec in self.flat_output_specs]
        )

        # Backbone
        self.backbone = GPTBackbone(model_config)

        # Spec routing for action/value sampling
        routing = _resolve_action_value_routing(self.flat_output_specs)
        self.mean_action_indices = routing["mean_action_indices"]
        self.log_std_action_indices = routing["log_std_action_indices"]
        self.value_index = routing["value_index"]

        mean_types = routing["mean_action_types"]
        self._is_continuous = bool(mean_types) and all(t == "continuous" for t in mean_types)
        self._is_discrete = bool(mean_types) and all(t in ("discrete", "multi_discrete") for t in mean_types)

        if self._is_continuous:
            mean_specs = routing["mean_action_specs"]
            lows = torch.cat([
                torch.as_tensor(s.low, dtype=torch.float32).reshape(-1) for s in mean_specs
            ]).reshape(1, -1)
            highs = torch.cat([
                torch.as_tensor(s.high, dtype=torch.float32).reshape(-1) for s in mean_specs
            ]).reshape(1, -1)
            self.register_buffer("_action_scale", (highs - lows) / 2.0)
            self.register_buffer("_action_bias", (highs + lows) / 2.0)

        self.to(device)

    # I/O helpers ---------------------------------------------------------

    def _split_input_heads(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.size(-1) != self.total_input_dim:
            raise ValueError(
                f"Expected input feature size {self.total_input_dim}, got {x.size(-1)}"
            )
        return list(torch.split(x, self.input_head_sizes, dim=-1))

    def adapt_input(self, x):
        """Map (B, T, sum(obs_sizes)) → (B, T, d_model)."""
        if not torch.is_tensor(x):
            input_heads = nested_to_flat_list_heads(x)
        else:
            input_heads = self._split_input_heads(x)

        if len(input_heads) != len(self.input_adapter):
            raise ValueError(
                f"Expected {len(self.input_adapter)} input heads, got {len(input_heads)}"
            )

        adapted_heads = [
            adapter(head)
            for adapter, head in zip(self.input_adapter, input_heads)
        ]
        x = torch.cat(adapted_heads, dim=-1)
        return self.input_proj(x)

    def project_output_heads(self, hidden):
        """Project hidden states (B, T, d_model) into per-spec head outputs."""
        return [head(hidden) for head in self.output_head]

    def adapt_output_heads(self, outputs):
        """Apply per-spec post-processing (e.g., tanh squash) to head outputs."""
        return [adapter(out) for out, adapter in zip(outputs, self.output_adapter)]

    # Forward variants ----------------------------------------------------

    def forward(self, x, padding_mask=None):
        embeded_x = self.adapt_input(x)
        hidden = self.backbone(embeded_x, padding_mask=padding_mask)
        return self.project_output_heads(hidden)

    def infer_windowed(self, x, padding_mask=None):
        embeded_x = self.adapt_input(x)
        hidden, _ = self.backbone.infer(embeded_x, padding_mask=padding_mask)
        return self.project_output_heads(hidden)

    def infer_cached(self, x, past_key_values=None, padding_mask=None):
        embeded_x = self.adapt_input(x)
        hidden, cache = self.backbone.infer(
            embeded_x,
            past_key_values=past_key_values,
            padding_mask=padding_mask,
        )
        pooled = hidden[:, -1:, ...]
        projected_output = self.project_output_heads(pooled)
        return projected_output, cache

    # Inference helpers ---------------------------------------------------

    def _extract_mean_action(self, out: list[torch.Tensor]) -> list[torch.Tensor]:
        if self._is_continuous:
            adapted = self.adapt_output_heads(out)
            return [adapted[i] for i in self.mean_action_indices]
        return [out[i] for i in self.mean_action_indices]

    @torch.inference_mode()
    def predict_incremental_cached(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        is_bos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_max_len: int | None = None,
    ) -> list[torch.Tensor]:
        """Eval one incremental cached step for policy inference."""
        self.eval()

        use_prefix_warm_start = not cache_has_history(past_key_values)
        if use_prefix_warm_start:
            full_input = torch.cat([states, actions, is_bos], dim=-1)
            model_input, model_padding_mask = prepare_cache_warm_start_inputs(
                full_input, padding_mask=padding_mask,
            )
        else:
            model_input = torch.cat(
                [states[:, -1:], actions[:, -1:], is_bos[:, -1:]], dim=-1
            )
            model_padding_mask = None

        past_key_values = prepare_kv_cache_inputs(
            backbone=self.backbone,
            past_key_values=past_key_values,
            max_len=cache_max_len,
        )

        out, past_key_values = self.infer_cached(
            model_input,
            past_key_values=past_key_values,
            padding_mask=model_padding_mask,
        )

        # Cap cache AFTER forward so the displayed/effective size stays at
        # cache_max_len (HF DynamicCache.update appends to the K/V tensors).
        if cache_max_len is not None and past_key_values is not None:
            past_key_values = _truncate_kv_cache_keep_latest(
                past_key_values, max_len=int(cache_max_len)
            )

        return self._extract_mean_action(out), past_key_values

    @torch.inference_mode()
    def predict_with_window(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        is_bos: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Eval policy outputs over a full context window."""
        self.eval()
        context = torch.cat([states, actions, is_bos], dim=-1)
        outputs = self.infer_windowed(context, padding_mask=padding_mask)
        return self._extract_mean_action(outputs)

    @torch.inference_mode()
    def sample_action_from_heads(self, out: list[torch.Tensor]) -> torch.Tensor:
        if self._is_continuous:
            return self._sample_continuous(out)
        if self._is_discrete:
            return self._sample_discrete(out)
        raise NotImplementedError("Mixed continuous/discrete action heads are not supported in action sampling.")

    def _sample_continuous(self, out: list[torch.Tensor]) -> torch.Tensor:
        mean = torch.cat([out[i] for i in self.mean_action_indices], dim=-1)
        log_std = torch.cat([out[i] for i in self.log_std_action_indices], dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        z = mean + log_std.exp() * torch.randn_like(mean)
        squashed = torch.tanh(z)
        scale = self._action_scale.to(dtype=squashed.dtype)
        bias = self._action_bias.to(dtype=squashed.dtype)
        return squashed * scale + bias

    def _sample_discrete(self, out: list[torch.Tensor]) -> torch.Tensor:
        sampled_actions = []
        for idx in self.mean_action_indices:
            logits = out[idx]
            sampled_idx = torch.distributions.Categorical(logits=logits).sample()
            one_hot = F.one_hot(sampled_idx, num_classes=logits.size(-1)).to(logits.dtype)
            sampled_actions.append(one_hot)
        return torch.cat(sampled_actions, dim=-1)


PolicyModel = AutoregressiveModel
