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
STATE_NORMALIZATION_EPS = 1e-8


def _resolve_action_value_routing(flat_output_specs):
    """Build index/spec routing for RL action sampling from output specs."""
    mean_action_indices = []
    mean_action_types = []
    mean_action_specs = []
    log_std_action_indices = []
    value_indices = []
    termination_indices = []
    for i, spec in enumerate(flat_output_specs):
        if spec.role == "action" and spec.sub_role == "mean":
            mean_action_indices.append(i)
            mean_action_types.append(spec.type)
            mean_action_specs.append(spec)
        elif spec.role == "action" and spec.sub_role == "log_std":
            log_std_action_indices.append(i)
        elif spec.role == "value":
            value_indices.append(i)
        elif spec.role == "termination":
            termination_indices.append(i)
    assert len(value_indices) == 1, "There should be exactly one value head in the model."
    assert len(termination_indices) <= 1, "There should be at most one termination head."
    return {
        "mean_action_indices": mean_action_indices,
        "mean_action_types": mean_action_types,
        "mean_action_specs": mean_action_specs,
        "log_std_action_indices": log_std_action_indices,
        "value_index": value_indices[0],
        "termination_index": termination_indices[0] if termination_indices else None,
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
        self.use_eos = bool(getattr(model_config, "use_eos", False))
        self.state_specs = list(state_specs)
        self.action_specs = list(action_specs)
        input_specs, output_specs = build_model_specs(
            state_specs, action_specs, use_eos=self.use_eos
        )
        self.input_specs = input_specs
        self.output_specs = output_specs
        self.flat_input_specs = flatten_specs(input_specs)
        self.flat_output_specs = flatten_specs(output_specs)  # for autoreg, input_specs includes output_specs
        self.device = device
        self.state_size = int(sum(spec.size for spec in self.state_specs))
        self.register_buffer(
            "state_normalization_enabled",
            torch.zeros(1, dtype=torch.float32),
        )
        self.register_buffer(
            "state_normalization_mean",
            torch.zeros(self.state_size, dtype=torch.float32),
        )
        self.register_buffer(
            "state_normalization_std",
            torch.ones(self.state_size, dtype=torch.float32),
        )

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
        self.termination_index = routing["termination_index"]

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

    # Representation helpers ---------------------------------------------

    def has_embedded_state_normalizer(self) -> bool:
        return bool(float(self.state_normalization_enabled.item()) > 0.5)

    @torch.no_grad()
    def set_state_normalization(
        self,
        *,
        mean: torch.Tensor,
        std: torch.Tensor | None = None,
        var: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> None:
        mean = torch.as_tensor(
            mean,
            device=self.state_normalization_mean.device,
            dtype=self.state_normalization_mean.dtype,
        ).reshape(-1)
        if std is None:
            if var is None:
                raise ValueError("Either std or var must be provided.")
            std = torch.sqrt(
                torch.as_tensor(
                    var,
                    device=self.state_normalization_std.device,
                    dtype=self.state_normalization_std.dtype,
                ).reshape(-1)
            ) + STATE_NORMALIZATION_EPS
        else:
            std = torch.as_tensor(
                std,
                device=self.state_normalization_std.device,
                dtype=self.state_normalization_std.dtype,
            ).reshape(-1)

        if mean.numel() != self.state_size or std.numel() != self.state_size:
            raise ValueError(
                f"Expected state normalization size {self.state_size}, "
                f"got mean={mean.numel()} std={std.numel()}."
            )

        self.state_normalization_mean.copy_(mean)
        self.state_normalization_std.copy_(std.clamp_min(STATE_NORMALIZATION_EPS))
        self.state_normalization_enabled.fill_(1.0 if enabled else 0.0)

    @torch.no_grad()
    def set_state_normalization_from_state_dict(
        self,
        state_dict: dict,
        *,
        enabled: bool = True,
    ) -> None:
        if "mean" not in state_dict:
            raise KeyError("state_dict must contain a 'mean' entry.")
        if "var" in state_dict:
            self.set_state_normalization(
                mean=state_dict["mean"],
                var=state_dict["var"],
                enabled=enabled,
            )
        elif "std" in state_dict:
            self.set_state_normalization(
                mean=state_dict["mean"],
                std=state_dict["std"],
                enabled=enabled,
            )
        else:
            raise KeyError("state_dict must contain either a 'var' or 'std' entry.")

    def normalize_states_for_inference(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(dtype=torch.float32)
        if not self.has_embedded_state_normalizer():
            return states

        feat = states.size(-1)
        mean = self.state_normalization_mean[-feat:].view(1, 1, -1)
        std = self.state_normalization_std[-feat:].view(1, 1, -1)
        if mean.device != states.device:
            mean = mean.to(states.device)
            std = std.to(states.device)
        return ((states - mean) / std.clamp_min(STATE_NORMALIZATION_EPS)).to(
            dtype=torch.float32
        )

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

    def _termination_prob(self, out: list[torch.Tensor]) -> torch.Tensor | None:
        """Sigmoid of the raw termination logit, or None when EOS is disabled.

        Read from the raw projected head (pre-adapter): the termination head
        is an unsquashed logit, so the adapter is a passthrough anyway.
        """
        if self.termination_index is None:
            return None
        return torch.sigmoid(out[self.termination_index])

    def _build_info(self, out: list[torch.Tensor]) -> dict:
        """Auxiliary per-step outputs that ride on the policy forward."""
        return {"termination_prob": self._termination_prob(out)}

    @torch.inference_mode()
    def predict_incremental_cached(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        is_bos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_max_len: int | None = None,
        return_info: bool = False,
    ) -> list[torch.Tensor]:
        """Eval one incremental cached step for policy inference.

        With `return_info=True`, also returns an auxiliary-output dict (e.g.
        `termination_prob`) as a trailing tuple element so callers that need
        EOS info can read it without a separate forward.
        """
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

        action = self._extract_mean_action(out)
        if return_info:
            return action, past_key_values, self._build_info(out)
        return action, past_key_values

    @torch.inference_mode()
    def predict_with_window(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        is_bos: torch.Tensor,
        padding_mask: torch.Tensor,
        return_info: bool = False,
    ) -> list[torch.Tensor]:
        """Eval policy outputs over a full context window.

        With `return_info=True`, also returns an auxiliary-output dict (e.g.
        `termination_prob`) as a trailing tuple element.
        """
        self.eval()
        context = torch.cat([states, actions, is_bos], dim=-1)
        outputs = self.infer_windowed(context, padding_mask=padding_mask)
        action = self._extract_mean_action(outputs)
        if return_info:
            return action, self._build_info(outputs)
        return action

    @torch.inference_mode()
    def sample_action_from_heads(self, out: list[torch.Tensor], std_scale: float = 1.0) -> torch.Tensor:
        if self._is_continuous:
            return self._sample_continuous(out, std_scale=std_scale)
        if self._is_discrete:
            return self._sample_discrete(out)
        raise NotImplementedError("Mixed continuous/discrete action heads are not supported in action sampling.")

    def _sample_continuous(self, out: list[torch.Tensor], std_scale: float = 1.0) -> torch.Tensor:
        mean = torch.cat([out[i] for i in self.mean_action_indices], dim=-1)
        log_std = torch.cat([out[i] for i in self.log_std_action_indices], dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        z = mean + std_scale * log_std.exp() * torch.randn_like(mean)
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
