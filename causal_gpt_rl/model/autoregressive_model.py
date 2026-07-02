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
        # BOS action gate: neutralize the (absent) previous-action channel at
        # episode start. Serving capability (persisted); default-off is
        # byte-identical. See .local/docs/dev/model-output/bos-gate-to-zero.md.
        self.use_bos_action_gate = bool(getattr(model_config, "use_bos_action_gate", False))
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
        mean_specs = routing["mean_action_specs"]
        # Per-head continuity flags are the source of truth; core logic
        # dispatches per head. The global `_is_continuous`/`_is_discrete` are
        # derived views (see properties below) kept only for the homogeneous
        # fast paths and back-compat.
        self._mean_is_continuous = [t == "continuous" for t in mean_types]
        # Per-head action type (continuous / discrete / multi_discrete /
        # multi_binary), in mean-head order. The source of truth for sampling
        # dispatch: continuous is squashed+scaled, categorical is argmax/one-hot,
        # multi_binary is independent Bernoulli — and they must not be confused.
        self._mean_types = list(mean_types)
        # Positions of continuous heads within the mean-head order. Plain
        # attributes derived from specs — deliberately NOT registered as buffers
        # so they stay out of state_dict (weight compatibility).
        self._continuous_mean_positions = [
            i for i, is_cont in enumerate(self._mean_is_continuous) if is_cont
        ]

        # (De)squash buffers cover the CONTINUOUS mean heads only. For an
        # all-continuous bundle this subset is the full set in the same order,
        # so the buffer is byte-identical to the legacy layout → state_dict
        # stays compatible. Pure-discrete bundles register no buffer (as before).
        if any(self._mean_is_continuous):
            cont_specs = [
                spec for spec, is_cont in zip(mean_specs, self._mean_is_continuous)
                if is_cont
            ]
            lows = torch.cat([
                torch.as_tensor(s.low, dtype=torch.float32).reshape(-1) for s in cont_specs
            ]).reshape(1, -1)
            highs = torch.cat([
                torch.as_tensor(s.high, dtype=torch.float32).reshape(-1) for s in cont_specs
            ]).reshape(1, -1)
            self.register_buffer("_action_scale", (highs - lows) / 2.0)
            self.register_buffer("_action_bias", (highs + lows) / 2.0)

        # BOS action-gate parameters (opt-in). One learned vector per mean-action
        # input head, zeros-init. Left at zero (frozen) this reproduces
        # gate-to-zero — action-col → 0 at bos, marker supplied for free by the
        # is_bos column's own projection. Trained (use_bos_action_prior) it
        # promotes to a learned "null action" prior. Registered before
        # `to(device)` so the params move with the module.
        if self.use_bos_action_gate:
            # Input specs carry only sub_role=="mean" action heads (log_std is
            # output-only), so `role == "action"` selects exactly the columns to
            # gate. This filter is INPUT-only — do not reuse on output specs.
            self._input_action_positions = [
                i for i, spec in enumerate(self.flat_input_specs)
                if spec.role == "action"
            ]
            self._input_bos_index = next(
                (i for i, spec in enumerate(self.flat_input_specs)
                 if spec.role == "bos_indicator"),  # NOTE: "bos_indicator", not "bos"
                None,
            )
            if self._input_bos_index is None:
                raise ValueError(
                    "use_bos_action_gate requires a bos_indicator input head"
                )
            self.bos_action_gate_emb = nn.ParameterDict({
                str(i): nn.Parameter(
                    torch.zeros(1, 1, self.flat_input_specs[i].size)
                )
                for i in self._input_action_positions
            })
            # gate-to-zero is the default: keep the emb at zero. Promotion to a
            # learned prior is a trainer-only toggle (not a persisted capability;
            # inference reproduces behavior from the loaded emb values alone).
            if not bool(getattr(model_config, "use_bos_action_prior", False)):
                for p in self.bos_action_gate_emb.parameters():
                    p.requires_grad_(False)

        self.to(device)

    # Derived action-type views (homogeneous fast paths / back-compat) -----

    @property
    def _is_continuous(self) -> bool:
        """True iff every mean-action head is continuous."""
        return bool(self._mean_is_continuous) and all(self._mean_is_continuous)

    @property
    def _is_discrete(self) -> bool:
        """True iff every mean-action head is categorical (discrete / multi_discrete).

        MultiBinary heads are independent Bernoulli, NOT categorical, so a bundle
        containing one is not "discrete": it routes to the per-head hybrid sampler
        rather than the argmax/one-hot fast path (which would mis-treat the n
        independent bits as a single n-way choice).
        """
        return bool(self._mean_types) and all(
            t in ("discrete", "multi_discrete") for t in self._mean_types
        )

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

        if self.use_bos_action_gate:
            # At bos=1 replace each mean-action column with its gate emb (zero for
            # gate-to-zero); at bos=0 keep the real action. Done in feature space
            # (pre-`input_proj`) so the single joint Linear is untouched. Both
            # input branches (tensor / nested) converge here, so gating lives in
            # one place. `input_heads` is aligned with flat_input_specs in both.
            bos = input_heads[self._input_bos_index].to(
                dtype=adapted_heads[0].dtype
            ).clamp(0.0, 1.0)
            for i in self._input_action_positions:
                act = adapted_heads[i]
                gate_emb = self.bos_action_gate_emb[str(i)].to(
                    device=act.device, dtype=act.dtype
                )
                adapted_heads[i] = (1.0 - bos) * act + bos * gate_emb.expand_as(act)

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
        # Per head: continuous → its squash+scale adapter; categorical → raw
        # logits passthrough (argmax stays the runner's single decode source).
        # Pure-continuous keeps adapting every head, pure-discrete adapts none —
        # both reduce to the legacy behavior exactly.
        adapted = self.adapt_output_heads(out) if any(self._mean_is_continuous) else None
        return [
            adapted[i] if is_cont else out[i]
            for is_cont, i in zip(self._mean_is_continuous, self.mean_action_indices)
        ]

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
        return self._sample_hybrid(out, std_scale=std_scale)

    def _sample_hybrid(self, out: list[torch.Tensor], std_scale: float = 1.0) -> torch.Tensor:
        """Per-head sampling for mixed continuous + categorical policies.

        Continuous heads are sampled jointly so the squash/scale buffer (a
        concatenation over the continuous heads, in head order) applies
        wholesale — identical to how the pure-continuous path treats those same
        columns. Categorical heads are sampled independently. Results are
        reassembled in mean-head order so the output layout matches the flat
        head schedule the runner decodes against.
        """
        cont_mean_indices = [
            i for i, t in zip(self.mean_action_indices, self._mean_types)
            if t == "continuous"
        ]
        cont_actions = None
        if cont_mean_indices:
            mean = torch.cat([out[i] for i in cont_mean_indices], dim=-1)
            log_std = torch.cat([out[i] for i in self.log_std_action_indices], dim=-1)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            z = mean + std_scale * log_std.exp() * torch.randn_like(mean)
            squashed = torch.tanh(z)
            scale = self._action_scale.to(dtype=squashed.dtype)
            bias = self._action_bias.to(dtype=squashed.dtype)
            cont_actions = squashed * scale + bias

        parts = []
        offset = 0
        for i, head_type in zip(self.mean_action_indices, self._mean_types):
            if head_type == "continuous":
                width = out[i].size(-1)
                parts.append(cont_actions[..., offset:offset + width])
                offset += width
            elif head_type == "multi_binary":
                # Independent Bernoulli per element: sample {0,1} from the head
                # logits (no one-hot — the n bits are independent). std_scale == 0
                # collapses to the deterministic threshold (== prob 0.5), matching
                # the runner's greedy decode, mirroring how continuous treats 0.
                logits = out[i]
                if std_scale == 0.0:
                    parts.append((logits > 0.0).to(logits.dtype))
                else:
                    parts.append(
                        torch.distributions.Bernoulli(logits=logits).sample().to(logits.dtype)
                    )
            else:
                logits = out[i]
                sampled_idx = torch.distributions.Categorical(logits=logits).sample()
                one_hot = F.one_hot(sampled_idx, num_classes=logits.size(-1)).to(logits.dtype)
                parts.append(one_hot)
        return torch.cat(parts, dim=-1)

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
