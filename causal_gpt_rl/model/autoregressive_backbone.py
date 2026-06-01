"""
Causal Autoregressive Reinforcement Learning
Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

import inspect

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from transformers import LlamaModel, LlamaConfig


def normalize_attention_mask(padding_mask):
    """Coerce `[B,T]` or `[B,T,1]` padding mask to a `[B,T]` bool tensor."""
    if padding_mask is None:
        return None
    if padding_mask.dim() == 3:
        if padding_mask.size(-1) != 1:
            raise ValueError(
                f"padding_mask last dim must be 1 when 3D, got {tuple(padding_mask.shape)}"
            )
        padding_mask = padding_mask.squeeze(-1)
    elif padding_mask.dim() != 2:
        raise ValueError(
            f"padding_mask must be [B,T] or [B,T,1], got {tuple(padding_mask.shape)}"
        )
    return padding_mask.bool()


def _llama_rope_kwargs(rope_theta: float) -> dict:
    """Return RoPE config kwargs for the installed Transformers version."""
    llama_params = inspect.signature(LlamaConfig.__init__).parameters
    if "rope_parameters" in llama_params:
        return {
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": float(rope_theta),
            }
        }
    if "rope_theta" in llama_params:
        return {"rope_theta": float(rope_theta)}
    return {}


class GPTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        # ModelConfig.__post_init__ resolves all derived fields; backbone reads
        # them directly. Defaults remain for legacy/duck-typed configs.
        network_name = str(getattr(config, "network_name", "GPT2")).lower()
        num_heads = int(getattr(config, "num_heads", 8))
        d_model = int(getattr(config, "d_model", 256))
        num_layers = int(getattr(config, "num_layers", 4))
        dropout = float(getattr(config, "dropout", 0.02))
        rope_theta = float(getattr(config, "rope_theta", 1e3))
        intermediate_size = int(getattr(config, "intermediate_size", None) or d_model * 4)
        context_length = int(
            getattr(
                config,
                "context_length",
                getattr(config, "train_context_length", None),
            )
            or 32
        )
        max_pos_attr = getattr(config, "max_position_embeddings", None)
        if max_pos_attr is None:
            if network_name in {"gpt2", "gpt", "gpt-2"}:
                max_pos = max(8, context_length * 2)
            else:
                max_pos = max(32, context_length * 8)
        else:
            max_pos = int(max_pos_attr)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        if network_name in {"gpt2", "gpt", "gpt-2"}:
            model_config = GPT2Config(
                vocab_size=1,
                n_positions=max_pos,
                n_ctx=max_pos,
                n_embd=d_model,
                n_layer=num_layers,
                n_head=num_heads,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
                use_cache=False,
                bos_token_id=1,
                eos_token_id=2,
            )
            self.net = GPT2Model(model_config)
        else:
            model_config = LlamaConfig(
                vocab_size=1,
                hidden_size=d_model,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                hidden_act="silu",
                max_position_embeddings=max_pos,
                rms_norm_eps=1e-6,
                use_cache=False,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                attention_dropout=dropout,
                **_llama_rope_kwargs(rope_theta),
            )
            self.net = LlamaModel(model_config)
        self.model_config = model_config

    def forward(self, x, padding_mask=None):
        attention_mask = normalize_attention_mask(padding_mask)
        out = self.net(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    @torch.inference_mode()
    def infer(self, input_tensor, past_key_values=None, padding_mask=None):
        attention_mask = normalize_attention_mask(padding_mask)
        out = self.net(
            inputs_embeds=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            return_dict=True,
        )
        return out.last_hidden_state, out.past_key_values
