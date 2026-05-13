"""
Causal Autoregressive Reinforcement Learning
Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

import copy
from typing import Any

import torch

try:
    from transformers import DynamicCache, StaticCache
except Exception:  # pragma: no cover - import error depends on transformers version.
    DynamicCache = None
    StaticCache = None


def _get_model_config(backbone: Any):
    if hasattr(backbone, "model_config"):
        return backbone.model_config
    if hasattr(backbone, "config"):
        return backbone.config
    raise AttributeError("Backbone must expose model_config or config for KV cache preparation.")


def cache_has_history(past_key_values) -> bool:
    if past_key_values is None:
        return False

    # Newer HF (>=4.46): DynamicCache.layers
    if hasattr(past_key_values, "layers") and past_key_values.layers:
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and layer.keys is not None and hasattr(layer.keys, "shape"):
                return int(layer.keys.shape[-2]) > 0

    # Older HF (4.40~4.45): DynamicCache.key_cache
    if hasattr(past_key_values, "key_cache"):
        key_cache = past_key_values.key_cache
        if key_cache and key_cache[0] is not None and hasattr(key_cache[0], "shape"):
            return int(key_cache[0].shape[-2]) > 0

    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first_layer = past_key_values[0]
        if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
            key_tensor = first_layer[0]
            if hasattr(key_tensor, "size"):
                return int(key_tensor.size(-2)) > 0

    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length()) > 0
        except Exception:
            return False

    return False


def prepare_cache_warm_start_inputs(
    full_input: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if padding_mask is None:
        return full_input, None

    if padding_mask.dim() == 3:
        padding_mask = padding_mask.squeeze(-1)
    if padding_mask.dim() != 2:
        return full_input, padding_mask

    valid_lengths = padding_mask.to(dtype=torch.long).sum(dim=1)
    if valid_lengths.numel() == 0:
        return full_input, None

    min_valid_len = int(valid_lengths.min().item())
    max_valid_len = int(valid_lengths.max().item())
    if min_valid_len <= 0:
        return full_input, padding_mask

    # If every batch item shares the same valid suffix length, we can
    # compact away the left padding and warm the cache without a mask.
    if min_valid_len == max_valid_len:
        compact_len = max(1, max_valid_len)
        return full_input[:, -compact_len:], None

    return full_input, padding_mask


def build_kv_cache(backbone, max_len: int):
    if DynamicCache is None:
        raise ImportError("transformers cache classes are unavailable. Upgrade transformers to a version with DynamicCache.")

    cache_config = copy.deepcopy(_get_model_config(backbone))
    # HF dynamic cache tracks past tokens only; using +1 aligns the observed
    # KV length cap with the user-facing context length.
    cache_config.sliding_window = int(max_len) + 1

    cache = DynamicCache(config=cache_config)
    # Keep an explicit length tag because HF cache objects do not reliably
    # expose the requested sliding-window size as a stable public attribute.
    setattr(cache, "_configured_max_len", int(max_len))
    return cache


def _truncate_kv_cache_keep_latest(past_key_values, max_len: int):
    max_len = max(1, int(max_len))

    for layer in getattr(past_key_values, "layers", ()) or ():
        if layer.keys is None:
            continue
        if layer.keys.shape[-2] > max_len:
            layer.keys = layer.keys[..., -max_len:, :].contiguous()
            layer.values = layer.values[..., -max_len:, :].contiguous()
        # DynamicSlidingWindowLayer drives get_seq_length() from
        # cumulative_length, not keys.shape — realign so position ids stay
        # capped at max_len after manual keep-latest slicing.
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = int(layer.keys.shape[-2])

    setattr(past_key_values, "_configured_max_len", max_len)
    return past_key_values


def prepare_kv_cache_inputs(
    backbone,
    past_key_values=None,
    max_len: int = 32,
):
    """
    Return a KV cache sized for `max_len` history.

    Build a fresh cache when none is provided; otherwise cap the existing cache
    to `max_len`. HF DynamicCache only grows — manual cap each step keeps the
    cache tensor size and `past_length` (used for new-token position embedding)
    bounded so positions stay within the trained input range.
    """
    if max_len is None:
        return past_key_values
    if past_key_values is None:
        return build_kv_cache(backbone=backbone, max_len=max_len)
    return _truncate_kv_cache_keep_latest(past_key_values, max_len=int(max_len))
