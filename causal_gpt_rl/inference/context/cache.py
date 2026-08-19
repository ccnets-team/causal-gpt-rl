"""KV-cache container for public inference context state.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

import numpy as np
import torch


def _grow_cache_batch(past_key_values, extra_rows: int) -> bool:
    """Append `extra_rows` zero-filled rows to a cache's batch axis, in place.

    A new row carries no history, so its keys/values are never read: the caller
    records it as owning none of the cache and `build_cached_attention_mask`
    hides them. The zeros are storage, not content — without that mask they
    would dilute attention rather than be ignored.

    Returns False for a cache layout this cannot grow, leaving it untouched so
    the caller can fall back to dropping the cache and recomputing.
    """
    k = int(extra_rows)
    if k <= 0:
        return True

    def _grown(tensor):
        pad = torch.zeros(
            (k, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, pad], dim=0)

    layers = getattr(past_key_values, "layers", None)
    if layers:
        for layer in layers:
            if getattr(layer, "keys", None) is None:
                continue
            layer.keys = _grown(layer.keys)
            layer.values = _grown(layer.values)
        return True

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        for index, keys in enumerate(key_cache):
            if keys is None:
                continue
            key_cache[index] = _grown(keys)
            value_cache[index] = _grown(value_cache[index])
        return True

    return False


class ContextCache:
    """
    KV-cache holder for autoregressive evaluation and precompute passes.

    This class intentionally does NOT store or transform trajectory tensors
    (states/actions/masks). Sequence semantics, including this project's
    non-standard (state, action) ordering, must remain in ContextBuffer.

    One cached tensor is shared by every agent row, so all rows carry the same
    cached *length*. What differs per row is how much of that length is still its
    own: a row whose episode restarts mid-batch keeps its slot in the tensor but
    owns none of what is in it. `valid_lengths()` is what the attention mask is
    built from, and it is why a partial reset does not have to throw the cache
    away.
    """

    def __init__(self, kv_cache_max_len: int | None = None, num_agents: int = 1):
        self.kv_cache_max_len = None if kv_cache_max_len is None else max(1, int(kv_cache_max_len))
        self.past_key_values = None
        # How much of the cache each row still owns. One integer per row is
        # enough: every row appends one token per step and trimming drops the
        # oldest positions for all of them at once, so a row's own region is
        # always a suffix of the cache.
        self.num_agents = max(1, int(num_agents))
        self._valid_len = np.zeros(self.num_agents, dtype=np.int64)

    def set_kv_cache_max_len(self, kv_cache_max_len: int | None) -> None:
        self.kv_cache_max_len = None if kv_cache_max_len is None else max(1, int(kv_cache_max_len))

    def get_kv_cache_length(self) -> int:
        if self.past_key_values is None:
            return 0

        # Newer HF (>=4.46): DynamicCache.layers[*].keys
        if hasattr(self.past_key_values, "layers") and self.past_key_values.layers:
            for layer in self.past_key_values.layers:
                if hasattr(layer, "keys") and layer.keys is not None and hasattr(layer.keys, "shape"):
                    return int(layer.keys.shape[-2])

        # Older HF (4.40~4.45): DynamicCache.key_cache list
        if hasattr(self.past_key_values, "key_cache"):
            key_cache = self.past_key_values.key_cache
            if key_cache and key_cache[0] is not None and hasattr(key_cache[0], "shape"):
                return int(key_cache[0].shape[-2])

        # Legacy tuple/list path
        if isinstance(self.past_key_values, (tuple, list)) and len(self.past_key_values) > 0:
            first_layer = self.past_key_values[0]
            if isinstance(first_layer, (tuple, list)) and len(first_layer) > 0:
                key_tensor = first_layer[0]
                if hasattr(key_tensor, "size"):
                    return int(key_tensor.size(-2))

        # Fallback only
        if hasattr(self.past_key_values, "get_seq_length"):
            try:
                return int(self.past_key_values.get_seq_length())
            except Exception:
                return 0

        return 0
    
    def set_past_key_values(self, past_key_values) -> None:
        self.past_key_values = past_key_values

    def get_past_key_values(self):
        return self.past_key_values

    def reset(self) -> None:
        self.past_key_values = None
        self._valid_len[:] = 0

    def valid_lengths(self) -> np.ndarray:
        """Per row, how many cached positions belong to its current episode."""
        return np.minimum(self._valid_len, self.get_kv_cache_length())

    def has_partial_rows(self) -> bool:
        """True when some row owns less of the cache than the cache holds."""
        return bool((self.valid_lengths() < self.get_kv_cache_length()).any())

    def record_append(self, appended: int) -> None:
        """Note that the last forward added `appended` tokens to every row.

        Kept apart from `set_past_key_values` so that method stays exactly what
        it was: a caller that only stores a cache is unaffected, and one that
        also tracks rows says so in its own call.
        """
        self._valid_len = np.minimum(
            self._valid_len + int(appended), self.get_kv_cache_length()
        )

    def set_valid_lengths(self, lengths) -> None:
        """Declare each row's owned length outright, after a full recompute."""
        lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        if lengths.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected valid lengths for {self.num_agents} agents, got {lengths.shape[0]}"
            )
        self._valid_len = np.minimum(lengths, self.get_kv_cache_length())

    def invalidate_rows(self, reset_mask) -> None:
        """Drop the flagged rows' history and leave every other row untouched."""
        mask = np.asarray(reset_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected reset_mask for {self.num_agents} agents, got {mask.shape[0]}"
            )
        self._valid_len[mask] = 0

    def add_agent_rows(self, extra_rows: int) -> bool:
        """Grow the batch by `extra_rows` fresh rows that own no history.

        Returns True when the existing rows' cache survived, False when the
        layout could not be grown and the cache was dropped instead — the caller
        then recomputes it, which is what every partial restart used to do.
        """
        k = int(extra_rows)
        if k <= 0:
            raise ValueError(f"add_agent_rows needs at least one row; got {k}")
        self.num_agents += k
        self._valid_len = np.concatenate(
            [self._valid_len, np.zeros(k, dtype=np.int64)]
        )
        if self.past_key_values is None:
            return True
        if _grow_cache_batch(self.past_key_values, k):
            return True
        self.reset()
        return False

    def __repr__(self) -> str:
        return f"ContextCache(kv_cache_max_len={self.kv_cache_max_len})"
