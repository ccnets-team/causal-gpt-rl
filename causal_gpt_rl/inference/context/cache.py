"""KV-cache container for public inference context state.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

class ContextCache:
    """
    KV-cache holder for autoregressive evaluation and precompute passes.

    This class intentionally does NOT store or transform trajectory tensors
    (states/actions/masks). Sequence semantics, including this project's
    non-standard (state, action) ordering, must remain in ContextBuffer.
    """

    def __init__(self, kv_cache_max_len: int | None = None):
        self.kv_cache_max_len = None if kv_cache_max_len is None else max(1, int(kv_cache_max_len))
        self.past_key_values = None

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

    def __repr__(self) -> str:
        return f"ContextCache(kv_cache_max_len={self.kv_cache_max_len})"
