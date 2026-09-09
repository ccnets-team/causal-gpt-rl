"""Independent cached timelines for asynchronously restarting retain rows.

Groups split only when their rows advance differently. Normal steps reuse each
group's cache in place; they never re-encode history or insert dummy positions.
"""
from copy import copy
from dataclasses import dataclass

import numpy as np
import torch

from ...model.schema import ensure_tensor_heads
from ...model.utils.kv_cache import cached_position_count


def select_cache_rows(cache, indices):
    if cache is None:
        return None
    layers = getattr(cache, "layers", None)
    if not layers:
        raise RuntimeError("Retain row transitions require a layers-based KV cache.")
    selected = copy(cache)
    selected.layers = []
    for layer in layers:
        item = copy(layer)
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if value is not None:
                index = torch.as_tensor(indices, device=value.device, dtype=torch.long)
                setattr(item, name, value.index_select(0, index))
        selected.layers.append(item)
    return selected


@dataclass
class CacheGroup:
    rows: np.ndarray
    cache: object
    valid: np.ndarray


class RetainedCache:
    """A cache bank; lengths/positions remain local to each group's timeline."""

    def __init__(self, cache, valid):
        self.num_rows = len(valid)
        # Validate splitting support before callers change lifecycle state.
        if cache is not None and not getattr(cache, "layers", None):
            raise RuntimeError("Retain row transitions require a layers-based KV cache.")
        self.groups = [CacheGroup(np.arange(self.num_rows), cache, np.array(valid, copy=True))]

    def get_seq_length(self):
        return max((cached_position_count(g.cache) for g in self.groups), default=0)

    def valid_lengths(self):
        result = np.zeros(self.num_rows, dtype=np.int64)
        for group in self.groups:
            result[group.rows] = group.valid
        return result

    def split(self, mask):
        groups = []
        for group in self.groups:
            active = mask[group.rows]
            if active.all() or not active.any():
                groups.append(group)
                continue
            for part in (active, ~active):
                indices = np.flatnonzero(part)
                groups.append(CacheGroup(group.rows[part], select_cache_rows(group.cache, indices), group.valid[part]))
        self.groups = groups

    def invalidate_rows(self, mask):
        self.split(mask)
        for group in self.groups:
            if mask[group.rows].all():
                group.cache = None
                group.valid[:] = 0

    def add_rows(self, count):
        self.groups.append(CacheGroup(np.arange(self.num_rows, self.num_rows + count), None, np.zeros(count, dtype=np.int64)))
        self.num_rows += count

    def _coalesce(self):
        """Rejoin compatible timelines, notably once both reach the cache cap."""
        buckets = {}
        for group in self.groups:
            signature = (type(group.cache), tuple(
                (type(layer), None if layer.keys is None else layer.keys.shape[-2],
                 getattr(layer, "cumulative_length", None))
                for layer in getattr(group.cache, "layers", ())
            ))
            buckets.setdefault(signature, []).append(group)
        merged = []
        for groups in buckets.values():
            if len(groups) == 1:
                merged.append(groups[0])
                continue
            cache = groups[0].cache
            if cache is not None:
                cache = copy(cache)
                cache.layers = [copy(layer) for layer in groups[0].cache.layers]
                for index, layer in enumerate(cache.layers):
                    for name in ("keys", "values"):
                        if getattr(layer, name, None) is not None:
                            setattr(layer, name, torch.cat([getattr(g.cache.layers[index], name) for g in groups], dim=0))
            merged.append(CacheGroup(np.concatenate([g.rows for g in groups]), cache,
                                     np.concatenate([g.valid for g in groups])))
        self.groups = merged

    def predict(self, model, states, actions, bos, padding, *, active, max_len,
                coordinate, ingest_only=False):
        self.split(active)
        heads = None
        info = {}
        for group in self.groups:
            if not active[group.rows].all():
                continue
            full_batch = np.array_equal(group.rows, np.arange(self.num_rows))
            rows = slice(None) if full_batch else torch.as_tensor(group.rows, device=states.device)
            result = model._predict_incremental_cached(
                states=states[rows], actions=actions[rows], is_bos=bos[rows],
                padding_mask=padding[rows], past_key_values=group.cache,
                cache_max_len=max_len, return_info=True,
                past_valid_len=torch.as_tensor(group.valid, device=states.device),
                action_context_coordinate=coordinate, ingest_only=ingest_only,
            )
            if ingest_only:
                group.cache = result
            else:
                values, group.cache, group_info = result
                values = ensure_tensor_heads(values)
                if full_batch:
                    heads, info = values, group_info
                elif heads is None:
                    heads = values.new_zeros((self.num_rows, *values.shape[1:]))
                if not full_batch:
                    heads[rows] = values
                for key, value in (() if full_batch else group_info.items()):
                    if value is None:
                        info[key] = None
                    else:
                        if key not in info:
                            info[key] = value.new_zeros((self.num_rows, *value.shape[1:]))
                        info[key][rows] = value
            group.valid = np.minimum(group.valid + 1, cached_position_count(group.cache))
        self._coalesce()
        return heads, info
