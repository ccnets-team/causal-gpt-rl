"""PolicyRunner public step-wise inference API.

Wraps the rolling context buffer, KV cache, and state representation handling
around a trained AutoregressiveModel and exposes a small
`reset(state) / act(state) -> env_action` interface. After reset, callers may
also use `act()` because the initial state is already seeded in the buffer.

This module deliberately does not contain training logic such as online
normalizer updates, rollout collection, or learning.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import torch

from ..model.autoregressive_model import AutoregressiveModel
from ..model.schema import DataSpec, ensure_tensor_heads
from ..model.utils.kv_cache import cache_has_history
from .checkpoint import load_inference_checkpoint
from .context.buffer import ContextBuffer
from .state_normalizer import StateNormalizer

ActionMode = Literal["continuous", "discrete", "multi_discrete"]
DEFAULT_KV_CACHE_CONTEXT_MULTIPLIER = 4


class PolicyRunner:
    """Step-wise interface for running a trained autoregressive policy."""

    def __init__(
        self,
        model: AutoregressiveModel,
        *,
        action_mode: ActionMode,
        action_head_sizes: Iterable[int],
        state_size: int,
        action_size: int,
        context_length: int,
        state_normalizer: Optional[StateNormalizer] = None,
        num_envs: int = 1,
        kv_cache_max_len: Optional[int] = None,
        use_windowed: bool = False,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ):
        if action_mode not in ("continuous", "discrete", "multi_discrete"):
            raise ValueError(f"Unsupported action_mode: {action_mode}")

        self.model = model
        self.action_mode: ActionMode = action_mode
        self.action_head_sizes = [int(s) for s in action_head_sizes]
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.context_length = int(context_length)
        self.num_envs = int(num_envs)
        self.use_windowed = bool(use_windowed)
        if self.context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {context_length}")
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        self.default_kv_cache_max_len = (
            self.context_length * DEFAULT_KV_CACHE_CONTEXT_MULTIPLIER
        )
        self.kv_cache_max_len = (
            self.default_kv_cache_max_len
            if kv_cache_max_len is None
            else int(kv_cache_max_len)
        )
        if self.kv_cache_max_len <= 0:
            raise ValueError(f"kv_cache_max_len must be > 0, got {kv_cache_max_len}")
        self.state_normalizer = self._resolve_state_normalizer(state_normalizer)
        self.action_low = (
            None if action_low is None else np.asarray(action_low, dtype=np.float32)
        )
        self.action_high = (
            None if action_high is None else np.asarray(action_high, dtype=np.float32)
        )

        kv_limit = None if self.use_windowed else self.kv_cache_max_len
        self.buffer = ContextBuffer(
            num_agents=self.num_envs,
            context_length=self.context_length,
            state_size=self.state_size,
            action_size=self.action_size,
            kv_cache_max_len=kv_limit,
        )
        self._last_buffer_action: Optional[np.ndarray] = None
        self._is_reset = False
        self._reset_kv_after_next_act = False

        self.model.eval()
        if self.state_normalizer is not None:
            self.state_normalizer.to(self.model.device).eval()

    def _resolve_state_normalizer(
        self,
        state_normalizer: Optional[StateNormalizer],
    ) -> Optional[StateNormalizer]:
        if state_normalizer is None:
            return None
        if hasattr(self.model, "has_embedded_state_normalizer") and (
            self.model.has_embedded_state_normalizer()
        ):
            return None
        if hasattr(self.model, "set_state_normalization_from_state_dict"):
            self.model.set_state_normalization_from_state_dict(
                state_normalizer.state_dict()
            )
            return None
        return state_normalizer

    def reset(self, initial_state) -> None:
        """Reset internal buffers and seed with the initial observation."""
        state = self._format_state(initial_state)
        self.buffer.reset_context()
        zeros = np.zeros((self.num_envs, self.action_size), dtype=np.float32)
        self.buffer.update_data(state, zeros, is_bos=1.0)
        self._last_buffer_action = None
        self._is_reset = True
        self._reset_kv_after_next_act = True

    @torch.inference_mode()
    def act(self, state=None) -> np.ndarray:
        """Predict the next env-ready action for the current state."""
        env_action, _ = self._step(state, return_info=False)
        return env_action

    @torch.inference_mode()
    def act_with_info(self, state=None) -> tuple[np.ndarray, dict]:
        """Like `act()`, but also returns auxiliary per-step outputs.

        The info dict carries `termination_prob` (float for `num_envs == 1`,
        else a per-env array; `None` when the model has no EOS head). This is
        the opt-in companion to `act()` — the action contract is unchanged.
        """
        return self._step(state, return_info=True)

    def _step(self, state, *, return_info: bool) -> tuple[np.ndarray, dict]:
        if state is not None:
            self.observe(state)
        elif not self._is_reset:
            raise RuntimeError("Call reset(initial_state) before act().")

        states, actions, is_bos, mask, past_kv = self.buffer.get_context()
        device = self.model.device
        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        is_bos_t = torch.as_tensor(is_bos, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).bool()

        states_t = self._normalize_states_for_inference(states_t)

        info_raw: Optional[dict] = None
        if self.use_windowed:
            result = self.model.predict_with_window(
                states=states_t,
                actions=actions_t,
                is_bos=is_bos_t,
                padding_mask=mask_t,
                return_info=return_info,
            )
            if return_info:
                next_action, info_raw = result
            else:
                next_action = result
        else:
            if not cache_has_history(past_kv):
                states_t = states_t[:, -1:]
                actions_t = actions_t[:, -1:]
                is_bos_t = is_bos_t[:, -1:]
                mask_t = mask_t[:, -1:]
            result = self.model.predict_incremental_cached(
                states=states_t,
                actions=actions_t,
                is_bos=is_bos_t,
                padding_mask=mask_t,
                past_key_values=past_kv,
                cache_max_len=self.kv_cache_max_len,
                return_info=return_info,
            )
            if return_info:
                next_action, past_kv, info_raw = result
            else:
                next_action, past_kv = result
            if self._reset_kv_after_next_act:
                self.buffer.set_past_key_values(None)
                self._reset_kv_after_next_act = False
            else:
                self.buffer.set_past_key_values(past_kv)

        last_step = ensure_tensor_heads(next_action)[:, -1]
        env_action, buffer_action = self._decode(last_step.detach().cpu().numpy())
        self._last_buffer_action = buffer_action
        return env_action, self._build_step_info(info_raw)

    def _build_step_info(self, info_raw: Optional[dict]) -> dict:
        """Reduce a raw model info dict to last-step, env-facing values."""
        info: dict = {}
        if not info_raw:
            return info
        term = info_raw.get("termination_prob")
        if term is None:
            info["termination_prob"] = None
        else:
            t = term[:, -1].detach().cpu().numpy().reshape(-1)
            info["termination_prob"] = float(t[0]) if self.num_envs == 1 else t
        return info

    def observe(self, state) -> None:
        """Record a new observation after the previously emitted action."""
        state_arr = self._format_state(state)
        if not self._is_reset:
            self.reset(state_arr)
            return
        if self._last_buffer_action is not None:
            self.buffer.update_data(state_arr, self._last_buffer_action, is_bos=0.0)
        # Otherwise the state was already placed via reset().

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        model: AutoregressiveModel,
        action_specs: list[DataSpec],
        state_size: int,
        context_length: int,
        map_location: str | torch.device = "cpu",
        num_envs: int = 1,
        kv_cache_max_len: Optional[int] = None,
        use_windowed: bool = False,
    ) -> "PolicyRunner":
        """Load a training checkpoint into `model` and build a runner."""
        ckpt = load_inference_checkpoint(checkpoint_path, map_location=map_location)
        model.load_state_dict(ckpt["model_state"], strict=False)

        normalizer: Optional[StateNormalizer] = None
        if "state_normalizer_state" in ckpt:
            normalizer = StateNormalizer.from_state_dict(ckpt["state_normalizer_state"])
            normalizer.to(model.device)

        action_mode, head_sizes, action_size, lows, highs = cls._resolve_action_specs(
            action_specs
        )
        return cls(
            model=model,
            action_mode=action_mode,
            action_head_sizes=head_sizes,
            state_size=state_size,
            action_size=action_size,
            context_length=context_length,
            state_normalizer=normalizer,
            num_envs=num_envs,
            kv_cache_max_len=kv_cache_max_len,
            use_windowed=use_windowed,
            action_low=lows,
            action_high=highs,
        )

    def _format_state(self, state) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (self.num_envs, self.state_size):
            raise ValueError(
                f"Expected state of shape ({self.num_envs}, {self.state_size}), "
                f"got {arr.shape}"
            )
        return arr

    def _normalize_states_for_inference(self, states: torch.Tensor) -> torch.Tensor:
        if self.state_normalizer is not None:
            return self.state_normalizer(states)
        if hasattr(self.model, "normalize_states_for_inference"):
            return self.model.normalize_states_for_inference(states)
        return states.to(dtype=torch.float32)

    def _decode(self, action: np.ndarray):
        """Return (env_action, buffer_action) from the model's last-step output."""
        if self.action_mode == "continuous":
            raw = action.astype(np.float32)
            if self.action_low is not None and self.action_high is not None:
                env_action = np.clip(raw, self.action_low, self.action_high).astype(
                    np.float32
                )
            else:
                env_action = raw
            if self.num_envs == 1:
                env_action = env_action[0]
            return env_action, raw

        if self.action_mode == "discrete":
            n = self.action_head_sizes[0]
            if action.shape[-1] == n:
                idx = np.argmax(action, axis=-1).astype(np.int64)
            else:
                idx = np.rint(action[..., 0]).astype(np.int64)
                idx = np.clip(idx, 0, n - 1)
            buffer_action = self._one_hot(idx, n)
            env_action = int(idx[0]) if self.num_envs == 1 else idx
            return env_action, buffer_action

        if action.shape[-1] == len(self.action_head_sizes):
            idxs = np.rint(action).astype(np.int64)
            for head_idx, n in enumerate(self.action_head_sizes):
                idxs[:, head_idx] = np.clip(idxs[:, head_idx], 0, n - 1)
        else:
            splits = np.split(action, np.cumsum(self.action_head_sizes)[:-1], axis=-1)
            idxs = np.stack(
                [np.argmax(split, axis=-1).astype(np.int64) for split in splits],
                axis=1,
            )
        parts = [
            self._one_hot(idxs[:, head_idx], n)
            for head_idx, n in enumerate(self.action_head_sizes)
        ]
        buffer_action = np.concatenate(parts, axis=1).astype(np.float32)
        env_action = idxs[0] if self.num_envs == 1 else idxs
        return env_action, buffer_action

    @staticmethod
    def _resolve_action_specs(specs: list[DataSpec]):
        types = [s.type for s in specs]
        if all(t == "continuous" for t in types):
            mode: ActionMode = "continuous"
        elif all(t == "discrete" for t in types):
            mode = "discrete"
        elif all(t == "multi_discrete" for t in types):
            mode = "multi_discrete"
        else:
            raise ValueError(f"Mixed action types are not supported: {types}")

        head_sizes = [int(s.size) for s in specs]
        action_size = int(sum(head_sizes))

        lows: Optional[np.ndarray] = None
        highs: Optional[np.ndarray] = None
        if mode == "continuous":
            lows = np.concatenate(
                [np.asarray(s.low, dtype=np.float32).reshape(-1) for s in specs]
            )
            highs = np.concatenate(
                [np.asarray(s.high, dtype=np.float32).reshape(-1) for s in specs]
            )
        return mode, head_sizes, action_size, lows, highs

    @staticmethod
    def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
        out = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
        out[np.arange(indices.shape[0]), indices] = 1.0
        return out

    def __repr__(self) -> str:
        return (
            "PolicyRunner("
            f"context_length={self.context_length}, "
            f"kv_cache_max_len={self.kv_cache_max_len}, "
            f"use_windowed={self.use_windowed}, "
            f"num_envs={self.num_envs})"
        )
