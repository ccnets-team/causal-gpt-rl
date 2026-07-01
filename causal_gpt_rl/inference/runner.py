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
from typing import Iterable, Optional

import gymnasium as gym
import numpy as np
import torch

from ..model.autoregressive_model import AutoregressiveModel
from ..model.schema import DataSpec, ensure_tensor_heads
from ..model.utils.kv_cache import cache_has_history
from .adapters import make_action_output_adapter, make_state_input_adapter
from .checkpoint import load_inference_checkpoint
from .context.buffer import ContextBuffer
from .state_normalizer import StateNormalizer

DEFAULT_KV_CACHE_CONTEXT_MULTIPLIER = 4


class PolicyRunner:
    """Step-wise interface for running a trained autoregressive policy."""

    def __init__(
        self,
        model: AutoregressiveModel,
        *,
        action_schedule: Iterable[tuple],
        state_size: int,
        context_length: int,
        state_normalizer: Optional[StateNormalizer] = None,
        num_envs: int = 1,
        kv_cache_max_len: Optional[int] = None,
        use_windowed: bool = False,
        obs_space=None,
        action_space=None,
    ):
        self.model = model
        # Per-head action schedule: [(type, size, low, high), ...]. Mixed
        # families are allowed; decoding dispatches per head. Restoring the
        # declared Gymnasium container (Tuple/Dict) is the output adapter's job
        # (P5), wired below from `action_space` and applied in `_decode`.
        self.action_schedule = [
            (
                str(t),
                int(size),
                None if low is None else np.asarray(low, dtype=np.float32).reshape(-1),
                None if high is None else np.asarray(high, dtype=np.float32).reshape(-1),
            )
            for (t, size, low, high) in action_schedule
        ]
        self.action_head_sizes = [size for (_, size, _, _) in self.action_schedule]
        types = [t for (t, _, _, _) in self.action_schedule]
        # Homogeneous fast-path family, or None for a mixed (hybrid) schedule.
        # Homogeneous decoding stays byte-for-byte identical to the legacy
        # single-family behavior.
        if bool(types) and all(t == "continuous" for t in types):
            self._homogeneous_mode: Optional[str] = "continuous"
        elif bool(types) and all(t == "discrete" for t in types):
            self._homogeneous_mode = "discrete"
        elif bool(types) and all(t == "multi_discrete" for t in types):
            self._homogeneous_mode = "multi_discrete"
        elif bool(types) and all(t == "multi_binary" for t in types):
            self._homogeneous_mode = "multi_binary"
        else:
            self._homogeneous_mode = None  # hybrid

        # Concatenated continuous bounds for the homogeneous-continuous clip
        # path (preserves legacy behavior).
        self.action_low: Optional[np.ndarray] = None
        self.action_high: Optional[np.ndarray] = None
        if self._homogeneous_mode == "continuous":
            lows = [low for (_, _, low, _) in self.action_schedule]
            highs = [high for (_, _, _, high) in self.action_schedule]
            if all(l is not None for l in lows) and all(h is not None for h in highs):
                self.action_low = np.concatenate([l.reshape(-1) for l in lows])
                self.action_high = np.concatenate([h.reshape(-1) for h in highs])

        self.state_size = int(state_size)
        # Declared Gymnasium spaces (deserialized by the loader). The input
        # adapter (P4) converts a structured env observation into the model's
        # canonical flat state; it is None for a plain Box / no space, leaving
        # the legacy raw-passthrough path byte-identical. The output adapter (P5)
        # restores the declared Dict/Tuple container on the emitted action; it is
        # None for a non-container action space, leaving the flat per-head path
        # byte-identical.
        self.obs_space = obs_space
        self.action_space = action_space
        self._input_adapter = make_state_input_adapter(obs_space)
        if (
            self._input_adapter is not None
            and self._input_adapter.flatdim != self.state_size
        ):
            raise ValueError(
                f"obs_space flatdim {self._input_adapter.flatdim} != state_size "
                f"{self.state_size}; the bundle's declared space and state specs "
                f"disagree."
            )
        self.action_size = int(sum(self.action_head_sizes))
        self._output_adapter = make_action_output_adapter(action_space)
        if (
            self._output_adapter is not None
            and self._output_adapter.flatdim != self.action_size
        ):
            raise ValueError(
                f"action_space flatdim {self._output_adapter.flatdim} != "
                f"action_size {self.action_size}; the bundle's declared space and "
                f"action specs disagree."
            )
        # A Discrete action carries a `start` offset, and gym.flatten subtracts it
        # (the model is trained on 0-based class indices), so the env-facing decode
        # must add it back. Read it from the declared space, NOT the model specs
        # (SpaceSpec drops start) — this keeps the fix serving-only, no schema or
        # trainer change. 0 when the space is absent (old bundles) or start == 0,
        # leaving those byte-identical. Container actions get start via the output
        # adapter's gym.unflatten, so only the bare-Discrete decode reads this.
        self._discrete_start = (
            int(action_space.start)
            if isinstance(action_space, gym.spaces.Discrete)
            else 0
        )
        # Same for a bare MultiDiscrete: a per-dimension start offset. None when
        # the space is absent / not MultiDiscrete / all-zero start, leaving the
        # decode untouched (byte-identical). Container leaves get start via the
        # output adapter's gym.unflatten, so only the bare path reads this.
        self._multi_discrete_start = None
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            md_start = np.asarray(
                getattr(action_space, "start", 0), dtype=np.int64
            ).reshape(-1)
            if md_start.size and np.any(md_start):
                self._multi_discrete_start = md_start
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
        # Rows flagged by `reset_rows` to be seeded as a fresh episode on the
        # next observe; empty (all-False) in the common all-envs-in-lockstep case.
        self._pending_bos_mask = np.zeros(self.num_envs, dtype=bool)

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
        self._pending_bos_mask[:] = False

    def reset_rows(self, done_mask) -> None:
        """Restart the episodes of a subset of envs; leave the rest untouched.

        In batched inference (``num_envs > 1``) envs terminate at different
        steps. `done_mask` is a boolean / 0-1 array of shape ``(num_envs,)``:
        True rows have their rolling context wiped and are seeded as a fresh
        episode on the next `observe`/`act`, while False rows keep their history
        and continue uninterrupted. The typical call order mirrors a vectorized
        env's auto-reset::

            action = runner.act(state)
            next_state, done = env.step(action)
            runner.reset_rows(done)   # next_state[done] is the new episode's obs
            runner.observe(next_state)

        The shared KV cache is invalidated and recomputed from the buffer on the
        next step, so surviving rows pay one warm-start recompute but never lose
        context. Call after at least one `act()`; use `reset()` to restart the
        whole batch.
        """
        if not self._is_reset:
            raise RuntimeError("Call reset(initial_state) before reset_rows().")
        mask = np.asarray(done_mask).reshape(-1).astype(bool)
        if mask.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected done_mask for {self.num_envs} envs, got {mask.shape[0]}"
            )
        if not mask.any():
            return
        # Wipe the flagged rows' buffered trajectory and drop the shared cache.
        self.buffer.reset_context_rows(mask)
        # Seed those rows as BOS on the next observe, and clear any stale action
        # so the fresh episode does not inherit the previous step's action.
        self._pending_bos_mask |= mask
        if self._last_buffer_action is not None:
            self._last_buffer_action[mask] = 0.0
        # Cache was just invalidated; recompute cleanly after the next act, the
        # same way a full reset does.
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
            is_bos = (
                self._pending_bos_mask.astype(np.float32)
                if self._pending_bos_mask.any()
                else 0.0
            )
            self.buffer.update_data(
                state_arr, self._last_buffer_action, is_bos=is_bos
            )
            self._pending_bos_mask[:] = False
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

        action_schedule = cls._resolve_action_specs(action_specs)
        return cls(
            model=model,
            action_schedule=action_schedule,
            state_size=state_size,
            context_length=context_length,
            state_normalizer=normalizer,
            num_envs=num_envs,
            kv_cache_max_len=kv_cache_max_len,
            use_windowed=use_windowed,
        )

    def _format_state(self, state) -> np.ndarray:
        if self._input_adapter is not None:
            arr = self._adapt_structured_state(state)
        else:
            arr = np.asarray(state, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        if arr.shape != (self.num_envs, self.state_size):
            raise ValueError(
                f"Expected state of shape ({self.num_envs}, {self.state_size}), "
                f"got {arr.shape}"
            )
        return arr

    def _adapt_structured_state(self, state) -> np.ndarray:
        """Structured env observation(s) -> ``(num_envs, state_size)`` canonical.

        For ``num_envs == 1``, `state` is a single structured observation. For
        ``num_envs > 1``, `state` must be an iterable of ``num_envs`` per-env
        observations (each flattened independently). Flatten happens here, before
        the buffer; normalization stays downstream in ``_step`` so the order is
        flatten -> normalize (canonical, one-hot tail identity).
        """
        if self.num_envs == 1:
            flat = np.asarray(self._input_adapter(state), dtype=np.float32)
            return flat.reshape(1, -1)
        observations = list(state)
        if len(observations) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} per-env observations, "
                f"got {len(observations)}"
            )
        rows = [
            np.asarray(self._input_adapter(obs), dtype=np.float32).reshape(-1)
            for obs in observations
        ]
        return np.stack(rows, axis=0)

    def _normalize_states_for_inference(self, states: torch.Tensor) -> torch.Tensor:
        if self.state_normalizer is not None:
            return self.state_normalizer(states)
        if hasattr(self.model, "normalize_states_for_inference"):
            return self.model.normalize_states_for_inference(states)
        return states.to(dtype=torch.float32)

    def _decode(self, action: np.ndarray):
        """Return (env_action, buffer_action) for the model's last-step output.

        ``_decode_flat`` decodes per head into the flat env representation and the
        AR-feedback buffer action. When the bundle declared a Dict/Tuple
        ``action_space``, the output adapter (P5) then restores that container on
        the env-facing side. The buffer action (model feedback) always stays flat
        and is never touched by the adapter — output == the next input (L2 §2).
        """
        env_action, buffer_action = self._decode_flat(action)
        if self._output_adapter is not None:
            env_action = self._restructure_env_action(action)
        return env_action, buffer_action

    def _restructure_env_action(self, action: np.ndarray):
        """Flat per-head model output -> the declared Dict/Tuple action container.

        Builds the env-ready ``gym.spaces.flatten``-convention vector (continuous
        heads clipped to bounds, categorical heads one-hot, declared/head order)
        and unflattens it per env through the output adapter. No continuous-first
        permutation — actions are declared order (L2 §2).
        """
        env_flat = self._gym_flatten_action(action)
        out = [self._output_adapter(env_flat[i]) for i in range(self.num_envs)]
        return out[0] if self.num_envs == 1 else out

    def _gym_flatten_action(self, action: np.ndarray) -> np.ndarray:
        """Per-head model output -> ``(num_envs, action_size)`` gym-flat, env-ready.

        Mirrors ``gym.spaces.flatten`` so the output adapter's unflatten inverts
        it exactly: continuous heads clipped to their bounds, categorical heads
        argmax'd to a one-hot, concatenated in declared (head) order. This is the
        clipped sibling of the AR ``buffer_action`` (which stays raw for feedback).
        """
        action = action.astype(np.float32)
        parts: list = []
        offset = 0
        for head_type, size, low, high in self.action_schedule:
            col = action[:, offset:offset + size]
            offset += size
            if head_type == "continuous":
                if low is not None and high is not None:
                    col = np.clip(col, low, high)
                parts.append(col.astype(np.float32))
            elif head_type == "multi_binary":
                # gym.flatten(MultiBinary) is the {0,1} n-vector itself, so the
                # head contributes its thresholded logits in place (no one-hot).
                parts.append((col > 0.0).astype(np.float32))
            else:
                idx = np.argmax(col, axis=-1).astype(np.int64)
                parts.append(self._one_hot(idx, size))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _decode_flat(self, action: np.ndarray):
        """Return (env_action, buffer_action) from the model's last-step output.

        Homogeneous single-family schedules use the legacy fast paths, kept
        byte-for-byte identical. Mixed (hybrid) schedules decode per head and
        return a flat per-head env representation; wrapping it into the declared
        Gymnasium container is the output adapter's job (see ``_decode``).
        """
        if self._homogeneous_mode == "continuous":
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

        if self._homogeneous_mode == "discrete":
            n = self.action_head_sizes[0]
            if action.shape[-1] == n:
                idx = np.argmax(action, axis=-1).astype(np.int64)
            else:
                idx = np.rint(action[..., 0]).astype(np.int64)
                idx = np.clip(idx, 0, n - 1)
            # one_hot stays over the 0-based class index (start-independent, AR
            # feedback); only the env-facing value carries the Discrete start.
            buffer_action = self._one_hot(idx, n)
            env_idx = idx + self._discrete_start
            env_action = int(env_idx[0]) if self.num_envs == 1 else env_idx
            return env_action, buffer_action

        if self._homogeneous_mode == "multi_discrete":
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
            # one_hot above used the 0-based class indices (AR feedback); only the
            # env-facing values carry the per-dimension MultiDiscrete start.
            if self._multi_discrete_start is not None:
                idxs = idxs + self._multi_discrete_start
            env_action = idxs[0] if self.num_envs == 1 else idxs
            return env_action, buffer_action

        if self._homogeneous_mode == "multi_binary":
            # Independent Bernoulli per element: threshold the raw logits at 0
            # (== prob 0.5). `gym.spaces.flatten(MultiBinary)` is the {0,1}
            # n-vector itself (no one-hot), so the thresholded vector is both the
            # env-facing action (int8) and the AR-feedback buffer (float32).
            binary = (action.astype(np.float32) > 0.0)
            buffer_action = binary.astype(np.float32)
            env_binary = binary.astype(np.int8)
            env_action = env_binary[0] if self.num_envs == 1 else env_binary
            return env_action, buffer_action

        return self._decode_hybrid(action)

    def _decode_hybrid(self, action: np.ndarray):
        """Flat per-head decode for mixed action families.

        Each head is decoded independently — continuous heads are clipped to
        their bounds, categorical heads are argmax'd to an integer index — and
        the flat buffer_action (continuous raw + categorical one-hot, in head
        order) is built for autoregressive feedback. The per-head env outputs
        are returned as a flat list; restoring the customer's declared container
        structure (Tuple/Dict) is deferred to the adapter layer.
        """
        action = action.astype(np.float32)
        env_parts: list = []
        buffer_parts: list = []
        offset = 0
        for head_type, size, low, high in self.action_schedule:
            col = action[:, offset:offset + size]
            offset += size
            if head_type == "continuous":
                if low is not None and high is not None:
                    env_col = np.clip(col, low, high).astype(np.float32)
                else:
                    env_col = col.astype(np.float32)
                env_parts.append(env_col)
                buffer_parts.append(col.astype(np.float32))
            elif head_type == "multi_binary":
                # Independent Bernoulli: threshold logits at 0. The {0,1} vector
                # is both the env value (int8, size n) and the AR feedback.
                binary = col > 0.0
                env_parts.append(binary.astype(np.int8))
                buffer_parts.append(binary.astype(np.float32))
            else:
                idx = np.argmax(col, axis=-1).astype(np.int64)
                env_parts.append(idx.reshape(self.num_envs, 1))
                buffer_parts.append(self._one_hot(idx, size))
        buffer_action = np.concatenate(buffer_parts, axis=1).astype(np.float32)
        env_action = [part[0] if self.num_envs == 1 else part for part in env_parts]
        return env_action, buffer_action

    @staticmethod
    def _resolve_action_specs(specs: list[DataSpec]):
        """Build the per-head action schedule ``[(type, size, low, high), ...]``.

        Mixed action families are allowed — the runner decodes per head. ``low``
        and ``high`` are populated only for continuous heads; categorical heads
        carry ``None``.
        """
        schedule: list[tuple] = []
        for s in specs:
            head_type = s.type
            size = int(s.size)
            if head_type == "continuous":
                low = np.asarray(s.low, dtype=np.float32).reshape(-1)
                high = np.asarray(s.high, dtype=np.float32).reshape(-1)
            else:
                low = None
                high = None
            schedule.append((head_type, size, low, high))
        return schedule

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
