"""Rolling context buffer for public inference.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

import numpy as np
from .cache import ContextCache

class ContextBuffer:
    def __init__(self, num_agents: int, context_length: int, state_size: int, action_size: int, kv_cache_max_len: int | None = None):
        """
        Fixed-size context buffer used by environment-backed evaluation.

        External `context_length` = number of tokens the model sees (= train
        input length). Internally we allocate context_length + 1 slots: the
        trailing slot holds the just-taken action that get_context() strips
        away via [:, :-1]. So get_context() always returns exactly
        `context_length` tokens.

        Args:
            num_agents: Number of parallel agents/environments.
            context_length: Tokens the model sees; get_context() returns this.
            state_size: Flattened state dimension.
            action_size: Flattened action dimension.
            kv_cache_max_len: Optional maximum retained KV-cache length.
        """
        if int(num_agents) <= 0:
            raise ValueError(f"num_agents must be > 0, got {num_agents}")
        if int(context_length) <= 0:
            raise ValueError(f"context_length must be > 0, got {context_length}")
        if int(state_size) <= 0:
            raise ValueError(f"state_size must be > 0, got {state_size}")
        if int(action_size) <= 0:
            raise ValueError(f"action_size must be > 0, got {action_size}")
        if kv_cache_max_len is not None and int(kv_cache_max_len) <= 0:
            raise ValueError(f"kv_cache_max_len must be > 0, got {kv_cache_max_len}")

        self.num_agents = int(num_agents)
        self.context_length = int(context_length)
        self._internal_len = self.context_length + 1

        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.state_type = np.float32
        self.action_type = np.float32
        # Create initial random data and mask
        self.states = np.zeros((self.num_agents, self._internal_len, self.state_size)).astype(self.state_type)
        self.actions = np.zeros((self.num_agents, self._internal_len, self.action_size)).astype(self.action_type)
        # Default to BOS=1 everywhere; real-step positions are flipped to 0
        # as update_data fills the buffer with actual rollout data.
        self.is_bos = np.ones((self.num_agents, self._internal_len, 1), dtype=np.float32)
        self.masks = np.zeros((self.num_agents, self._internal_len), dtype=np.float32)
        self.cache = ContextCache(kv_cache_max_len=kv_cache_max_len)

    def set_kv_cache_max_len(self, kv_cache_max_len: int | None) -> None:
        self.cache.set_kv_cache_max_len(kv_cache_max_len)

    def set_past_key_values(self, past_key_values) -> None:
        self.cache.set_past_key_values(past_key_values)

    def get_past_key_values(self):
        return self.cache.get_past_key_values()

    def get_kv_cache_length(self):
        return self.cache.get_kv_cache_length()

    def reset_context(self) -> None:
        """Reset masks, cached keys/values, and buffered state/action tensors."""
        self.masks.fill(0.0)
        self.cache.reset()
        self.states = np.zeros((self.num_agents, self._internal_len, self.state_size)).astype(self.state_type)
        self.actions = np.zeros((self.num_agents, self._internal_len, self.action_size)).astype(self.action_type)
        self.is_bos = np.ones((self.num_agents, self._internal_len, 1), dtype=np.float32)

    def reset_context_rows(self, reset_mask) -> None:
        reset_mask = np.asarray(reset_mask, dtype=bool).reshape(-1)
        if reset_mask.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected reset_mask for {self.num_agents} agents, got {reset_mask.shape[0]}"
            )
        if not reset_mask.any():
            return

        self.states[reset_mask] = 0.0
        self.actions[reset_mask] = 0.0
        self.is_bos[reset_mask] = 1.0
        self.masks[reset_mask] = 0.0
        # Cache entries are shared across agents in the current implementation,
        # so a partial reset must invalidate the shared cache to avoid mixing
        # a fresh episode with stale keys/values from a previous one.
        self.cache.reset()

    def update_data(
        self,
        dec_next_states: np.ndarray,
        dec_actions: np.ndarray,
        is_bos: float = 0.0,
    ) -> None:
        """
        Shift the rolling context and append the newest state/action pair.

        `is_bos` marks whether the action just placed in the buffer is a
        BOS/episode-start placeholder (1.0) or a real emitted action (0.0).
        """

        # Shift the sequence by one step to the left
        self.states = np.roll(self.states, shift=-1, axis=1)
        # Place the new observations at the last position
        self.states[:, -1] = dec_next_states
        if is_bos != 0.0:
            self.states[:, -2] = dec_next_states

        # Shift the sequence by one step to the left
        self.actions = np.roll(self.actions, shift=-1, axis=1)
        # Place the new observations at the last position
        self.actions[:, -2] = dec_actions

        # Mirror action layout for is_bos so the indicator stays aligned
        # with the action it describes (slot -2 in the rolling window).
        self.is_bos = np.roll(self.is_bos, shift=-1, axis=1)
        self.is_bos[:, -2] = float(is_bos)

        # Update the mask similarly
        self.masks = np.roll(self.masks, shift=-1, axis=1)
        self.masks[:, -2] = 1.0

    def get_context(self):
        # IMPORTANT: keep project-specific ordering as (state, action), not (next_state, action)
        states  = self.states.copy()
        actions = self.actions.copy()
        is_bos = self.is_bos.copy()
        masks   = self.masks.copy()
        states_cut  = states[:, :-1]
        actions_cut = actions[:, :-1]
        is_bos_cut = is_bos[:, :-1]
        masks_cut   = masks[:, :-1]
        past_key_values = self.get_past_key_values()

        return states_cut, actions_cut, is_bos_cut, masks_cut, past_key_values
    

    def __repr__(self) -> str:
        return (f"ContextBuffer(num_agents={self.num_agents}x, "
                f"state_size={self.state_size})")
