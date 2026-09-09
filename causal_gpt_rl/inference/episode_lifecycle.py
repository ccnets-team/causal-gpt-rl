"""Opt-in retain lifecycle. Discard continues through the original runner."""
import numpy as np
import torch

from .context.retained_cache import RetainedCache


class RetainEpisodeLifecycle:
    def _episode_mask(self, mask):
        if not self._is_reset:
            raise RuntimeError("Call reset(initial_state) first.")
        if (self.bos_cache_mode == "retain"
                and self._session_components != (id(self.model), id(self.state_normalizer))):
            raise RuntimeError("Model/normalizer replacement requires reset() before inference.")
        if mask is None:
            return np.ones(self.num_envs, dtype=bool)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != self.num_envs:
            raise ValueError(f"Expected a mask for {self.num_envs} rows.")
        return mask

    def _retain_cache_bank(self):
        cache = self.buffer.get_past_key_values()
        if not isinstance(cache, RetainedCache):
            cache = RetainedCache(cache, self.buffer.get_kv_valid_lengths())
            self.buffer.set_past_key_values(cache)
        return cache

    @torch.inference_mode()
    def _flush_retain(self, rows):
        rows = rows & self._retain_dirty
        if self.use_windowed or not rows.any():
            return
        cache = self._retain_cache_bank()
        states, actions, bos, mask, _ = self.buffer.get_context()
        device = self.model.device
        states = self._normalize_states_for_inference(torch.as_tensor(states, device=device))
        cache.predict(
            self.model, states, torch.as_tensor(actions, device=device),
            torch.as_tensor(bos, device=device), torch.as_tensor(mask, device=device).bool(),
            active=rows, max_len=self.kv_cache_max_len,
            coordinate=self.rollout_action_context_coordinate, ingest_only=True,
        )
        self._retain_dirty[rows] = False

    def _observe_retain(self, state, rows):
        if (rows & self._retain_awaiting).any():
            raise RuntimeError("Ended rows need restart_episode(initial_state), not observe().")
        seed = rows & self._pending_bos_mask
        consume = rows & self._retain_pending_action
        selected = seed | consume
        if not selected.any():
            return  # initial observation was already seeded by reset()
        actions = np.zeros((self.num_envs, self.action_size), dtype=np.float32)
        if self._last_buffer_action is not None:
            actions[consume] = self._last_buffer_action[consume]
        actions[seed] = 0
        self.buffer.update_rows(state, actions, selected, is_bos=seed.astype(np.float32))
        self._pending_bos_mask[selected] = False
        self._retain_pending_action[selected] = False
        self._retain_dirty[selected] = True

    def finish_episode(self, terminal_state=None, *, done_mask=None):
        """Finalize real feedback without generating an action or a new BOS.

        Retain only. Ended rows pause until restart_episode supplies reset
        observations; act() supplies placeholders for those paused rows.
        terminal_state, when supplied, must contain actual final observations.
        """
        if self.bos_cache_mode != "retain":
            raise RuntimeError("finish_episode requires bos_cache_mode='retain'.")
        rows = self._episode_mask(done_mask)
        if not rows.any():
            return
        if (rows & (self._retain_awaiting | ~self._retain_has_action)).any():
            raise RuntimeError("Each episode must emit an action and may be finished only once.")
        state = self.buffer.states[:, -1].copy() if terminal_state is None else self._format_state(terminal_state)
        # Check cache support before committing feedback or changing row state.
        if not self.use_windowed:
            self._retain_cache_bank()
        self._observe_retain(state, rows)
        self._flush_retain(rows)
        self._retain_awaiting[rows] = True

    def restart_episode(self, initial_state, *, terminal_state=None, done_mask=None):
        """Start natural next episodes; explicit reset/reset_rows still erase history.

        Inputs have the runner's full-batch observation shape; only done_mask
        rows are consumed. A preceding observe(final_state) or finish_episode
        is supported without inserting the final action twice. Discard accepts
        only all-row restarts; use advance() for partial discard transitions.
        """
        rows = self._episode_mask(done_mask)
        state = self._format_state(initial_state)
        if terminal_state is not None:
            self._format_state(terminal_state)  # validate before changing any rows
        if not rows.any():
            return
        if self.bos_cache_mode == "discard":
            if not rows.all():
                raise ValueError(
                    "Partial restart_episode in discard mode is unsupported; "
                    "use advance(observations, done_mask=...) to consume all rows once."
                )
            self.reset(initial_state)
            return
        unfinished = rows & ~self._retain_awaiting
        if (unfinished & ~self._retain_has_action).any():
            raise RuntimeError("Cannot restart twice or before the episode emits an action.")
        if unfinished.any():
            self.finish_episode(terminal_state, done_mask=unfinished)
        self.buffer.update_rows(state, np.zeros((self.num_envs, self.action_size), np.float32), rows, is_bos=1.0)
        self._retain_awaiting[rows] = False
        self._retain_has_action[rows] = False
        self._retain_pending_action[rows] = False
        self._retain_dirty[rows] = True
        self._pending_bos_mask[rows] = False
        if self._last_buffer_action is not None:
            self._last_buffer_action[rows] = 0

    def advance(self, observations, *, done_mask, terminal_states=None):
        """Consume one same-step/explicit-reset vector step exactly once.

        observations holds next observations for survivors and reset observations
        for done rows. Do not also observe/reset_rows for this environment step.
        NEXT_STEP adapters use finish_episode, then restart_episode on arrival.
        """
        rows = self._episode_mask(done_mask)
        state = self._format_state(observations)
        if terminal_states is not None:
            self._format_state(terminal_states)
        if self.bos_cache_mode == "discard":
            self.reset_rows(rows)
            self.observe(observations)
            return
        if (rows & ~self._retain_awaiting & ~self._retain_has_action).any():
            raise RuntimeError("Done rows must have emitted an action.")
        if ((~rows) & self._retain_awaiting).any():
            raise RuntimeError("Paused rows need a reset observation.")
        self.restart_episode(observations, terminal_state=terminal_states, done_mask=rows)
        self._observe_retain(state, ~rows)
