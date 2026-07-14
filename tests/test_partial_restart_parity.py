"""End-to-end warm-start parity for partial batch restarts.

`reset_rows` / `add_rows` drop the shared KV cache. The next act must re-prime
each row that still carries history from its *full* buffered trajectory (a real
full-window warm-start), not collapse every row to its newest token. The
regression guarded here is the `_step` empty-cache slice (`[:, -1:]`) that used
to wipe surviving / pre-existing rows' context.

A `use_windowed=True` runner is the ground truth: it always reprocesses the full
masked window, so its action for a history-carrying row reflects that row's true
context at every step. Right after a restart, the cached runner's recompute must
reproduce that full-window action for the history-carrying row. The buggy slice
recomputed it from a single token, so it would diverge.

Both runners are built from identically-seeded weights so the only difference is
the cached vs. full-window forward path. `bos_cache_mode="retain"` is used so the
cached path targets the full-window path exactly (no discard offset); the fixed
warm-start code path is shared by both modes, so this still guards the fix.
"""
import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_CFG = ModelConfig(d_model=32, num_heads=4)
_ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
_TOL = 1e-5  # recompute must match full-window; the pre-fix slice diverged ~1e-3


def _model() -> AutoregressiveModel:
    # Seed so a cached runner and a windowed runner share identical weights and
    # differ only in the forward path under test.
    torch.manual_seed(0)
    return AutoregressiveModel(
        _CFG,
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=extract_data_specs_from_space(_ACTION_SPACE),
        device=torch.device("cpu"),
    )


def _runner(num_envs, *, windowed=False, ctx=6) -> PolicyRunner:
    return PolicyRunner(
        model=_model(),
        action_schedule=[("continuous", 2, None, None)],
        state_size=2,
        context_length=ctx,
        num_envs=num_envs,
        use_windowed=windowed,
        bos_cache_mode="retain",
    )


def _obs(rng, n, envs=2):
    return [rng.standard_normal((envs, 2)).astype(np.float32) for _ in range(n)]


def test_reset_rows_survivor_matches_full_window_at_recompute():
    """A staggered reset_rows must not wipe the surviving row's context: its
    recompute-step action equals the full-window action for that row."""
    cached = _runner(2)
    windowed = _runner(2, windowed=True)
    rng = np.random.default_rng(0)
    obs = _obs(rng, 6)

    cached.reset(obs[0].copy())
    windowed.reset(obs[0].copy())
    # Accrue several steps of real history for both rows.
    for t in range(1, 4):
        cached.act()
        windowed.act()
        cached.observe(obs[t].copy())
        windowed.observe(obs[t].copy())

    # Row 1 restarts mid-episode; row 0 survives and must keep its history.
    done = np.array([False, True])
    cached.reset_rows(done)
    windowed.reset_rows(done)
    cached.observe(obs[4].copy())
    windowed.observe(obs[4].copy())

    a_cached = np.asarray(cached.act())
    a_windowed = np.asarray(windowed.act())
    # Survivor (row 0): the recompute reprocessed its full trajectory.
    np.testing.assert_allclose(a_cached[0], a_windowed[0], atol=_TOL)


def test_add_rows_existing_row_matches_full_window_at_recompute():
    """Growing the batch must not wipe an existing row's context: its
    recompute-step action equals the full-window action for that row."""
    cached = _runner(1)
    windowed = _runner(1, windowed=True)
    rng = np.random.default_rng(1)
    obs = _obs(rng, 6, envs=1)

    cached.reset(obs[0].copy())
    windowed.reset(obs[0].copy())
    for t in range(1, 4):
        cached.act()
        windowed.act()
        cached.observe(obs[t].copy())
        windowed.observe(obs[t].copy())

    # Append one fresh row; existing row 0 keeps its full history.
    new_row = rng.standard_normal((1, 2)).astype(np.float32)
    cached.add_rows(new_row.copy())
    windowed.add_rows(new_row.copy())

    a_cached = np.asarray(cached.act())
    a_windowed = np.asarray(windowed.act())
    # Existing row (row 0): recompute reprocessed its full trajectory.
    np.testing.assert_allclose(a_cached[0], a_windowed[0], atol=_TOL)


def test_lockstep_reset_rows_tracks_full_window_after_restart():
    """When every row restarts together (same phase), the cached path stays in
    lockstep with the full-window path across subsequent steps."""
    cached = _runner(2)
    windowed = _runner(2, windowed=True)
    rng = np.random.default_rng(2)
    obs = _obs(rng, 8)

    cached.reset(obs[0].copy())
    windowed.reset(obs[0].copy())
    for t in range(1, 4):
        cached.act()
        windowed.act()
        cached.observe(obs[t].copy())
        windowed.observe(obs[t].copy())

    done = np.array([True, True])
    cached.reset_rows(done)
    windowed.reset_rows(done)
    cached.observe(obs[4].copy())
    windowed.observe(obs[4].copy())

    # All rows are fresh and advance together, so parity holds every step.
    for t in range(5, 8):
        a_cached = np.asarray(cached.act())
        a_windowed = np.asarray(windowed.act())
        np.testing.assert_allclose(a_cached, a_windowed, atol=_TOL)
        cached.observe(obs[t].copy())
        windowed.observe(obs[t].copy())
