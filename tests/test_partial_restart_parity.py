"""Partial batch restarts must not touch the rows that did not restart.

`reset_rows` and `add_rows` change one part of a batch while the rest keeps
running. The rows that keep running are the contract: their next action has to
come out the same as if nothing had happened to their neighbours, because each
row is a separate environment and a separate episode.

The cached keys and values live in one tensor shared by every row, which is what
makes this worth pinning. A restarted row cannot have its columns cut out of
that tensor, so it is recorded as owning none of it and the next forward masks
its previous episode away. The alternative the runtime used to take — dropping
the whole cache and rebuilding it from the rolling window — silently cut every
surviving row's history down to `context_length`.

The control in each test is the same runner driven over the same observations
without the restart. Comparing against that, rather than against a full-window
runner, is deliberate: the cached and full-window paths differ slightly on their
own, and that difference is not what these tests are about.
"""
import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference.runner import PolicyRunner
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

_ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
# The measured difference is 0.0 — masking a row's whole past is bit-identical to
# having no past. The tolerance is here so a floating-point path change in a new
# transformers release does not fail the suite; it is still three orders of
# magnitude under the ~1e-3 shift a lost context produces.
_TOL = 1e-6


def _model(network="Llama") -> AutoregressiveModel:
    # Seed so every runner in a test shares identical weights and differs only in
    # what happens to the batch around the row under test.
    torch.manual_seed(0)
    return AutoregressiveModel(
        ModelConfig(d_model=32, num_heads=4, network_name=network),
        state_specs=[SpaceSpec(type="continuous", size=2, dtype=torch.float32,
                               low=[-1.0, -1.0], high=[1.0, 1.0])],
        action_specs=extract_data_specs_from_space(_ACTION_SPACE),
        device=torch.device("cpu"),
    )


def _runner(num_envs, *, windowed=False, ctx=6, kv=None, bos="retain",
            network="Llama") -> PolicyRunner:
    return PolicyRunner(
        model=_model(network),
        action_schedule=[("continuous", 2, None, None)],
        state_size=2,
        context_length=ctx,
        num_envs=num_envs,
        use_windowed=windowed,
        kv_cache_max_len=kv,
        bos_cache_mode=bos,
    )


def _obs(rng, n, envs=2):
    return [rng.standard_normal((envs, 2)).astype(np.float32) for _ in range(n)]


def test_reset_rows_leaves_a_survivor_bit_identical():
    """One row restarting must not move the other row's next action."""
    restarted = _runner(2)
    control = _runner(2)
    rng = np.random.default_rng(0)
    obs = _obs(rng, 6)

    for runner in (restarted, control):
        runner.reset(obs[0].copy())
        for t in range(1, 4):
            runner.act()
            runner.observe(obs[t].copy())

    # Row 1 restarts mid-episode in one runner and not in the other. Row 0 sees
    # the same observations either way, so its action must not move.
    restarted.reset_rows(np.array([False, True]))
    restarted.observe(obs[4].copy())
    control.observe(obs[4].copy())

    np.testing.assert_allclose(
        np.asarray(restarted.act())[0], np.asarray(control.act())[0], atol=_TOL
    )


def test_reset_rows_keeps_a_survivors_retention_past_the_window():
    """A survivor keeps the history it accrued, not `context_length` of it.

    With `kv_cache_max_len` above `context_length` the rolling window is no
    longer a complete record of the rollout, so a rebuild from it would be
    lossy. Nothing is rebuilt: the survivor's owned length only grows.
    """
    runner = _runner(2, ctx=6, kv=48)
    rng = np.random.default_rng(3)
    obs = _obs(rng, 24)

    runner.reset(obs[0].copy())
    for t in range(1, 20):
        runner.act()
        runner.observe(obs[t].copy())
    runner.act()
    before = runner.buffer.get_kv_valid_lengths().copy()
    assert before[0] > runner.context_length, "test needs history past the window"

    runner.reset_rows(np.array([False, True]))
    runner.observe(obs[20].copy())
    runner.act()

    after = runner.buffer.get_kv_valid_lengths()
    assert after[0] == before[0] + 1, "the survivor lost cached history"
    assert after[1] == 1, "the restarted row kept history it should not have"


def test_survivor_isolation_holds_on_a_learned_position_backbone():
    """The survivor guarantee is not a property of rotary embeddings.

    A restarted row keeps the position it restarted at, which is why the *other*
    half of the contract is backbone-dependent (see the test below). A surviving
    row's positions do not move either way, so its isolation is exact here too.
    """
    restarted = _runner(2, network="GPT-2", kv=48)
    control = _runner(2, network="GPT-2", kv=48)
    rng = np.random.default_rng(5)
    obs = _obs(rng, 26)

    for runner in (restarted, control):
        runner.reset(obs[0].copy())
        for t in range(1, 20):
            runner.act()
            runner.observe(obs[t].copy())

    restarted.reset_rows(np.array([False, True]))
    for t in range(20, 25):
        restarted.observe(obs[t].copy())
        control.observe(obs[t].copy())
        np.testing.assert_allclose(
            np.asarray(restarted.act())[0], np.asarray(control.act())[0], atol=_TOL
        )


def test_a_restarted_row_starts_as_though_freshly_reset():
    """The restarted row's first action equals a brand-new runner's first action.

    Its previous episode is still physically in the shared cache; being masked
    out has to be the same thing as it never having been there.

    Rotary backbones only, which is what every published bundle is. Masking the
    past cancels a uniform position offset under RoPE; a backbone with learned
    absolute positions (`GPT-2`) still embeds the position the row restarted at,
    and its first steps land ~1e-2 away from a fresh runner. This diff narrows
    that gap rather than closing it, and the docs say so.
    """
    for bos in ("retain", "discard"):
        batched = _runner(2, bos=bos)
        rng = np.random.default_rng(1)
        obs = _obs(rng, 6)

        batched.reset(obs[0].copy())
        for t in range(1, 4):
            batched.act()
            batched.observe(obs[t].copy())

        batched.reset_rows(np.array([False, True]))
        batched.observe(obs[4].copy())
        restarted_action = np.asarray(batched.act())[1]

        fresh = _runner(1, bos=bos)
        fresh.reset(obs[4][1:].copy())
        fresh_action = np.asarray(fresh.act()).reshape(-1)

        np.testing.assert_allclose(
            restarted_action, fresh_action, atol=_TOL,
            err_msg=f"bos_cache_mode={bos}",
        )


def test_add_rows_leaves_an_existing_row_bit_identical():
    """Growing the batch must not move an existing row's next action."""
    grown = _runner(1)
    control = _runner(1)
    rng = np.random.default_rng(2)
    obs = _obs(rng, 6, envs=1)

    for runner in (grown, control):
        runner.reset(obs[0].copy())
        for t in range(1, 4):
            runner.act()
            runner.observe(obs[t].copy())

    grown.add_rows(np.array([[0.25, -0.5]], dtype=np.float32))

    np.testing.assert_allclose(
        np.asarray(grown.act())[0],
        np.asarray(control.act()).reshape(-1),
        atol=_TOL,
    )


def test_add_rows_keeps_the_existing_rows_retention_past_the_window():
    """The same guarantee as `reset_rows`, for a batch that grows instead."""
    runner = _runner(1, ctx=6, kv=48)
    rng = np.random.default_rng(4)
    obs = _obs(rng, 24, envs=1)

    runner.reset(obs[0].copy())
    for t in range(1, 20):
        runner.act()
        runner.observe(obs[t].copy())
    runner.act()
    before = runner.buffer.get_kv_valid_lengths().copy()
    assert before[0] > runner.context_length, "test needs history past the window"

    runner.add_rows(np.array([[0.1, 0.2]], dtype=np.float32))
    runner.act()

    after = runner.buffer.get_kv_valid_lengths()
    assert after[0] == before[0] + 1, "the existing row lost cached history"
    assert after[1] == 1, "the appended row started with history"


def test_lockstep_reset_rows_tracks_full_window_after_restart():
    """When every row restarts together, nothing is masked and the cached path
    stays in lockstep with the full-window path across subsequent steps."""
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
        np.testing.assert_allclose(a_cached, a_windowed, atol=1e-5)
        cached.observe(obs[t].copy())
        windowed.observe(obs[t].copy())
