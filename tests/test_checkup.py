"""The checkup example is exercised on structured and hybrid bundles.

Its first version only worked on flat continuous Box spaces: it indexed Minari
episodes as arrays of samples (a `KeyError` on a Dict episode), flattened
observations without the continuous-first permutation the runtime applies, and
scored every non-continuous action with a single argmax over the whole vector.
Continuous MuJoCo data walks none of those paths.

Self-contained: builds tiny bundles in a temp dir — a Dict observation space, a
Tuple action space with a Discrete head, and one with a MultiBinary head — and
stands a stub dataset in for Minari, so flattening and per-head scoring are the
only things under test.
"""
import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.inference.spaces import extract_data_specs_from_space
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig
from examples.deploy import checkup

_CONTEXT = 8
_CFG = ModelConfig(
    d_model=32,
    num_heads=4,
    context_length=_CONTEXT,
    use_eos=True,
)

_DICT_OBS = gym.spaces.Dict(
    {
        "kind": gym.spaces.Discrete(4),
        "pos": gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
    }
)
_FLAT_OBS = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
_BOX_ACTION = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
_DISCRETE_ACTION = gym.spaces.Tuple(
    (gym.spaces.Box(-1.0, 1.0, (2,), np.float32), gym.spaces.Discrete(3))
)
_BINARY_ACTION = gym.spaces.Tuple(
    (gym.spaces.Box(-1.0, 1.0, (2,), np.float32), gym.spaces.MultiBinary(3))
)


class _Norm:
    """Minimal duck-typed state normalizer (mean/var state_dict)."""

    def __init__(self, n: int):
        self.n = n

    def state_dict(self):
        return {"mean": torch.zeros(self.n), "var": torch.ones(self.n)}


def _build(tmp_path, obs_space, action_space):
    """Export a tiny bundle for these spaces and return its directory."""
    state_specs = extract_data_specs_from_space(obs_space)
    action_specs = extract_data_specs_from_space(action_space)
    state_size = int(gym.spaces.flatdim(obs_space))
    model = AutoregressiveModel(
        _CFG,
        state_specs=state_specs,
        action_specs=action_specs,
        device=torch.device("cpu"),
    )
    bundle.export_bundle(
        tmp_path,
        model=model,
        model_config=_CFG,
        state_specs=model.state_specs,
        action_specs=model.action_specs,
        context_length=_CONTEXT,
        obs_space=obs_space,
        action_space=action_space,
        state_normalizer=_Norm(state_size),
    )
    return tmp_path


def _stack(samples):
    """[sample, ...] -> the dict/tuple-of-arrays layout Minari writes."""
    first = samples[0]
    if isinstance(first, dict):
        return {k: _stack([s[k] for s in samples]) for k in first}
    if isinstance(first, tuple):
        return tuple(_stack([s[i] for s in samples]) for i in range(len(first)))
    return np.asarray(samples)


class _StubEpisode:
    """One episode stored the way Minari stores it: fields of stacked arrays."""

    def __init__(self, obs_space, action_space, steps, rng):
        self.observations = _stack([obs_space.sample() for _ in range(steps + 1)])
        self.actions = _stack([action_space.sample() for _ in range(steps)])
        self.rewards = rng.standard_normal(steps)
        self.terminations = np.zeros(steps, dtype=bool)
        self.truncations = np.zeros(steps, dtype=bool)
        self.terminations[-1] = True


class _StubDataset:
    def __init__(self, obs_space, action_space, episodes, steps):
        self.observation_space = obs_space
        self.action_space = action_space
        rng = np.random.default_rng(0)
        obs_space.seed(0)
        action_space.seed(0)
        self._episodes = [
            _StubEpisode(obs_space, action_space, steps, rng) for _ in range(episodes)
        ]

    def iterate_episodes(self):
        return iter(self._episodes)


@pytest.fixture
def stub_minari(monkeypatch):
    """Install a fake `minari` module returning the dataset handed to it."""

    def install(dataset):
        module = types.ModuleType("minari")
        module.load_dataset = lambda _id: dataset
        monkeypatch.setitem(sys.modules, "minari", module)
        return dataset

    return install


def _checks(report) -> dict:
    return {check["check"]: check for check in report["checks"]}


@pytest.mark.parametrize(
    "obs_space, action_space, expected",
    [
        # Dict observation: flattening must apply the continuous-first
        # permutation, and indexing must pick a timestep out of a dict.
        (_DICT_OBS, _BOX_ACTION, ["action[continuous:2] rmse"]),
        # Tuple action: the Discrete head is scored on its own slice, not by one
        # argmax over the concatenated vector.
        (
            _FLAT_OBS,
            _DISCRETE_ACTION,
            ["action[continuous:2] rmse", "action[discrete:3] top-1 match"],
        ),
        # MultiBinary decodes per bit rather than as a one-hot class.
        (
            _FLAT_OBS,
            _BINARY_ACTION,
            ["action[continuous:2] rmse", "action[multi_binary:3] bit match"],
        ),
    ],
    ids=["dict-obs", "tuple-discrete-action", "tuple-multibinary-action"],
)
def test_structured_and_hybrid_bundles_are_scored_per_head(
    tmp_path, stub_minari, obs_space, action_space, expected
):
    bundle_dir = _build(tmp_path, obs_space, action_space)
    stub_minari(_StubDataset(obs_space, action_space, episodes=4, steps=_CONTEXT + 4))
    report = checkup.run_checkup(str(bundle_dir), "stub/dataset-v0", 4, "cpu")

    checks = _checks(report)
    for label in expected:
        assert label in checks, f"missing {label}; got {sorted(checks)}"
    assert "normalized state" in checks
    assert "value vs return-to-go" in checks


def test_mismatched_spaces_are_refused(tmp_path, stub_minari):
    """Equal flat widths must not pass as a matching dataset."""
    bundle_dir = _build(tmp_path, _FLAT_OBS, _DISCRETE_ACTION)
    # Tuple(Box(2), Discrete(3)) flattens to width 5, and so does this Box.
    wrong = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
    stub_minari(_StubDataset(_FLAT_OBS, wrong, episodes=2, steps=_CONTEXT + 4))

    with pytest.raises(SystemExit):
        checkup.run_checkup(str(bundle_dir), "stub/dataset-v0", 2, "cpu")


def test_termination_check_reports_only_real_endings(tmp_path, stub_minari):
    """A window edge is not an episode end.

    Every stub episode ends well past the context window, so the check must say
    nothing ended inside it rather than labelling the last scored position as
    the ending.
    """
    bundle_dir = _build(tmp_path, _FLAT_OBS, _DISCRETE_ACTION)
    stub_minari(
        _StubDataset(_FLAT_OBS, _DISCRETE_ACTION, episodes=3, steps=_CONTEXT * 3)
    )
    report = checkup.run_checkup(str(bundle_dir), "stub/dataset-v0", 3, "cpu")

    check = _checks(report)["termination prob"]
    assert check["episodes_ending_in_window"] == 0
    assert check["at_episode_end"] is None


def test_short_episodes_keep_their_real_ending(tmp_path, stub_minari):
    """An episode that does end inside the window is counted."""
    bundle_dir = _build(tmp_path, _FLAT_OBS, _DISCRETE_ACTION)
    stub_minari(
        _StubDataset(_FLAT_OBS, _DISCRETE_ACTION, episodes=3, steps=_CONTEXT - 2)
    )
    report = checkup.run_checkup(str(bundle_dir), "stub/dataset-v0", 3, "cpu")

    check = _checks(report)["termination prob"]
    assert check["episodes_ending_in_window"] == 3
    assert check["at_episode_end"] is not None


def test_value_targets_are_one_step_ahead():
    """The head at t values step t+1, so its target skips the current step.

    A correlation cannot see a one-step shift, so the pairing is pinned here
    instead: with unit rewards the return remaining from t+1 is one less than
    the return remaining from t.
    """
    targets = checkup.value_targets([[1.0, 1.0, 1.0, 1.0]], context_length=4)

    # Return-to-go is [4, 3, 2, 1]; positions 0..2 take the value of the *next*
    # step, so 3, 2, 1 — not 4, 3, 2.
    np.testing.assert_allclose(targets, [[3.0, 2.0, 1.0]])


def test_value_targets_pad_short_episodes():
    targets = checkup.value_targets([[2.0, 2.0]], context_length=4)
    np.testing.assert_allclose(targets, [[2.0, 0.0, 0.0]])


def test_normalized_state_statistics_exclude_the_one_hot_tail(tmp_path, stub_minari):
    """Only the continuous block is normalized.

    A Dict observation carries a one-hot tail that passes through raw; folding
    it into the statistic would drag the mean toward the one-hot mean and read
    as a bundle/dataset mismatch on data that fits perfectly.
    """
    bundle_dir = _build(tmp_path, _DICT_OBS, _BOX_ACTION)
    stub_minari(_StubDataset(_DICT_OBS, _BOX_ACTION, episodes=4, steps=_CONTEXT + 4))
    report = checkup.run_checkup(str(bundle_dir), "stub/dataset-v0", 4, "cpu")

    # Dict(kind=Discrete(4), pos=Box(3)) -> 3 continuous dims, 4 one-hot dims.
    check = _checks(report)["normalized |z| > 5"]
    assert check["dims"].startswith("3 of 7")


def test_episodes_must_be_positive(tmp_path, stub_minari):
    bundle_dir = _build(tmp_path, _FLAT_OBS, _BOX_ACTION)
    stub_minari(_StubDataset(_FLAT_OBS, _BOX_ACTION, episodes=1, steps=_CONTEXT))

    with pytest.raises(SystemExit):
        checkup.run_checkup(str(bundle_dir), "stub/dataset-v0", 0, "cpu")
