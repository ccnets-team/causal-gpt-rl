"""The recording example maps an episode's ending onto the right flag.

`record_episodes` is where a time limit could quietly become a termination: the
loop sees `terminated`, `truncated`, and its own `--max-steps` cap, and has to
tell the environment reaching a terminal state apart from running out of time.
The rest of the recording is `CollectionRunner`'s and is tested there.
"""

import numpy as np
import pytest

from collection.runner import CollectionRunner
from examples.deploy.record import record_episodes


class _StubRunner:
    """The `PolicyRunner` surface `CollectionRunner` reads, with a fixed action."""

    num_envs = 1
    obs_space = None
    action_space = None
    state_size = 2
    action_schedule = [
        (
            "continuous",
            2,
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )
    ]
    context_length = 8
    kv_cache_max_len = 8
    bos_cache_mode = "discard"
    use_windowed = False

    def reset(self, observation):
        pass

    def act(self):
        return np.zeros(2, dtype=np.float32)

    def observe(self, observation):
        pass


class _FakeEnv:
    """Terminates after `terminate_after` steps, or never."""

    def __init__(self, *, terminate_after=None):
        self.terminate_after = terminate_after
        self.seeds = []
        self._step = 0

    def reset(self, seed=None):
        self.seeds.append(seed)
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        terminated = (
            self.terminate_after is not None and self._step >= self.terminate_after
        )
        return np.zeros(2, dtype=np.float32), 1.0, terminated, False, {}

    def close(self):
        pass


def _collector(tmp_path):
    return CollectionRunner(_StubRunner(), tmp_path)


def _episode(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def test_the_step_cap_is_recorded_as_a_truncation(tmp_path):
    env = _FakeEnv()
    collector = _collector(tmp_path)

    results = record_episodes(
        env, collector, episodes=2, max_steps=5, seed_start=0
    )

    assert [r["steps"] for r in results] == [5, 5]
    assert [r["truncated"] for r in results] == [True, True]
    assert [r["terminated"] for r in results] == [False, False]
    for name in ("ep_000000.npz", "ep_000001.npz"):
        episode = _episode(tmp_path / name)
        # Running out of time is not the environment reaching a terminal state.
        assert episode["truncations"].tolist() == [False] * 4 + [True]
        assert not episode["terminations"].any()


def test_a_real_termination_is_recorded_as_one(tmp_path):
    env = _FakeEnv(terminate_after=3)
    collector = _collector(tmp_path)

    results = record_episodes(
        env, collector, episodes=1, max_steps=10, seed_start=0
    )

    assert results[0]["terminated"] is True and results[0]["steps"] == 3
    episode = _episode(tmp_path / "ep_000000.npz")
    assert episode["terminations"].tolist() == [False, False, True]
    assert not episode["truncations"].any()
    assert len(episode["observations"]) == 4  # the terminal state is recorded


def test_each_episode_gets_its_own_seed(tmp_path):
    env = _FakeEnv()
    collector = _collector(tmp_path)

    record_episodes(env, collector, episodes=3, max_steps=2, seed_start=7)

    assert env.seeds == [7, 8, 9]
    assert len(list(tmp_path.glob("ep_*.npz"))) == 3


def test_a_termination_on_the_cap_step_stays_a_termination(tmp_path):
    # Both endings land on the same step; the terminal state is the stronger
    # claim, and the flags must not contradict each other.
    env = _FakeEnv(terminate_after=4)
    collector = _collector(tmp_path)

    results = record_episodes(env, collector, episodes=1, max_steps=4, seed_start=0)

    assert results[0]["terminated"] is True and results[0]["truncated"] is False
    episode = _episode(tmp_path / "ep_000000.npz")
    assert episode["terminations"][-1] and not episode["truncations"].any()
