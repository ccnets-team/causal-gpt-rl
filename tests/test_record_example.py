"""The recording example maps an episode's ending onto the right flag, and says
so on a console that cannot spell.

`record_episodes` is where a time limit could quietly become a termination: the
loop sees `terminated`, `truncated`, and its own `--max-steps` cap, and has to
tell the environment reaching a terminal state apart from running out of time.
What it prints matters too — a Windows console on cp949 raises on a non-ASCII
write, which would fail a run whose episodes are already safely on disk.
The rest of the recording is `CollectionRunner`'s and is tested there.
"""

import sys

import numpy as np
import pytest

from collection.runner import CollectionRunner
from examples.deploy import record
from examples.deploy.record import record_episodes, summarize


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
    """Ends after `terminate_after` / `truncate_after` steps, or never.

    Both can fire on the same step, which is what a Gymnasium `TimeLimit` does
    when the wrapped env terminates on its last allowed step.
    """

    def __init__(self, *, terminate_after=None, truncate_after=None):
        self.terminate_after = terminate_after
        self.truncate_after = truncate_after
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
        truncated = (
            self.truncate_after is not None and self._step >= self.truncate_after
        )
        return np.zeros(2, dtype=np.float32), 1.0, terminated, truncated, {}

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


def test_an_env_raising_both_flags_is_recorded_as_a_termination(tmp_path):
    # A Gymnasium TimeLimit returns truncated=True on the step it runs out on,
    # even when the wrapped env terminated on that same step.
    env = _FakeEnv(terminate_after=3, truncate_after=3)
    collector = _collector(tmp_path)

    results = record_episodes(env, collector, episodes=1, max_steps=10, seed_start=0)

    assert results[0]["terminated"] is True and results[0]["truncated"] is False
    episode = _episode(tmp_path / "ep_000000.npz")
    assert episode["terminations"][-1] and not episode["truncations"].any()


def test_an_env_truncation_alone_stays_a_truncation(tmp_path):
    env = _FakeEnv(truncate_after=3)
    collector = _collector(tmp_path)

    results = record_episodes(env, collector, episodes=1, max_steps=10, seed_start=0)

    assert results[0]["truncated"] is True and results[0]["terminated"] is False
    episode = _episode(tmp_path / "ep_000000.npz")
    assert episode["truncations"][-1] and not episode["terminations"].any()


def test_a_termination_on_the_cap_step_stays_a_termination(tmp_path):
    # Both endings land on the same step; the terminal state is the stronger
    # claim, and the flags must not contradict each other.
    env = _FakeEnv(terminate_after=4)
    collector = _collector(tmp_path)

    results = record_episodes(env, collector, episodes=1, max_steps=4, seed_start=0)

    assert results[0]["terminated"] is True and results[0]["truncated"] is False
    episode = _episode(tmp_path / "ep_000000.npz")
    assert episode["terminations"][-1] and not episode["truncations"].any()


# -- what it prints -----------------------------------------------------


@pytest.mark.parametrize("terminated", [0, 1])
def test_the_run_output_survives_a_legacy_code_page(tmp_path, capsys, terminated):
    env = _FakeEnv(terminate_after=2 if terminated else None)
    collector = _collector(tmp_path)

    results = record_episodes(env, collector, episodes=1, max_steps=3, seed_start=0)
    print(collector)
    summarize(results, tmp_path)

    # Raises UnicodeEncodeError on any character cp949 cannot spell, which is
    # what a Windows console does mid-run. The zero-termination note only
    # prints on one of these two paths, so both are walked.
    capsys.readouterr().out.encode("cp949")


def test_the_help_text_survives_a_legacy_code_page(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["record", "--help"])

    with pytest.raises(SystemExit):
        record.parse_args()

    capsys.readouterr().out.encode("cp949")


# -- the vector loop ----------------------------------------------------


class _VecStubRunner(_StubRunner):
    """The same surface, batched, with the per-row restart a vector run needs."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.restarts = []

    def act(self):
        return np.zeros((self.num_envs, 2), dtype=np.float32)

    def reset_rows(self, done_mask):
        self.restarts.append(np.asarray(done_mask).copy())


class _FakeVecEnv:
    """A vector env with Gymnasium's `NEXT_STEP` auto-reset.

    Each row ends after its own number of steps: the step it ends on carries the
    true final observation and flags, and the next one carries the new episode's
    first observation with a zero reward, no flags, and that row's action
    ignored. Reproducing that here is the point — the loop's one-step wait is
    what a same-step reset would get wrong.
    """

    def __init__(self, lengths, *, truncate=False, both_flags=False):
        self.num_envs = len(lengths)
        self.lengths = list(lengths)
        self.truncate = truncate
        self.both_flags = both_flags
        self.seeds = None
        self._step = np.zeros(self.num_envs, dtype=np.int64)
        self._seeding = np.zeros(self.num_envs, dtype=bool)

    def reset(self, seed=None):
        self.seeds = seed
        self._step[:] = 0
        self._seeding[:] = False
        return np.zeros((self.num_envs, 2), dtype=np.float32), {}

    def step(self, actions):
        observations = np.zeros((self.num_envs, 2), dtype=np.float32)
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for row in range(self.num_envs):
            if self._seeding[row]:
                self._seeding[row] = False
                self._step[row] = 0
                rewards[row] = 0.0
                continue
            self._step[row] += 1
            observations[row] = float(self._step[row])
            if self._step[row] >= self.lengths[row]:
                terminated[row] = self.both_flags or not self.truncate
                truncated[row] = self.both_flags or self.truncate
                self._seeding[row] = True
        return observations, rewards, terminated, truncated, {}

    def close(self):
        pass


def _vec_collector(tmp_path, num_envs):
    return CollectionRunner(_VecStubRunner(num_envs), tmp_path)


def test_the_vector_loop_writes_one_episode_per_row_ending(tmp_path):
    env = _FakeVecEnv([2, 3, 4])
    collector = _vec_collector(tmp_path, 3)

    results = record.record_vector_episodes(
        env, collector, episodes=4, seed_start=0
    )

    # Rows finish at 2, 3, 4 and 4 steps, so the target is reached on the step
    # row 0 ends for the second time.
    assert [r["steps"] for r in results[:4]] == [2, 3, 4, 2]
    assert [r["row"] for r in results[:4]] == [0, 1, 2, 0]
    assert all(r["terminated"] for r in results[:4])
    # Reward is 1.0 a step, and the step that seeds a row pays none of it.
    assert [r["return"] for r in results[:4]] == [2.0, 3.0, 4.0, 2.0]


def test_a_row_mid_episode_at_the_target_is_flushed_as_a_truncation(tmp_path):
    env = _FakeVecEnv([2, 3, 4])
    collector = _vec_collector(tmp_path, 3)

    results = record.record_vector_episodes(
        env, collector, episodes=4, seed_start=0
    )

    # Asking for 4 gets 5: row 1 was one transition into its second episode when
    # the target landed, and close() writes what a row has rather than dropping
    # it. The summary counts it the same as the files do.
    assert collector.episodes_written == 5
    assert len(results) == 5
    assert results[-1] == {
        "row": 1,
        "return": 1.0,
        "steps": 1,
        "terminated": False,
        "truncated": True,
    }


def test_only_the_first_episodes_are_seeded(tmp_path):
    env = _FakeVecEnv([2, 2, 2])
    collector = _vec_collector(tmp_path, 3)

    record.record_vector_episodes(env, collector, episodes=3, seed_start=7)

    # One seed per row on the one reset the loop performs; the auto-resets after
    # it are the env's and take none.
    assert env.seeds == [7, 8, 9]


def test_a_vector_env_raising_both_flags_is_recorded_as_a_termination(tmp_path):
    # A Gymnasium TimeLimit raises both when the wrapped env terminates on its
    # last allowed step; the recorded transition has to say one thing.
    env = _FakeVecEnv([3, 3], both_flags=True)
    collector = _vec_collector(tmp_path, 2)

    results = record.record_vector_episodes(
        env, collector, episodes=2, seed_start=0
    )

    assert [(r["terminated"], r["truncated"]) for r in results[:2]] == [
        (True, False),
        (True, False),
    ]
    episode = _episode(tmp_path / "ep_000000.npz")
    assert episode["terminations"].tolist() == [False, False, True]
    assert not episode["truncations"].any()


def test_a_vector_env_truncation_alone_stays_a_truncation(tmp_path):
    env = _FakeVecEnv([3, 3], truncate=True)
    collector = _vec_collector(tmp_path, 2)

    results = record.record_vector_episodes(
        env, collector, episodes=2, seed_start=0
    )

    assert [(r["terminated"], r["truncated"]) for r in results[:2]] == [
        (False, True),
        (False, True),
    ]
    episode = _episode(tmp_path / "ep_000000.npz")
    assert not episode["terminations"].any()
    assert episode["truncations"].tolist() == [False, False, True]


def test_only_the_rows_that_ended_are_restarted(tmp_path):
    env = _FakeVecEnv([2, 7])
    collector = _vec_collector(tmp_path, 2)

    record.record_vector_episodes(env, collector, episodes=3, seed_start=0)

    # Row 0 ends twice before row 1 ends once, and row 1 is never restarted
    # alongside it.
    assert [mask.tolist() for mask in collector.runner.restarts] == [
        [True, False],
        [True, False],
    ]


def test_the_vector_run_output_survives_a_legacy_code_page(tmp_path, capsys):
    env = _FakeVecEnv([2, 3])
    collector = _vec_collector(tmp_path, 2)

    results = record.record_vector_episodes(
        env, collector, episodes=2, seed_start=0
    )
    summarize(results, tmp_path)

    printed = capsys.readouterr().out
    printed.encode("cp949")  # raises if anything non-ASCII slipped in
