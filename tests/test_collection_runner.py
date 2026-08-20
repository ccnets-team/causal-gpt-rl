"""CollectionRunner writes what the policy actually did, in the contract's shape.

The pairing, the terminal observation, and the two ways an episode closes are
what the wrapper exists to get right, so they are checked against
`_internal.contract.validate_raw_directory` — the same gate the packager
applies — rather than against the wrapper's own idea of a correct file.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from collection._internal.contract import ContractError, validate_raw_directory
from collection.runner import CollectionRunner


_BOX_SCHEDULE = [
    (
        "continuous",
        2,
        np.array([-1.0, -1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    )
]


class _StubRunner:
    """The `PolicyRunner` surface the wrapper reads, with scripted actions.

    Attribute-for-attribute what `CollectionRunner` touches; the real runner is
    driven end-to-end further down, which is what pins this list.
    """

    def __init__(
        self,
        *,
        actions,
        obs_space=None,
        action_space=None,
        state_size=2,
        action_schedule=None,
        num_envs=1,
    ):
        self.num_envs = num_envs
        self.obs_space = obs_space
        self.action_space = action_space
        self.state_size = state_size
        self.action_schedule = action_schedule or _BOX_SCHEDULE
        self.context_length = 8
        self.kv_cache_max_len = 16
        self.bos_cache_mode = "retain"
        self.use_windowed = False
        self._actions = list(actions)
        self._emitted = 0
        self.calls = []

    def reset(self, observation):
        self.calls.append(("reset", observation))

    def act(self):
        self.calls.append(("act", None))
        action = self._actions[self._emitted % len(self._actions)]
        self._emitted += 1
        return action

    def observe(self, observation):
        self.calls.append(("observe", observation))

    def reset_rows(self, done_mask):
        self.calls.append(("reset_rows", np.asarray(done_mask).copy()))


def _box_actions(count=4):
    return [np.full(2, 0.1 * (i + 1), dtype=np.float32) for i in range(count)]


def _observation(step):
    return np.full(2, float(step), dtype=np.float32)


def _drive(collector, *, steps, terminated=False, truncated=False, start=0):
    """One episode of `steps` transitions; the last carries the flags."""
    collector.reset(_observation(start))
    for step in range(steps):
        collector.act()
        last = step == steps - 1
        collector.observe(
            _observation(start + step + 1),
            float(step),
            terminated and last,
            truncated and last,
        )


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# -- pairing ------------------------------------------------------------


def test_records_the_action_act_returned_for_the_state_it_saw(tmp_path):
    actions = _box_actions(3)
    collector = CollectionRunner(_StubRunner(actions=actions), tmp_path)

    _drive(collector, steps=3, terminated=True)

    episode = _load(tmp_path / "ep_000000.npz")
    # actions[t] is what act() returned while the model held observations[t] —
    # not the context token's `(state, previous action)` pairing.
    np.testing.assert_allclose(episode["actions"], np.stack(actions))
    np.testing.assert_allclose(
        episode["observations"], np.stack([_observation(i) for i in range(4)])
    )
    np.testing.assert_allclose(episode["rewards"], [0.0, 1.0, 2.0])


def test_the_terminal_observation_is_recorded_but_not_fed_back(tmp_path):
    runner = _StubRunner(actions=_box_actions())
    collector = CollectionRunner(runner, tmp_path)

    _drive(collector, steps=2, terminated=True)

    fed = [value for kind, value in runner.calls if kind == "observe"]
    assert len(fed) == 1  # observation 1 only; observation 2 ended the episode
    np.testing.assert_allclose(fed[0], _observation(1))
    assert len(_load(tmp_path / "ep_000000.npz")["observations"]) == 3


def test_a_second_act_without_an_observe_is_refused(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)
    collector.reset(_observation(0))
    collector.act()

    with pytest.raises(RuntimeError, match="twice in a row"):
        collector.act()


def test_the_runners_state_taking_act_is_refused(tmp_path):
    # PolicyRunner.act(state) is observe(state) then act(): it would feed the
    # runner while bypassing the reward and flag channel, losing the step.
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)
    collector.reset(_observation(0))

    with pytest.raises(RuntimeError, match="observe"):
        collector.act(_observation(1))


def test_the_loop_calls_are_refused_before_reset(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)

    with pytest.raises(RuntimeError, match="before act"):
        collector.act()
    with pytest.raises(RuntimeError, match="before observe"):
        collector.observe(_observation(0), 0.0, True, False)


def test_an_observe_with_no_action_to_pair_is_refused(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)
    collector.reset(_observation(0))

    with pytest.raises(RuntimeError, match="without a preceding act"):
        collector.observe(_observation(1), 1.0, False, False)


# -- how an episode closes ----------------------------------------------


@pytest.mark.parametrize("flag", ["terminations", "truncations"])
def test_both_close_paths_flag_only_the_final_transition(tmp_path, flag):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)

    _drive(
        collector,
        steps=3,
        terminated=flag == "terminations",
        truncated=flag == "truncations",
    )

    episode = _load(tmp_path / "ep_000000.npz")
    other = "truncations" if flag == "terminations" else "terminations"
    assert episode[flag].tolist() == [False, False, True]
    assert not episode[other].any()
    validate_raw_directory(tmp_path)


def test_a_reset_mid_episode_truncates_and_drops_the_dangling_action(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)

    collector.reset(_observation(0))
    collector.act()
    collector.observe(_observation(1), 1.0, False, False)
    collector.act()  # no observe: this action did nothing that was seen
    collector.reset(_observation(0))

    episode = _load(tmp_path / "ep_000000.npz")
    assert len(episode["actions"]) == 1
    assert len(episode["observations"]) == 2
    assert episode["truncations"].tolist() == [True]
    assert not episode["terminations"].any()


def test_an_episode_with_no_completed_transition_writes_nothing(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)

    collector.reset(_observation(0))
    collector.act()
    collector.reset(_observation(0))
    collector.close()

    assert sorted(p.name for p in tmp_path.glob("*.npz")) == []
    assert collector.episodes_written == 0


def test_close_flushes_an_episode_the_loop_left_open(tmp_path):
    with CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path) as collector:
        collector.reset(_observation(0))
        collector.act()
        collector.observe(_observation(1), 1.0, False, False)

    episode = _load(tmp_path / "ep_000000.npz")
    assert episode["truncations"].tolist() == [True]
    validate_raw_directory(tmp_path)


def test_record_false_drives_the_runner_without_writing(tmp_path):
    runner = _StubRunner(actions=_box_actions())
    collector = CollectionRunner(runner, tmp_path)

    collector.reset(_observation(0), record=False)
    collector.act()
    collector.observe(_observation(1), 1.0, False, False)
    collector.observe(_observation(2), 1.0, True, False)

    assert list(tmp_path.glob("*.npz")) == []
    assert [kind for kind, _ in runner.calls] == ["reset", "act", "observe"]


def test_episodes_are_numbered_in_order(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)

    _drive(collector, steps=2, terminated=True)
    _drive(collector, steps=3, truncated=True)

    assert sorted(p.name for p in tmp_path.glob("*.npz")) == [
        "ep_000000.npz",
        "ep_000001.npz",
    ]
    files, _ = validate_raw_directory(tmp_path)
    assert len(files) == 2


# -- declared spaces -> the contract's declaration ----------------------


def test_a_structured_observation_keeps_one_channel_per_leaf(tmp_path):
    obs_space = gym.spaces.Dict(
        {
            "a_pos": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "b_kind": gym.spaces.Discrete(4),
        }
    )
    collector = CollectionRunner(
        _StubRunner(actions=_box_actions(), obs_space=obs_space, state_size=7),
        tmp_path,
    )
    observation = {"a_pos": np.zeros(3, dtype=np.float32), "b_kind": 2}

    collector.reset(observation)
    collector.act()
    collector.observe(observation, 1.0, True, False)

    assert collector.spec["obs_channels"] == [3, 4]
    row = _load(tmp_path / "ep_000000.npz")["observations"][0]
    # Declared (`gym.spaces.flatten`) order, not the model's continuous-first
    # canonical order: the file matches the spaces spec.json declares.
    np.testing.assert_allclose(row, gym.spaces.flatten(obs_space, observation))


def test_a_discrete_action_is_recorded_as_a_zero_based_index(tmp_path):
    action_space = gym.spaces.Discrete(4, start=2)
    collector = CollectionRunner(
        _StubRunner(
            actions=[3],
            action_space=action_space,
            action_schedule=[("discrete", 4, None, None)],
        ),
        tmp_path,
    )

    _drive(collector, steps=2, terminated=True)

    episode = _load(tmp_path / "ep_000000.npz")
    assert collector.spec["branches"] == [4]
    assert episode["actions"].dtype == np.int64
    # start is env-facing; the model is trained on 0-based classes, so 3 -> 1.
    assert episode["actions"].tolist() == [[1], [1]]
    validate_raw_directory(tmp_path)


def test_a_multi_discrete_action_is_recorded_per_branch(tmp_path):
    collector = CollectionRunner(
        _StubRunner(
            actions=[np.array([0, 2, 1], dtype=np.int64)],
            action_space=gym.spaces.MultiDiscrete([3, 3, 3]),
            action_schedule=[("multi_discrete", 3, None, None)] * 3,
        ),
        tmp_path,
    )

    _drive(collector, steps=2, terminated=True)

    assert collector.spec["branches"] == [3, 3, 3]
    assert _load(tmp_path / "ep_000000.npz")["actions"].tolist() == [[0, 2, 1]] * 2
    validate_raw_directory(tmp_path)


def test_a_multi_binary_action_becomes_one_two_way_branch_per_bit(tmp_path):
    collector = CollectionRunner(
        _StubRunner(
            actions=[np.array([1, 0, 1], dtype=np.int8)],
            action_space=gym.spaces.MultiBinary(3),
            action_schedule=[("multi_binary", 3, None, None)],
        ),
        tmp_path,
    )

    _drive(collector, steps=2, terminated=True)

    assert collector.spec["branches"] == [2, 2, 2]
    assert _load(tmp_path / "ep_000000.npz")["actions"].tolist() == [[1, 0, 1]] * 2
    validate_raw_directory(tmp_path)


def test_a_hybrid_action_stores_continuous_columns_then_indices(tmp_path):
    action_space = gym.spaces.Tuple(
        (
            gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            gym.spaces.Discrete(3),
        )
    )
    collector = CollectionRunner(
        _StubRunner(
            actions=[(np.array([0.5, -0.5], dtype=np.float32), 2)],
            action_space=action_space,
            action_schedule=[
                (
                    "continuous",
                    2,
                    np.array([-1.0, -1.0], dtype=np.float32),
                    np.array([1.0, 1.0], dtype=np.float32),
                ),
                ("discrete", 3, None, None),
            ],
        ),
        tmp_path,
    )

    _drive(collector, steps=2, terminated=True)

    assert collector.spec["continuous_size"] == 2
    assert collector.spec["branches"] == [3]
    # One array carries both parts, so the index rides along as an integral float.
    np.testing.assert_allclose(
        _load(tmp_path / "ep_000000.npz")["actions"], [[0.5, -0.5, 2.0]] * 2
    )
    validate_raw_directory(tmp_path)


def test_an_action_space_outside_the_contract_is_refused_up_front(tmp_path):
    action_space = gym.spaces.Dict(
        {"move": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)}
    )
    with pytest.raises(ContractError, match="no form in the input contract"):
        CollectionRunner(
            _StubRunner(actions=_box_actions(), action_space=action_space), tmp_path
        )
    assert not (tmp_path / "spec.json").exists()


def test_an_unbounded_action_box_is_refused_up_front(tmp_path):
    action_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    with pytest.raises(ContractError, match="non-finite bounds"):
        CollectionRunner(
            _StubRunner(actions=_box_actions(), action_space=action_space), tmp_path
        )


def test_a_batched_runner_without_per_row_restarts_is_refused(tmp_path):
    # Rows end at different steps, so ending one episode without ending every
    # row's needs `reset_rows`. A runner without it cannot record a batch.
    class _NoRestartRunner(_StubRunner):
        reset_rows = None

    with pytest.raises(ContractError, match="reset_rows"):
        CollectionRunner(
            _NoRestartRunner(actions=_box_actions(), num_envs=4), tmp_path
        )


def test_the_action_space_is_read_from_the_heads_when_undeclared(tmp_path):
    # Bundles predating declared spaces carry only the per-head schedule.
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)

    assert collector.spec["action_kind"] == "continuous"
    assert collector.spec["act_low"] == [-1.0, -1.0]
    assert collector.spec["act_high"] == [1.0, 1.0]
    spec = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert spec["provenance"][0]["action_space_declared"] is False


# -- provenance ---------------------------------------------------------


def test_spec_json_declares_the_spaces_and_keeps_the_provenance(tmp_path):
    runner = _StubRunner(actions=_box_actions())
    CollectionRunner(runner, tmp_path, bundle="ccnets/causal-gpt-rl@ant-v5")

    spec = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert spec["action_kind"] == "continuous"
    assert spec["obs_channels"] == [2]
    entry = spec["provenance"][0]
    assert entry["bundle"] == "ccnets/causal-gpt-rl@ant-v5"
    assert entry["context_length"] == runner.context_length
    assert entry["kv_cache_max_len"] == runner.kv_cache_max_len
    assert entry["bos_cache_mode"] == "retain"
    assert entry["recorder"] == "collection.CollectionRunner"
    assert entry["action_space"]["type"] == "Box"


def test_provenance_survives_the_packager_reading_the_same_file(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)
    _drive(collector, steps=2, terminated=True)

    _, spec = validate_raw_directory(tmp_path)

    assert spec["action_kind"] == "continuous"
    assert spec["obs_channels"] == [2]
    assert spec["provenance"][0]["recorder"] == "collection.CollectionRunner"


def test_a_directory_with_episodes_is_not_written_over(tmp_path):
    collector = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)
    _drive(collector, steps=2, terminated=True)

    with pytest.raises(FileExistsError, match="already holds 1 episodes"):
        CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)


def test_resume_continues_the_numbering_and_appends_a_provenance_entry(tmp_path):
    first = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)
    _drive(first, steps=2, terminated=True)

    second = CollectionRunner(
        _StubRunner(actions=_box_actions()), tmp_path, resume=True, bundle="second"
    )
    _drive(second, steps=3, terminated=True)

    assert sorted(p.name for p in tmp_path.glob("*.npz")) == [
        "ep_000000.npz",
        "ep_000001.npz",
    ]
    spec = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert [entry["bundle"] for entry in spec["provenance"]] == [None, "second"]
    files, _ = validate_raw_directory(tmp_path)
    assert len(files) == 2


def test_resume_refuses_a_run_that_declares_other_spaces(tmp_path):
    first = CollectionRunner(_StubRunner(actions=_box_actions()), tmp_path)
    _drive(first, steps=2, terminated=True)

    other = _StubRunner(
        actions=[1],
        action_space=gym.spaces.Discrete(4),
        action_schedule=[("discrete", 4, None, None)],
    )
    with pytest.raises(ContractError, match="not what this runner records"):
        CollectionRunner(other, tmp_path, resume=True)


# -- the real runner ----------------------------------------------------


def _policy_runner(**kwargs):
    """A small real `PolicyRunner`, so the coupling above is not self-referential."""
    torch = pytest.importorskip("torch")
    from causal_gpt_rl.inference.runner import PolicyRunner
    from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
    from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec

    model = AutoregressiveModel(
        ModelConfig(d_model=32, num_heads=4),
        state_specs=[
            SpaceSpec(
                type="continuous",
                size=3,
                dtype=torch.float32,
                low=[-1.0] * 3,
                high=[1.0] * 3,
            )
        ],
        action_specs=[
            SpaceSpec(
                type="continuous",
                size=2,
                dtype=torch.float32,
                low=[-1.0] * 2,
                high=[1.0] * 2,
                squash="tanh",
            )
        ],
        device=torch.device("cpu"),
    )
    return PolicyRunner(
        model=model,
        action_schedule=PolicyRunner._resolve_action_specs(model.action_specs),
        state_size=3,
        context_length=8,
        obs_space=gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        **kwargs,
    )


def test_a_real_rollout_records_what_act_returned(tmp_path):
    collector = CollectionRunner(_policy_runner(), tmp_path, bundle="test-bundle")

    emitted = []
    collector.reset(np.zeros(3, dtype=np.float32), record=True)
    for step in range(5):
        emitted.append(np.asarray(collector.act(), dtype=np.float32))
        observation = np.full(3, 0.1 * (step + 1), dtype=np.float32)
        collector.observe(observation, float(step), step == 4, False)

    episode = _load(tmp_path / "ep_000000.npz")
    np.testing.assert_allclose(episode["actions"], np.stack(emitted))
    assert episode["observations"].shape == (6, 3)
    assert episode["terminations"].tolist() == [False] * 4 + [True]
    files, spec = validate_raw_directory(tmp_path)
    assert len(files) == 1
    assert spec["obs_channels"] == [3]


def test_the_wrapper_delegates_everything_it_does_not_own(tmp_path):
    runner = _policy_runner(kv_cache_max_len=32)
    collector = CollectionRunner(runner, tmp_path)

    # Attributes the wrapper reads off the runner, pinned on the real class.
    for name in (
        "num_envs",
        "obs_space",
        "action_space",
        "state_size",
        "action_schedule",
        "context_length",
        "kv_cache_max_len",
        "bos_cache_mode",
        "use_windowed",
    ):
        assert hasattr(runner, name), name

    assert collector.kv_cache_max_len == 32
    assert collector.model is runner.model
    with pytest.raises(AttributeError):
        collector.no_such_attribute


def test_two_policy_runner_conventions_the_wrapper_absorbs(tmp_path):
    """`reset` seeds `(o0, zeros)` and `observe` writes the *previous* action."""
    runner = _policy_runner()
    collector = CollectionRunner(runner, tmp_path)
    collector.reset(np.zeros(3, dtype=np.float32))

    action = collector.act()
    # The buffer's newest action column is still the reset's zeros: the context
    # token is `(state, previous action)`, which is why the recorded pairing is
    # taken at the API boundary instead.
    _, actions, _, _, _ = runner.buffer.get_context()
    np.testing.assert_allclose(actions[0, -1], np.zeros(2, dtype=np.float32))
    assert np.asarray(action).shape == (2,)


# -- a batch of rows ----------------------------------------------------
#
# A vectorized env autoresets in `NEXT_STEP` mode: the step a row ends on
# carries its true final observation and flags, and the next one carries the new
# episode's first observation with a zero reward, no flags, and that row's
# action ignored. These drive that shape by hand; the integration test at the
# bottom drives it through a real `SyncVectorEnv` instead.


def _batched_actions(steps, num_envs=2):
    """Distinct per (step, row), so a misfiled action is visible in the file."""
    return [
        np.stack(
            [np.full(2, 0.1 * (step + 1) - 0.5 * row, dtype=np.float32)
             for row in range(num_envs)]
        )
        for step in range(steps)
    ]


def _batched_observation(step, num_envs=2):
    return np.stack(
        [np.full(2, float(step) + row, dtype=np.float32) for row in range(num_envs)]
    )


def test_rows_that_end_at_different_steps_each_write_their_own_episode(tmp_path):
    actions = _batched_actions(4)
    runner = _StubRunner(actions=actions, num_envs=2)
    collector = CollectionRunner(runner, tmp_path)

    collector.reset(_batched_observation(0))
    # step 0: both running.
    collector.act()
    collector.observe(_batched_observation(1), [0.0, 0.0], [False, False], [False, False])
    # step 1: row 0 terminates on its true final observation.
    collector.act()
    collector.observe(_batched_observation(2), [1.0, 1.0], [True, False], [False, False])
    # step 2: row 0's autoreset seed — zero reward, no flags, action ignored.
    collector.act()
    collector.observe(_batched_observation(3), [0.0, 2.0], [False, False], [False, False])
    # step 3: row 1 terminates; row 1 has run four transitions to row 0's two.
    collector.act()
    collector.observe(_batched_observation(4), [3.0, 3.0], [False, True], [False, False])
    collector.close()

    # Numbered in the order rows finish: row 0's first episode, then row 1's,
    # then row 0's second flushed by close().
    first = _load(tmp_path / "ep_000000.npz")
    second = _load(tmp_path / "ep_000001.npz")
    third = _load(tmp_path / "ep_000002.npz")

    assert first["actions"].shape == (2, 2)
    np.testing.assert_allclose(
        first["actions"], np.stack([actions[0][0], actions[1][0]])
    )
    np.testing.assert_allclose(
        first["observations"],
        np.stack([_batched_observation(s)[0] for s in range(3)]),
    )
    assert first["terminations"].tolist() == [False, True]

    assert second["actions"].shape == (4, 2)
    np.testing.assert_allclose(
        second["actions"], np.stack([actions[s][1] for s in range(4)])
    )
    assert second["rewards"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert second["terminations"].tolist() == [False, False, False, True]

    # Row 0's second episode: seeded at step 2, one transition at step 3, cut
    # short by close() — which is a truncation, not a termination.
    assert third["actions"].shape == (1, 2)
    np.testing.assert_allclose(third["actions"][0], actions[3][0])
    assert third["truncations"].tolist() == [True]

    files, _ = validate_raw_directory(tmp_path)
    assert len(files) == 3


def test_the_action_of_an_ignored_autoreset_step_is_not_recorded(tmp_path):
    actions = _batched_actions(3)
    collector = CollectionRunner(_StubRunner(actions=actions, num_envs=2), tmp_path)

    collector.reset(_batched_observation(0))
    collector.act()
    collector.observe(_batched_observation(1), [0.0, 0.0], [True, False], [False, False])
    # Row 0 is mid-autoreset here: the env ignores actions[1][0], so the file
    # must not carry it, and the observation seeds the next episode instead of
    # closing a transition.
    collector.act()
    collector.observe(_batched_observation(2), [0.0, 1.0], [False, False], [False, False])
    collector.act()
    collector.observe(_batched_observation(3), [2.0, 2.0], [True, True], [False, False])
    collector.close()

    reborn = _load(tmp_path / "ep_000001.npz")
    np.testing.assert_allclose(reborn["actions"], actions[2][0].reshape(1, 2))
    np.testing.assert_allclose(
        reborn["observations"],
        np.stack([_batched_observation(2)[0], _batched_observation(3)[0]]),
    )
    assert reborn["rewards"].tolist() == [2.0]


def test_only_the_rows_that_ended_are_restarted_in_the_runner(tmp_path):
    runner = _StubRunner(actions=_batched_actions(3), num_envs=2)
    collector = CollectionRunner(runner, tmp_path)

    collector.reset(_batched_observation(0))
    collector.act()
    collector.observe(_batched_observation(1), [0.0, 0.0], [True, False], [False, False])
    collector.act()
    collector.observe(_batched_observation(2), [0.0, 1.0], [False, False], [False, False])

    restarts = [call for call in runner.calls if call[0] == "reset_rows"]
    assert len(restarts) == 1, "the surviving row must not be restarted"
    assert restarts[0][1].tolist() == [True, False]
    # And it happens before the runner is fed, so the ended episode leaves the
    # row's context rather than preceding the new one.
    order = [name for name, _ in runner.calls]
    assert order.index("reset_rows") < len(order) - 1
    assert order[order.index("reset_rows") + 1] == "observe"


def test_a_batch_argument_must_carry_one_value_per_row(tmp_path):
    collector = CollectionRunner(
        _StubRunner(actions=_batched_actions(2), num_envs=2), tmp_path
    )
    collector.reset(_batched_observation(0))
    collector.act()

    with pytest.raises(ContractError, match="rewards for 2 envs"):
        collector.observe(_batched_observation(1), [0.0], [False, False], [False, False])


def test_a_batched_observation_of_the_wrong_width_is_refused(tmp_path):
    collector = CollectionRunner(
        _StubRunner(actions=_batched_actions(2), num_envs=2), tmp_path
    )
    with pytest.raises(ContractError, match="observations for 2 envs"):
        collector.reset(np.zeros((3, 2), dtype=np.float32))


def test_a_batched_run_records_how_many_envs_produced_it(tmp_path):
    CollectionRunner(_StubRunner(actions=_batched_actions(2), num_envs=2), tmp_path)

    spec = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert spec["provenance"][0]["num_envs"] == 2


# -- a batch of rows, through a real vector env -------------------------


class _CountdownEnv(gym.Env):
    """Terminates after a fixed number of steps, so rows stagger by construction."""

    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def __init__(self, length: int):
        self.length = length
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        observation = np.full(3, 0.01 * self._t, dtype=np.float32)
        return observation, float(self._t), self._t >= self.length, False, {}


def test_a_real_vector_env_rollout_splits_into_one_file_per_episode(tmp_path):
    pytest.importorskip("torch")
    venv = gym.vector.SyncVectorEnv(
        [lambda length=length: _CountdownEnv(length) for length in (2, 3)]
    )
    # The recorder owns a one-step wait that only NEXT_STEP autoreset needs.
    # Pin the mode rather than assume it, so a Gymnasium default change is a
    # failure here instead of a silently mispaired dataset.
    assert str(venv.metadata["autoreset_mode"]).endswith("NEXT_STEP")

    collector = CollectionRunner(_policy_runner(num_envs=2), tmp_path)
    try:
        observations, _ = venv.reset(seed=[0, 1])
        collector.reset(observations)
        for _ in range(8):
            actions = collector.act()
            observations, rewards, terminations, truncations, _ = venv.step(actions)
            collector.observe(observations, rewards, terminations, truncations)
        collector.close()
    finally:
        venv.close()

    files, spec = validate_raw_directory(tmp_path)
    # Row 0 ends every 2 steps and row 1 every 3, interleaved by finish order.
    lengths = [len(_load(path)["actions"]) for path in sorted(files)]
    assert lengths == [2, 3, 2, 3, 2]
    assert spec["obs_channels"] == [3]
    for path in files:
        episode = _load(path)
        assert episode["terminations"][-1], "each row ended on a terminal state"
        assert not episode["truncations"].any()


# -- the packager's library entry point ---------------------------------


def test_build_dataset_raises_the_contract_error_the_cli_turns_into_an_exit():
    pytest.importorskip("minari")
    from collection.build_minari import build_dataset

    with pytest.raises(ContractError, match="Malformed dataset id"):
        build_dataset(Path("."), "no-version-suffix")
