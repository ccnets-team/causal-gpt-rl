"""Regression tests for Unity multi-agent trajectory collection bookkeeping."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNITY_DIR = ROOT / "examples" / "unity_collection"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_collection_stubs(monkeypatch):
    environment = types.ModuleType("mlagents_envs.environment")
    environment.UnityEnvironment = type(
        "UnityEnvironment", (), {"BASE_ENVIRONMENT_PORT": 5005}
    )
    environment.ActionTuple = type("ActionTuple", (), {})
    engine = types.ModuleType(
        "mlagents_envs.side_channel.engine_configuration_channel"
    )
    engine.EngineConfigurationChannel = type("EngineConfigurationChannel", (), {})
    monkeypatch.setitem(sys.modules, "mlagents_envs", types.ModuleType("mlagents_envs"))
    monkeypatch.setitem(sys.modules, "mlagents_envs.environment", environment)
    monkeypatch.setitem(
        sys.modules, "mlagents_envs.side_channel", types.ModuleType("side_channel")
    )
    monkeypatch.setitem(
        sys.modules,
        "mlagents_envs.side_channel.engine_configuration_channel",
        engine,
    )
    for name, attr in (
        ("unity_env", "UnityEnv"),
        ("onnx_policy", "OnnxPolicy"),
        ("noisy_policy", "NoisyPolicy"),
    ):
        module = types.ModuleType(name)
        setattr(module, attr, type(attr, (), {}))
        monkeypatch.setitem(sys.modules, name, module)


def test_soccer_groups_are_paired_into_four_player_fields(monkeypatch):
    _install_collection_stubs(monkeypatch)
    collect = _load_module("unity_collect_test", UNITY_DIR / "collect.py")
    contexts = [
        {"env_index": 0, "team_id": team, "group_id": group, "agent_id": agent}
        for team, group, agents in (
            (0, 10, (1, 2)),
            (0, 11, (3, 4)),
            (1, 20, (5, 6)),
            (1, 21, (7, 8)),
        )
        for agent in agents
    ]

    annotated, sizes, members = collect._assign_field_ids(contexts)

    assert sizes == {(0, 0): 4, (0, 1): 4}
    assert members == {(0, 0): [0, 1, 4, 5], (0, 1): [2, 3, 6, 7]}
    assert [row["field_id"] for row in annotated] == [0, 0, 1, 1, 0, 0, 1, 1]
    assert all("field_id" not in row for row in contexts)  # inputs are not mutated


def test_soccer_team_noise_is_shared_by_teammates_and_reproducible(monkeypatch):
    _install_collection_stubs(monkeypatch)
    collect = _load_module("unity_collect_noise_test", UNITY_DIR / "collect.py")
    contexts = [
        {
            "env_index": 0,
            "field_id": 0,
            "team_id": team,
            "group_id": team + 1,
            "agent_id": agent,
        }
        for team, agents in ((0, (10, 11)), (1, (20, 21)))
        for agent in agents
    ]
    schedule = collect.TeamNoiseSchedule(
        contexts, {0: (0.02, 0.05), 1: (0.2, 0.4)}, seed=123
    )

    first = schedule.epsilon_vector()
    assert first[0] == first[1]
    assert first[2] == first[3]
    assert first[0] in (0.02, 0.05)
    assert first[2] in (0.2, 0.4)
    assert schedule.opponent_epsilon_for_agent(0) == first[2]

    schedule.start_match((0, 0), 7)
    resumed = collect.TeamNoiseSchedule(
        contexts,
        {0: (0.02, 0.05), 1: (0.2, 0.4)},
        seed=123,
        match_indices={(0, 0): 7},
    )
    np.testing.assert_array_equal(schedule.epsilon_vector(), resumed.epsilon_vector())


def test_noisy_policy_accepts_per_agent_discrete_epsilon():
    noisy_policy = _load_module("unity_noisy_policy_test", UNITY_DIR / "noisy_policy.py")

    class BasePolicy:
        kind = "discrete"
        branches = (3, 3, 3)
        act_dim = 3

        def act(self, observations):
            return np.zeros((2, 3), dtype=np.float32)

    policy = noisy_policy.NoisyPolicy(BasePolicy(), rng=np.random.default_rng(5))
    policy.set_epsilon_by_agent([0.0, 1.0])
    observations = [np.ones((2, 1), dtype=np.float32)]
    actions = policy.act(observations)

    np.testing.assert_array_equal(actions[0], np.zeros(3))
    assert np.all((actions[1] >= 0) & (actions[1] < 3))


class _Steps:
    def __init__(self, agent_ids, group_ids):
        self.agent_id = np.asarray(agent_ids, dtype=np.int64)
        self.group_id = np.asarray(group_ids, dtype=np.int64)


def test_dungeon_reissued_ids_reuse_stable_group_slots(monkeypatch):
    _install_collection_stubs(monkeypatch)
    unity_env = _load_module("unity_env_test", UNITY_DIR / "unity_env.py")
    wrapper = object.__new__(unity_env.UnityEnv)
    key = (0, "DungeonEscape?team=0")
    wrapper.slot_maps = {key: {10: 0, 11: 1, 12: 2}}
    wrapper.offsets = {key: 0}
    wrapper.counts = {key: 3}
    wrapper.agent_context = [
        {"group_id": 7},
        {"group_id": 7},
        {"group_id": 7},
    ]
    wrapper._reissued_ids = {}

    slots, restarted = wrapper._resolve_step_slots(
        key,
        _Steps([20, 21, 22], [7, 7, 7]),
        _Steps([10, 11, 12], [7, 7, 7]),
    )

    assert restarted == {7}
    assert {slots[20], slots[21], slots[22]} == {0, 1, 2}
    assert slots[10] == slots[20]
    assert slots[11] == slots[21]
    assert slots[12] == slots[22]
    assert wrapper._reissued_ids[(key, 7)] == set()
