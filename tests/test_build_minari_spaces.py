"""Regression tests for the Minari packager's declared spaces.

`build_minari.py` runs in a separate `minari==0.5.3` packaging env, so minari is
stubbed here: the space construction under test needs only gymnasium.

The ego-agent expectations are the schema the published SoccerTwos and
DungeonEscape datasets ship, so a change that breaks them breaks reproduction of
those datasets.
"""

import importlib.util
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD_MINARI = ROOT / "collection" / "build_minari.py"


@pytest.fixture
def build_minari(monkeypatch):
    """Load `collection/build_minari.py` with minari stubbed out."""
    minari = types.ModuleType("minari")
    minari.create_dataset_from_buffers = lambda *a, **k: None
    data_collector = types.ModuleType("minari.data_collector")
    data_collector.EpisodeBuffer = type("EpisodeBuffer", (), {})
    monkeypatch.setitem(sys.modules, "minari", minari)
    monkeypatch.setitem(sys.modules, "minari.data_collector", data_collector)

    spec = importlib.util.spec_from_file_location(
        "collection.build_minari_test", BUILD_MINARI
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_sensor_stays_a_bare_box(build_minari):
    space = build_minari._build_observation_space([64])
    assert space == gym.spaces.Box(-np.inf, np.inf, shape=(64,), dtype=np.float32)


def test_multi_sensor_becomes_a_tuple_of_boxes(build_minari):
    space = build_minari._build_observation_space([126, 32])
    assert isinstance(space, gym.spaces.Tuple)
    assert [s.shape[0] for s in space.spaces] == [126, 32]


def test_ego_wrap_is_absent_without_a_key(build_minari):
    inner = build_minari._build_observation_space([126, 32])
    assert build_minari._wrap_ego_space(inner, None) is inner
    value = (np.zeros((3, 126)), np.zeros((3, 32)))
    assert build_minari._wrap_ego_value(value, None) is value


def test_ego_wrap_matches_published_soccer_twos_schema(build_minari):
    """SoccerTwos: `Tuple(Box(264), Box(72))` + `MultiDiscrete([3, 3, 3])`."""
    obs = build_minari._wrap_ego_space(
        build_minari._build_observation_space([264, 72]), "agent_0"
    )
    act = build_minari._wrap_ego_space(
        build_minari._build_action_space("discrete", {"branches": [3, 3, 3]}),
        "agent_0",
    )
    assert str(obs) == (
        "Dict('agents': Dict('agent_0': Tuple(Box(-inf, inf, (264,), float32), "
        "Box(-inf, inf, (72,), float32))))"
    )
    assert str(act) == "Dict('agents': Dict('agent_0': MultiDiscrete([3 3 3])))"


def test_ego_wrap_matches_published_dungeon_escape_schema(build_minari):
    """DungeonEscape: `Tuple(Box(10), Box(360), Box(1))` + `Discrete(7)`."""
    obs = build_minari._wrap_ego_space(
        build_minari._build_observation_space([10, 360, 1]), "agent_0"
    )
    act = build_minari._wrap_ego_space(
        build_minari._build_action_space("discrete", {"branches": [7]}),
        "agent_0",
    )
    assert str(obs) == (
        "Dict('agents': Dict('agent_0': Tuple(Box(-inf, inf, (10,), float32), "
        "Box(-inf, inf, (360,), float32), Box(-inf, inf, (1,), float32))))"
    )
    assert str(act) == "Dict('agents': Dict('agent_0': Discrete(7)))"


@pytest.mark.parametrize(
    "action_kind, spec, expected",
    [
        (
            "continuous",
            {"act_dim": 20, "act_low": -1.0, "act_high": 1.0},
            "Box(-1.0, 1.0, (20,), float32)",
        ),
        (
            "hybrid",
            {
                "continuous_size": 3,
                "branches": [2],
                "act_low": -1.0,
                "act_high": 1.0,
            },
            "Tuple(Box(-1.0, 1.0, (3,), float32), Discrete(2))",
        ),
    ],
)
def test_ego_wrap_covers_continuous_and_hybrid(
    build_minari, action_kind, spec, expected
):
    """The wrapper is orthogonal to action kind, not discrete-only."""
    act = build_minari._wrap_ego_space(
        build_minari._build_action_space(action_kind, spec), "agent_0"
    )
    assert str(act) == f"Dict('agents': Dict('agent_0': {expected}))"


def test_continuous_space_uses_declared_per_dimension_bounds(build_minari):
    space = build_minari._build_action_space(
        "continuous",
        {
            "act_dim": 2,
            "act_low": np.asarray([-2.0, 10.0], dtype=np.float32),
            "act_high": np.asarray([2.0, 30.0], dtype=np.float32),
        },
    )

    np.testing.assert_array_equal(space.low, [-2.0, 10.0])
    np.testing.assert_array_equal(space.high, [2.0, 30.0])


def test_ego_wrap_nests_the_episode_payload_to_match_the_space(build_minari):
    obs = (np.zeros((6, 264), np.float32), np.zeros((6, 72), np.float32))
    wrapped = build_minari._wrap_ego_value(obs, "agent_0")
    assert wrapped["agents"]["agent_0"] is obs


def test_ego_spaces_reduce_to_the_same_leaf_specs_as_the_flat_form():
    """The declared space is the interface: nesting changes no leaf spec."""
    from causal_gpt_rl.inference.spaces import extract_data_specs_from_space

    inner = gym.spaces.Tuple(
        [
            gym.spaces.Box(-np.inf, np.inf, shape=(264,), dtype=np.float32),
            gym.spaces.Box(-np.inf, np.inf, shape=(72,), dtype=np.float32),
        ]
    )
    ego = gym.spaces.Dict({"agents": gym.spaces.Dict({"agent_0": inner})})

    flat_specs = extract_data_specs_from_space(inner)
    ego_specs = extract_data_specs_from_space(ego)

    assert [(s.type, s.size) for s in flat_specs] == [(s.type, s.size) for s in ego_specs]
