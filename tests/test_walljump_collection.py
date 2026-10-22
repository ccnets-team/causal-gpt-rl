"""WallJump episode boundaries must survive changes of behavior and decision gaps."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class Steps:
    def __init__(self, rows=()):
        self.agent_id = np.array([r[0] for r in rows], dtype=np.int32)
        self.obs = [np.array([[r[1], r[1]] for r in rows], np.float32).reshape(-1, 2)]
        self.reward = np.array([r[2] for r in rows], np.float32)
        self.group_reward = np.zeros(len(rows), np.float32)
        self.interrupted = np.zeros(len(rows), bool)
        self.action_mask = None

    def __len__(self):
        return len(self.agent_id)


def load_collector(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "examples" / "unity_collection"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("walljump_test_collector", folder / "collect_walljump.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fake(module, monkeypatch, tmp_path, bad_switch=False):
    class FakeEnv:
        instance = None

        def __init__(self, **kwargs):
            FakeEnv.instance = self
            self.tick, self.sent, self.closed = 0, {}, False
            spec = SimpleNamespace(observation_specs=[SimpleNamespace(shape=(2,))],
                action_spec=SimpleNamespace(continuous_size=0,
                                            discrete_branches=np.array([3, 3, 3, 2])))
            self.behavior_specs = {b + "?team=0": spec for b in module.BEHAVIORS}

        def reset(self):
            pass

        def get_steps(self, name):
            small = name.startswith("Small")
            if self.tick == 0:
                return Steps([(1, 0, 0)] if small else [(2, 10, 0)]), Steps()
            if self.tick == 1:
                return Steps([] if small else [(2, 11, .1)]), Steps()
            if self.tick == 2:
                return ((Steps(), Steps([] if bad_switch else [(1, 2, 1)])) if small
                        else (Steps([(2, 12, .1), (1, 20, 0)]), Steps()))
            if self.tick == 3:
                return Steps(), Steps([] if small else [(1, 21, -1), (2, 13, 1)])
            raise AssertionError("Collector did not finish its complete episodes")

        def set_actions(self, name, actions):
            decision = self.get_steps(name)[0]
            for aid, action in zip(decision.agent_id, actions.discrete):
                self.sent[(self.tick, name.split("?")[0], int(aid))] = action.copy()

        def step(self):
            self.tick += 1

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "UnityEnvironment", FakeEnv)
    args = SimpleNamespace(out=tmp_path / "raw", build=tmp_path / "WallJump.exe",
        worker_id=1, seed=7, time_scale=20, random=True, target=5, quality="random",
        source_revision="test", small_onnx=tmp_path / "small.onnx",
        big_onnx=tmp_path / "big.onnx", small_temperature_range=None,
        big_temperature_range=None)
    return FakeEnv, args


def test_behavior_switch_and_repeated_action_gap(monkeypatch, tmp_path):
    module = load_collector(monkeypatch)
    fake, args = run_fake(module, monkeypatch, tmp_path)
    result = module.collect(args)
    assert result["transitions"] == 5 and result["episodes"] == 3
    assert fake.instance.closed
    rows = [json.loads(s) for s in (args.out / "episode_metadata.jsonl").read_text().splitlines()]
    by_key = {(r["behavior_name"], r["agent_id"]): r for r in rows}
    for key, expected_obs, expected_ticks, action_ticks in (
        (("SmallWallJump", 1), [0, 2], [2], [0]),
        (("BigWallJump", 1), [20, 21], [1], [2]),
        (("BigWallJump", 2), [10, 11, 12, 13], [1, 1, 1], [0, 1, 2]),
    ):
        row = by_key[key]
        with np.load(args.out / row["episode_file"]) as data:
            np.testing.assert_array_equal(data["observations"][:, 0], expected_obs)
            np.testing.assert_array_equal(data["unity_tick_deltas"], expected_ticks)
            np.testing.assert_array_equal(data["actions"],
                [fake.instance.sent[(t, *key)] for t in action_ticks])
            assert data["terminations"][-1] and not data["truncations"].any()
    assert by_key[("BigWallJump", 1)]["outcome"] == "fall"


def test_unclosed_behavior_switch_fails(monkeypatch, tmp_path):
    module = load_collector(monkeypatch)
    fake, args = run_fake(module, monkeypatch, tmp_path, bad_switch=True)
    with pytest.raises(RuntimeError, match="Behavior changed without terminal"):
        module.collect(args)
    assert fake.instance.closed
