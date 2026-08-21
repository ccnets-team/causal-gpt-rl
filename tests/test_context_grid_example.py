"""The context-grid recipe records one dataset per level, on shared seeds.

What the grid is for is comparison, so what this pins is the two things that
make its levels comparable — every episode on a seed the caller chose, and one
batch width for the whole grid — plus the parsing that turns `--context` into
the levels, because a grid that quietly dropped or reordered a level would
mislabel every dataset after it.

The recording itself is `CollectionRunner`'s and is tested there; the fakes here
stand in for the policy and the environment so the recipe can be driven without
a bundle.
"""
import json
import sys

import numpy as np
import pytest

from collection.runner import CollectionRunner
from examples.mujoco_collection import record_context_grid as grid
# A sibling test module, imported bare: `tests/` has no `__init__.py`, so
# `tests.test_record_example` resolves against whatever regular `tests`
# package is installed on sys.path instead of this directory.
from test_record_example import _FakeVecEnv, _VecStubRunner


# -- the grid -----------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("8,16,32", [8, 16, 32]),
        ("8-11", [8, 9, 10, 11]),
        ("128,8,32", [8, 32, 128]),          # sorted: the grid is an axis
        ("8,8,16", [8, 16]),                 # a level recorded twice is one level
        ("1-3,64", [1, 2, 3, 64]),
    ],
)
def test_the_grid_is_sorted_and_deduplicated(spec, expected):
    assert grid.parse_context_grid(spec) == expected


@pytest.mark.parametrize(
    "spec, message",
    [
        ("8,,32", "empty entry"),
        ("8,x", "not a context length"),
        ("0-4", ">= 1"),
        ("64-8", "counts down"),
    ],
)
def test_a_malformed_grid_is_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        grid.parse_context_grid(spec)


def test_a_level_is_named_so_a_listing_sorts_as_the_grid_does():
    names = [grid.level_name(c) for c in (8, 64, 1000)]
    assert names == ["kv0008", "kv0064", "kv1000"]
    assert sorted(names) == names


# -- the two rules the levels rest on -----------------------------------


def _argv(*extra):
    return [
        "record_context_grid",
        "--env-id", "Humanoid-v5",
        "--out", "raw",
        "--context", "8,32",
        *extra,
    ]


def test_the_grid_reaches_the_parsed_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv("--episodes", "64"))

    args = grid.parse_args()

    assert args.episodes == 64
    assert args.context == [8, 32]


def test_the_batch_is_as_wide_as_the_episode_count(tmp_path, monkeypatch):
    """One row per episode, so every episode carries a chosen seed.

    There is no width to set: `--episodes` is it, and this is what pins that the
    env is actually built that wide rather than the two drifting apart.
    """
    widths = []

    def _make_vec(env_id, *, num_envs, max_episode_steps):
        widths.append(num_envs)
        return _FakeVecEnv([2] * num_envs)

    runner = _VecStubRunner(3)
    monkeypatch.setattr(grid, "load_policy", lambda args, ctx: (runner, "bundle/fake"))
    monkeypatch.setattr(grid.gym, "make_vec", _make_vec)

    grid.record_level(_Args(tmp_path, [16], episodes=3), 16)

    assert widths == [3]


def test_one_level_is_not_a_grid(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_context_grid", "--env-id", "Humanoid-v5", "--out", "raw",
            "--context", "32",
        ],
    )

    with pytest.raises(SystemExit):
        grid.parse_args()
    assert "--context-length" in capsys.readouterr().err


def test_the_namespace_defaults_to_the_published_one(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv())
    assert grid.parse_args().namespace == "ccnets/humanoid"


def test_every_level_names_its_own_dataset(monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv("--version", "1"))
    args = grid.parse_args()

    ids = [f"{args.namespace}/{grid.level_name(c)}-v{args.version}" for c in args.context]
    assert ids == ["ccnets/humanoid/kv0008-v1", "ccnets/humanoid/kv0032-v1"]


# -- recording a level --------------------------------------------------


class _Args:
    """The parsed arguments `record_level` reads, without the parser."""

    def __init__(self, out, context, episodes):
        self.env_id = "Fake-v0"
        self.out = out
        self.context = context
        self.episodes = episodes
        self.seed_start = 0
        self.max_steps = 10
        self.namespace = "ccnets/fake"
        self.version = 0
        self.build = False
        self.device = "cpu"
        self.bundle = None
        self.repo_id = "ccnets/causal-gpt-rl"
        self.subfolder = None


def test_a_level_writes_one_episode_per_row_into_its_own_directory(
    tmp_path, monkeypatch
):
    lengths = [2, 3]
    runner = _VecStubRunner(len(lengths))
    monkeypatch.setattr(
        grid, "load_policy", lambda args, context: (runner, "bundle/fake")
    )
    monkeypatch.setattr(
        grid.gym, "make_vec", lambda *a, **k: _FakeVecEnv(lengths)
    )

    args = _Args(tmp_path, [16], episodes=len(lengths))
    level = grid.record_level(args, 16)

    assert level["level"] == "kv0016"
    assert level["dataset_id"] == "ccnets/fake/kv0016-v0"
    assert level["episodes"] == len(lengths)
    # One directory per level, and one file per row inside it.
    assert sorted(p.name for p in (tmp_path / "kv0016").glob("ep_*.npz")) == [
        "ep_000000.npz",
        "ep_000001.npz",
    ]


def test_the_context_length_of_a_level_reaches_its_provenance(tmp_path, monkeypatch):
    """`spec.json` is what says which level a directory holds."""
    runner = _VecStubRunner(2)
    runner.kv_cache_max_len = 64
    monkeypatch.setattr(
        grid, "load_policy", lambda args, context: (runner, "bundle/fake")
    )
    monkeypatch.setattr(grid.gym, "make_vec", lambda *a, **k: _FakeVecEnv([2, 2]))

    grid.record_level(_Args(tmp_path, [64], episodes=2), 64)

    spec = json.loads((tmp_path / "kv0064" / "spec.json").read_text(encoding="utf-8"))
    provenance = spec["provenance"][0]
    assert provenance["kv_cache_max_len"] == 64
    assert provenance["bundle"] == "bundle/fake"
    # One level per directory is why no per-episode manifest is needed.
    assert provenance["episodes_per_row"] == 1


# -- the summary --------------------------------------------------------


def _level(context, return_mean, return_std, terminated=1):
    return {
        "context": context,
        "level": grid.level_name(context),
        "bundle": "bundle/fake",
        "raw_dir": "raw/fake",
        "dataset_id": f"ccnets/fake/{grid.level_name(context)}-v0",
        "episodes": 4,
        "transitions": 40,
        "length_mean": 10.0,
        "return_mean": return_mean,
        "return_std": return_std,
        "return_min": return_mean - return_std,
        "terminated": terminated,
        "returns": [return_mean] * 4,
    }


def test_a_grid_narrower_than_its_own_episodes_says_so(capsys, tmp_path):
    args = _Args(tmp_path, [8, 32], episodes=4)
    grid.summarize(args, [_level(8, 100.0, 500.0), _level(32, 120.0, 500.0)])

    printed = capsys.readouterr().out
    assert "the levels differ by less than the episodes" in printed


def test_a_separated_grid_gets_no_note(capsys, tmp_path):
    args = _Args(tmp_path, [8, 32], episodes=4)
    grid.summarize(args, [_level(8, 100.0, 10.0), _level(32, 900.0, 10.0)])

    printed = capsys.readouterr().out
    assert "the levels differ by less than the episodes" not in printed


def test_a_level_with_no_terminal_state_is_called_out(capsys, tmp_path):
    args = _Args(tmp_path, [8, 32], episodes=4)
    grid.summarize(
        args, [_level(8, 100.0, 10.0, terminated=0), _level(32, 900.0, 10.0)]
    )

    assert "reached no terminal state" in capsys.readouterr().out


def test_the_summary_survives_a_legacy_code_page(capsys, tmp_path):
    args = _Args(tmp_path, [8, 32], episodes=4)
    grid.summarize(args, [_level(8, 100.0, 500.0, terminated=0), _level(32, 120.0, 5.0)])

    capsys.readouterr().out.encode("cp949")  # raises if anything non-ASCII slipped in
