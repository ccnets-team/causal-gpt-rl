"""Record a tier ladder from one bundle, and package each tier as a dataset.

`deploy/record.py` records one dataset at one retention. This records several in
one run -- a `simple` / `medium` / `expert` ladder cut out of the same weights by
changing how much past the rollout keeps -- and lays the result out the way the
published dataset repositories are laid out::

    <namespace>/simple-v0/data/main_data.hdf5
    <namespace>/medium-v0/data/main_data.hdf5
    <namespace>/expert-v0/data/main_data.hdf5

Every tier runs the same seeds, so the ladder is one policy at three retentions
rather than three unrelated draws. `--num-envs` records a tier as one batch
instead of one episode at a time, and must be 1 or equal to `--episodes`: a
vector env takes its seeds at reset and auto-resets unseeded after that, so one
episode per row is the only batched form that keeps the draws shared. Which retention belongs to which tier is a
measurement, not a guess: `calibrate_retention.py` reports the normalized score
of each level and picks them.

Recording needs torch; `--build` additionally needs `minari==0.5.3`. Leave it off
to record now and package in a packaging environment later -- the command is
printed either way.

Example:
    python -m examples.mujoco_collection.record_tiers \
        --env-id Hopper-v5 --out raw/hopper-v5 \
        --tier simple=16 --tier medium=32 --tier expert=128 \
        --episodes 200 --build
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import load_runner, load_runner_from_hub

# `collection/` sits beside `examples/` in the repository; it is the packager
# and, here, the recorder -- not part of the installed runtime package.
from collection import CollectionRunner
from examples.deploy.record import record_episodes, record_vector_episodes

DEFAULT_REPO_ID = "ccnets/causal-gpt-rl"

STACK = ("torch", "gymnasium", "mujoco")


def tier_pair(value: str) -> tuple[str, int]:
    """Parse one `--tier name=retention` argument."""
    name, _, retention = value.partition("=")
    name = name.strip()
    if not name or not retention.strip().isdigit() or int(retention) < 1:
        raise argparse.ArgumentTypeError(
            f"expected name=retention with retention >= 1, got {value!r}"
        )
    return name, int(retention)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-id", required=True, help="Gymnasium environment id.")
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory to write one raw subdirectory per tier into.",
    )
    p.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        type=tier_pair,
        metavar="NAME=RETENTION",
        help="A tier and the retention that lands it, e.g. medium=32. Repeatable; "
        "recorded in the order given.",
    )
    p.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Local bundle directory. Omit to download from the Hub.",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hub repo to load from.")
    p.add_argument(
        "--subfolder",
        default=None,
        help="Bundle subfolder in the repo. Defaults to the lowercased env id.",
    )
    p.add_argument(
        "--namespace",
        default=None,
        help="Minari namespace the tiers are packaged under. Defaults to the "
        "lowercased env id, which is how the published repositories name them.",
    )
    p.add_argument("--episodes", type=int, default=100, help="Episodes per tier.")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Environments per tier, recorded as one batch. Must be 1 or equal "
        "to --episodes: a vector env is seeded once, so anything between would "
        "leave most episodes unseeded and the tiers would no longer be the same "
        "draws.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Safety cap per episode, recorded as a truncation. The env's own "
        "TimeLimit usually fires first.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--build",
        action="store_true",
        help="Also package each tier with minari, instead of only printing the "
        "commands that would.",
    )
    p.add_argument(
        "--version",
        type=int,
        default=0,
        help="Dataset version suffix: <tier>-v<N>. Minari refuses to overwrite an "
        "existing id, so a re-recording needs the next one.",
    )
    args = p.parse_args()
    if not args.tiers:
        p.error("pass at least one --tier NAME=RETENTION")
    names = [name for name, _ in args.tiers]
    if len(set(names)) != len(names):
        p.error(f"tier names must be distinct, got {names}")
    if args.episodes < 1:
        p.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.max_steps < 1:
        p.error(f"--max-steps must be >= 1, got {args.max_steps}")
    if args.num_envs < 1:
        p.error(f"--num-envs must be >= 1, got {args.num_envs}")
    if 1 < args.num_envs != args.episodes:
        # The ladder is one policy at several retentions, which only holds if
        # every tier draws the same initial states. A vector env takes its seeds
        # at reset and auto-resets unseeded after that, so one episode per row is
        # the only batched form where every episode is still a chosen seed.
        p.error(
            f"--num-envs {args.num_envs} would seed only {args.num_envs} of "
            f"{args.episodes} episodes per tier, so the tiers would not be the "
            "same draws. Pass --num-envs 1, or --num-envs equal to --episodes "
            f"({args.episodes})."
        )
    if args.version < 0:
        p.error(f"--version must be >= 0, got {args.version}")
    if args.namespace is None:
        args.namespace = args.env_id.lower()
    return args


def load_policy(args: argparse.Namespace, retention: int):
    """A fresh runner at one retention, and its bundle identity."""
    if args.bundle is not None:
        if not args.bundle.is_dir():
            raise FileNotFoundError(args.bundle)
        runner = load_runner(
            args.bundle,
            device=args.device,
            num_envs=args.num_envs,
            kv_cache_max_len=retention,
        )
        return runner, str(args.bundle)
    subfolder = args.subfolder if args.subfolder is not None else args.env_id.lower()
    runner = load_runner_from_hub(
        repo_id=args.repo_id,
        subfolder=subfolder,
        device=args.device,
        num_envs=args.num_envs,
        kv_cache_max_len=retention,
    )
    return runner, f"{args.repo_id}/{subfolder}"


def record_tier(args: argparse.Namespace, name: str, retention: int) -> dict:
    """Record one tier's episodes into its own raw directory."""
    raw_dir = args.out / name
    if args.num_envs > 1:
        # One episode per row, so every episode is a seeded one; the cap rides in
        # each sub-env's own TimeLimit because only the env can restart a row.
        env = gym.make_vec(
            args.env_id, num_envs=args.num_envs, max_episode_steps=args.max_steps
        )
    else:
        env = gym.make(args.env_id)
    try:
        runner, bundle_id = load_policy(args, retention)
        print(
            f"\n[{name}] retention={runner.kv_cache_max_len} "
            f"trained={runner.context_length} -> {raw_dir}",
            flush=True,
        )
        collector = CollectionRunner(
            runner,
            raw_dir,
            bundle=bundle_id,
            episodes_per_row=1 if args.num_envs > 1 else None,
        )
        if args.num_envs > 1:
            # Closes the collector itself once every row has written its one.
            results = record_vector_episodes(
                env, collector, seed_start=args.seed_start
            )
        else:
            results = record_episodes(
                env,
                collector,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed_start=args.seed_start,
            )
            collector.close()
    finally:
        env.close()

    returns = np.asarray([r["return"] for r in results], dtype=np.float64)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float64)
    return {
        "tier": name,
        "retention": retention,
        "bundle": bundle_id,
        "raw_dir": raw_dir,
        "dataset_id": f"{args.namespace}/{name}-v{args.version}",
        "episodes": len(results),
        "transitions": int(steps.sum()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "terminated": sum(r["terminated"] for r in results),
    }


def describe(args: argparse.Namespace, tier: dict) -> str:
    """The one-line description that travels with the packaged dataset."""
    return (
        f"{args.env_id} driven by {tier['bundle']} with "
        f"kv_cache_max_len={tier['retention']} ({tier['tier']} tier)."
    )


def build(args: argparse.Namespace, tiers: list[dict]) -> None:
    """Package every recorded tier under one namespace."""
    from collection import build_dataset

    for tier in tiers:
        print(f"\n[{tier['tier']}] packaging -> {tier['dataset_id']}", flush=True)
        build_dataset(
            tier["raw_dir"], tier["dataset_id"], description=describe(args, tier)
        )


def summarize(args: argparse.Namespace, tiers: list[dict]) -> None:
    print(f"\n{'tier':<10} {'retention':>9} {'episodes':>9} {'transitions':>12} "
          f"{'return':>22} {'terminated':>11}")
    for tier in tiers:
        returns = f"{tier['return_mean']:.2f} +- {tier['return_std']:.2f}"
        print(
            f"{tier['tier']:<10} {tier['retention']:>9} {tier['episodes']:>9} "
            f"{tier['transitions']:>12} {returns:>22} {tier['terminated']:>11}"
        )

    if any(tier["terminated"] == 0 for tier in tiers):
        # A dataset where nothing ever ends teaches a termination head that
        # nothing ever ends. Worth knowing before packaging it.
        print(
            "\n[note] a tier reached no terminal state - every episode ran out of\n"
            "       time. Raise --max-steps, or expect no terminations in it."
        )

    if args.build:
        print(f"\npackaged under the {args.namespace!r} namespace.")
        return
    print("\nNext:")
    for tier in tiers:
        print(
            f"  python collection/build_minari.py --raw {tier['raw_dir']} "
            f"--dataset-id {tier['dataset_id']} \\\n"
            f"      --description \"{describe(args, tier)}\""
        )


def installed(package: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    args = parse_args()
    print(
        "[stack] device=" + str(args.device) + "  "
        + "  ".join(f"{name}={installed(name)}" for name in STACK)
    )
    print(
        f"[tiers] {args.env_id} -> {args.namespace}/<tier>-v{args.version}  "
        f"seeds {args.seed_start}..{args.seed_start + args.episodes - 1}"
        + (f"  ({args.num_envs} rows per tier)" if args.num_envs > 1 else "")
    )

    tiers = [record_tier(args, name, retention) for name, retention in args.tiers]
    if args.build:
        build(args, tiers)
    summarize(args, tiers)


if __name__ == "__main__":
    main()
