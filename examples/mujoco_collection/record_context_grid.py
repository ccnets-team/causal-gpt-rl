"""Record one dataset per rollout context length, all on the same seeds.

`record_tiers.py` cuts a `simple` / `medium` / `expert` ladder out of one bundle
by naming a context length per tier. This is the measurement that ladder is
supposed to rest on: the same policy recorded at several context lengths, one
dataset each, so what the length does to the trajectories can be read off
instead of assumed.

Two rules make the grid comparable, and both are enforced rather than suggested:

  - **every episode carries a chosen seed.** A vector env is seeded once, at
    reset, so a level is one batch of `--episodes` rows recording one episode
    each. There is no separate width to set: a narrower batch would leave the
    rest of the episodes to unseeded auto-resets, where the initial state
    depends on how many steps the policy took before it -- which is the thing
    that differs between the levels being compared.
  - **every level runs the same batch width.** Fifty rows and a hundred rows
    reduce floating point in a different order, and in a closed loop that
    difference compounds. The width is part of the measurement, so it follows
    `--episodes` and is one number for the whole grid.

The context length is a load-time argument, so every level is the same weights
read again -- nothing is retrained between them.

Recording needs torch; `--build` additionally needs `minari==0.5.3`. Leave it
off to record now and package in a packaging environment later -- the command is
printed either way.

Example:
    python -m examples.mujoco_collection.record_context_grid \
        --env-id Humanoid-v5 --out raw/humanoid \
        --context 8,16,32,64,128 --episodes 100 --build
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import load_runner, load_runner_from_hub

# `collection/` sits beside `examples/` in the repository; it is the packager
# and, here, the recorder -- not part of the installed runtime package.
from collection import CollectionRunner
from examples.deploy.record import record_vector_episodes

DEFAULT_REPO_ID = "ccnets/causal-gpt-rl"

STACK = ("torch", "gymnasium", "mujoco")


def parse_context_grid(spec: str) -> list[int]:
    """`'8,16-18'` -> `[8, 16, 17, 18]`, sorted and deduplicated.

    Sorted because the grid is an axis, not a sequence of instructions: the
    order levels are recorded in does not change what any of them is, and a
    sorted grid is what the summary table has to be read against. Ranges are
    inclusive at both ends because they name context lengths, not indices.
    """
    values: list[int] = []
    for part in str(spec).split(","):
        piece = part.strip()
        if not piece:
            raise ValueError(f"empty entry in --context {spec!r}")
        low, dash, high = piece.partition("-")
        try:
            start = int(low)
            stop = int(high) if dash else start
        except ValueError:
            raise ValueError(
                f"{piece!r} is not a context length or an 'a-b' range"
            ) from None
        if start < 1 or stop < 1:
            raise ValueError(f"context lengths must be >= 1, got {piece!r}")
        if stop < start:
            raise ValueError(f"range {piece!r} counts down; write it as low-high")
        values.extend(range(start, stop + 1))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-id", required=True, help="Gymnasium environment id.")
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Directory to write one raw subdirectory per context length into.",
    )
    p.add_argument(
        "--context",
        required=True,
        metavar="SPEC",
        help="The grid, as a comma-separated list of lengths and 'a-b' ranges, "
        "e.g. '8,16,32,64,128' or '8-16'. One dataset is recorded per value.",
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
        help="Minari namespace the levels are packaged under. Defaults to "
        "ccnets/<env name>, which is how the published recordings name them.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per context length, and with them the batch width: a "
        "level is one batch of this many rows, each recording one episode, so "
        "every episode carries a seed the caller chose.",
    )
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed. Every context length runs the same seeds, which is "
        "what separates the grid from the seed draw.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Per-episode cap, applied as each sub-env's own TimeLimit and "
        "recorded as a truncation.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--build",
        action="store_true",
        help="Also package each level with minari, instead of only printing the "
        "commands that would.",
    )
    p.add_argument(
        "--version",
        type=int,
        default=0,
        help="Dataset version suffix: kv<NNNN>-v<N>. Minari refuses to overwrite "
        "an existing id, so a re-recording needs the next one.",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the summary, including per-episode returns, here.",
    )
    args = p.parse_args()
    try:
        args.context = parse_context_grid(args.context)
    except ValueError as exc:
        p.error(str(exc))
    if len(args.context) < 2:
        p.error(
            f"--context is a grid; {args.context} is one level. Record a single "
            "level with examples/deploy/record.py --context-length"
        )
    if args.episodes < 1:
        p.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.max_steps < 1:
        p.error(f"--max-steps must be >= 1, got {args.max_steps}")
    if args.version < 0:
        p.error(f"--version must be >= 0, got {args.version}")
    if args.namespace is None:
        args.namespace = f"ccnets/{args.env_id.split('-')[0].lower()}"
    return args


def level_name(context: int) -> str:
    """`kv0032` — zero-padded so a directory listing sorts as the grid does."""
    return f"kv{context:04d}"


def load_policy(args: argparse.Namespace, context: int):
    """A fresh runner at one context length, and its bundle identity.

    Fresh per level rather than one runner reconfigured: the cache size is fixed
    when the buffer is built, and a level that inherited another's state would
    not be the level it claims to be.
    """
    if args.bundle is not None:
        if not args.bundle.is_dir():
            raise FileNotFoundError(args.bundle)
        runner = load_runner(
            args.bundle,
            device=args.device,
            num_envs=args.episodes,
            kv_cache_max_len=context,
        )
        return runner, str(args.bundle)
    subfolder = args.subfolder if args.subfolder is not None else args.env_id.lower()
    runner = load_runner_from_hub(
        repo_id=args.repo_id,
        subfolder=subfolder,
        device=args.device,
        num_envs=args.episodes,
        kv_cache_max_len=context,
    )
    return runner, f"{args.repo_id}/{subfolder}"


def record_level(args: argparse.Namespace, context: int) -> dict:
    """Record one context length's episodes into its own raw directory."""
    name = level_name(context)
    raw_dir = args.out / name
    # The per-episode cap rides in each sub-env's own TimeLimit: only the env can
    # restart a row it truncates, so the loop cannot apply one.
    env = gym.make_vec(
        args.env_id, num_envs=args.episodes, max_episode_steps=args.max_steps
    )
    try:
        runner, bundle_id = load_policy(args, context)
        print(
            f"\n[{name}] context={runner.kv_cache_max_len} "
            f"trained={runner.context_length} rows={args.episodes} -> {raw_dir}",
            flush=True,
        )
        collector = CollectionRunner(
            runner, raw_dir, bundle=bundle_id, episodes_per_row=1
        )
        # Closes the collector itself once every row has written its one.
        results = record_vector_episodes(env, collector, seed_start=args.seed_start)
    finally:
        env.close()

    returns = np.asarray([r["return"] for r in results], dtype=np.float64)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float64)
    return {
        "context": context,
        "level": name,
        "bundle": bundle_id,
        "raw_dir": raw_dir,
        "dataset_id": f"{args.namespace}/{name}-v{args.version}",
        "episodes": len(results),
        "transitions": int(steps.sum()),
        "length_mean": float(steps.mean()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "terminated": sum(r["terminated"] for r in results),
        "returns": returns.tolist(),
    }


def describe(args: argparse.Namespace, level: dict) -> str:
    """The one-line description that travels with the packaged dataset."""
    return (
        f"{args.env_id} driven by {level['bundle']} with "
        f"kv_cache_max_len={level['context']}, {level['episodes']} episodes on "
        f"seeds {args.seed_start}..{args.seed_start + level['episodes'] - 1}."
    )


def build(args: argparse.Namespace, levels: list[dict]) -> None:
    """Package every recorded level under one namespace."""
    from collection import build_dataset

    for level in levels:
        print(f"\n[{level['level']}] packaging -> {level['dataset_id']}", flush=True)
        build_dataset(
            level["raw_dir"], level["dataset_id"], description=describe(args, level)
        )


def summarize(args: argparse.Namespace, levels: list[dict]) -> None:
    print(
        f"\n{'context':>8} {'episodes':>9} {'transitions':>12} {'return':>22} "
        f"{'worst':>10} {'length':>8} {'terminated':>11}"
    )
    print("-" * 84)
    for level in levels:
        returns = f"{level['return_mean']:.2f} +- {level['return_std']:.2f}"
        print(
            f"{level['context']:>8} {level['episodes']:>9} {level['transitions']:>12} "
            f"{returns:>22} {level['return_min']:>10.2f} "
            f"{level['length_mean']:>8.1f} {level['terminated']:>11}"
        )

    spread = max(l["return_mean"] for l in levels) - min(
        l["return_mean"] for l in levels
    )
    widest = max(l["return_std"] for l in levels)
    print(
        f"\nspread across the grid: {spread:.2f}, against a within-level spread of "
        f"up to {widest:.2f}"
    )
    if spread < widest:
        # ASCII only, here and everywhere this script prints: a console on a
        # legacy code page raises UnicodeEncodeError on the write, which would
        # fail a run whose episodes are already on disk.
        print(
            "  [note] the levels differ by less than the episodes within one of\n"
            "         them. Read the worst-episode and terminated columns before\n"
            "         concluding anything from the means."
        )

    if any(level["terminated"] == 0 for level in levels):
        # A dataset where nothing ever ends teaches a termination head that
        # nothing ever ends. Worth knowing before packaging it.
        print(
            "\n[note] a level reached no terminal state - every episode ran out\n"
            "       of time. Raise --max-steps, or expect no terminations in it."
        )

    if args.build:
        print(f"\npackaged under the {args.namespace!r} namespace.")
        return
    print("\nNext:")
    for level in levels:
        print(
            f"  python collection/build_minari.py --raw {level['raw_dir']} "
            f"--dataset-id {level['dataset_id']} \\\n"
            f"      --description \"{describe(args, level)}\""
        )


def write_json(args: argparse.Namespace, levels: list[dict]) -> None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "env_id": args.env_id,
                "namespace": args.namespace,
                "episodes": args.episodes,
                "seed_start": args.seed_start,
                "max_steps": args.max_steps,
                "device": str(args.device),
                "grid": [
                    {**level, "raw_dir": str(level["raw_dir"])} for level in levels
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.json}")


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
        f"[grid] {args.env_id} -> {args.namespace}/kv<NNNN>-v{args.version}  "
        f"{len(args.context)} levels x {args.episodes} episodes  "
        f"seeds {args.seed_start}..{args.seed_start + args.episodes - 1}"
    )

    levels = [record_level(args, context) for context in args.context]
    if args.build:
        build(args, levels)
    summarize(args, levels)
    if args.json is not None:
        write_json(args, levels)


if __name__ == "__main__":
    main()
