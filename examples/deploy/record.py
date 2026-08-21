"""Record episodes with a Causal-GPT-RL bundle, ready for Minari packaging.

The rollout of `deploy/mujoco.py`, writing what it drives. `CollectionRunner`
wraps the runner and produces a raw directory — one `ep_%06d.npz` per episode
plus a `spec.json` — which `collection/build_minari.py` packages unchanged.
That is the second collection cycle: the policy a dataset trained records the
next dataset.

Run it from a checkout, not from an installed wheel: `collection/` is a
directory of this repository, not part of the `causal-gpt-rl` package.

Example:
    python -m examples.deploy.record \
        --env-id Hopper-v5 --out raw/ --episodes 20 --context-length 256
    python collection/build_minari.py --raw raw/ --dataset-id review/hopper-v0

`--num-envs` records from a vectorized env instead, one episode file per row.
Rows end at different steps and the env resets them itself, so episodes are
numbered in the order they finish rather than by row. Each row records
`ceil(--episodes / --num-envs)` of them and then retires, which is what keeps
the total exact; because a vector env is seeded once, only each row's first
episode carries a `--seed-start` seed, so `--num-envs == --episodes` is the form
to use when the seeds have to line up across runs.

Retention above the bundle's `context_length` is what this cycle turns on, and
whether it pays is environment-dependent — across the published bundles it helps
some and hurts others. Measure the retention you mean to record at before
spending a collection run on it.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import load_runner, load_runner_from_hub

# `collection/` sits beside `examples/` in the repository; it is the packager
# and, here, the recorder — not part of the installed runtime package.
from collection import CollectionRunner

DEFAULT_REPO_ID = "ccnets/causal-gpt-rl"

# Trajectories are the simulator's output, so the stack that produced them is
# part of what they are. The raw directory does not record it — print it, and
# keep the log beside the dataset.
STACK = ("torch", "gymnasium", "mujoco")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-id", required=True, help="Gymnasium environment id.")
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Raw directory to write ep_*.npz and spec.json into.",
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
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Environments to record in one batch. Above 1 the env is built "
        "with gym.make_vec and each row records ceil(--episodes / --num-envs) "
        "episodes, so the total is exact. A vector env is seeded once, so only "
        "each row's first episode carries a --seed-start seed; pass "
        "--num-envs == --episodes to have every episode seeded.",
    )
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Safety cap per episode, recorded as a truncation. The env's own "
        "TimeLimit usually fires first.",
    )
    p.add_argument(
        "--context-length",
        "--kv-cache-max-len",
        dest="kv_cache_max_len",
        type=int,
        default=None,
        help="Rollout context retained in the KV cache. Defaults to the bundle's "
        "trained context length. --kv-cache-max-len is kept as an alias.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Add to a raw directory that already holds episodes, instead of "
        "refusing it. The declared spaces must match.",
    )
    args = p.parse_args()
    if args.episodes < 1:
        p.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.max_steps < 1:
        p.error(f"--max-steps must be >= 1, got {args.max_steps}")
    if args.num_envs < 1:
        p.error(f"--num-envs must be >= 1, got {args.num_envs}")
    if args.kv_cache_max_len is not None and args.kv_cache_max_len < 1:
        p.error(
            "--context-length must be >= 1, got "
            f"{args.kv_cache_max_len}"
        )
    return args


def load_policy(args: argparse.Namespace) -> tuple[object, str]:
    """The runner, and the bundle identity to keep as provenance."""
    if args.bundle is not None:
        if not args.bundle.is_dir():
            raise FileNotFoundError(args.bundle)
        runner = load_runner(
            args.bundle,
            device=args.device,
            num_envs=args.num_envs,
            kv_cache_max_len=args.kv_cache_max_len,
        )
        return runner, str(args.bundle)
    subfolder = args.subfolder if args.subfolder is not None else args.env_id.lower()
    runner = load_runner_from_hub(
        repo_id=args.repo_id,
        subfolder=subfolder,
        device=args.device,
        num_envs=args.num_envs,
        kv_cache_max_len=args.kv_cache_max_len,
    )
    return runner, f"{args.repo_id}/{subfolder}"


def record_episodes(
    env: gym.Env,
    collector: CollectionRunner,
    *,
    episodes: int,
    max_steps: int,
    seed_start: int,
) -> list[dict]:
    """Drive `episodes` episodes through the collector, one seed each.

    The `max_steps` cap is passed as a truncation, not a termination: running
    out of time is not the environment reaching a terminal state, and the two
    mean different things to whatever trains on the result.
    """
    results = []
    for index in range(episodes):
        observation, _ = env.reset(seed=seed_start + index)
        collector.reset(observation, record=True)
        total = 0.0
        for step in range(max_steps):
            action = collector.act()
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            # A terminal state is the stronger claim, so it clears truncation —
            # whether the truncation came from the cap or from the env itself
            # (a TimeLimit can raise both flags on the same step). Recording
            # both would say two different things about one transition.
            truncated = not terminated and (
                bool(truncated) or step + 1 == max_steps
            )
            collector.observe(observation, reward, terminated, truncated)
            if terminated or truncated:
                break
        results.append(
            {
                "seed": seed_start + index,
                "return": total,
                "steps": step + 1,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        print(
            f"[ep {index + 1:03d}] seed={seed_start + index} return={total:.2f} "
            f"steps={step + 1} {'terminated' if terminated else 'truncated'}"
        )
    return results


def share_per_row(episodes: int, num_envs: int) -> int:
    """Each row's share of `episodes`, and with it the exact total written.

    The total is `num_envs * share`, which is `episodes` exactly when the two
    divide and rounds up otherwise. Rounding up rather than down keeps
    `--episodes` a floor the run is guaranteed to clear.
    """
    return -(-int(episodes) // int(num_envs))


def record_vector_episodes(
    venv,
    collector: CollectionRunner,
    *,
    seed_start: int,
) -> list[dict]:
    """Drive a vectorized env until every row has written its share.

    The collector's `episodes_per_row` is the stop condition, so the count is
    exact: rows that finish early retire instead of churning out unseeded
    repeats, and nothing is in flight to be flushed at the end. `--max-steps`
    rides in the sub-envs' own `TimeLimit` (see `main`) rather than being applied
    here, because only the env can restart a row it truncates.

    A vector env is seeded once, at reset, so only each row's first episode
    carries a seed the caller chose; the rest are the env restarting itself.
    Each result says which it is, because it is what decides whether two runs
    can be compared episode for episode.

    The returns and lengths tracked alongside are for the printed summary; the
    files are the collector's, and which rows wrote is read back from it so the
    two cannot drift. A row being seeded after an auto-reset contributes no
    transition — zero reward, no flags, its action ignored — so it is left out of
    the tally for that step.
    """
    if collector.episodes_per_row is None:
        raise ValueError(
            "record_vector_episodes needs a collector built with "
            "episodes_per_row; without a share per row there is no stop "
            "condition that lands on an exact count."
        )
    rows = venv.num_envs
    returns = np.zeros(rows, dtype=np.float64)
    lengths = np.zeros(rows, dtype=np.int64)
    seeding = np.zeros(rows, dtype=bool)
    results: list[dict] = []

    observations, _ = venv.reset(seed=[seed_start + row for row in range(rows)])
    collector.reset(observations, record=True)

    while not collector.all_rows_retired:
        written = collector.row_episodes
        actions = collector.act()
        observations, rewards, terminations, truncations, _ = venv.step(actions)
        terminations = np.asarray(terminations, dtype=bool)
        # A terminal state is the stronger claim, so it clears truncation — a
        # TimeLimit can raise both flags on the same step, and recording both
        # would say two different things about one transition.
        truncations = np.asarray(truncations, dtype=bool) & ~terminations
        collector.observe(observations, rewards, terminations, truncations)

        returns[seeding] = 0.0
        lengths[seeding] = 0
        live = ~seeding
        returns[live] += np.asarray(rewards, dtype=np.float64)[live]
        lengths[live] += 1

        # Which rows actually wrote — a retired row can end its episode without
        # a file landing, so the flags alone would over-count.
        for row in np.flatnonzero(collector.row_episodes > written):
            row = int(row)
            seed = seed_start + row if written[row] == 0 else None
            results.append(
                {
                    "row": row,
                    "seed": seed,
                    "return": float(returns[row]),
                    "steps": int(lengths[row]),
                    "terminated": bool(terminations[row]),
                    "truncated": bool(truncations[row]),
                }
            )
            print(
                f"[ep {len(results):03d}] row={row} "
                + (f"seed={seed}" if seed is not None else "unseeded")
                + f" return={returns[row]:.2f} steps={lengths[row]} "
                + ("terminated" if terminations[row] else "truncated")
            )
        seeding = (terminations | truncations) & live

    collector.close()
    return results


def _installed(package: str) -> str:
    try:
        return _pkg_version(package)
    except PackageNotFoundError:
        return "not installed"


def summarize(results: list[dict], out: Path) -> None:
    returns = np.asarray([r["return"] for r in results], dtype=np.float64)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float64)
    terminated = sum(r["terminated"] for r in results)
    print(
        f"\nepisodes={len(results)}  transitions={int(steps.sum())}  "
        f"mean_length={steps.mean():.1f}"
    )
    print(
        f"return mean={returns.mean():.2f}  std={returns.std():.2f}  "
        f"min={returns.min():.2f}  max={returns.max():.2f}"
    )
    print(f"terminated={terminated}  truncated={len(results) - terminated}")
    if terminated == 0:
        # A dataset where nothing ever ends teaches a termination head that
        # nothing ever ends. Worth knowing before packaging it.
        # ASCII only, here and everywhere this script prints: a console on a
        # legacy code page (cp949, cp1252) raises UnicodeEncodeError on the
        # write, which would fail a run whose episodes are already on disk.
        print(
            "[note] no episode reached a terminal state - every one ran out of "
            "time. Raise --max-steps, or expect a dataset with no terminations "
            "in it."
        )
    print(
        f"\nNext:\n  python collection/build_minari.py --raw {out} "
        "--dataset-id <namespace>/<name>-v0"
    )


def main() -> None:
    args = parse_args()
    print(
        "[stack] device=" + str(args.device) + "  "
        + "  ".join(f"{name}={_installed(name)}" for name in STACK)
    )

    if args.num_envs > 1:
        # The per-episode cap rides in each sub-env's own TimeLimit: only the
        # env can restart a row it truncates, so the loop cannot apply one.
        env = gym.make_vec(
            args.env_id, num_envs=args.num_envs, max_episode_steps=args.max_steps
        )
    else:
        env = gym.make(args.env_id)
    try:
        runner, bundle_id = load_policy(args)
        print(
            f"[context] trained={runner.context_length}  "
            f"rollout={runner.kv_cache_max_len}"
        )
        share = share_per_row(args.episodes, args.num_envs) if args.num_envs > 1 else None
        collector = CollectionRunner(
            runner,
            args.out,
            bundle=bundle_id,
            resume=args.resume,
            episodes_per_row=share,
        )
        print(collector)
        if args.num_envs > 1:
            total = args.num_envs * share
            print(
                f"[batch] {args.num_envs} rows x {share} episode(s) = {total} "
                f"episodes; seeds {args.seed_start}.."
                f"{args.seed_start + args.num_envs - 1} cover the first episode "
                "of each row"
                + ("" if share == 1 else ", the rest are unseeded auto-resets")
            )
            # This one closes the collector itself: with a share per row there
            # is nothing in flight left to flush.
            results = record_vector_episodes(
                env,
                collector,
                seed_start=args.seed_start,
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
        summarize(results, args.out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
