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
        --env-id Hopper-v5 --out raw/ --episodes 20 --kv-cache-max-len 256
    python collection/build_minari.py --raw raw/ --dataset-id review/hopper-v0

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
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Safety cap per episode, recorded as a truncation. The env's own "
        "TimeLimit usually fires first.",
    )
    p.add_argument(
        "--kv-cache-max-len",
        type=int,
        default=None,
        help="KV cache cap. Defaults to the bundle's context length; raising it "
        "is what a second collection cycle turns on.",
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
    return args


def load_policy(args: argparse.Namespace) -> tuple[object, str]:
    """The runner, and the bundle identity to keep as provenance."""
    if args.bundle is not None:
        if not args.bundle.is_dir():
            raise FileNotFoundError(args.bundle)
        runner = load_runner(
            args.bundle,
            device=args.device,
            num_envs=1,
            kv_cache_max_len=args.kv_cache_max_len,
        )
        return runner, str(args.bundle)
    subfolder = args.subfolder if args.subfolder is not None else args.env_id.lower()
    runner = load_runner_from_hub(
        repo_id=args.repo_id,
        subfolder=subfolder,
        device=args.device,
        num_envs=1,
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
            # The cap only truncates an episode the environment did not already
            # end: a terminal state is the stronger claim, and flagging both
            # would say two different things about one transition.
            truncated = bool(truncated) or (
                not terminated and step + 1 == max_steps
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
        print(
            "[note] no episode reached a terminal state — every one ran out of "
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

    env = gym.make(args.env_id)
    try:
        runner, bundle_id = load_policy(args)
        collector = CollectionRunner(
            runner, args.out, bundle=bundle_id, resume=args.resume
        )
        print(collector)
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
