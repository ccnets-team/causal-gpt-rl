"""Find the retention lengths that land a simple / medium / expert ladder.

A tier is defined by its normalized score, the same quantity the public tables
use::

    norm = 100 * (return - random_ref) / (expert_ref - random_ref)

so before picking a retention you need both endpoints and a return-vs-retention
curve. This measures all three under one protocol:

  - ``random_ref`` : uniform action sampling, same seeds, same batch width
  - ``expert_ref`` : the best retention in ``--grid`` -- the ladder's own top
  - the curve      : mean return at every retention in ``--grid``

then reports the normalized score at each level and picks the grid value whose
score is closest to ``--target-simple`` / ``--target-medium``. Feed those two
numbers back into ``record_tiers.py``.

Retention is the degradation dial that reaches the recorder. A dial that
perturbs the emitted action does not: `CollectionRunner` writes the action
`act()` returned, and this policy conditions on its own past actions, so a
perturbation applied outside it would leave the trajectory and the policy's
context disagreeing about what was taken.

Whether retention spreads a given environment far enough to *be* a ladder is the
question this script exists to answer before a collection run is spent on it. It
does not always: the published Humanoid bundle moves about 9% in mean return
across its whole grid. The normalized spread at the bottom is where to look.

Example:
    python -m examples.mujoco_collection.calibrate_retention \
        --env-id Hopper-v5 --target-simple 40 --target-medium 70
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import torch

from causal_gpt_rl.inference import load_runner, load_runner_from_hub
from examples.deploy.reproduce import (
    DEFAULT_REPO_ID,
    installed_versions,
    print_stack_report,
    run_seed_batch,
)

# Half the trained window, the trained window, and up to thirty times past it.
DEFAULT_GRID = (16, 32, 64, 128, 256, 512, 1000)

# Below this the grid has not produced a ladder, whatever the closest picks say.
MIN_LADDER_SPREAD = 20.0

# Printed in place of tier picks when the grid produced no usable score at all,
# which is the collapsed twin of the spread note above. ASCII only, like
# everything this script prints: a console on a legacy code page raises
# UnicodeEncodeError on the write, which would fail a run whose measurements
# are already done.
NO_LADDER_NOTE = (
    "\n  [note] no retention beat the random reference, so every normalized\n"
    "         score is undefined and there are no tiers to cut here. What to\n"
    "         change is the bundle or the environment, not the retention."
)


class RandomPolicy:
    """Uniform action sampling, wearing the runner's three-call interface.

    `run_seed_batch` only calls `reset` / `act` / `observe`, so the reference
    return comes out of the same loop, the same seeds, and the same batch width
    as every retention measured below it. Measuring it any other way would put
    the two endpoints of the normalization on different protocols.
    """

    def __init__(self, action_space, seed: int):
        self._action_space = action_space
        self._action_space.seed(seed)

    def reset(self, observation) -> None:
        pass

    def observe(self, observation) -> None:
        pass

    def act(self):
        return self._action_space.sample()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-id", required=True, help="Gymnasium environment id.")
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
        "--grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_GRID),
        metavar="N",
        help="Retention lengths to measure. Default: "
        + " ".join(str(value) for value in DEFAULT_GRID),
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per level, one per seed - and the width of the batch, "
        "which is part of the measurement.",
    )
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--target-simple", type=float, default=40.0)
    p.add_argument("--target-medium", type=float, default=70.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full curve, including per-seed returns, here.",
    )
    args = p.parse_args()
    if args.episodes < 1:
        p.error(f"--episodes must be >= 1, got {args.episodes}")
    if args.max_steps < 1:
        p.error(f"--max-steps must be >= 1, got {args.max_steps}")
    invalid = [value for value in args.grid if value < 1]
    if invalid:
        p.error(f"--grid takes retention lengths >= 1, got {invalid}")
    args.grid = sorted(set(args.grid))
    return args


def make_envs(env_id: str, count: int) -> gym.vector.VectorEnv:
    """One environment per seed, advanced by a single batched call per step.

    A fresh set per level, matching the sweep protocol: reusing one set across
    levels would be one more thing to have to argue is equivalent.
    """
    return gym.vector.SyncVectorEnv(
        [lambda eid=env_id: gym.make(eid) for _ in range(count)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def load_policy(args: argparse.Namespace, retention: int):
    """A fresh runner at one retention, and the bundle identity behind it."""
    if args.bundle is not None:
        if not args.bundle.is_dir():
            raise FileNotFoundError(args.bundle)
        runner = load_runner(
            args.bundle,
            device=args.device,
            num_envs=args.episodes,
            kv_cache_max_len=retention,
        )
        return runner, str(args.bundle)
    subfolder = args.subfolder if args.subfolder is not None else args.env_id.lower()
    runner = load_runner_from_hub(
        repo_id=args.repo_id,
        subfolder=subfolder,
        device=args.device,
        num_envs=args.episodes,
        kv_cache_max_len=retention,
    )
    return runner, f"{args.repo_id}/{subfolder}"


def measure(args: argparse.Namespace, policy, seeds: list[int]) -> dict:
    """One level's returns, under the protocol every other level also ran."""
    envs = make_envs(args.env_id, len(seeds))
    try:
        returns, lengths, ends = run_seed_batch(
            envs, policy, seeds, args.max_steps, per_seed=False
        )
    finally:
        envs.close()
    return {
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "length_mean": float(lengths.mean()),
        "full": int(ends["truncated"].sum()),
        "terminated": int(ends["terminated"].sum()),
        "returns": returns.tolist(),
    }


def normalized(value: float, random_ref: float, expert_ref: float) -> float:
    """The published normalization, guarded against a collapsed denominator."""
    spread = expert_ref - random_ref
    if spread <= 0:
        # The policy did not beat uniform sampling. There is no ladder here, and
        # a normalized score would only dress that up as a number.
        return float("nan")
    return 100.0 * (value - random_ref) / spread


def ladder_collapsed(rows: list[dict]) -> bool:
    """True when the grid produced no usable normalized score.

    `normalized` returns NaN for a policy that did not beat uniform sampling,
    and it collapses the whole grid at once: the denominator is the best row's
    margin over random, which every row shares. Reading that NaN back rather
    than re-deriving the comparison keeps one definition of "no ladder".
    """
    return all(math.isnan(row["norm"]) for row in rows)


def print_curve(rows: list[dict], episodes: int, expert_retention: int) -> None:
    print(
        f"\n{'retention':>9} {'return':>12} {'std':>10} {'worst':>12} "
        f"{'full':>9} {'norm':>8}"
    )
    print("-" * 66)
    for row in rows:
        mark = "  <- expert_ref" if row["retention"] == expert_retention else ""
        full = f"{row['full']}/{episodes}"
        print(
            f"{row['retention']:>9} {row['return_mean']:>12.2f} "
            f"{row['return_std']:>10.2f} {row['return_min']:>12.2f} "
            f"{full:>9} {row['norm']:>8.1f}{mark}"
        )


def pick(rows: list[dict], target: float) -> dict:
    """The measured level whose normalized score sits closest to `target`."""
    return min(rows, key=lambda row: abs(row["norm"] - target))


def report_picks(rows: list[dict], args: argparse.Namespace, expert: dict) -> None:
    print("\ntiers")
    targets = (("simple", args.target_simple), ("medium", args.target_medium))
    for name, target in targets:
        chosen = pick(rows, target)
        miss = abs(chosen["norm"] - target)
        note = "" if miss <= 10 else f"   [note] {miss:.1f} off - refine --grid"
        print(
            f"  {name:<8} target {target:>5.1f}   retention {chosen['retention']:<5} "
            f"norm {chosen['norm']:>6.1f}{note}"
        )
    print(
        f"  {'expert':<8} {'':>12}   retention {expert['retention']:<5} norm  100.0"
    )

    spread = max(row["norm"] for row in rows) - min(row["norm"] for row in rows)
    print(f"\nnormalized spread across the grid: {spread:.1f}")
    if spread < MIN_LADDER_SPREAD:
        # ASCII only, here and everywhere this script prints: a console on a
        # legacy code page raises UnicodeEncodeError on the write, which would
        # fail a run whose measurements are already done.
        print(
            "  [note] retention barely separates this environment. Three tiers cut\n"
            "         out of this curve would differ by less than the seed draw -\n"
            "         record one dataset at the best retention instead of a ladder."
        )


def next_command(args: argparse.Namespace, rows: list[dict], expert: dict) -> str:
    tiers = " ".join(
        f"--tier {name}={pick(rows, target)['retention']}"
        for name, target in (
            ("simple", args.target_simple),
            ("medium", args.target_medium),
        )
    )
    return (
        "  python -m examples.mujoco_collection.record_tiers "
        f"--env-id {args.env_id} \\\n"
        f"      --out raw/{args.env_id.lower()} {tiers} "
        f"--tier expert={expert['retention']}"
    )


def main() -> None:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    print_stack_report(installed_versions())
    print(
        f"\n{args.env_id} - {args.episodes} episodes as one batch . "
        f"seeds {seeds[0]}..{seeds[-1]} . {args.device}",
        flush=True,
    )

    probe = gym.make(args.env_id)
    try:
        action_space = gym.vector.utils.batch_space(probe.action_space, args.episodes)
    finally:
        probe.close()
    random_row = measure(args, RandomPolicy(action_space, args.seed_start), seeds)
    random_ref = random_row["return_mean"]
    print(
        f"  random policy   {random_ref:>12.2f} +- {random_row['return_std']:.2f}",
        flush=True,
    )

    rows: list[dict] = []
    bundle_id = None
    for retention in args.grid:
        policy, bundle_id = load_policy(args, retention)
        row = measure(args, policy, seeds)
        row["retention"] = retention
        row["trained_context"] = int(policy.context_length)
        rows.append(row)
        print(
            f"  retention {retention:<5} {row['return_mean']:>12.2f} +- "
            f"{row['return_std']:.2f}",
            flush=True,
        )

    expert = max(rows, key=lambda row: row["return_mean"])
    for row in rows:
        row["norm"] = normalized(row["return_mean"], random_ref, expert["return_mean"])

    print_curve(rows, args.episodes, expert["retention"])
    # Neither the picks nor the next command survive an all-NaN curve. `pick`
    # is a min over abs(norm - target), and with every key NaN each comparison
    # is False, so it returns the first grid entry — the smallest retention,
    # printed as both tiers. `report_picks`' own spread guard cannot catch that
    # either: NaN < MIN_LADDER_SPREAD is False, so the guidance is suppressed
    # exactly where it applies. The measurements above still print; they cost a
    # run, and they are what says there is nothing here.
    collapsed = ladder_collapsed(rows)
    if collapsed:
        print(NO_LADDER_NOTE)
    else:
        report_picks(rows, args, expert)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "env_id": args.env_id,
                    "bundle": bundle_id,
                    "episodes": args.episodes,
                    "seed_start": args.seed_start,
                    "max_steps": args.max_steps,
                    "device": args.device,
                    "random_ref": random_ref,
                    "expert_ref": expert["return_mean"],
                    "expert_retention": expert["retention"],
                    "curve": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    if not collapsed:
        print("\nNext:")
        print(next_command(args, rows, expert))


if __name__ == "__main__":
    main()
