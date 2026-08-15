"""Measure a published bundle under the reproduction protocol.

The quickstart in the README is a smoke test: a handful of episodes off one
seed, enough to see a policy move. A return from five episodes of a
high-variance environment lands wherever the draw put it.

The protocol is fifty episodes, seeds 0..49, capped at `--max-steps`, with the
KV cache left at the bundle's own context length::

    python -m examples.deploy.reproduce --env-id Ant-v5
    python -m examples.deploy.reproduce --env-id all --json card.json

Three things fix the number, and dropping any one of them changes it.

*The seeds.* `run_episodes` seeds only its first reset and lets the
environment's RNG carry the rest, so fifty of its episodes are fifty draws from
the same distribution rather than seeds 0..49 — comparable in the mean, never
equal episode by episode.

*The batch width.* The fifty episodes run together, as one fifty-row batch: one
environment per seed, advanced by a single batched policy call per step. A
batch-of-fifty forward and a batch-of-one forward do not reduce in the same
order, and in a closed autoregressive loop that last-bit difference compounds
over a thousand steps until the trajectories separate outright. Fifty
sequential rollouts are a different measurement, not a slower one.

*The runtime.* A simulator release change is a different measurement even with
identical weights and seeds, so the installed versions are printed beside the
ones the protocol is defined on.

A row that terminates early is auto-reset by the vector env and keeps stepping,
so only each row's first episode is scored.

A local bundle works the same way::

    python -m examples.deploy.reproduce --env-id Hopper-v5 --bundle path/to/bundle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import load_runner, load_runner_from_hub

# The runtime the protocol is defined on. Simulator releases change contact
# dynamics, so a different stack is a different measurement — hence a warning
# rather than a pin, which would force the whole package onto one runtime.
REFERENCE_STACK = {
    "torch": "2.8.0",
    "gymnasium": "1.2.3",
    "mujoco": "3.2.3",
}

DEFAULT_REPO_ID = "ccnets/causal-gpt-rl"

# What `--env-id all` expands to: the published MuJoCo bundles, in card order.
PUBLISHED_ENVS = [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--env-id",
        required=True,
        nargs="+",
        metavar="ENV_ID",
        help="One or more Gymnasium environment ids, e.g. Ant-v5. Pass 'all' "
        "for every published bundle.",
    )
    p.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Local bundle directory. Omit to download from the Hub. Single "
        "environment only.",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hub repo to load from.")
    p.add_argument(
        "--subfolder",
        default=None,
        help="Bundle subfolder in the repo. Defaults to the lowercased env id. "
        "Single environment only.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes, one per seed — and the width of the batch, which is "
        "part of the measurement. The protocol is 50.",
    )
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument(
        "--kv-cache-max-len",
        type=int,
        default=None,
        help="KV cache cap. Defaults to the bundle's context length, which is "
        "what the protocol uses.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--per-seed",
        action="store_true",
        help="Print each seed's return as its episode ends, not just the summary.",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full result, including per-seed returns, here.",
    )
    args = p.parse_args()

    if args.episodes < 1:
        p.error(f"--episodes must be >= 1, got {args.episodes}")
    if len(args.env_id) == 1 and args.env_id[0].lower() == "all":
        args.env_id = list(PUBLISHED_ENVS)
    if len(args.env_id) > 1:
        for flag, value in (("--bundle", args.bundle), ("--subfolder", args.subfolder)):
            if value is not None:
                p.error(f"{flag} names a single bundle; pass one --env-id with it.")
    return args


def installed_versions() -> dict[str, str]:
    """Report the versions that matter to a measurement, as installed."""
    versions = {"torch": torch.__version__, "gymnasium": gym.__version__}
    try:
        import mujoco
    except ImportError:
        versions["mujoco"] = "not installed"
    else:
        versions["mujoco"] = mujoco.__version__
    return versions


def print_stack_report(versions: dict[str, str]) -> bool:
    """Print installed versus reference versions; return True if they agree."""
    print("runtime")
    matched = True
    for name, reference in REFERENCE_STACK.items():
        installed = versions.get(name, "not installed")
        # A local version segment ("2.8.0+cu129") is a build of the same
        # release, so compare on the part before it.
        agrees = installed.split("+")[0] == reference
        matched &= agrees
        mark = " " if agrees else "*"
        print(f"  {mark} {name:<11} {installed:<16} protocol: {reference}")
    if not matched:
        print(
            "  * differs from the stack the protocol is defined on — expect the\n"
            "    returns to differ too, seeds held equal."
        )
    return bool(matched)


def load(args: argparse.Namespace, env_id: str, num_envs: int):
    if args.bundle is not None:
        if not args.bundle.is_dir():
            raise FileNotFoundError(args.bundle)
        return load_runner(
            args.bundle,
            device=args.device,
            num_envs=num_envs,
            kv_cache_max_len=args.kv_cache_max_len,
        )
    subfolder = args.subfolder if args.subfolder is not None else env_id.lower()
    return load_runner_from_hub(
        repo_id=args.repo_id,
        subfolder=subfolder,
        device=args.device,
        num_envs=num_envs,
        kv_cache_max_len=args.kv_cache_max_len,
    )


def run_seed_batch(
    envs: gym.vector.VectorEnv,
    runner,
    seeds: list[int],
    max_steps: int,
    per_seed: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Score every seed's first episode, all rows advancing in lockstep.

    Returns per-row totals and lengths, plus boolean masks for how each row's
    first episode ended: `completed` at all, and split into `terminated` (the
    environment reached a terminal state) and `truncated` (it ran out of time).
    The two mean different things on a card — a Humanoid that falls at step 300
    and one that is still walking at the time limit are not the same result.
    """
    count = len(seeds)
    totals = np.zeros(count, dtype=np.float64)
    lengths = np.zeros(count, dtype=np.int64)
    completed = np.zeros(count, dtype=bool)
    ended_terminated = np.zeros(count, dtype=bool)
    ended_truncated = np.zeros(count, dtype=bool)

    obs, _ = envs.reset(seed=seeds)
    runner.reset(obs)

    for _ in range(max_steps):
        # At num_envs=1 the runner squeezes the batch dimension away, which is
        # what a bare env wants and what a vector env of one does not. Put it
        # back rather than special-casing the single-seed run into its own path.
        action = np.asarray(runner.act())
        if action.shape[:1] != (count,):
            action = action[None]
        obs, reward, terminated, truncated, _ = envs.step(action)

        # A completed row keeps stepping under auto-reset; its later rewards
        # belong to a second episode nobody asked for, so they are dropped.
        active = ~completed
        totals[active] += np.asarray(reward, dtype=np.float64)[active]
        lengths[active] += 1

        # The vector env auto-resets on either kind of ending, so the loop
        # treats them alike — but they are recorded apart.
        term = np.asarray(terminated, dtype=bool)
        trunc = np.asarray(truncated, dtype=bool)
        just_ended = active & (term | trunc)
        ended_terminated |= just_ended & term
        ended_truncated |= just_ended & ~term
        if per_seed:
            for row in np.flatnonzero(just_ended):
                print(
                    f"    [seed {seeds[row]:>3}] return={totals[row]:10.2f}  "
                    f"steps={lengths[row]}  "
                    f"{'terminated' if term[row] else 'truncated'}",
                    flush=True,
                )
        completed |= just_ended
        if completed.all():
            break

        # Under SAME_STEP auto-reset, a row that just ended already carries its
        # next episode's first observation, so its context is wiped before that
        # is handed over. Rows still running keep their history. Rows that ended
        # earlier are reset again here: they too were auto-reset by the env.
        runner.reset_rows(term | trunc)
        runner.observe(obs)

    return totals, lengths, {
        "completed": completed,
        "terminated": ended_terminated,
        "truncated": ended_truncated,
    }


def measure(args: argparse.Namespace, env_id: str, seeds: list[int]) -> dict:
    """Run one environment's protocol and return its record."""
    # One runner row and one environment per seed. `--episodes 1` is the same
    # code path, a batch of one.
    runner = load(args, env_id, num_envs=args.episodes)
    envs = gym.vector.SyncVectorEnv(
        [lambda eid=env_id: gym.make(eid) for _ in seeds],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        print(
            f"\n{env_id} — {args.episodes} episodes as one batch · "
            f"seeds {seeds[0]}..{seeds[-1]} · kv {runner.kv_cache_max_len} · "
            f"{args.device}",
            flush=True,
        )
        returns, lengths, ends = run_seed_batch(
            envs, runner, seeds, args.max_steps, args.per_seed
        )
    finally:
        envs.close()

    completed = int(ends["completed"].sum())
    terminated = int(ends["terminated"].sum())
    truncated = int(ends["truncated"].sum())

    # Population std, matching `run_episodes` and the published card.
    print(
        f"  return  {returns.mean():.2f} ± {returns.std():.2f}"
        f"     min {returns.min():.2f}   max {returns.max():.2f}"
    )
    print(
        f"  length  {lengths.mean():.1f} ± {lengths.std():.1f}"
        f"     {completed}/{args.episodes} episodes completed"
        f"  ({terminated} terminated, {truncated} truncated)"
    )
    return {
        "env_id": env_id,
        "bundle": str(args.bundle) if args.bundle else None,
        "repo_id": None if args.bundle else args.repo_id,
        "subfolder": None if args.bundle else (args.subfolder or env_id.lower()),
        "episodes": args.episodes,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "max_steps": args.max_steps,
        "kv_cache_max_len": runner.kv_cache_max_len,
        "device": args.device,
        "vectorized": True,
        "num_envs": args.episodes,
        "completed": completed,
        "terminated": terminated,
        "truncated": truncated,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "returns": returns.tolist(),
        "lengths": lengths.tolist(),
    }


def print_table(records: list[dict], episodes: int) -> None:
    """One row per environment, in the shape the model card reports."""
    print(
        f"\n{'bundle':<22} {'return':>22} {'length':>18} "
        f"{'completed':>11} {'term/trunc':>12}"
    )
    for r in records:
        returns = f"{r['return_mean']:.2f} ± {r['return_std']:.2f}"
        lengths = f"{r['length_mean']:.1f} ± {r['length_std']:.1f}"
        print(
            f"{r['subfolder'] or r['env_id']:<22} {returns:>22} {lengths:>18} "
            f"{str(r['completed']) + '/' + str(episodes):>11} "
            f"{str(r['terminated']) + '/' + str(r['truncated']):>12}"
        )


def main() -> None:
    args = parse_args()

    versions = installed_versions()
    stack_matches = print_stack_report(versions)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    records: list[dict] = []
    failed: list[tuple[str, str]] = []
    for env_id in args.env_id:
        try:
            records.append(measure(args, env_id, seeds))
        except Exception as exc:  # one missing bundle should not lose the sweep
            if len(args.env_id) == 1:
                raise
            print(f"\n{env_id} — failed: {type(exc).__name__}: {exc}", flush=True)
            failed.append((env_id, f"{type(exc).__name__}: {exc}"))

    if len(records) > 1:
        print_table(records, args.episodes)
    if failed:
        print(f"\n{len(failed)} environment(s) failed: {', '.join(e for e, _ in failed)}")
    if not stack_matches:
        print("\nMeasured on a different runtime than the protocol — see above.")

    if args.json is not None:
        payload = {
            "versions": versions,
            "reference_stack": REFERENCE_STACK,
            "reference_stack_matched": stack_matches,
            "runs": records,
            "failed": [{"env_id": e, "error": m} for e, m in failed],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
