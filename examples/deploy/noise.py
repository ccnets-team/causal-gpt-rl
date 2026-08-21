"""Stress a published bundle with noise and watch where its return falls apart.

The reproduction protocol measures a bundle in the environment it was trained
for, clean. Deployment is rarely clean: a sensor drifts, an actuator undershoots,
a state estimate arrives smeared. This runs the same protocol with noise injected
at one of the two points the inference surface exposes, and reports the return
against the same run's clean baseline::

    python -m examples.deploy.noise --env-id Ant-v5 --channel obs
    python -m examples.deploy.noise --env-id Ant-v5 --channel action \
        --sigmas 0 0.05 0.1 0.2 --episodes 50

Two channels, and they are not the same experiment.

*Observation noise* is added to the state before `runner.observe` sees it. The
simulator's own state is untouched — the policy is misinformed, not disturbed.
Sigma is in units of the bundle's own training-distribution standard deviation,
per dimension, read out of the embedded normalizer, so `--sigmas 0.1` is a tenth
of a training sigma on every perturbed coordinate and means the same thing in
Ant as in Humanoid, whose raw units are nothing alike.

*Action noise* is added to the action the environment executes, then clipped
back into the action space. The runner keeps conditioning on the action it
emitted, because that is the only action the inference surface lets it see — so
this is the case where the world does not do what the policy asked, and the
policy learns about it only through the next observation. Sigma is in units of
the action space's half-range, so 0.1 is a tenth of full travel regardless of
whether the space is [-1, 1] or [-0.4, 0.4].

**Choose the dimensions.** A MuJoCo observation is not one sensor. Ant-v5's 105
values are 13 joint positions, 14 velocities, and then 78 contact forces that
are clipped to [-1, 1] and read exactly zero most of the time; Humanoid-v5's 348
put 45 kinematic values in front of inertias, com velocities, actuator forces and
contacts. Gaussian noise on a clipped, mostly-zero contact channel invents
contacts that are not there, which is not a failure mode any real sensor has.
`--dims` restricts the noise to index ranges, and the kinematic prefix is usually
what you want::

    --dims 0:27     # Ant-v5:      qpos + qvel
    --dims 0:45     # Humanoid-v5: qpos + qvel

Left unset, every dimension is perturbed and the report says how many of them
were constant in the training data — those have a training sigma at the
normalizer's epsilon floor, so what reaches the model there is a rounding
artifact rather than a controlled insult. Read the note; do not average over it.

A sweep always runs sigma 0 first, in the same process, on the same seeds. The
percentage column is against that row rather than against the model card: a card
number measured on a different day and a different runtime would fold a runtime
difference into what is supposed to be the noise response.

The noise stream is seeded (`--noise-seed`) and redrawn identically for every
sigma, so a sweep is reproducible and two sigmas differ by scale rather than by
draw.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import load_runner, load_runner_from_hub

from examples.deploy.reproduce import (
    DEFAULT_REPO_ID,
    PUBLISHED_ENVS,
    installed_versions,
    print_stack_report,
)

DEFAULT_SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.4]

# A dimension that never moved in training normalizes with `sqrt(var) + EPS`
# where `var` is zero, so its sigma sits at the floor. Perturbing it by a
# fraction of that floor is a change of order 1e-9 in raw units, which float32
# may or may not carry into the model at all — the effect is a precision
# accident, not a measurement. Such dims are counted and reported, never
# silently dropped.
DEGENERATE_SIGMA = 1e-6


def parse_dims(spec: str) -> tuple[int, int]:
    """Parse one `START:END` observation slice, end-exclusive."""
    try:
        start_text, end_text = spec.split(":")
        start, end = int(start_text), int(end_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--dims wants START:END with integers, got {spec!r}"
        ) from None
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(f"--dims needs 0 <= START < END, got {spec!r}")
    return start, end


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--env-id",
        required=True,
        nargs="+",
        metavar="ENV_ID",
        help="One or more Gymnasium environment ids. Pass 'all' for every "
        "published bundle.",
    )
    p.add_argument(
        "--channel",
        choices=["obs", "action", "both"],
        default="obs",
        help="Where the noise enters. 'both' drives one sigma into each channel "
        "at once, which measures the pair and not either one.",
    )
    p.add_argument(
        "--dims",
        type=parse_dims,
        nargs="+",
        default=None,
        metavar="START:END",
        help="Observation index ranges to perturb, end-exclusive; repeatable. "
        "Defaults to every dimension. Ignored by the action channel.",
    )
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMAS,
        help="Noise levels to sweep. 0 is the in-run clean baseline and is "
        "prepended if you leave it out.",
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
        help="Bundle subfolder in the repo. Defaults to the lowercased env id.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes, one per seed, run as one batch. The protocol is 50.",
    )
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Seeds the noise stream, which is redrawn identically for every "
        "sigma in the sweep.",
    )
    p.add_argument("--kv-cache-max-len", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    if args.episodes < 1:
        p.error(f"--episodes must be >= 1, got {args.episodes}")
    # `s < 0` is False for NaN and for +inf alike, so a bare sign test lets both
    # through. NaN poisons the clean baseline that every later row is divided by
    # and never prepends the 0 row; inf scales the noise past anything the
    # simulator can integrate.
    if not all(np.isfinite(s) and s >= 0.0 for s in args.sigmas):
        p.error(f"--sigmas must be finite and non-negative, got {args.sigmas}")
    if len(args.env_id) == 1 and args.env_id[0].lower() == "all":
        args.env_id = list(PUBLISHED_ENVS)
    if len(args.env_id) > 1:
        for flag, value in (("--bundle", args.bundle), ("--subfolder", args.subfolder)):
            if value is not None:
                p.error(f"{flag} names a single bundle; pass one --env-id with it.")

    sigmas = sorted(set(args.sigmas))
    if sigmas[0] > 0.0:
        sigmas.insert(0, 0.0)
    args.sigmas = sigmas
    return args


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


def training_sigma(runner, obs_size: int) -> tuple[np.ndarray, str]:
    """Per-dimension noise unit: the training-distribution standard deviation.

    A v2 bundle carries its normalizer inside the model, a v1 bundle beside it,
    and either way this is the scale that makes `0.1` mean the same thing in Ant
    and in Humanoid. With neither, fall back to raw units and say so — a sweep in
    raw units is still a sweep, it just is not comparable across environments.
    """
    model = runner.model
    if getattr(model, "has_embedded_state_normalizer", lambda: False)():
        std = model.state_normalization_std.detach().to("cpu").numpy().reshape(-1)
        return std[-obs_size:].astype(np.float64), "training sigma (embedded)"
    normalizer = getattr(runner, "state_normalizer", None)
    if normalizer is not None:
        std = normalizer.std.detach().to("cpu").numpy().reshape(-1)
        return std[-obs_size:].astype(np.float64), "training sigma (sidecar)"
    return np.ones(obs_size, dtype=np.float64), "raw observation units"


def build_obs_mask(
    dims: list[tuple[int, int]] | None, obs_size: int
) -> np.ndarray:
    """Boolean mask over observation indices, True where noise is applied."""
    if dims is None:
        return np.ones(obs_size, dtype=bool)
    mask = np.zeros(obs_size, dtype=bool)
    for start, end in dims:
        if start >= obs_size:
            raise ValueError(
                f"--dims {start}:{end} starts past the {obs_size}-dim observation"
            )
        mask[start : min(end, obs_size)] = True
    return mask


def describe_obs_noise(
    mask: np.ndarray, scale: np.ndarray, note: str
) -> tuple[str, dict]:
    """One line saying what the observation channel actually perturbs."""
    perturbed = int(mask.sum())
    degenerate = int((mask & (scale <= DEGENERATE_SIGMA)).sum())
    line = f"{perturbed}/{mask.size} observation dims, unit: {note}"
    if degenerate:
        line += (
            f"\n    {degenerate} of them were constant in training (sigma at the "
            f"normalizer floor);\n    what reaches the model there is a precision "
            f"artifact — exclude them with --dims"
        )
    return line, {
        "dims_perturbed": perturbed,
        "dims_total": int(mask.size),
        "dims_degenerate": degenerate,
        "obs_scale_note": note,
    }


def action_scale(space: gym.spaces.Space) -> np.ndarray:
    """Half-range of the action space, the unit the action channel is quoted in."""
    low = np.asarray(space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(space.high, dtype=np.float64).reshape(-1)
    return (high - low) / 2.0


def run_seed_batch(
    envs: gym.vector.VectorEnv,
    runner,
    seeds: list[int],
    max_steps: int,
    *,
    obs_sigma: float,
    act_sigma: float,
    obs_scale: np.ndarray,
    act_scale: np.ndarray,
    act_low: np.ndarray,
    act_high: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Score every seed's first episode with noise on the configured channels.

    The loop is `examples.deploy.reproduce.run_seed_batch` with two insertions:
    the observation is perturbed on its way into the runner, and the action is
    perturbed on its way into the environment. Everything else — the shared
    batch, the auto-reset that only the scoring ignores, the first-episode-only
    accounting — is unchanged, so sigma 0 reproduces that function exactly.

    `obs_scale` is already zero on the dimensions `--dims` excluded, so the mask
    costs nothing per step.
    """
    count = len(seeds)
    totals = np.zeros(count, dtype=np.float64)
    lengths = np.zeros(count, dtype=np.int64)
    completed = np.zeros(count, dtype=bool)
    ended_terminated = np.zeros(count, dtype=bool)
    ended_truncated = np.zeros(count, dtype=bool)

    def perturb_obs(obs):
        if obs_sigma <= 0.0:
            return obs
        arr = np.asarray(obs, dtype=np.float64)
        return arr + rng.standard_normal(arr.shape) * (obs_sigma * obs_scale)

    def perturb_action(action: np.ndarray) -> np.ndarray:
        if act_sigma <= 0.0:
            return action
        noisy = np.asarray(action, dtype=np.float64)
        noisy = noisy + rng.standard_normal(noisy.shape) * (act_sigma * act_scale)
        return np.clip(noisy, act_low, act_high).astype(np.float32)

    obs, _ = envs.reset(seed=seeds)
    runner.reset(perturb_obs(obs))

    for _ in range(max_steps):
        action = np.asarray(runner.act())
        if action.shape[:1] != (count,):
            action = action[None]
        obs, reward, terminated, truncated, _ = envs.step(perturb_action(action))

        active = ~completed
        totals[active] += np.asarray(reward, dtype=np.float64)[active]
        lengths[active] += 1

        term = np.asarray(terminated, dtype=bool)
        trunc = np.asarray(truncated, dtype=bool)
        just_ended = active & (term | trunc)
        ended_terminated |= just_ended & term
        ended_truncated |= just_ended & ~term
        completed |= just_ended
        if completed.all():
            break

        runner.observe(perturb_obs(obs))

    return totals, lengths, {
        "completed": completed,
        "terminated": ended_terminated,
        "truncated": ended_truncated,
    }


def measure_one(
    args: argparse.Namespace,
    env_id: str,
    seeds: list[int],
    sigma: float,
) -> dict:
    """One sigma, one environment: a fresh runner, fresh envs, fresh noise stream."""
    runner = load(args, env_id, num_envs=args.episodes)
    envs = gym.vector.SyncVectorEnv(
        [lambda eid=env_id: gym.make(eid) for _ in seeds],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        space = envs.single_action_space
        obs_size = int(np.prod(envs.single_observation_space.shape))
        scale, note = training_sigma(runner, obs_size)
        mask = build_obs_mask(args.dims, obs_size)
        description, meta = describe_obs_noise(mask, scale, note)
        # Folding the mask into the scale keeps the hot loop a single multiply.
        obs_scale = np.where(mask, scale, 0.0)

        act_scale = action_scale(space)
        act_low = np.asarray(space.low, dtype=np.float64)
        act_high = np.asarray(space.high, dtype=np.float64)

        obs_sigma = sigma if args.channel in ("obs", "both") else 0.0
        act_sigma = sigma if args.channel in ("action", "both") else 0.0

        returns, lengths, ends = run_seed_batch(
            envs,
            runner,
            seeds,
            args.max_steps,
            obs_sigma=obs_sigma,
            act_sigma=act_sigma,
            obs_scale=obs_scale,
            act_scale=act_scale,
            act_low=act_low,
            act_high=act_high,
            rng=np.random.default_rng(args.noise_seed),
        )
    finally:
        envs.close()

    record = {
        "env_id": env_id,
        "sigma": sigma,
        "channel": args.channel,
        "obs_sigma": obs_sigma,
        "act_sigma": act_sigma,
        "noise_description": description,
        "kv_cache_max_len": runner.kv_cache_max_len,
        "completed": int(ends["completed"].sum()),
        "terminated": int(ends["terminated"].sum()),
        "truncated": int(ends["truncated"].sum()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "returns": returns.tolist(),
        "lengths": lengths.tolist(),
    }
    record.update(meta)
    return record


def sweep(args: argparse.Namespace, env_id: str, seeds: list[int]) -> list[dict]:
    print(
        f"\n{env_id} — {args.channel} noise · {args.episodes} episodes as one "
        f"batch · seeds {seeds[0]}..{seeds[-1]} · {args.device}",
        flush=True,
    )
    print(
        f"  {'sigma':>7} {'return':>22} {'vs clean':>9} "
        f"{'length':>16} {'completed':>11} {'term/trunc':>12}"
    )

    records: list[dict] = []
    baseline = None
    for sigma in args.sigmas:
        record = measure_one(args, env_id, seeds, sigma)
        if baseline is None:
            baseline = record["return_mean"]
        # A baseline at or below zero has no meaningful ratio, so the column
        # goes blank rather than printing a percentage of nothing.
        usable = baseline is not None and baseline > 0.0
        record["pct_of_clean"] = (
            100.0 * record["return_mean"] / baseline if usable else None
        )
        share = f"{record['pct_of_clean']:.1f}%" if usable else "-"
        returns = f"{record['return_mean']:.2f} ± {record['return_std']:.2f}"
        lengths = f"{record['length_mean']:.1f} ± {record['length_std']:.1f}"
        print(
            f"  {sigma:>7.3f} {returns:>22} {share:>9} {lengths:>16} "
            f"{str(record['completed']) + '/' + str(args.episodes):>11} "
            f"{str(record['terminated']) + '/' + str(record['truncated']):>12}",
            flush=True,
        )
        records.append(record)

    if records:
        if args.channel in ("obs", "both"):
            print(f"  obs noise: {records[0]['noise_description']}")
        if args.channel in ("action", "both"):
            print("  action noise: every dim, unit: action-space half-range")
    return records


def main() -> None:
    args = parse_args()

    versions = installed_versions()
    stack_matches = print_stack_report(versions)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    records: list[dict] = []
    failed: list[tuple[str, str]] = []
    for env_id in args.env_id:
        try:
            records.extend(sweep(args, env_id, seeds))
        except Exception as exc:  # one missing bundle should not lose the sweep
            if len(args.env_id) == 1:
                raise
            print(f"\n{env_id} — failed: {type(exc).__name__}: {exc}", flush=True)
            failed.append((env_id, f"{type(exc).__name__}: {exc}"))

    if failed:
        print(f"\n{len(failed)} environment(s) failed: {', '.join(e for e, _ in failed)}")
    if not stack_matches:
        print("\nMeasured on a different runtime than the protocol — see above.")

    if args.json is not None:
        payload = {
            "versions": versions,
            "channel": args.channel,
            "dims": None if args.dims is None else [list(d) for d in args.dims],
            "sigmas": args.sigmas,
            "noise_seed": args.noise_seed,
            "episodes": args.episodes,
            "runs": records,
            "failed": [{"env_id": e, "error": m} for e, m in failed],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
