"""Collect transitions from a model-removed ML-Agents Unity build, driven by the
env's baked ONNX policy run in onnxruntime, and write one `.npz` per finished
episode.

Runs in the collection env (Python 3.10): mlagents_envs 1.1.0 + onnxruntime.
`collection/build_minari.py` (a separate env with minari) turns the raw episodes
into a Minari dataset. The two-env split keeps mlagents_envs's older numpy/gym
pins away from minari's newer gymnasium.

Per-agent trajectories are assembled independently (each Crawler agent is one
single-agent MDP with the same policy), so each finished episode -> one raw file
with the standard Minari layout: observations length T+1, everything else T.

Reward alignment follows the gym contract the wrapper implements: the reward
returned by `step(a_t)` is the consequence of `a_t`, paired with obs_t. Use
`--probe K` to confirm this empirically before the full run.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np

from unity_env import UnityEnv
from onnx_policy import OnnxPolicy
from noisy_policy import NoisyPolicy


def _concat_obs(observations, g, n_ch):
    """Flatten agent g's obs channels (spec order) into one vector."""
    return np.concatenate(
        [np.asarray(observations[i][g], dtype=np.float32).reshape(-1) for i in range(n_ch)]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="Path to the Unity build (exe).")
    ap.add_argument("--onnx", required=True, help="Path to the baked policy .onnx.")
    ap.add_argument("--out", required=True, help="Output dir for raw episode .npz files.")
    ap.add_argument("--env-id", default="crawler")
    ap.add_argument("--target", type=int, default=1_000_000, help="Transitions to collect.")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel Unity build instances (each launches the exe with its own "
        "agents). >1 runs headless; total agents = per-scene * num_envs.",
    )
    ap.add_argument("--graphics", action="store_true", help="Render (default headless, single env only).")
    ap.add_argument(
        "--probe",
        type=int,
        default=0,
        help="If >0, stop after this many finished episodes and print alignment diagnostics.",
    )
    # Noise dials to synthesize a lower-quality tier from the stock policy. Both
    # zero (default) records the expert tier. See noisy_policy.py; pick values
    # for a target return with calibrate_noise.py.
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Gaussian action noise std (continuous). >0 degrades the policy.")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="Per-agent probability of a uniform-random action. 1.0 = random policy.")
    ap.add_argument("--noise-seed", type=int, default=0, help="Seed for the noise RNG (reproducible tiers).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    UnityEnv.register(args.env_id, None, args.build)
    if args.num_envs > 1:
        # Each instance launches the exe (headless) with its own agents; the
        # wrapper aggregates them into one global agent index range.
        env = UnityEnv.make_vec(args.env_id, args.num_envs, time_scale=args.time_scale)
    else:
        env = UnityEnv.make(args.env_id, time_scale=args.time_scale, use_graphics=args.graphics)
    n = env.num_agents
    print(f"[env] instances={args.num_envs}")
    n_ch = len(env.observation_shapes)
    obs_dims = [int(np.prod(s)) for s in env.observation_shapes]
    action_spec = env.spec.action_spec
    print(
        f"[env] agents={n} obs_channels={obs_dims} -> obs_dim={sum(obs_dims)} "
        f"cont={action_spec.continuous_size} disc_branches={tuple(action_spec.discrete_branches)}"
    )

    policy = OnnxPolicy(
        args.onnx, num_agents=n, obs_shapes=env.observation_shapes, action_spec=action_spec
    )
    noised = args.noise_std > 0.0 or args.epsilon > 0.0
    if noised:
        policy = NoisyPolicy(
            policy, noise_std=args.noise_std, epsilon=args.epsilon,
            rng=np.random.default_rng(args.noise_seed),
        )
        print(f"[noise] noise_std={args.noise_std} epsilon={args.epsilon} seed={args.noise_seed}")

    # Record the obs/action spec so build_minari.py can build the right Minari
    # spaces: a per-sensor Tuple observation (`obs_channels`; a single channel
    # stays a bare Box) and the action space (Box for continuous,
    # Discrete/MultiDiscrete for discrete, Tuple(Box, Discrete) for hybrid). Raw
    # obs/actions stay flat here — build_minari splits them by these dims into the
    # leaf arrays Minari stores. `obs_dim` is kept for backward-compatible readers.
    spec_meta = {
        "obs_dim": int(sum(obs_dims)),
        "obs_channels": [int(d) for d in obs_dims],
        "action_kind": policy.kind,
    }
    if noised:
        spec_meta["noise"] = {
            "noise_std": args.noise_std, "epsilon": args.epsilon, "noise_seed": args.noise_seed,
        }
    if policy.kind == "continuous":
        spec_meta["act_dim"] = int(policy.act_dim)
    elif policy.kind == "discrete":
        spec_meta["branches"] = [int(b) for b in policy.branches]
    else:  # hybrid: continuous head(s) + discrete branch(es), read from the spec
        spec_meta["continuous_size"] = int(action_spec.continuous_size)
        spec_meta["branches"] = [int(b) for b in policy.branches]
    (out_dir / "spec.json").write_text(json.dumps(spec_meta, indent=2), encoding="utf-8")
    print(f"[env] action_kind={policy.kind} obs_channels={obs_dims} spec={spec_meta}")

    # Per-agent working buffers. Invariant while active[g]: buf_obs[g] holds
    # obs_0..obs_k and buf_act/buf_rew hold the k actions/rewards taken so far,
    # so len(buf_obs[g]) == len(buf_act[g]) + 1.
    buf_obs = [[] for _ in range(n)]
    buf_act = [[] for _ in range(n)]
    buf_rew = [[] for _ in range(n)]
    active = [False] * n

    def reset_agent(g):
        buf_obs[g], buf_act[g], buf_rew[g] = [], [], []
        active[g] = False

    def seed(observations, g):
        reset_agent(g)
        buf_obs[g].append(_concat_obs(observations, g, n_ch))
        active[g] = True

    ep_count = 0
    total = 0
    probe_rows = []

    def flush(g, terminated, truncated):
        nonlocal ep_count
        T = len(buf_act[g])
        if T == 0:
            return
        obs_arr = np.stack(buf_obs[g], axis=0).astype(np.float32)  # [T+1, obs_dim]
        assert obs_arr.shape[0] == T + 1, (obs_arr.shape, T)
        if policy.kind == "discrete":
            act_arr = np.stack(buf_act[g], axis=0).astype(np.int64)  # [T, num_branches] indices
        else:
            act_arr = np.stack(buf_act[g], axis=0).astype(np.float32)  # [T, act_dim]
        rew_arr = np.asarray(buf_rew[g], dtype=np.float32)  # [T]
        term = np.zeros(T, dtype=bool)
        trunc = np.zeros(T, dtype=bool)
        term[-1] = terminated
        trunc[-1] = truncated
        np.savez(
            out_dir / f"ep_{ep_count:06d}.npz",
            observations=obs_arr,
            actions=act_arr,
            rewards=rew_arr,
            terminations=term,
            truncations=trunc,
        )
        if args.probe:
            probe_rows.append(
                (ep_count, T, float(rew_arr.sum()), float(rew_arr[-1]), bool(terminated), bool(truncated))
            )
        ep_count += 1

    observations, _ = env.reset()
    for g in range(n):
        if observations[0][g] is not None:
            seed(observations, g)

    next_report = 20_000
    t0 = time.time()  # after reset: excludes build launch/handshake overhead
    while total < args.target:
        actions = policy.act(observations)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        final = info["final_observation"]

        for g in range(n):
            if rewards[g] is None:
                continue  # agent absent this step (mid decision-period / desynced)

            if not active[g] or len(buf_obs[g]) == 0:
                # A reward with no paired obs — the gap right after a terminal-only
                # step. Treat the returned obs as the new episode's seed; drop the
                # (reset) reward, which is not a real transition.
                if next_obs[0][g] is not None:
                    seed(next_obs, g)
                else:
                    reset_agent(g)
                continue

            # Normal transition: acted from buf_obs[g][-1].
            buf_act[g].append(np.asarray(actions[g], dtype=np.float32).reshape(-1))
            buf_rew[g].append(float(rewards[g]))
            total += 1

            if bool(terminated[g]) or bool(truncated[g]):
                if final[0][g] is not None:
                    # Common case: terminal obs in final_observation, new episode's
                    # first obs in the returned obs.
                    buf_obs[g].append(_concat_obs(final, g, n_ch))
                    new_seed = next_obs if next_obs[0][g] is not None else None
                else:
                    # Terminal-only step: wrapper places the terminal obs in the
                    # returned obs, and there is no new-episode obs yet.
                    buf_obs[g].append(_concat_obs(next_obs, g, n_ch))
                    new_seed = None
                flush(g, bool(terminated[g]), bool(truncated[g]))
                if new_seed is not None:
                    seed(new_seed, g)
                else:
                    reset_agent(g)
            else:
                buf_obs[g].append(_concat_obs(next_obs, g, n_ch))

        observations = next_obs

        if args.probe and ep_count >= args.probe:
            break
        if total >= next_report:
            print(f"[collect] transitions={total} episodes={ep_count}")
            next_report += 20_000

    # Flush in-flight episodes as truncated so transitions already counted in
    # `total` (but not yet at a terminal/truncation boundary) are saved rather
    # than discarded at the target cutoff. Each active buffer holds the standard
    # invariant len(buf_obs) == len(buf_act) + 1, so flush() writes it directly.
    if not args.probe:
        for g in range(n):
            if active[g] and len(buf_act[g]) > 0:
                flush(g, terminated=False, truncated=True)

    elapsed = time.time() - t0
    env.close()
    rate = total / elapsed if elapsed > 0 else 0.0
    print(f"[done] transitions={total} episodes={ep_count} -> {out_dir}")
    print(
        f"[timing] {elapsed:.1f}s steady-state (launch excluded), "
        f"{rate:.0f} transitions/s @ time_scale={args.time_scale}"
    )
    if rate > 0:
        print(f"[estimate] 1,000,000 transitions ~ {1_000_000 / rate / 60:.1f} min at this rate")

    if args.probe:
        print("\n[probe] ep    len    sum_r     last_r   term   trunc")
        for r in probe_rows:
            print(
                f"        {r[0]:<5d} {r[1]:<6d} {r[2]:>8.2f} {r[3]:>9.3f}   {str(r[4]):<5s}  {r[5]}"
            )
        print(
            "\nReward-alignment check: terminated episodes should carry the fall\n"
            "penalty on last_r, and episode lengths should VARY. If every episode\n"
            "has identical length and trunc=True, terminations aren't firing."
        )


if __name__ == "__main__":
    main()
