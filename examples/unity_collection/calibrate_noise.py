"""Find the noise levels that land the `simple` / `medium` tiers.

A tier is defined by its normalized score, exactly as the public table computes it:

    norm = 100 * (return - random_ref) / (expert_ref - random_ref)

so before picking noise you need the two endpoints and a return-vs-noise curve.
This measures all three against the live build in ONE launch (the launch is the
expensive part, so switching noise between blocks and re-measuring beats
relaunching per level):

  - expert_ref : stock policy, no noise                 (noise_std=0, epsilon=0)
  - random_ref : uniform-random policy                  (epsilon=1)
  - the curve  : mean return at each noise_std in --grid

then reports the normalized score at each level and picks the grid value whose
score is closest to --target-simple / --target-medium. Feed those two noise_std
values back into collect.py (`--noise-std ...`) to record the datasets.

Return accounting mirrors collect.py's reward/obs pairing (the reset-reward gap
after a terminal-only step is dropped), so the numbers are comparable to what a
collected episode would sum to. It measures closed-loop return only — no
transition buffers — so it stays independent of the collection buffer logic.

Run in the collection env (onnxruntime + mlagents_envs), same as collect.py:
    python calibrate_noise.py --build path/to/Crawler.exe --onnx path/to/Crawler.onnx
"""
import argparse
import numpy as np

from unity_env import UnityEnv
from onnx_policy import OnnxPolicy
from noisy_policy import NoisyPolicy


def measure(env, policy, n, episodes, max_ticks):
    """Roll until `episodes` episodes finish; return their per-episode returns.

    Faithful to collect.py's active/seed state machine: an episode starts only
    from a real observation, the terminal reward is counted, and the reset-reward
    that follows a terminal-only step (a reward with no active episode) is dropped.
    """
    running = np.zeros(n)
    active = [False] * n
    returns = []

    obs, _ = env.reset()
    for g in range(n):
        if obs[0][g] is not None:
            active[g] = True
            running[g] = 0.0

    ticks = 0
    while len(returns) < episodes and ticks < max_ticks:
        ticks += 1
        actions = policy.act(obs)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        final = info["final_observation"]

        for g in range(n):
            if rewards[g] is None:
                continue
            if not active[g]:
                # Reset-reward gap after a terminal-only step: seed a new episode
                # from the returned obs if present, and drop this (reset) reward.
                if next_obs[0][g] is not None:
                    active[g] = True
                    running[g] = 0.0
                continue

            running[g] += float(rewards[g])
            if bool(terminated[g]) or bool(truncated[g]):
                returns.append(running[g])
                # Common case: new episode seed arrives in next_obs. Terminal-only
                # step (no final obs, no seed yet): go idle and let the next
                # reward gap seed the following episode.
                if final[0][g] is not None and next_obs[0][g] is not None:
                    active[g] = True
                    running[g] = 0.0
                else:
                    active[g] = False
        obs = next_obs

    return returns


def _block(env, base, n, episodes, max_ticks, noise_std, epsilon, seed):
    policy = NoisyPolicy(base, noise_std=noise_std, epsilon=epsilon,
                         rng=np.random.default_rng(seed))
    r = np.asarray(measure(env, policy, n, episodes, max_ticks), dtype=np.float64)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--env-id", default="crawler")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--episodes", type=int, default=30, help="Episodes measured per noise level.")
    ap.add_argument("--max-ticks", type=int, default=200_000, help="Per-block safety cap on env steps.")
    ap.add_argument("--grid", default="0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8",
                    help="Comma-separated noise_std values to sweep.")
    ap.add_argument("--target-simple", type=float, default=40.0, help="Normalized score for the simple tier.")
    ap.add_argument("--target-medium", type=float, default=70.0, help="Normalized score for the medium tier.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed for the noise RNG.")
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",") if x.strip()]

    UnityEnv.register(args.env_id, None, args.build)
    env = UnityEnv.make(args.env_id, time_scale=args.time_scale, use_graphics=False)
    try:
        n = env.num_agents
        action_spec = env.spec.action_spec
        base = OnnxPolicy(args.onnx, num_agents=n, obs_shapes=env.observation_shapes,
                          action_spec=action_spec)
        if base.kind == "discrete":
            print("[warn] discrete behavior: noise_std has no effect; sweep epsilon instead.")
        elif base.kind == "hybrid":
            print("[note] hybrid behavior: noise_std perturbs the continuous half; "
                  "epsilon randomizes the whole action.")
        print(f"[env] agents={n} kind={base.kind} episodes/level={args.episodes}\n")

        def stats(r):
            return (float(r.mean()), float(r.std()), int(r.size)) if r.size else (float("nan"), float("nan"), 0)

        # Endpoints: expert (no noise) and random (epsilon=1).
        e_mean, e_std, e_n = stats(_block(env, base, n, args.episodes, args.max_ticks, 0.0, 0.0, args.seed))
        r_mean, r_std, r_n = stats(_block(env, base, n, args.episodes, args.max_ticks, 0.0, 1.0, args.seed + 1))
        scale = e_mean - r_mean
        print(f"[expert_ref] mean={e_mean:8.2f} std={e_std:7.2f} n={e_n}")
        print(f"[random_ref] mean={r_mean:8.2f} std={r_std:7.2f} n={r_n}")
        print(f"[scale] expert-random = {scale:.2f}\n")

        def norm(m):
            return 100.0 * (m - r_mean) / scale if scale != 0 else float("nan")

        print(f"{'noise_std':>10} {'mean':>9} {'std':>8} {'n':>4} {'norm':>8}")
        print(f"{0.0:>10.3f} {e_mean:>9.2f} {e_std:>8.2f} {e_n:>4} {norm(e_mean):>8.2f}   (expert)")
        rows = []
        for i, s in enumerate(grid):
            m, sd, cnt = stats(_block(env, base, n, args.episodes, args.max_ticks, s, 0.0, args.seed + 2 + i))
            rows.append((s, m, norm(m)))
            print(f"{s:>10.3f} {m:>9.2f} {sd:>8.2f} {cnt:>4} {norm(m):>8.2f}")
        print(f"{'eps=1':>10} {r_mean:>9.2f} {r_std:>8.2f} {r_n:>4} {norm(r_mean):>8.2f}   (random)")

        def pick(target):
            return min(rows, key=lambda row: abs(row[2] - target)) if rows else None

        s_pick, m_pick = pick(args.target_simple), pick(args.target_medium)
        print("\n[pick] closest grid level to each target:")
        if s_pick:
            print(f"  simple (~{args.target_simple:.0f}): --noise-std {s_pick[0]:.3f}  "
                  f"-> norm={s_pick[2]:.2f} return={s_pick[1]:.2f}")
        if m_pick:
            print(f"  medium (~{args.target_medium:.0f}): --noise-std {m_pick[0]:.3f}  "
                  f"-> norm={m_pick[2]:.2f} return={m_pick[1]:.2f}")
        print("\nRefine the grid around a pick if no level is close enough, then "
              "record each tier with collect.py --noise-std <value>.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
