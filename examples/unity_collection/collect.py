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
    ap.add_argument(
        "--complete-matches",
        action="store_true",
        help=(
            "After reaching --target, finish each field's in-flight match and "
            "discard later matches instead of truncating at the transition cutoff."
        ),
    )


def _assign_field_ids(agent_context):
    """Return copied agent contexts annotated with stable per-env field IDs.

    Soccer exposes one behavior/group per team, so equal-sized sorted group
    lists are paired across the two teams. Cooperative/single-team scenes use
    one field per group. The returned sizes/members let the collector determine
    when every agent trajectory belonging to a match has finished.
    """
    contexts = [dict(item) for item in agent_context]
    field_by_group = {}
    contexts_by_env = {}
    for context in contexts:
        contexts_by_env.setdefault(context["env_index"], []).append(context)

    for env_index, env_contexts in contexts_by_env.items():
        teams = sorted({context["team_id"] for context in env_contexts})
        groups_by_team = {
            team: sorted(
                {
                    context["group_id"]
                    for context in env_contexts
                    if context["team_id"] == team
                }
            )
            for team in teams
        }
        if (
            len(teams) == 2
            and len(groups_by_team[teams[0]]) == len(groups_by_team[teams[1]])
        ):
            paired_groups = zip(
                groups_by_team[teams[0]], groups_by_team[teams[1]]
            )
            for field_id, group_pair in enumerate(paired_groups):
                for group_id in group_pair:
                    field_by_group[(env_index, group_id)] = field_id
        else:
            all_groups = sorted({context["group_id"] for context in env_contexts})
            for field_id, group_id in enumerate(all_groups):
                field_by_group[(env_index, group_id)] = field_id

    field_sizes = {}
    field_members = {}
    for global_index, context in enumerate(contexts):
        context["field_id"] = field_by_group[
            (context["env_index"], context["group_id"])
        ]
        key = (context["env_index"], context["field_id"])
        field_sizes[key] = field_sizes.get(key, 0) + 1
        field_members.setdefault(key, []).append(global_index)
    return contexts, field_sizes, field_members
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue from complete episodes already present in --out after an interrupted run.",
    )
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument(
        "--worker-id",
        type=int,
        default=100,
        help="ML-Agents worker id; choose another value when its TCP port is in use.",
    )
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
    ap.add_argument("--policy-id", default="stock", help="Identity of the recorded policy for episode metadata.")
    ap.add_argument("--opponent-policy-id", default="stock", help="Identity of the opponent policy for episode metadata.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_episode_files = sorted(out_dir.glob("ep_*.npz"))
    if existing_episode_files and not args.resume:
        raise FileExistsError(
            f"{out_dir} already contains {len(existing_episode_files)} episodes; "
            "pass --resume or choose a new --out directory"
        )

    UnityEnv.register(args.env_id, None, args.build)
    if args.num_envs > 1:
        # Each instance launches the exe (headless) with its own agents; the
        # wrapper aggregates them into one global agent index range.
        env = UnityEnv.make_vec(
            args.env_id, args.num_envs, time_scale=args.time_scale, worker_id=args.worker_id
        )
    else:
        env = UnityEnv.make(
            args.env_id,
            time_scale=args.time_scale,
            use_graphics=args.graphics,
            worker_id=args.worker_id,
        )
    n = env.num_agents
    print(f"[env] instances={args.num_envs}")
    n_ch = len(env.observation_shapes)
    obs_dims = [int(np.prod(s)) for s in env.observation_shapes]
    action_spec = env.spec.action_spec
    print(
        f"[env] agents={n} obs_channels={obs_dims} -> obs_dim={sum(obs_dims)} "
        f"cont={action_spec.continuous_size} disc_branches={tuple(action_spec.discrete_branches)}"
    )

    # Build a stable field mapping from the two Soccer team behaviors. Each
    # ML-Agents SimpleMultiAgentGroup supplies a group_id; Soccer creates one
    # two-player group per team per field. Pair the sorted team groups to recover
    # the shared field without relying on raw agent IDs.
    agent_context, field_sizes, field_members = _assign_field_ids(env.agent_context)
    print(f"[env] fields={len(field_sizes)} agents_per_field={sorted(set(field_sizes.values()))}")

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
    field_episode_counts = {key: 0 for key in field_sizes}
    manifest_path = out_dir / "episode_metadata.jsonl"
    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"cannot resume without {manifest_path}")
        previous_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(previous_rows) != len(existing_episode_files):
            raise RuntimeError(
                "resume refused: episode file/manifest count mismatch "
                f"({len(existing_episode_files)} files vs {len(previous_rows)} rows)"
            )
        if previous_rows:
            ep_count = max(int(row["episode_index"]) for row in previous_rows) + 1
            total = sum(int(row["transition_count"]) for row in previous_rows)
            for row in previous_rows:
                field_key = (int(row["env_index"]), int(row["field_id"]))
                field_episode_counts[field_key] += 1
        print(f"[resume] transitions={total} episodes={ep_count}")
    else:
        manifest_path.write_text("", encoding="utf-8")

    def flush(g, terminated, truncated):
        nonlocal ep_count
        T = len(buf_act[g])
        if T == 0:
            return False
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
        context = agent_context[g]
        field_key = (context["env_index"], context["field_id"])
        match_index = field_episode_counts[field_key] // field_sizes[field_key]
        episode_return = float(rew_arr.sum())
        result = "win" if episode_return > 0 else "loss" if episode_return < 0 else "draw"
        metadata = {
            "episode_file": f"ep_{ep_count:06d}.npz",
            "episode_index": ep_count,
            "transition_count": T,
            "return": episode_return,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "match_id": (
                f"env{context['env_index']}-field{context['field_id']}-match{match_index}"
            ),
            "field_id": context["field_id"],
            "env_index": context["env_index"],
            "behavior_name": context["behavior_name"],
            "team_id": context["team_id"],
            "group_id": context["group_id"],
            "agent_id": context["agent_id"],
            "policy_id": args.policy_id,
            "opponent_policy_id": args.opponent_policy_id,
            "match_result": result,
        }
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(metadata, sort_keys=True) + "\n")
        if args.probe:
            probe_rows.append(
                (ep_count, T, float(rew_arr.sum()), float(rew_arr[-1]), bool(terminated), bool(truncated))
            )
        ep_count += 1
        field_episode_counts[field_key] += 1
        return True

    observations, _ = env.reset()
    for g in range(n):
        if observations[0][g] is not None:
            seed(observations, g)

    next_report = 20_000
    t0 = time.time()  # after reset: excludes build launch/handshake overhead
    pending_fields = None
    retired_fields = set()
    while total < args.target or (pending_fields is not None and pending_fields):
        actions = policy.act(observations)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        final = info["final_observation"]

        flushed_fields = set()

        for g in range(n):
            context = agent_context[g]
            field_key = (context["env_index"], context["field_id"])
            if field_key in retired_fields:
                reset_agent(g)
                continue
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
                if flush(g, bool(terminated[g]), bool(truncated[g])):
                    flushed_fields.add(field_key)
                if new_seed is not None:
                    seed(new_seed, g)
                else:
                    reset_agent(g)
            else:
                buf_obs[g].append(_concat_obs(next_obs, g, n_ch))

        ended_fields = {
            field_key
            for field_key in flushed_fields
            if field_episode_counts[field_key] % field_sizes[field_key] == 0
        }

        if args.complete_matches and total >= args.target:
            if pending_fields is None:
                # A field that ended on the cutoff step has only a fresh seed
                # and no in-flight transition, so it need not play another match.
                pending_fields = {
                    (agent_context[g]["env_index"], agent_context[g]["field_id"])
                    for g in range(n)
                    if active[g] and len(buf_act[g]) > 0
                }
                retired_fields = set(field_sizes) - pending_fields
                print(
                    f"[collect] target reached at {total}; finishing "
                    f"{len(pending_fields)} in-flight fields"
                )
            completed_now = pending_fields & ended_fields
            pending_fields -= completed_now
            retired_fields |= completed_now
            for g in range(n):
                context = agent_context[g]
                if (context["env_index"], context["field_id"]) in retired_fields:
                    reset_agent(g)

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
    if not args.probe and not args.complete_matches:
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
