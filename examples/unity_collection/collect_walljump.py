"""Collect WallJump with its two external policies and episode-boundary switching.

One transition runs from a decision observation to the next observation returned
for that agent. ML-Agents accumulates rewards while repeating the submitted
action between decisions. Terminal observations close the old behavior before
new decisions open an episode under the next behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from onnx_policy import OnnxPolicy

BEHAVIORS = ("SmallWallJump", "BigWallJump")
REVISION = "abbffcd63c17c30ca834846ec4b913d6ca4433a2"


def band(value):
    values = tuple(float(x) for x in value.split(","))
    if len(values) != 2 or not 0 < values[0] <= values[1]:
        raise argparse.ArgumentTypeError("expected positive lo,hi")
    return values


def collect(args):
    if args.target < 1 or args.time_scale <= 0:
        raise ValueError("target and time_scale must be positive")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("ep_*.npz")):
        raise FileExistsError(f"Existing episodes in {out}; use a new directory")
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=args.time_scale)
    env = UnityEnvironment(file_name=str(args.build.resolve()), no_graphics=True,
                           worker_id=args.worker_id, seed=args.seed,
                           side_channels=[channel], timeout_wait=90,
                           log_folder=str(out / "player-logs"))
    rng = np.random.default_rng(args.seed)
    total = saved = episodes = tick = 0
    active, policies, paths, bands = {}, {}, {}, {}
    started = time.monotonic()
    finishing = False
    masked = 0
    manifest = None
    try:
        env.reset()
        names = {n.split("?team=")[0]: n for n in env.behavior_specs}
        if set(names) != set(BEHAVIORS):
            raise ValueError(f"Expected WallJump behaviors, got {names}")
        specs = {b: env.behavior_specs[names[b]] for b in BEHAVIORS}
        shapes = [tuple(x.shape) for x in specs[BEHAVIORS[0]].observation_specs]
        branches = [int(x) for x in specs[BEHAVIORS[0]].action_spec.discrete_branches]
        for index, behavior in enumerate(BEHAVIORS):
            spec = specs[behavior]
            if ([tuple(x.shape) for x in spec.observation_specs] != shapes
                    or list(spec.action_spec.discrete_branches) != branches
                    or spec.action_spec.continuous_size):
                raise ValueError("WallJump behavior schemas differ")
            paths[behavior] = args.small_onnx if index == 0 else args.big_onnx
            bands[behavior] = args.small_temperature_range if index == 0 else args.big_temperature_range
            if not args.random:
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                policies[behavior] = OnnxPolicy(str(paths[behavior]), 1, shapes,
                    spec.action_spec, rng=np.random.default_rng(args.seed + index + 17),
                    session_options=options)
                if bands[behavior] is not None and not policies[behavior].has_logits:
                    raise ValueError(f"{behavior} needs a logits-exposing policy")
        spec = {"environment": "WallJump", "source_revision": args.source_revision,
                "obs_dim": sum(int(np.prod(s)) for s in shapes),
                "obs_channels": [int(np.prod(s)) for s in shapes],
                "obs_shapes": [list(s) for s in shapes], "action_kind": "discrete",
                "branches": branches, "dataset_quality": args.quality,
                "policy_sha256": {b: hashlib.sha256(paths[b].read_bytes()).hexdigest()
                                  for b in BEHAVIORS} if not args.random else {},
                "noise": {"mode": "per_agent_per_episode", "dial": "temperature",
                          "ranges": bands, "seed": args.seed},
                "random_policy": args.random,
                "transition_clock": "agent decision to next decision or terminal; rewards accumulated by ML-Agents",
                "seed": args.seed, "time_scale": args.time_scale}
        (out / "spec.json").write_bytes((json.dumps(spec, indent=2) + "\n").encode())
        manifest = (out / "episode_metadata.jsonl").open("w", encoding="utf-8", newline="\n")
        stats = {b: [] for b in BEHAVIORS}

        def append(key, steps, row):
            nonlocal total
            buf = active[key]
            if buf["action"] is None:
                raise RuntimeError(f"No submitted action for {key}")
            obs = np.concatenate([x[row].reshape(-1) for x in steps.obs]).astype(np.float32)
            reward = float(steps.reward[row] + steps.group_reward[row])
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise ValueError("Non-finite observation/reward")
            buf["observations"].append(obs)
            buf["actions"].append(buf["action"].copy())
            buf["rewards"].append(reward)
            buf["ticks"].append(tick - buf["last_tick"])
            buf["last_tick"] = tick
            total += 1

        def finish(key, interrupted):
            nonlocal saved, episodes
            buf = active.pop(key)
            length = len(buf["rewards"])
            if not length:
                raise RuntimeError("Empty terminal episode")
            terminal = np.zeros(length, bool)
            truncated = np.zeros(length, bool)
            (truncated if interrupted else terminal)[-1] = True
            filename = f"ep_{episodes:06d}.npz"
            arrays = {"observations": np.stack(buf["observations"]),
                      "actions": np.stack(buf["actions"]).astype(np.int64),
                      "rewards": np.asarray(buf["rewards"], np.float32),
                      "terminations": terminal, "truncations": truncated,
                      "unity_tick_deltas": np.asarray(buf["ticks"], np.int32)}
            np.savez_compressed(out / filename, **arrays)
            result = {"episode_file": filename, "episode_index": episodes,
                      "behavior_name": key[0], "agent_id": key[1],
                      "transition_count": length, "return": float(arrays["rewards"].sum()),
                      "terminated": not interrupted, "truncated": interrupted,
                      "temperature": buf["temperature"], "dataset_quality": args.quality,
                      "policy_id": "random" if args.random else key[0] + ".onnx",
                      "unity_tick_deltas": buf["ticks"]}
            # Success is determined by the terminal reward event, not return sign.
            result["outcome"] = ("timeout" if interrupted else "goal" if
                                  arrays["rewards"][-1] > 0 else "fall")
            manifest.write(json.dumps(result, separators=(",", ":")) + "\n")
            manifest.flush()
            stats[key[0]].append(result)
            saved += length
            episodes += 1

        next_log = 20000
        while True:
            steps = {b: env.get_steps(names[b]) for b in BEHAVIORS}
            # A single GameObject may change its behavior after reset.
            for behavior in BEHAVIORS:
                terminal = steps[behavior][1]
                for row, aid in enumerate(terminal.agent_id):
                    key = (behavior, int(aid))
                    if key in active:
                        append(key, terminal, row)
                        finish(key, bool(terminal.interrupted[row]))
            if total >= args.target:
                finishing = True
            if finishing and not active:
                break
            for behavior in BEHAVIORS:
                decision = steps[behavior][0]
                count = len(decision)
                if not count:
                    continue
                temperatures = np.ones(count)
                for row, aid in enumerate(decision.agent_id):
                    key = (behavior, int(aid))
                    if key in active:
                        append(key, decision, row)
                    elif not finishing:
                        if any(other[1] == int(aid) for other in active):
                            raise RuntimeError(f"Behavior changed without terminal: {aid}")
                        temperature = float(rng.uniform(*bands[behavior])) if bands[behavior] else 1.0
                        active[key] = {"observations": [np.concatenate([x[row].reshape(-1)
                                       for x in decision.obs]).astype(np.float32)],
                                       "actions": [], "rewards": [], "ticks": [],
                                       "action": None, "last_tick": tick,
                                       "temperature": temperature}
                    if key in active:
                        temperatures[row] = active[key]["temperature"]
                if decision.action_mask is not None:
                    masked += sum(int(m.sum()) for m in decision.action_mask)
                    if any(m.any() for m in decision.action_mask):
                        raise ValueError("WallJump masks changed; driver must apply live masks")
                if args.random:
                    actions = np.column_stack([rng.integers(width, size=count) for width in branches])
                else:
                    policy = policies[behavior]
                    policy.num_agents = count
                    policy.set_temperature_by_agent(temperatures if bands[behavior] else None)
                    actions = policy.act(decision.obs).astype(np.int32)
                for row, aid in enumerate(decision.agent_id):
                    key = (behavior, int(aid))
                    if key in active:
                        active[key]["action"] = actions[row].copy()
                env.set_actions(names[behavior], ActionTuple(discrete=actions.astype(np.int32)))
            env.step()
            tick += 1
            if total >= next_log:
                print(f"transitions={total} saved={saved} episodes={episodes} ticks={tick} finishing={finishing}", flush=True)
                next_log += 20000
        manifest.close()
        summary = {"transitions": saved, "episodes": episodes, "ticks": tick,
                   "seconds": time.monotonic() - started, "masked_actions": masked,
                   "behaviors": {}}
        for behavior, rows in stats.items():
            if not rows:
                raise RuntimeError(f"No episodes for {behavior}")
            summary["behaviors"][behavior] = {
                "episodes": len(rows), "transitions": sum(r["transition_count"] for r in rows),
                "mean_return": float(np.mean([r["return"] for r in rows])),
                "mean_episode_step_reward": float(np.mean([r["return"] / r["transition_count"] for r in rows])),
                "goal_rate": float(np.mean([r["outcome"] == "goal" for r in rows])),
                "mean_length": float(np.mean([r["transition_count"] for r in rows]))}
        (out / "summary.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode())
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        if manifest is not None:
            manifest.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--small-onnx", type=Path, required=True)
    parser.add_argument("--big-onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000000)
    parser.add_argument("--quality", choices=("expert", "medium", "simple", "random"), required=True)
    parser.add_argument("--small-temperature-range", type=band)
    parser.add_argument("--big-temperature-range", type=band)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=2310000)
    parser.add_argument("--worker-id", type=int, default=811)
    parser.add_argument("--time-scale", type=float, default=20)
    parser.add_argument("--source-revision", default=REVISION)
    collect(parser.parse_args())
