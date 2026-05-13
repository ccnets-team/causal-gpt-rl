"""Run an exported Causal-GPT-RL bundle in a MuJoCo Gymnasium env.

Minimal deployment example using the public inference surface
(`causal_gpt_rl.inference.load_runner`). It only needs an exported bundle,
the Gymnasium environment, and the inference API.

Example:
    python -m examples.deploy.mujoco \
        --env-id Hopper-v5 \
        --bundle examples/models/hopper-v5/export-bundle \
        --episodes 5 --render human
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import PolicyRunner, load_runner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Local exported bundle directory containing model.safetensors and config.json.",
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--use-windowed",
        action="store_true",
        help="Use windowed prediction instead of cached KV.",
    )
    p.add_argument(
        "--kv-cache-max-len",
        type=int,
        default=None,
        help="Optional KV cache cap. Defaults to 4x the bundle context_length.",
    )
    return p.parse_args()


def build_env(env_id: str, render: str, seed: int) -> gym.Env:
    render_mode = None if render == "none" else render
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def run_episode(env: gym.Env, runner: PolicyRunner, max_steps: int) -> tuple[float, int]:
    obs, _ = env.reset()
    runner.reset(obs)
    total_reward = 0.0
    for step in range(max_steps):
        action = runner.act()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            return total_reward, step + 1
        runner.observe(obs)
    return total_reward, max_steps


def main() -> None:
    args = parse_args()
    if not args.bundle.is_dir():
        raise FileNotFoundError(args.bundle)

    device = torch.device(args.device)
    env = build_env(args.env_id, args.render, args.seed)
    try:
        runner = load_runner(
            args.bundle,
            device=device,
            num_envs=1,
            kv_cache_max_len=args.kv_cache_max_len,
            use_windowed=args.use_windowed,
        )
        print(runner)

        rewards: list[float] = []
        for ep in range(args.episodes):
            total, steps = run_episode(env, runner, args.max_steps)
            rewards.append(total)
            print(f"[ep {ep + 1:02d}] return={total:.2f}  steps={steps}")

        arr = np.asarray(rewards, dtype=np.float64)
        print(
            f"\nmean={arr.mean():.2f}  std={arr.std():.2f}  "
            f"min={arr.min():.2f}  max={arr.max():.2f}  n={len(arr)}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
