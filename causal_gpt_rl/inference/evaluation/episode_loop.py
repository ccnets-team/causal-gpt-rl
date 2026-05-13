"""Public single-env episode rollout — per-episode and aggregate stats.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..runner import PolicyRunner


def run_episodes(
    env,
    runner: PolicyRunner,
    *,
    num_episodes: int,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> dict:
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    if runner.num_envs != 1:
        raise ValueError(
            f"run_episodes supports single-env runners only; got num_envs={runner.num_envs}"
        )

    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(num_episodes):
        # Seed once on the first reset; subsequent resets advance env RNG naturally.
        reset_kwargs = {"seed": int(seed)} if (ep == 0 and seed is not None) else {}
        reset_out = env.reset(**reset_kwargs)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        runner.reset(obs)

        ep_return = 0.0
        ep_length = 0
        done = False
        while not done:
            action = runner.act(obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, term, trunc, _ = step_out
                done = bool(term) or bool(trunc)
            else:
                obs, reward, done_flag, _ = step_out
                done = bool(done_flag)
            ep_return += float(reward)
            ep_length += 1
            if max_steps is not None and ep_length >= max_steps:
                break

        returns.append(ep_return)
        lengths.append(ep_length)

    returns_arr = np.asarray(returns, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)

    return {
        "num_episodes": int(num_episodes),
        "returns": returns,
        "lengths": lengths,
        "return_mean": float(returns_arr.mean()),
        "return_std": float(returns_arr.std()),
        "length_mean": float(lengths_arr.mean()),
        "length_std": float(lengths_arr.std()),
    }
