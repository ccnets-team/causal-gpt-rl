"""Run a Causal GPT-RL Unity policy (ONNX) in a Unity ML-Agents build.

The Unity deploy + measurement example: it drives the `crawler.onnx` policy from
[ccnets/causal-gpt-rl-unity](https://huggingface.co/ccnets/causal-gpt-rl-unity)
in a Unity Crawler build and reports closed-loop return per agent. ONNX-only — it
needs `onnxruntime` and the ML-Agents stepping wrapper, but NOT PyTorch and NOT
the `causal_gpt_rl` runtime.

Inputs (both public):
  - policy: ccnets/causal-gpt-rl-unity           (`crawler.onnx`)
  - build:  ccnets/causal-gpt-rl-unity-envs       (model-removed Crawler build)

The policy is a windowed graph: it consumes a 32-step rolling window of raw
observations + past actions and returns the next action. This script reproduces,
in plain numpy, the context buffer the window needs:

    reset -> push (s0, zero-action, is_bos=1, mask=1)
    step  -> push (s_next, previous raw action, is_bos=0, mask=1); the window rolls

The environment action is `clip(raw, -1, 1)`; the RAW action is fed back into the
window (autoregressive feedback). Observations go in raw — state normalization is
inside the graph.

First-episode-per-agent: each scene agent is measured for one episode, then frozen.

Run (in an env with onnxruntime + mlagents_envs; no torch needed):
    python examples/deploy/mlagents.py \
        --build path/to/Crawler.exe \
        --onnx  path/to/crawler.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# The ML-Agents -> gymnasium stepping wrapper lives with the collection recipe.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unity_collection"))

CTX = 32


class Window:
    """Plain-numpy replica of the policy's context buffer (32-step window)."""

    def __init__(self, n, ctx, ss, acts):
        self.n, self.L, self.ss, self.acts = n, ctx + 1, ss, acts
        self.reset()

    def reset(self):
        self.states = np.zeros((self.n, self.L, self.ss), np.float32)
        self.actions = np.zeros((self.n, self.L, self.acts), np.float32)
        self.is_bos = np.ones((self.n, self.L, 1), np.float32)
        self.masks = np.zeros((self.n, self.L), np.float32)

    def update(self, next_states, actions, bos):
        bos = np.broadcast_to(np.asarray(bos, np.float32), (self.n,))
        bos_rows = bos != 0.0
        self.states = np.roll(self.states, -1, axis=1)
        self.states[:, -1] = next_states
        if bos_rows.any():
            self.states[bos_rows, -2] = np.asarray(next_states)[bos_rows]
        self.actions = np.roll(self.actions, -1, axis=1)
        self.actions[:, -2] = actions
        self.is_bos = np.roll(self.is_bos, -1, axis=1)
        self.is_bos[:, -2, 0] = bos
        self.masks = np.roll(self.masks, -1, axis=1)
        self.masks[:, -2] = 1.0

    def context(self):
        return (self.states[:, :-1], self.actions[:, :-1],
                self.is_bos[:, :-1], self.masks[:, :-1])


def _pack(observations, g, n_ch):
    """Flatten agent g's obs channels (spec order) into one flat vector."""
    return np.concatenate(
        [np.asarray(observations[i][g], np.float32).reshape(-1) for i in range(n_ch)]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--build", required=True, type=Path, help="Path to the Unity Crawler build (.exe).")
    p.add_argument("--onnx", required=True, type=Path, help="Path to the policy crawler.onnx.")
    p.add_argument("--env-id", default="crawler")
    p.add_argument("--time-scale", type=float, default=20.0)
    p.add_argument("--graphics", action="store_true", help="Render the Unity window (default headless).")
    p.add_argument("--max-ticks", type=int, default=20000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from unity_env import UnityEnv

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    acts = sess.get_inputs()[1].shape[-1]  # actions channel width

    UnityEnv.register(args.env_id, None, str(args.build))
    env = UnityEnv.make(args.env_id, time_scale=args.time_scale, use_graphics=args.graphics)
    try:
        n = env.num_agents
        n_ch = len(env.observation_shapes)
        ss = int(sum(int(np.prod(s)) for s in env.observation_shapes))
        print(f"[env] agents={n} obs_dim={ss} act={acts}", flush=True)

        w = Window(n, CTX, ss, acts)
        obs, _ = env.reset()
        last_state = np.stack([_pack(obs, g, n_ch) for g in range(n)], axis=0)
        w.reset()
        w.update(last_state, np.zeros((n, acts), np.float32), bos=1.0)

        ret = np.zeros(n, np.float64)
        recorded = [None] * n
        last_env_act = np.zeros((n, acts), np.float32)
        last_buffer_act = np.zeros((n, acts), np.float32)
        ticks = 0

        while any(r is None for r in recorded) and ticks < args.max_ticks:
            ticks += 1
            present = any(obs[0][g] is not None for g in range(n))
            if present:                                  # decision tick -> run ONNX
                st, ac, ib, mk = w.context()
                raw = np.zeros((n, acts), np.float32)
                for g in range(n):
                    raw[g] = sess.run(
                        ["action"],
                        {"states": st[g:g + 1], "actions": ac[g:g + 1],
                         "is_bos": ib[g:g + 1], "mask": mk[g:g + 1]},
                    )[0][0]
                last_buffer_act = raw
                last_env_act = np.clip(raw, -1.0, 1.0).astype(np.float32)

            next_obs, rewards, terminated, truncated, _ = env.step(last_env_act)

            for g in range(n):
                if recorded[g] is not None:
                    continue
                if rewards[g] is not None:
                    ret[g] += float(rewards[g])
                if terminated[g] is True or truncated[g] is True:
                    recorded[g] = (float(ret[g]), "term" if terminated[g] else "trunc")

            next_present = any(next_obs[0][g] is not None for g in range(n))
            if next_present:                             # observe: advance the window
                for g in range(n):
                    if next_obs[0][g] is not None:
                        last_state[g] = _pack(next_obs, g, n_ch)
                w.update(last_state, last_buffer_act, bos=0.0)
            obs = next_obs

        done = [r for r in recorded if r is not None]
        if not done:
            raise SystemExit(f"No episode finished within {args.max_ticks} ticks.")
        vals = np.asarray([r[0] for r in done], np.float64)
        n_term = sum(1 for r in done if r[1] == "term")
        print("\n[per-agent first-episode return]")
        for g, r in enumerate(recorded):
            print(f"  agent {g:2d}: {'unfinished' if r is None else f'{r[0]:8.2f} ({r[1]})'}")
        print(
            f"\n[return] n={len(vals)} mean={vals.mean():.2f} std={vals.std():.2f} "
            f"min={vals.min():.2f} max={vals.max():.2f} "
            f"term={n_term}/{len(done)} ticks={ticks}"
        )
    finally:
        env.close()
        print("closed", flush=True)


if __name__ == "__main__":
    main()
