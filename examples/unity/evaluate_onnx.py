"""Measure a Causal GPT-RL ONNX policy in a Unity ML-Agents build.

The policy and model-removed Unity build can be downloaded from Hugging Face and
passed as local paths.  The ONNX graph must use the windowed policy contract:

    states  [B, T, state_size]     actions [B, T, action_size]
    is_bos  [B, T, 1]             mask    [B, T]
    -> action [B, action_size]

Both batch-1 graphs and graphs exported for all scene agents are supported.  A
fixed full-scene batch is substantially faster because it makes one ONNX call
per decision tick instead of one call per agent.

Examples:
    python examples/unity/evaluate_onnx.py \
        --build path/to/UnityEnvironment.exe \
        --onnx path/to/dungeonescape.onnx

    python examples/unity/evaluate_onnx.py \
        --build path/to/Crawler.exe --onnx path/to/crawler.onnx

Requires ``onnxruntime`` and ``mlagents_envs``; PyTorch is not required.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Reuse the public ML-Agents stepping wrapper from the collection example.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unity_collection"))


class Window:
    """Numpy implementation of the windowed policy's autoregressive context."""

    def __init__(
        self,
        num_agents: int,
        context_length: int,
        state_size: int,
        action_size: int,
        bos_cache_mode: str = "discard",
    ):
        if bos_cache_mode not in {"discard", "retain"}:
            raise ValueError("bos_cache_mode must be 'discard' or 'retain'")
        self.num_agents = num_agents
        self.length = context_length + 1
        self.state_size = state_size
        self.action_size = action_size
        self.bos_cache_mode = bos_cache_mode
        self.states = np.zeros((num_agents, self.length, state_size), np.float32)
        self.actions = np.zeros((num_agents, self.length, action_size), np.float32)
        self.is_bos = np.ones((num_agents, self.length, 1), np.float32)
        self.mask = np.zeros((num_agents, self.length), np.float32)

    def update(self, states: np.ndarray, actions: np.ndarray, is_bos) -> None:
        bos = np.broadcast_to(np.asarray(is_bos, np.float32), (self.num_agents,))
        bos_rows = bos != 0.0
        self.states = np.roll(self.states, -1, axis=1)
        self.states[:, -1] = states
        if bos_rows.any():
            self.states[bos_rows, -2] = states[bos_rows]
        self.actions = np.roll(self.actions, -1, axis=1)
        self.actions[:, -2] = actions
        self.is_bos = np.roll(self.is_bos, -1, axis=1)
        self.is_bos[:, -2, 0] = bos
        self.mask = np.roll(self.mask, -1, axis=1)
        self.mask[:, -2] = 1.0

    def after_act(self) -> None:
        """Apply the bundle's post-first-action BOS retention convention.

        A cached ``PolicyRunner`` in discard mode drops the episode-start
        token's KV immediately after using it for act#0.  A stateless windowed
        ONNX graph has no persisted KV, so reproduce that convention by
        removing every visible BOS token from subsequent attention masks.
        """
        if self.bos_cache_mode == "discard":
            self.mask[self.is_bos[..., 0] != 0.0] = 0.0

    def inputs(self) -> dict[str, np.ndarray]:
        return {
            "states": self.states[:, :-1],
            "actions": self.actions[:, :-1],
            "is_bos": self.is_bos[:, :-1],
            "mask": self.mask[:, :-1],
        }


def _pack_observation(observations, agent: int, num_channels: int) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observations[channel][agent], np.float32).reshape(-1)
            for channel in range(num_channels)
        ]
    )


def _decode(raw: np.ndarray, continuous_size: int, branches: list[int]):
    """Return ``(Unity action, autoregressive feedback action)``."""
    num_agents = raw.shape[0]
    env_parts = []
    feedback_parts = []

    if continuous_size:
        continuous = raw[:, :continuous_size].astype(np.float32)
        env_parts.append(np.clip(continuous, -1.0, 1.0))
        feedback_parts.append(continuous)

    offset = continuous_size
    for branch_size in branches:
        logits = raw[:, offset : offset + branch_size]
        offset += branch_size
        index = np.argmax(logits, axis=1).astype(np.int32)
        env_parts.append(index.reshape(num_agents, 1).astype(np.float32))
        one_hot = np.zeros((num_agents, branch_size), np.float32)
        one_hot[np.arange(num_agents), index] = 1.0
        feedback_parts.append(one_hot)

    if offset != raw.shape[1]:
        raise ValueError(
            f"ONNX output width {raw.shape[1]} does not match Unity action spec "
            f"({continuous_size} continuous + {sum(branches)} discrete logits)."
        )
    return np.concatenate(env_parts, axis=1), np.concatenate(feedback_parts, axis=1)


def _run_onnx(session: ort.InferenceSession, inputs: dict[str, np.ndarray], batch: int):
    num_agents = inputs["states"].shape[0]
    if batch == num_agents:
        return session.run(["action"], inputs)[0]

    rows = []
    for agent in range(num_agents):
        feed = {name: value[agent : agent + 1] for name, value in inputs.items()}
        rows.append(session.run(["action"], feed)[0][0])
    return np.stack(rows)


def _group_key(context: dict) -> tuple:
    return (
        int(context["env_index"]),
        str(context["behavior_name"]),
        int(context["group_id"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", required=True, type=Path, help="Unity executable path.")
    parser.add_argument("--onnx", required=True, type=Path, help="Downloaded policy ONNX path.")
    parser.add_argument("--env-id", default="unity-evaluation")
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--max-ticks", type=int, default=20_000)
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--graphics", action="store_true")
    parser.add_argument(
        "--bos-cache-mode",
        choices=("discard", "retain"),
        default="discard",
        help="Drop or retain the BOS token after the first action (default: discard).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from unity_env import UnityEnv

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    inputs_by_name = {item.name: item for item in session.get_inputs()}
    required = {"states", "actions", "is_bos", "mask"}
    if set(inputs_by_name) != required:
        raise SystemExit(
            f"Expected ONNX inputs {sorted(required)}, got {sorted(inputs_by_name)}."
        )
    if not session.get_outputs() or session.get_outputs()[0].name != "action":
        raise SystemExit("Expected an ONNX output named 'action'.")

    states_shape = inputs_by_name["states"].shape
    actions_shape = inputs_by_name["actions"].shape
    model_batch = states_shape[0]
    context_length = int(states_shape[1])
    model_state_size = int(states_shape[2])
    model_action_size = int(actions_shape[2])

    UnityEnv.register(args.env_id, None, str(args.build))
    make_kwargs = {
        "time_scale": args.time_scale,
        "use_graphics": args.graphics,
    }
    if args.worker_id is not None:
        make_kwargs["worker_id"] = args.worker_id
    env = UnityEnv.make(args.env_id, **make_kwargs)
    try:
        num_agents = env.num_agents
        num_channels = len(env.observation_shapes)
        env_state_size = int(sum(np.prod(shape) for shape in env.observation_shapes))
        if model_batch not in (1, num_agents):
            raise SystemExit(
                f"ONNX batch must be 1 or match the scene's {num_agents} agents; "
                f"got {model_batch}."
            )
        if model_state_size != env_state_size:
            raise SystemExit(
                f"ONNX state size {model_state_size} != Unity observation size "
                f"{env_state_size}."
            )

        action_spec = env.spec.action_spec
        continuous_size = int(action_spec.continuous_size)
        branches = [int(size) for size in action_spec.discrete_branches]
        expected_action_size = continuous_size + sum(branches)
        if model_action_size != expected_action_size:
            raise SystemExit(
                f"ONNX action size {model_action_size} != Unity model-space action "
                f"size {expected_action_size}."
            )

        print(
            f"[env] agents={num_agents} obs={env_state_size} "
            f"continuous={continuous_size} branches={branches}",
            flush=True,
        )
        print(
            f"[onnx] batch={model_batch} context={context_length} "
            f"calls_per_decision={1 if model_batch == num_agents else num_agents}",
            flush=True,
        )

        observations, _ = env.reset()
        states = np.stack(
            [_pack_observation(observations, i, num_channels) for i in range(num_agents)]
        )
        window = Window(
            num_agents,
            context_length,
            env_state_size,
            model_action_size,
            bos_cache_mode=args.bos_cache_mode,
        )
        feedback_action = np.zeros((num_agents, model_action_size), np.float32)
        window.update(states, feedback_action, is_bos=1.0)

        env_action_size = continuous_size + len(branches)
        env_action = np.zeros((num_agents, env_action_size), np.float32)
        returns = np.zeros(num_agents, np.float64)
        results: list[tuple[float, str] | None] = [None] * num_agents
        ticks = 0

        while any(result is None for result in results) and ticks < args.max_ticks:
            ticks += 1
            if any(observations[0][agent] is not None for agent in range(num_agents)):
                raw = _run_onnx(session, window.inputs(), int(model_batch))
                window.after_act()
                env_action, feedback_action = _decode(raw, continuous_size, branches)

            next_observations, rewards, terminated, truncated, _ = env.step(env_action)
            for agent in range(num_agents):
                if results[agent] is not None:
                    continue
                if rewards[agent] is not None:
                    returns[agent] += float(rewards[agent])
                if terminated[agent] is True or truncated[agent] is True:
                    reason = "term" if terminated[agent] else "trunc"
                    results[agent] = (float(returns[agent]), reason)

            if any(next_observations[0][agent] is not None for agent in range(num_agents)):
                for agent in range(num_agents):
                    if next_observations[0][agent] is not None:
                        states[agent] = _pack_observation(
                            next_observations, agent, num_channels
                        )
                window.update(states, feedback_action, is_bos=0.0)
            observations = next_observations

        completed = [(index, result) for index, result in enumerate(results) if result]
        if not completed:
            raise SystemExit(f"No agent episode finished within {args.max_ticks} ticks.")

        agent_returns = np.asarray([result[0] for _, result in completed], np.float64)
        print("\n[agent return]")
        print(
            f"n={len(agent_returns)} mean={agent_returns.mean():.6f} "
            f"std={agent_returns.std():.6f} min={agent_returns.min():.6f} "
            f"max={agent_returns.max():.6f} ticks={ticks}"
        )

        grouped: dict[tuple, list[float]] = defaultdict(list)
        for agent, result in completed:
            grouped[_group_key(env.agent_context[agent])].append(float(result[0]))
        complete_groups = [values for values in grouped.values() if len(values) > 1]
        if complete_groups:
            group_returns = np.asarray([sum(values) for values in complete_groups])
            group_success = np.asarray([any(value > 0 for value in values) for values in complete_groups])
            print("\n[cooperative group return]")
            print(
                f"n={len(group_returns)} mean={group_returns.mean():.6f} "
                f"std={group_returns.std():.6f} success_rate={group_success.mean():.2%}"
            )

        unfinished = sum(result is None for result in results)
        if unfinished:
            print(f"\n[warning] {unfinished}/{num_agents} agent episodes unfinished.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
