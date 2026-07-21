"""Evaluate a Causal GPT-RL team against a stock ML-Agents ONNX policy.

This runner is intended for two-team decentralized environments such as
SoccerTwos.  The Causal policy controls exactly one team and keeps one temporal
context per controlled agent; the stock policy controls the opposing team.
Both teams' actions are routed into one Unity step.  By default the evaluation
is repeated with the Causal policy on each side to remove team-side bias.

Example:
    python examples/unity/evaluate_matchup.py \
        --build path/to/SoccerTwos/UnityEnvironment.exe \
        --causal-onnx path/to/soccertwos-b16.onnx \
        --stock-onnx path/to/SoccerTwos-release23-ort.onnx
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "unity_collection"))
sys.path.insert(0, str(HERE))

from collect import _assign_field_ids  # noqa: E402
from evaluate_onnx import Window, _decode, _pack_observation, _run_onnx  # noqa: E402
from onnx_policy import OnnxPolicy  # noqa: E402
from unity_env import UnityEnv  # noqa: E402


@dataclass
class SideResult:
    causal_team: int | None
    wins: int
    draws: int
    losses: int
    causal_agent_mean: float | None
    causal_team_mean: float
    stock_team_mean: float
    ticks: int

    @property
    def matches(self) -> int:
        return self.wins + self.draws + self.losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--causal-onnx", required=True, type=Path)
    parser.add_argument("--stock-onnx", required=True, type=Path)
    parser.add_argument(
        "--causal-team",
        choices=("0", "1", "both"),
        default="both",
        help="Run one side or swap sides and aggregate (default: both).",
    )
    parser.add_argument(
        "--stock-baseline",
        action="store_true",
        help="Also run stock-vs-stock as an environment symmetry check.",
    )
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--max-ticks", type=int, default=20_000)
    parser.add_argument("--worker-id", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graphics", action="store_true")
    return parser.parse_args()


def _session_contract(session: ort.InferenceSession) -> tuple[int, int, int, int]:
    inputs = {item.name: item for item in session.get_inputs()}
    required = {"states", "actions", "is_bos", "mask"}
    if set(inputs) != required:
        raise ValueError(f"expected Causal ONNX inputs {sorted(required)}, got {sorted(inputs)}")
    if not session.get_outputs() or session.get_outputs()[0].name != "action":
        raise ValueError("expected Causal ONNX output named 'action'")
    states = inputs["states"].shape
    actions = inputs["actions"].shape
    return int(states[0]), int(states[1]), int(states[2]), int(actions[2])


def _print_result(label: str, result: SideResult) -> None:
    print(f"\n[{label}]")
    print(
        f"matches={result.matches} W/D/L={result.wins}/{result.draws}/{result.losses} "
        f"win_rate={result.wins / result.matches:.2%} ticks={result.ticks}"
    )
    if result.causal_agent_mean is not None:
        print(
            f"causal_agent_return={result.causal_agent_mean:.6f} "
            f"causal_team_return={result.causal_team_mean:.6f} "
            f"stock_team_return={result.stock_team_mean:.6f}"
        )
    else:
        print(
            f"team0_return={result.causal_team_mean:.6f} "
            f"team1_return={result.stock_team_mean:.6f} (stock-vs-stock)"
        )


def evaluate_side(args: argparse.Namespace, causal_team: int | None, run_index: int) -> SideResult:
    env_id = f"soccer-twos-matchup-{run_index}"
    UnityEnv.register(env_id, None, str(args.build))
    env = UnityEnv.make(
        env_id,
        time_scale=args.time_scale,
        use_graphics=args.graphics,
        worker_id=args.worker_id + run_index,
    )
    try:
        contexts, field_sizes, field_members = _assign_field_ids(env.agent_context)
        teams = sorted({int(context["team_id"]) for context in contexts})
        if teams != [0, 1]:
            raise ValueError(f"expected exactly Soccer teams [0, 1], got {teams}")
        if any(size != 4 for size in field_sizes.values()):
            raise ValueError(f"expected four agents per Soccer field, got {field_sizes}")

        num_agents = env.num_agents
        causal_indices = (
            []
            if causal_team is None
            else [i for i, context in enumerate(contexts) if context["team_id"] == causal_team]
        )
        stock_indices = [i for i in range(num_agents) if i not in set(causal_indices)]
        num_channels = len(env.observation_shapes)
        state_size = int(sum(np.prod(shape) for shape in env.observation_shapes))
        action_spec = env.spec.action_spec
        continuous_size = int(action_spec.continuous_size)
        branches = [int(size) for size in action_spec.discrete_branches]

        causal_session = ort.InferenceSession(
            str(args.causal_onnx), providers=["CPUExecutionProvider"]
        )
        batch, context_length, model_state_size, model_action_size = _session_contract(
            causal_session
        )
        expected_action_size = continuous_size + sum(branches)
        if model_state_size != state_size or model_action_size != expected_action_size:
            raise ValueError(
                f"Causal ONNX contract obs/action={model_state_size}/{model_action_size}, "
                f"Unity={state_size}/{expected_action_size}"
            )
        if causal_team is not None and batch not in (1, len(causal_indices)):
            raise ValueError(
                f"Causal ONNX batch must be 1 or controlled team size {len(causal_indices)}, "
                f"got {batch}"
            )

        stock = OnnxPolicy(
            str(args.stock_onnx),
            num_agents=num_agents,
            obs_shapes=env.observation_shapes,
            action_spec=action_spec,
            rng=np.random.default_rng(args.seed + run_index),
        )
        print(
            f"[contract] agents={num_agents} fields={len(field_sizes)} teams=16+16 "
            f"obs={state_size} branches={branches} causal_batch={batch}"
        )

        observations, _ = env.reset()
        window = None
        feedback = None
        if causal_team is not None:
            states = np.stack(
                [_pack_observation(observations, i, num_channels) for i in causal_indices]
            )
            window = Window(
                len(causal_indices), context_length, state_size, model_action_size
            )
            feedback = np.zeros((len(causal_indices), model_action_size), np.float32)
            window.update(states, feedback, is_bos=1.0)

        returns = np.zeros(num_agents, np.float64)
        finished = np.zeros(num_agents, dtype=bool)
        action = np.zeros((num_agents, continuous_size + len(branches)), np.float32)
        ticks = 0
        while not finished.all() and ticks < args.max_ticks:
            ticks += 1
            stock_action = stock.act(observations)
            action[stock_indices] = stock_action[stock_indices]

            if causal_team is not None:
                raw = _run_onnx(causal_session, window.inputs(), batch)
                causal_action, feedback = _decode(raw, continuous_size, branches)
                action[causal_indices] = causal_action

            next_observations, rewards, terminated, truncated, _ = env.step(action)
            for agent in range(num_agents):
                if not finished[agent] and rewards[agent] is not None:
                    returns[agent] += float(rewards[agent])
                if not finished[agent] and (terminated[agent] is True or truncated[agent] is True):
                    finished[agent] = True

            if causal_team is not None:
                next_states = np.zeros((len(causal_indices), state_size), np.float32)
                for row, agent in enumerate(causal_indices):
                    if next_observations[0][agent] is not None:
                        next_states[row] = _pack_observation(
                            next_observations, agent, num_channels
                        )
                    else:
                        next_states[row] = window.states[row, -1]
                window.update(next_states, feedback, is_bos=0.0)
            observations = next_observations

        if not finished.all():
            raise RuntimeError(
                f"only {int(finished.sum())}/{num_agents} agents finished in {ticks} ticks"
            )

        wins = draws = losses = 0
        causal_team_returns = []
        stock_team_returns = []
        for members in field_members.values():
            by_team = {
                team: [i for i in members if contexts[i]["team_id"] == team]
                for team in teams
            }
            team_returns = {
                team: float(returns[by_team[team]].sum()) for team in teams
            }
            if causal_team is None:
                controlled, opponent = team_returns[0], team_returns[1]
            else:
                controlled, opponent = team_returns[causal_team], team_returns[1 - causal_team]
            causal_team_returns.append(controlled)
            stock_team_returns.append(opponent)
            if controlled > opponent + 1e-8:
                wins += 1
            elif controlled < opponent - 1e-8:
                losses += 1
            else:
                draws += 1

        return SideResult(
            causal_team=causal_team,
            wins=wins,
            draws=draws,
            losses=losses,
            causal_agent_mean=(
                None if causal_team is None else float(returns[causal_indices].mean())
            ),
            causal_team_mean=float(np.mean(causal_team_returns)),
            stock_team_mean=float(np.mean(stock_team_returns)),
            ticks=ticks,
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    teams = [0, 1] if args.causal_team == "both" else [int(args.causal_team)]
    results = []
    for run_index, team in enumerate(teams):
        result = evaluate_side(args, causal_team=team, run_index=run_index)
        results.append(result)
        _print_result(f"Causal team {team} vs stock team {1 - team}", result)

    if len(results) == 2:
        wins = sum(result.wins for result in results)
        draws = sum(result.draws for result in results)
        losses = sum(result.losses for result in results)
        matches = wins + draws + losses
        print("\n[side-swapped aggregate]")
        print(
            f"matches={matches} W/D/L={wins}/{draws}/{losses} "
            f"win_rate={wins / matches:.2%}"
        )

    if args.stock_baseline:
        baseline = evaluate_side(
            args, causal_team=None, run_index=len(teams)
        )
        _print_result("stock team 0 vs stock team 1 baseline", baseline)


if __name__ == "__main__":
    main()
