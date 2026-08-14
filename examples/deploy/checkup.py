"""Basic health check of a delivered bundle against the dataset it was trained on.

No environment required. The dataset you trained on already answers most first
questions about a delivered policy: does it reproduce the actions in that data,
is its action spread sane, does its value head track anything, and — the one
that catches real mistakes — do your observations land where the bundle's
normalization expects them.

Run it on the bundle you were delivered and the Minari dataset the job was
given::

    python -m examples.deploy.checkup \
        --bundle path/to/bundle \
        --dataset mujoco/hopper/simple-v0 \
        --episodes 50

What it does NOT tell you: the return the policy earns. That takes the
environment — a policy can match recorded actions closely and still fail
closed-loop, because its own outputs steer where it goes next. Treat this as a
pre-flight check, not a substitute for measuring.

Method — teacher forcing on the first context window of each episode. The model
reads recorded `(state, action)` pairs, and the head at position `t` speaks
about `t+1`: it predicts the next action and the value of the next step. Only
the first window is scored, because it is the one window whose context matches
how an episode actually starts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from causal_gpt_rl.inference import load_runner
from causal_gpt_rl.inference.adapters import make_state_input_adapter

# The model clamps log_std before exponentiating when it samples, so mirroring
# the clamp here makes the reported spread the spread the model would use.
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

# The policy is not an imitator, so agreement with recorded actions is a
# diagnostic and not a quality score. It reads comparatively: run the same
# bundle against each dataset you have and see which one it belongs with.
_AGREEMENT_NOTE = "agreement with this data, not policy quality"


def _at(value, t):
    """Index one timestep out of a Minari episode field.

    Structured episodes store a dict / tuple of arrays rather than an array of
    samples, so a plain `value[t]` picks a leaf instead of a timestep.
    """
    if isinstance(value, dict):
        return {k: _at(v, t) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_at(v, t) for v in value)
    return value[t]


def _field_length(value) -> int:
    """Timesteps stored in a Minari episode field, structured or not."""
    if isinstance(value, dict):
        return min(_field_length(v) for v in value.values())
    if isinstance(value, tuple):
        return min(_field_length(v) for v in value)
    return len(value)


def _state_flattener(runner):
    """(flatten, n_cont) for this bundle's observation space.

    Structured spaces need `gym.flatten` *and* the continuous-first permutation;
    the runtime's own adapter owns that rule, so borrow it rather than restating
    it here. A plain Box needs neither.

    `n_cont` is where the normalized block ends: the canonical layout puts the
    continuous dimensions first and normalization is applied to those only, so
    the one-hot tail passes through raw and must stay out of any statistic about
    normalization.
    """
    adapter = make_state_input_adapter(runner.obs_space)
    if adapter is None:
        return (
            lambda obs: np.asarray(obs, dtype=np.float32).reshape(-1),
            runner.state_size,
        )
    return lambda obs: np.asarray(adapter(obs), dtype=np.float32), adapter.cf.n_cont


def _action_flattener(runner):
    """Env action -> the flat per-head vector the model emits.

    Declared head order, no permutation — action heads are not reordered,
    because the emitted action is the next input.
    """
    space = runner.action_space
    if space is None:
        return lambda act: np.asarray(act, dtype=np.float32).reshape(-1)
    return lambda act: np.asarray(
        gym.spaces.flatten(space, act), dtype=np.float32
    )


def _episode_arrays(episode, flat_state, flat_action):
    """(states, actions, rewards, terminal_index) for one episode."""
    actions = episode.actions
    rewards = np.asarray(episode.rewards, dtype=np.float64)
    n = min(
        len(rewards),
        len(np.atleast_1d(episode.terminations)),
        _field_length(actions),
    )
    states = np.stack([flat_state(_at(episode.observations, t)) for t in range(n)])
    acts = np.stack([flat_action(_at(actions, t)) for t in range(n)])
    ended = np.asarray(episode.terminations, dtype=bool) | np.asarray(
        episode.truncations, dtype=bool
    )
    terminal = int(np.argmax(ended[:n])) if ended[:n].any() else None
    return states, acts, rewards[:n], terminal


def _window(states, actions, length):
    """First `length` tokens of an episode, padded, with BOS on token 0."""
    t = min(length, len(states))
    s = np.zeros((length, states.shape[1]), dtype=np.float32)
    a = np.zeros((length, actions.shape[1]), dtype=np.float32)
    mask = np.zeros((length,), dtype=np.float32)
    is_bos = np.zeros((length, 1), dtype=np.float32)
    s[:t] = states[:t]
    a[:t] = actions[:t]
    mask[:t] = 1.0
    is_bos[0, 0] = 1.0
    return s, a, is_bos, mask, t


def _mean_heads(model, out, schedule):
    """Per-head predictions in declared order, continuous heads post-adapted."""
    adapted = model.adapt_output_heads(out)
    heads = []
    for (head_type, *_), index in zip(schedule, model.mean_action_indices):
        heads.append(adapted[index] if head_type == "continuous" else out[index])
    return heads


def value_targets(rewards_per_episode, context_length: int) -> np.ndarray:
    """Return-to-go lined up with the value the head at each position emits.

    The head at `t` values step `t+1`, so the target for position `t` is the
    return remaining from `t+1` onward. Shape `(episodes, context_length - 1)`,
    matching the scored positions.
    """
    out = np.zeros((len(rewards_per_episode), context_length), dtype=np.float64)
    for i, rewards in enumerate(rewards_per_episode):
        remaining = np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1]
        take = min(context_length, len(remaining))
        out[i, :take] = remaining[:take]
    return out[:, 1:]


def _r2(err: np.ndarray, target: np.ndarray) -> float | None:
    """Per-dimension R², averaged — so heads whose dims differ in scale count
    equally instead of the widest-swinging dimension setting the score."""
    variance = np.var(target, axis=0)
    usable = variance > 0
    if not usable.any():
        return None
    per_dim = 1.0 - np.mean(err[:, usable] ** 2, axis=0) / variance[usable]
    return float(np.mean(per_dim))


def _summary(name: str, x: np.ndarray) -> dict:
    return {
        "check": name,
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _check_spaces(runner, data) -> None:
    """Refuse a dataset whose declared spaces are not the bundle's.

    Equal flat widths are not enough: two different `Dict` layouts can flatten
    to the same length and mean entirely different things.
    """
    for kind, bundle_space, data_space in (
        ("observation", runner.obs_space, getattr(data, "observation_space", None)),
        ("action", runner.action_space, getattr(data, "action_space", None)),
    ):
        if bundle_space is None or data_space is None:
            continue
        if bundle_space != data_space:
            raise SystemExit(
                f"dataset does not fit this bundle: {kind} space\n"
                f"  bundle:  {bundle_space}\n"
                f"  dataset: {data_space}"
            )


def run_checkup(bundle: str, dataset_id: str, episodes: int, device: str) -> dict:
    if episodes < 1:
        raise SystemExit(f"--episodes must be at least 1, got {episodes}")
    try:
        import minari
    except ImportError as exc:
        raise SystemExit(
            "minari is required to read a training dataset. Install it with "
            "`pip install minari==0.5.3`."
        ) from exc

    runner = load_runner(bundle, device=device)
    model = runner.model
    ctx = runner.context_length
    schedule = runner.action_schedule

    data = minari.load_dataset(dataset_id)
    _check_spaces(runner, data)

    flat_state, n_cont = _state_flattener(runner)
    flat_action = _action_flattener(runner)
    eps = []
    for episode in data.iterate_episodes():
        eps.append(_episode_arrays(episode, flat_state, flat_action))
        if len(eps) >= episodes:
            break
    if not eps:
        raise SystemExit(f"dataset {dataset_id!r} yielded no episodes")

    state_dim, action_dim = eps[0][0].shape[1], eps[0][1].shape[1]
    if state_dim != runner.state_size or action_dim != runner.action_size:
        raise SystemExit(
            "dataset does not fit this bundle: "
            f"states {state_dim} vs {runner.state_size}, "
            f"actions {action_dim} vs {runner.action_size}"
        )

    windows = [_window(s, a, ctx) for s, a, _, _ in eps]
    as_tensor = lambda i: torch.as_tensor(  # noqa: E731 - four identical stacks
        np.stack([w[i] for w in windows]), device=model.device
    )
    states, actions, is_bos, mask = (as_tensor(i) for i in range(4))
    valid_len = np.array([w[4] for w in windows])

    normalized = model.normalize_states_for_inference(states)
    with torch.inference_mode():
        context = torch.cat([normalized, actions, is_bos], dim=-1)
        out = model.infer_windowed(context, padding_mask=mask.bool())

    report: dict = {
        "bundle": str(bundle),
        "dataset": dataset_id,
        "episodes": len(eps),
        "context_length": ctx,
        "checks": [],
    }

    # The head at t speaks about t+1, so the last valid token has no target.
    scored = np.zeros((len(windows), ctx - 1), dtype=bool)
    for i, t in enumerate(valid_len):
        scored[i, : max(t - 1, 0)] = True
    if not scored.any():
        raise SystemExit("episodes are too short to score against this context length")

    # --- Do your observations land where the bundle expects? ----------------
    # Only the continuous block is normalized; the one-hot tail passes through
    # raw and would drag these statistics toward its own mean.
    if n_cont > 0:
        z = normalized.cpu().numpy()[:, :-1, :n_cont][scored]
        report["checks"].append(_summary("normalized state", z))
        report["checks"].append(
            {
                "check": "normalized |z| > 5",
                "fraction": float(np.mean(np.abs(z) > 5.0)),
                "dims": f"{n_cont} of {runner.state_size} (continuous block)",
                "note": "large means this data is not what the bundle was normalized on",
            }
        )

    # --- Does it reproduce the recorded actions, head by head? --------------
    recorded = actions.cpu().numpy()[:, 1:]
    offset = 0
    continuous_pairs = []
    for head, (head_type, size, *_) in zip(
        _mean_heads(model, out, schedule), schedule
    ):
        predicted_all = head.cpu().numpy()[:, :-1]
        target_all = recorded[..., offset : offset + size]
        offset += size
        if head_type == "continuous":
            continuous_pairs.append((predicted_all, target_all))
        predicted, target = predicted_all[scored], target_all[scored]
        label = f"action[{head_type}:{size}]"
        if head_type == "continuous":
            err = predicted - target
            report["checks"].append(
                {
                    "check": f"{label} rmse",
                    "value": float(np.sqrt(np.mean(err**2))),
                    "r2": _r2(err, target),
                    "note": _AGREEMENT_NOTE,
                }
            )
        elif head_type == "multi_binary":
            # Independent bits: the decode thresholds each logit at zero.
            matched = (predicted > 0.0) == (target > 0.5)
            report["checks"].append(
                {
                    "check": f"{label} bit match",
                    "value": float(np.mean(matched)),
                    "note": _AGREEMENT_NOTE,
                }
            )
        else:
            # One-hot categorical head: the decode takes the most-likely class.
            matched = np.argmax(predicted, axis=-1) == np.argmax(target, axis=-1)
            report["checks"].append(
                {
                    "check": f"{label} top-1 match",
                    "value": float(np.mean(matched)),
                    "note": _AGREEMENT_NOTE,
                }
            )

    # --- Is the action spread sane? -----------------------------------------
    if model.log_std_action_indices:
        log_std = torch.cat(
            [out[i] for i in model.log_std_action_indices], dim=-1
        ).cpu().numpy()[:, :-1][scored]
        clamped = np.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        report["checks"].append(_summary("action std", np.exp(clamped)))
        report["checks"].append(
            {
                "check": "log_std at clamp",
                "at_floor": float(np.mean(log_std <= LOG_STD_MIN + 1e-3)),
                "at_ceiling": float(np.mean(log_std >= LOG_STD_MAX - 1e-3)),
                "note": "either near 1.0 means the spread collapsed or saturated",
            }
        )

    # --- Does the value head track anything? ---------------------------------
    # The head at t values step t+1, so it lines up with the return remaining
    # from t+1 onward.
    value = out[model.value_index].cpu().numpy()[..., 0][:, :-1][scored]
    target_rtg = value_targets([rewards for _, _, rewards, _ in eps], ctx)[scored]
    corr = (
        float(np.corrcoef(value, target_rtg)[0, 1])
        if np.std(value) > 0 and np.std(target_rtg) > 0
        else None
    )
    report["checks"].append(_summary("value head", value))
    report["checks"].append(
        {
            "check": "value vs return-to-go",
            "pearson": corr,
            "note": "a rollout never reads the value head; this is diagnostic only",
        }
    )

    # --- Does the termination head fire at the real end? ---------------------
    # Only episodes that actually end inside the scored window can answer this;
    # in a long episode the window's edge is not an ending.
    if model.termination_index is not None:
        term = torch.sigmoid(out[model.termination_index]).cpu().numpy()[..., 0]
        term = term[:, :-1]
        ends = np.zeros_like(scored)
        for i, (_, _, _, terminal) in enumerate(eps):
            if terminal is not None and 0 < terminal < ctx:
                ends[i, terminal - 1] = True
        elsewhere = scored & ~ends
        report["checks"].append(
            {
                "check": "termination prob",
                "episodes_ending_in_window": int(ends.any(axis=1).sum()),
                "at_episode_end": float(np.mean(term[ends])) if ends.any() else None,
                "elsewhere": float(np.mean(term[elsewhere])) if elsewhere.any() else None,
                "note": "no episode ends inside the window when the count is 0"
                if not ends.any()
                else None,
            }
        )

    # --- Does prediction improve as context accumulates? ---------------------
    if continuous_pairs:
        squared = np.concatenate(
            [(pred - target) ** 2 for pred, target in continuous_pairs], axis=-1
        )
        series = [
            float(np.sqrt(np.mean(squared[scored[:, t], t])))
            for t in range(ctx - 1)
            if scored[:, t].any()
        ]
        report["checks"].append(
            {
                "check": "continuous rmse by position",
                "first": series[0] if series else None,
                "last": series[-1] if series else None,
                "series": series,
            }
        )

    return report


def format_report(report: dict) -> str:
    lines = [
        f"bundle   {report['bundle']}",
        f"dataset  {report['dataset']}  ({report['episodes']} episodes, "
        f"context {report['context_length']})",
        "",
    ]
    for check in report["checks"]:
        body = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in check.items()
            if k not in ("check", "note", "series") and v is not None
        )
        lines.append(f"  {check['check']:<28} {body}")
        if check.get("note"):
            lines.append(f"  {'':<28} ({check['note']})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, help="bundle directory")
    parser.add_argument("--dataset", required=True, help="Minari dataset id")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args()

    report = run_checkup(args.bundle, args.dataset, args.episodes, args.device)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
