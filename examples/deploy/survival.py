"""Survival statistics for long-horizon rollouts.

A return mean stops describing anything once episode lengths go bimodal — some
trajectories die early, the rest run the full horizon — because the mean lands
between the two modes and the standard deviation reports the split rather than
run-to-run noise. Past a horizon where that happens, report survival.

`survival_stats` is a pure function of episode lengths (and optionally returns),
so it composes with `run_episodes` output or with lengths collected by any other
loop. It is analysis over results, not runtime behaviour, so it lives here beside
the other example evaluators rather than in the inference package — copy it or
import it as `examples.deploy.survival`.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def survival_stats(
    lengths: Sequence[int],
    returns: Optional[Sequence[float]] = None,
    *,
    horizon: Optional[int] = None,
    bucket: Optional[int] = None,
) -> dict:
    """Survival, per-interval hazard, and completer-only return per step.

    Args:
        lengths: Episode lengths, one per episode.
        returns: Episode returns, aligned with `lengths`. Enables the
            return-per-step fields; omit to compute survival only.
        horizon: The full-episode target. Defaults to `max(lengths)`. An episode
            counts as a completer when its length reaches it.
        bucket: Interval width for the hazard table. Defaults to `horizon // 5`
            (a single bucket when the horizon is short). The final interval is
            truncated when `bucket` does not divide `horizon`.

    Returns:
        A dict with `horizon`, `bucket`, `num_episodes`, `completers`,
        `completion_rate`, `intervals`, `conditional_mean`,
        `constant_rate_prediction`, and — when `returns` is given —
        `return_per_step_all` and `return_per_step_completers`.

        Each entry of `intervals` covers one bucket and carries `start`, `end`,
        `entered`, `died`, `survived`, `conditional` (the share of episodes
        entering the interval that also leave it) and `cumulative`.

    Reading it: a *flat* `conditional` across intervals means failure is a
    constant per-step risk rather than a degradation that compounds with rollout
    depth — an episode that survived this far is no more fragile than a fresh
    one. A falling `conditional` is the compounding case.
    """
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    if lengths_arr.ndim != 1 or lengths_arr.size == 0:
        raise ValueError("lengths must be a non-empty 1-D sequence")
    if np.any(lengths_arr < 0):
        raise ValueError("lengths must be non-negative")

    total = int(lengths_arr.size)
    horizon = int(horizon) if horizon is not None else int(lengths_arr.max())
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    bucket = int(bucket) if bucket is not None else max(1, horizon // 5)
    if bucket < 1:
        raise ValueError(f"bucket must be >= 1, got {bucket}")

    intervals: list[dict] = []
    start = 0
    while start < horizon:
        end = min(start + bucket, horizon)
        entered = int(np.count_nonzero(lengths_arr >= start))
        survived = int(np.count_nonzero(lengths_arr >= end))
        intervals.append(
            {
                "start": start + 1,
                "end": end,
                "entered": entered,
                "died": entered - survived,
                "survived": survived,
                # None rather than 0.0: no episode entered, so nothing was measured.
                "conditional": (survived / entered) if entered else None,
                "cumulative": survived / total,
            }
        )
        start = end

    measured = [i["conditional"] for i in intervals if i["conditional"] is not None]
    conditional_mean = float(np.mean(measured)) if measured else None
    # A constant hazard predicts completion as rate ** num_intervals. Comparing it
    # with the measured completion rate is the flatness check.
    constant_rate_prediction = (
        float(conditional_mean ** len(intervals)) if conditional_mean is not None else None
    )

    completers_mask = lengths_arr >= horizon
    completers = int(np.count_nonzero(completers_mask))

    stats = {
        "horizon": horizon,
        "bucket": bucket,
        "num_episodes": total,
        "completers": completers,
        "completion_rate": completers / total,
        "intervals": intervals,
        "conditional_mean": conditional_mean,
        "constant_rate_prediction": constant_rate_prediction,
    }

    if returns is not None:
        returns_arr = np.asarray(returns, dtype=np.float64)
        if returns_arr.shape != lengths_arr.shape:
            raise ValueError(
                f"returns shape {returns_arr.shape} != lengths shape {lengths_arr.shape}"
            )
        stats["return_per_step_all"] = _return_per_step(returns_arr, lengths_arr)
        # Completers only: mixing in episodes that died early drags the mean down,
        # so this is the figure that separates "still performing" from "alive but
        # no longer doing the task".
        stats["return_per_step_completers"] = _return_per_step(
            returns_arr[completers_mask], lengths_arr[completers_mask]
        )

    return stats


def _return_per_step(returns_arr: np.ndarray, lengths_arr: np.ndarray) -> Optional[float]:
    steps = int(lengths_arr.sum())
    if steps == 0:
        return None
    return float(returns_arr.sum() / steps)


def format_survival_table(stats: dict) -> str:
    """Render `survival_stats` as a fixed-width table for a terminal or a log."""
    lines = [
        f"horizon={stats['horizon']}  bucket={stats['bucket']}  "
        f"episodes={stats['num_episodes']}",
        f"completed {stats['completers']}/{stats['num_episodes']} "
        f"({stats['completion_rate']:.1%})",
        "",
        f"{'interval':>17} {'entered':>8} {'died':>6} {'survived':>9} "
        f"{'conditional':>12} {'cumulative':>11}",
    ]
    for row in stats["intervals"]:
        conditional = "—" if row["conditional"] is None else f"{row['conditional']:.1%}"
        lines.append(
            f"{row['start']:>7}-{row['end']:<9} {row['entered']:>8} {row['died']:>6} "
            f"{row['survived']:>9} {conditional:>12} {row['cumulative']:>10.1%}"
        )

    if stats["conditional_mean"] is not None:
        lines += [
            "",
            f"mean conditional survival per {stats['bucket']} steps: "
            f"{stats['conditional_mean']:.1%}",
            f"a constant rate predicts {stats['constant_rate_prediction']:.1%} completion "
            f"against {stats['completion_rate']:.1%} measured",
        ]

    if "return_per_step_all" in stats:
        completers = stats["return_per_step_completers"]
        lines += [
            "",
            f"return/step, all episodes: {stats['return_per_step_all']:.4f}",
            "return/step, completers:   "
            + ("—" if completers is None else f"{completers:.4f}"),
        ]

    return "\n".join(lines)
