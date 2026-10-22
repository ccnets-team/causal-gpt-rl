"""Validate WallJump policy identities, decision timing, and Minari load-back."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collection._internal.contract import (
    ContractError, REQUIRED_ARRAYS, load_spec, normalize_spec, validate_episode,
)


def validate(raw, target, dataset_id=None):
    raw = Path(raw)
    files = sorted(raw.glob("ep_*.npz"))
    if not files:
        raise ContractError("No WallJump episode files")
    spec = normalize_spec(load_spec(raw), observation_width=444, action_width=4)
    rows = [json.loads(s) for s in (raw / "episode_metadata.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    if len(rows) != len(files):
        raise ContractError("WallJump manifest count differs from episode count")
    if spec["obs_channels"] != [210, 210, 24] or spec["branches"] != [3, 3, 3, 2]:
        raise ContractError("Unexpected WallJump observation/action contract")
    by_behavior = {name: [] for name in ("SmallWallJump", "BigWallJump")}
    total = 0
    for i, (path, row) in enumerate(zip(files, rows)):
        if row["episode_index"] != i or row["episode_file"] != path.name:
            raise ContractError(f"Manifest order mismatch: {path.name}")
        behavior = row["behavior_name"]
        if behavior not in by_behavior:
            raise ContractError(f"Unknown behavior: {behavior}")
        if row["policy_id"] != ("random" if spec["random_policy"] else behavior + ".onnx"):
            raise ContractError(f"Wrong policy identity: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            # Read each compressed member once and run both contracts in one pass.
            episode = {name: archive[name] for name in (*REQUIRED_ARRAYS, "unity_tick_deltas")}
            validate_episode(episode, spec, source=path.name)
            length = len(episode["rewards"])
            if length != row["transition_count"] or not np.isclose(
                    episode["rewards"].sum(), row["return"], atol=1e-5):
                raise ContractError(f"Manifest rewards/count mismatch: {path.name}")
            deltas = episode["unity_tick_deltas"]
            if (deltas.shape != (length,) or not np.issubdtype(deltas.dtype, np.integer)
                    or np.any(deltas <= 0)
                    or not np.array_equal(deltas, row["unity_tick_deltas"])):
                raise ContractError(f"Bad decision timing: {path.name}")
            terminated, truncated = bool(episode["terminations"][-1]), bool(episode["truncations"][-1])
            if (terminated, truncated) != (row["terminated"], row["truncated"]) or terminated == truncated:
                raise ContractError(f"Boundary mismatch: {path.name}")
            outcome = "timeout" if truncated else "goal" if episode["rewards"][-1] > 0 else "fall"
            if outcome != row["outcome"]:
                raise ContractError(f"Outcome mismatch: {path.name}")
        limits = spec["noise"]["ranges"][behavior]
        if limits and not limits[0] <= row["temperature"] <= limits[1]:
            raise ContractError(f"Temperature outside band: {path.name}")
        total += length
        by_behavior[behavior].append(i)
    if total < target or not all(by_behavior.values()):
        raise ContractError("Collection target or behavior coverage not reached")
    checked = []
    if dataset_id:
        import minari
        dataset = minari.load_dataset(dataset_id)
        if (dataset.total_episodes, dataset.total_steps) != (len(rows), total):
            raise ContractError("Minari counts differ from raw")
        selected = sorted({indices[j] for indices in by_behavior.values()
                           for j in (0, len(indices) // 2, -1)})
        for index, episode in zip(selected, dataset.iterate_episodes(episode_indices=selected)):
            with np.load(files[index], allow_pickle=False) as original:
                flat = np.concatenate(episode.observations, axis=1)
                for actual, expected in ((flat, original["observations"]),
                    (episode.actions, original["actions"]), (episode.rewards, original["rewards"]),
                    (episode.terminations, original["terminations"]),
                    (episode.truncations, original["truncations"])):
                    if not np.array_equal(actual, expected):
                        raise ContractError(f"Minari values differ at episode {index}")
                checked.append(index)
    result = {"episodes": len(rows), "transitions": total, "dataset_id": dataset_id,
              "sampled_minari_episodes": checked,
              "behaviors": {b: {"episodes": len(indices),
                  "transitions": sum(rows[i]["transition_count"] for i in indices),
                  "mean_return": float(np.mean([rows[i]["return"] for i in indices])),
                  "mean_episode_step_reward": float(np.mean([rows[i]["return"] / rows[i]["transition_count"] for i in indices])),
                  "goal_rate": float(np.mean([rows[i]["outcome"] == "goal" for i in indices]))}
                  for b, indices in by_behavior.items()}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.raw, args.target, args.dataset_id)
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_bytes(text.encode())
    print(text, flush=True)
