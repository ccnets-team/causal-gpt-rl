"""Check raw Unity cooperative groups and an optional Minari load-back.

The generic raw contract validates shapes, action ranges and episode boundaries.
This check adds the Unity-specific requirement that all physical agents in one
group match are present, synchronized and use the same group noise setting.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from collection._internal.contract import ContractError, validate_raw_directory


def validate(raw: Path, group_size: int, target: int, dataset_id: str | None = None):
    if group_size < 1 or target < 1:
        raise ValueError("group_size and target must be positive")
    files, _ = validate_raw_directory(raw)
    manifest_path = raw / "episode_metadata.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != len(files):
        raise ContractError(f"manifest has {len(rows)} rows for {len(files)} episodes")
    groups = defaultdict(list)
    transitions = 0
    step_rewards = []
    for index, (path, row) in enumerate(zip(files, rows)):
        if row["episode_index"] != index or row["episode_file"] != path.name:
            raise ContractError(f"manifest order mismatch at {path.name}")
        with np.load(path, allow_pickle=False) as episode:
            length = len(episode["rewards"])
            if length != row["transition_count"]:
                raise ContractError(f"manifest transition count mismatch at {path.name}")
            if not np.isclose(episode["rewards"].sum(), row["return"], atol=1e-4):
                raise ContractError(f"manifest return mismatch at {path.name}")
            step_rewards.append(float(episode["rewards"].mean()))
        transitions += length
        groups[row["match_id"]].append((path, row))
    if transitions < target:
        raise ContractError(f"{transitions} transitions is below target {target}")
    for match_id, members in groups.items():
        if len(members) != group_size:
            raise ContractError(f"{match_id}: expected {group_size} agents, found {len(members)}")
        if len({row["agent_id"] for _, row in members}) != group_size:
            raise ContractError(f"{match_id}: duplicated physical agent")
        if len({row["transition_count"] for _, row in members}) != 1:
            raise ContractError(f"{match_id}: agent episodes have different lengths")
        if len({(row["terminated"], row["truncated"]) for _, row in members}) != 1:
            raise ContractError(f"{match_id}: boundary flags differ within group")
        noise = [row.get("group_temperature") for _, row in members]
        if len(set(noise)) != 1:
            raise ContractError(f"{match_id}: teammates used different temperatures")
        with np.load(members[0][0], allow_pickle=False) as reference:
            reference_rewards = reference["rewards"].copy()
        for path, _ in members[1:]:
            with np.load(path, allow_pickle=False) as episode:
                if not np.array_equal(episode["rewards"], reference_rewards):
                    raise ContractError(f"{match_id}: teammate reward streams differ")
    if dataset_id is not None:
        import minari

        dataset = minari.load_dataset(dataset_id)
        if (dataset.total_episodes, dataset.total_steps) != (len(files), transitions):
            raise ContractError("Minari load-back counts differ from raw collection")
    return {"episodes": len(files), "matches": len(groups),
            "transitions": transitions, "mean_episode_step_reward": float(np.mean(step_rewards)),
            "dataset_id": dataset_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--dataset-id")
    args = parser.parse_args()
    print(json.dumps(validate(args.raw, args.group_size, args.target, args.dataset_id),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
