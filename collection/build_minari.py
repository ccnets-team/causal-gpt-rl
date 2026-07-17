"""Build a Minari dataset from raw per-episode `.npz` files.

Runs in a packaging env with minari==0.5.3 (separate from whatever env recorded
the episodes).

Source-agnostic: the episodes can come from any environment. Each `.npz` holds
`observations` (length T+1) and `actions`/`rewards`/`terminations`/`truncations`
(length T); an optional sibling `spec.json` declares the action kind.

The dataset is env-less: observation/action spaces are declared explicitly and no
gym env is attached, so it has no `recover_environment()`. Load it with
`recover_env=False`; it follows the same flat-Box convention as the Gymnasium /
MuJoCo Minari datasets (obs is a single 1-D Box, action is Box[-1, 1]).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from minari import create_dataset_from_buffers
from minari.data_collector import EpisodeBuffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Dir of ep_*.npz files from collect.py.")
    ap.add_argument("--dataset-id", required=True, help="Minari id, e.g. 'unity/crawler/expert-v0'.")
    ap.add_argument("--author", default=None)
    ap.add_argument("--author-email", default=None)
    ap.add_argument("--description", default="Recorded episodes packaged as an env-less Minari dataset.")
    args = ap.parse_args()

    files = sorted(Path(args.raw).glob("ep_*.npz"))
    if not files:
        raise SystemExit(f"No ep_*.npz found in {args.raw}")

    # Action/obs spec written by the collector; fall back to continuous Box for
    # older runs recorded before spec.json existed.
    spec_path = Path(args.raw) / "spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {"action_kind": "continuous"}
    action_kind = spec.get("action_kind", "continuous")

    first = np.load(files[0])
    obs_dim = int(first["observations"].shape[1])
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    if action_kind == "discrete":
        branches = [int(b) for b in spec["branches"]]
        action_space = (
            gym.spaces.Discrete(branches[0])
            if len(branches) == 1
            else gym.spaces.MultiDiscrete(branches)
        )
    else:
        act_dim = int(first["actions"].shape[1])
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)

    def _load_actions(raw):
        if action_kind != "discrete":
            return raw.astype(np.float32)
        raw = raw.astype(np.int64)
        # Discrete -> scalar per step; MultiDiscrete -> vector per step.
        return raw[:, 0] if raw.shape[1] == 1 else raw

    buffers = []
    total = 0
    for i, f in enumerate(files):
        d = np.load(f)
        obs = d["observations"].astype(np.float32)
        act = _load_actions(d["actions"])
        rew = d["rewards"].astype(np.float32)
        term = d["terminations"].astype(bool)
        trunc = d["truncations"].astype(bool)
        T = len(rew)
        assert obs.shape[0] == T + 1, (f.name, obs.shape, T)
        buffers.append(
            EpisodeBuffer(
                id=i,
                observations=obs,
                actions=act,
                rewards=list(rew),
                terminations=list(term),
                truncations=list(trunc),
            )
        )
        total += T

    print(
        f"[build] episodes={len(buffers)} transitions={total} obs_dim={obs_dim} "
        f"action_space={action_space}"
    )

    ds = create_dataset_from_buffers(
        dataset_id=args.dataset_id,
        buffer=buffers,
        env=None,
        observation_space=observation_space,
        action_space=action_space,
        author=args.author,
        author_email=args.author_email,
        description=args.description,
    )
    print(f"[done] created Minari dataset '{args.dataset_id}' ({ds.total_episodes} episodes)")


if __name__ == "__main__":
    main()
