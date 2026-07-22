"""Build a Minari dataset from raw per-episode `.npz` files.

Runs in a packaging env with minari==0.5.3 (separate from whatever env recorded
the episodes).

Source-agnostic: the episodes can come from any environment. Each `.npz` holds
`observations` (length T+1) and `actions`/`rewards`/`terminations`/`truncations`
(length T); a sibling `spec.json` declares the obs channels and action kind.

The dataset is env-less: observation/action spaces are declared explicitly and no
gym env is attached, so it has no `recover_environment()`. Load it with
`recover_env=False`.

Spaces follow the declared structure, not a forced flat layout:

  - **observation** — `obs_channels` (per-sensor dims) becomes a `Tuple` of one
    `Box` per sensor, so distinct sensors stay distinguished; a single channel
    stays a bare `Box` (the flat MuJoCo-Minari convention). A consumer that wants
    one flat vector just concatenates the leaves, so the `Tuple` loses nothing —
    it is simply the honest, per-sensor form.
  - **action** — `Box[-1, 1]` (continuous), `Discrete`/`MultiDiscrete`
    (discrete), or `Tuple(Box, Discrete/MultiDiscrete)` (hybrid).

Raw `.npz` obs/actions are stored flat; this packager splits them by the declared
dims into the structured leaf arrays Minari stores. Older raw dirs without
`obs_channels` fall back to a single flat `Box` observation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from minari import create_dataset_from_buffers
from minari.data_collector import EpisodeBuffer


def _build_observation_space(obs_channels):
    """Per-sensor `Tuple(Box, ...)`; a single channel stays a bare `Box`."""
    boxes = [
        gym.spaces.Box(-np.inf, np.inf, shape=(int(d),), dtype=np.float32)
        for d in obs_channels
    ]
    return boxes[0] if len(boxes) == 1 else gym.spaces.Tuple(boxes)


def _split_obs(flat, obs_channels):
    """Flat `[T+1, sum]` -> per-sensor tuple of `[T+1, d]` (or the flat array)."""
    if len(obs_channels) == 1:
        return flat.astype(np.float32)
    out, off = [], 0
    for d in obs_channels:
        d = int(d)
        out.append(flat[:, off:off + d].astype(np.float32))
        off += d
    return tuple(out)


def _build_action_space(action_kind, spec, first):
    if action_kind == "continuous":
        act_dim = int(spec.get("act_dim") or first["actions"].shape[1])
        return gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
    if action_kind == "discrete":
        branches = [int(b) for b in spec["branches"]]
        return (
            gym.spaces.Discrete(branches[0])
            if len(branches) == 1
            else gym.spaces.MultiDiscrete(branches)
        )
    if action_kind == "hybrid":
        cont = int(spec["continuous_size"])
        branches = [int(b) for b in spec["branches"]]
        disc = (
            gym.spaces.Discrete(branches[0])
            if len(branches) == 1
            else gym.spaces.MultiDiscrete(branches)
        )
        # Tuple order (Box, Discrete) mirrors ML-Agents' (continuous, discrete)
        # ActionTuple and the ONNX (continuous_actions, discrete_actions) outputs.
        return gym.spaces.Tuple(
            (gym.spaces.Box(-1.0, 1.0, shape=(cont,), dtype=np.float32), disc)
        )
    raise SystemExit(f"Unknown action_kind {action_kind!r}")


def _split_actions(raw, action_kind, spec):
    """Flat stored action `[T, cols]` -> the structured per-step action.

    Discrete indices are stored as columns (not one-hot): one column per branch.
    Hybrid = continuous columns first, then one index column per discrete branch.
    """
    if action_kind == "continuous":
        return raw.astype(np.float32)
    if action_kind == "discrete":
        idx = raw.astype(np.int64)
        return idx[:, 0] if idx.shape[1] == 1 else idx
    if action_kind == "hybrid":
        cont = int(spec["continuous_size"])
        branches = spec["branches"]
        c = raw[:, :cont].astype(np.float32)
        d = raw[:, cont:].astype(np.int64)
        d = d[:, 0] if len(branches) == 1 else d
        return (c, d)  # matches Tuple((Box, Discrete/MultiDiscrete))
    raise SystemExit(f"Unknown action_kind {action_kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Dir of ep_*.npz files from collect.py.")
    ap.add_argument("--dataset-id", required=True, help="Minari id, e.g. 'unity/crawler/expert-v0'.")
    ap.add_argument("--author", default=None)
    ap.add_argument("--author-email", default=None)
    ap.add_argument("--description", default="Recorded episodes packaged as an env-less Minari dataset.")
    ap.add_argument(
        "--batch-episodes",
        type=int,
        default=1000,
        help="Episodes appended per HDF5 write (bounds memory for many short episodes).",
    )
    args = ap.parse_args()

    if args.batch_episodes < 1:
        raise SystemExit("--batch-episodes must be at least 1")

    files = sorted(Path(args.raw).glob("ep_*.npz"))
    if not files:
        raise SystemExit(f"No ep_*.npz found in {args.raw}")

    # Obs/action spec written by the collector; fall back to a single flat
    # continuous Box for older runs recorded before spec.json carried structure.
    spec_path = Path(args.raw) / "spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {"action_kind": "continuous"}
    action_kind = spec.get("action_kind", "continuous")

    first = np.load(files[0])
    flat_obs_dim = int(first["observations"].shape[1])
    obs_channels = [int(c) for c in spec.get("obs_channels", [])] or [flat_obs_dim]
    if sum(obs_channels) != flat_obs_dim:
        raise SystemExit(
            f"obs_channels {obs_channels} (sum={sum(obs_channels)}) != stored obs dim {flat_obs_dim}."
        )

    observation_space = _build_observation_space(obs_channels)
    action_space = _build_action_space(action_kind, spec, first)

    buffers = []
    ds = None
    total = 0

    def flush():
        nonlocal buffers, ds
        if not buffers:
            return
        if ds is None:
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
        else:
            ds.update_dataset_from_buffer(buffers)
        buffers = []

    for i, f in enumerate(files):
        d = np.load(f)
        obs = _split_obs(d["observations"].astype(np.float32), obs_channels)
        act = _split_actions(d["actions"], action_kind, spec)
        rew = d["rewards"].astype(np.float32)
        term = d["terminations"].astype(bool)
        trunc = d["truncations"].astype(bool)
        T = len(rew)
        obs_len = obs[0].shape[0] if isinstance(obs, tuple) else obs.shape[0]
        assert obs_len == T + 1, (f.name, obs_len, T)
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
        if len(buffers) >= args.batch_episodes:
            flush()

    print(
        f"[build] episodes={len(files)} transitions={total} "
        f"obs_channels={obs_channels} observation_space={observation_space} "
        f"action_space={action_space}"
    )

    flush()
    assert ds is not None
    print(f"[done] created Minari dataset '{args.dataset_id}' ({ds.total_episodes} episodes)")


if __name__ == "__main__":
    main()
