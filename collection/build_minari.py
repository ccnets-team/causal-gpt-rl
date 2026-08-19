"""Build a Minari dataset from raw per-episode `.npz` files.

Runs in a packaging env with minari==0.5.3 (separate from whatever env recorded
the episodes).

Source-agnostic: the episodes can come from any environment. Each `.npz` holds
`observations` (length T+1) and `actions`/`rewards`/`terminations`/`truncations`
(length T); a sibling `spec.json` declares the obs channels and action kind.

The dataset is env-less: observation/action spaces are declared explicitly and no
gym env is attached, so `minari.load_dataset(id)` returns a dataset with no
`recover_environment()`.

Spaces follow the declared structure, not a forced flat layout:

  - **observation** — `obs_channels` (per-sensor dims) becomes a `Tuple` of one
    `Box` per sensor, so distinct sensors stay distinguished; a single channel
    stays a bare `Box` (the flat MuJoCo-Minari convention). A consumer that wants
    one flat vector just concatenates the leaves, so the `Tuple` loses nothing —
    it is simply the honest, per-sensor form.
  - **action** — a declared `Box` (continuous; defaults to `[-1, 1]`),
    `Discrete`/`MultiDiscrete` (discrete), or
    `Tuple(Box, Discrete/MultiDiscrete)` (hybrid).
  - **ego-agent wrapper** (`--ego-agent`) — for multi-agent recordings, both of
    the above are nested under `Dict{"agents": {<key>: ...}}`, so a consumer
    reads `observations["agents"]["agent_0"]`. One ego episode per physical
    agent; the wrapper names whose trajectory the episode is.

Raw `.npz` obs/actions are stored flat; this packager splits them by the declared
dims into the structured leaf arrays Minari stores. Older raw dirs without
`obs_channels` fall back to a single flat `Box` observation.
"""
import argparse
import numpy as np
import gymnasium as gym
import minari
from minari import create_dataset_from_buffers
from minari.data_collector import EpisodeBuffer

try:  # Supports both ``python -m collection.build_minari`` and the documented path.
    from ._internal.contract import ContractError, validate_dataset_id, validate_raw_directory
except ImportError:  # pragma: no cover - exercised by CLI subprocess tests
    from _internal.contract import ContractError, validate_dataset_id, validate_raw_directory


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


def _build_action_space(action_kind, spec):
    if action_kind == "continuous":
        act_dim = int(spec["act_dim"])
        low = spec["act_low"]
        high = spec["act_high"]
        return gym.spaces.Box(low, high, shape=(act_dim,), dtype=np.float32)
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
        low = spec["act_low"]
        high = spec["act_high"]
        return gym.spaces.Tuple(
            (gym.spaces.Box(low, high, shape=(cont,), dtype=np.float32), disc)
        )
    raise ContractError(f"Unknown action_kind {action_kind!r}")


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
    raise ContractError(f"Unknown action_kind {action_kind!r}")


# Group key of the ego-agent schema. The collector writes one episode per
# physical agent, so there is exactly one ego key per episode; the nesting names
# whose trajectory it is without changing the leaf spaces underneath.
_EGO_GROUP = "agents"


def _wrap_ego_space(space, ego_agent):
    """`Dict{"agents": {<ego>: space}}`, or `space` unchanged when not requested."""
    if ego_agent is None:
        return space
    return gym.spaces.Dict({_EGO_GROUP: gym.spaces.Dict({ego_agent: space})})


def _wrap_ego_value(value, ego_agent):
    """Nest one episode's obs/action payload to match `_wrap_ego_space`."""
    if ego_agent is None:
        return value
    return {_EGO_GROUP: {ego_agent: value}}


DEFAULT_DESCRIPTION = "Recorded episodes packaged as an env-less Minari dataset."


def build_dataset(
    raw,
    dataset_id,
    *,
    author=None,
    author_email=None,
    description=DEFAULT_DESCRIPTION,
    batch_episodes=1000,
    ego_agent=None,
):
    """Package a raw directory of `ep_*.npz` files into a Minari dataset.

    Preflights every episode before Minari creates anything, writes in bounded
    batches, then loads the result back and verifies its counts and spaces.
    Contract violations raise `ContractError`; turning those into an exit status
    is `main()`'s job, so callers embedding this get an exception to handle.

    Returns a summary dict: `dataset_id`, `episodes`, `transitions`, and the two
    declared spaces.
    """
    if batch_episodes < 1:
        raise ContractError("batch_episodes must be at least 1")
    if ego_agent is not None and not ego_agent.strip():
        raise ContractError("ego_agent must be a non-empty key (e.g. 'agent_0')")
    validate_dataset_id(dataset_id)
    if dataset_id in minari.list_local_datasets():
        raise ContractError(f"Dataset {dataset_id!r} already exists; choose a new id")
    # Validate every file before Minari creates an output directory. This
    # deliberately makes a second, bounded-memory pass when writing.
    files, spec = validate_raw_directory(raw)

    action_kind = spec["action_kind"]
    obs_channels = spec["obs_channels"]

    observation_space = _wrap_ego_space(
        _build_observation_space(obs_channels), ego_agent
    )
    action_space = _wrap_ego_space(_build_action_space(action_kind, spec), ego_agent)

    buffers = []
    ds = None
    total = 0

    def flush():
        nonlocal buffers, ds
        if not buffers:
            return
        if ds is None:
            ds = create_dataset_from_buffers(
                dataset_id=dataset_id,
                buffer=buffers,
                env=None,
                observation_space=observation_space,
                action_space=action_space,
                author=author,
                author_email=author_email,
                description=description,
            )
        else:
            ds.update_dataset_from_buffer(buffers)
        buffers = []

    for i, f in enumerate(files):
        with np.load(f, allow_pickle=False) as d:
            obs = _split_obs(d["observations"].astype(np.float32), obs_channels)
            act = _split_actions(d["actions"], action_kind, spec)
            rew = d["rewards"].astype(np.float32)
            term = d["terminations"].astype(bool)
            trunc = d["truncations"].astype(bool)
        T = len(rew)
        buffers.append(
            EpisodeBuffer(
                id=i,
                observations=_wrap_ego_value(obs, ego_agent),
                actions=_wrap_ego_value(act, ego_agent),
                rewards=list(rew),
                terminations=list(term),
                truncations=list(trunc),
            )
        )
        total += T
        if len(buffers) >= batch_episodes:
            flush()

    flush()
    if ds is None:  # Defensive; preflight guarantees at least one episode.
        raise RuntimeError("Minari did not create a dataset")
    loaded = minari.load_dataset(dataset_id)
    if loaded.total_episodes != len(files) or loaded.total_steps != total:
        raise RuntimeError(
            f"Load-back count mismatch: episodes={loaded.total_episodes}/{len(files)} "
            f"steps={loaded.total_steps}/{total}"
        )
    if (
        loaded.observation_space != observation_space
        or loaded.action_space != action_space
    ):
        raise RuntimeError("Load-back spaces differ from the declared spaces")
    return {
        "dataset_id": dataset_id,
        "episodes": len(files),
        "transitions": total,
        "obs_channels": obs_channels,
        "observation_space": observation_space,
        "action_space": action_space,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Directory containing ep_*.npz files.")
    ap.add_argument("--dataset-id", required=True, help="Minari id, e.g. 'unity/crawler/expert-v0'.")
    ap.add_argument("--author", default=None)
    ap.add_argument("--author-email", default=None)
    ap.add_argument("--description", default=DEFAULT_DESCRIPTION)
    ap.add_argument(
        "--batch-episodes",
        type=int,
        default=1000,
        help="Episodes appended per HDF5 write (bounds memory for many short episodes).",
    )
    ap.add_argument(
        "--ego-agent",
        default=None,
        metavar="KEY",
        help="Multi-agent: nest obs/action under Dict{'agents': {KEY: ...}} "
             "(the published SoccerTwos / DungeonEscape datasets use 'agent_0'). "
             "Omit for the flat single-agent schema.",
    )
    args = ap.parse_args()

    try:
        summary = build_dataset(
            args.raw,
            args.dataset_id,
            author=args.author,
            author_email=args.author_email,
            description=args.description,
            batch_episodes=args.batch_episodes,
            ego_agent=args.ego_agent,
        )
    except ContractError as exc:
        raise SystemExit(f"collection input error: {exc}") from exc

    print(
        f"[build] episodes={summary['episodes']} transitions={summary['transitions']} "
        f"obs_channels={summary['obs_channels']} "
        f"observation_space={summary['observation_space']} "
        f"action_space={summary['action_space']}"
    )
    print(
        f"[done] created and verified Minari dataset '{summary['dataset_id']}' "
        f"({summary['episodes']} episodes, {summary['transitions']} transitions)"
    )


if __name__ == "__main__":
    main()
