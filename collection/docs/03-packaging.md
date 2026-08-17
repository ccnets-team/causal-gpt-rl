# Packaging

```bash
python collection/build_minari.py \
    --raw raw/ \
    --dataset-id <namespace>/<name>-v0
```

The final segment must end in `-v<integer>`; `review/badinput-v0` is valid while
`review/badinput/v0` is not. The id is also the path, so
`mujoco/humanoid/simple-v0` is written to
`~/.minari/datasets/mujoco/humanoid/simple-v0/`.

## Preflight

Before creating that directory, the command checks every episode for required
arrays, aligned lengths, finite observations/actions/rewards, consistent widths,
declared continuous bounds or discrete branches, and the single end-of-episode
boundary [the contract](01-the-input-contract.md) requires. Only after the full
preflight succeeds does it write Minari, then it loads the result back and
verifies its counts and spaces.

## Flags

- `--description`, `--author`, `--author-email` — recorded in the dataset metadata.
- `--batch-episodes N` — episodes appended per HDF5 write (default 1000). Lower
  it if many short episodes strain memory.
- `--ego-agent KEY` — the multi-agent wrapper below.

Runs with `minari==0.5.3`.

## Multi-agent recordings

For a recording with one episode per physical agent, `--ego-agent` nests both
spaces under an ego key, so a consumer reads
`observations["agents"]["agent_0"]`:

```bash
python collection/build_minari.py \
    --raw raw/ \
    --dataset-id <namespace>/<name>-v0 \
    --ego-agent agent_0
```

The leaf spaces are identical either way; the wrapper only names whose
trajectory the episode is. This is the schema the published SoccerTwos and
DungeonEscape datasets use.

## Check what you built

That automatic load-back proves the dataset is readable and the counts survived.
It cannot tell you the spaces are the ones you *meant* — a wrong declaration
produces the wrong interface even when packaging succeeds, so read them yourself
before you publish or upload. A bare `Box`, a per-sensor `Tuple`, and an ego
`Dict` are all walked down to the same leaf specs, so no shape here costs you an
adapter later; what it does decide is which leaves the model sees.

```python
import minari
import numpy as np

print(minari.list_local_datasets().keys())

ds = minari.load_dataset("unity/crawler/expert-v0")
print(ds.total_episodes, ds.total_steps)
print(ds.observation_space)   # Tuple(Box(126,), Box(32,))
print(ds.action_space)        # Box(-1.0, 1.0, (20,))

ds.set_seed(0)
ep = ds.sample_episodes(n_episodes=1)[0]
print(np.asarray(ep.actions).shape)          # (1000, 20)
print(np.asarray(ep.rewards).sum())          # episode return
```

An episode carries `observations`, `actions`, `rewards`, `terminations`,
`truncations`, `id`, and `infos`. Observations run one step longer than actions,
which is the `T + 1` from
[the contract](01-the-input-contract.md#each-ep_npz-is-one-episode) surviving the
round trip.

A structured value follows the declared space rather than being one matrix, so
walk it the way the space is shaped — a multi-channel `obs_channels` gives a
tuple of leaves, and `--ego-agent` gives a nested dict:

```python
for leaf in ep.observations:                 # Tuple space
    print(np.asarray(leaf).shape)            # (1001, 126) then (1001, 32)
```

What to confirm: the spaces match the sensors and actions you meant to record;
`total_episodes` and `total_steps` are what you collected; and a sampled episode
return is plausible for the policy that generated it. A dataset that loads but
declares the wrong action kind will train a model with the wrong output heads.

## Dropping and merging

`filter_episodes` and `split_dataset` return **views** — a new index list over the
same stored episodes, keeping the original `dataset_id` and directory. They are
for looking, not for producing an artifact: uploading after filtering uploads the
unfiltered dataset.

With this packager, drop unwanted episodes **before** packaging by removing their
`ep_*.npz` files from the raw directory.

`combine_datasets` is different — it takes a `new_dataset_id` and writes a real
dataset, which is how several collection runs become one id:

```python
merged = minari.combine_datasets([ds_a, ds_b], new_dataset_id="unity/crawler/expert-v1")
```

One directory caveat — the dataset id is stored in the metadata *and* is the
directory path. Renaming the directory without rebuilding makes Minari warn that
the namespace location does not match the id. Move the whole tree, do not rename
parts of it.
