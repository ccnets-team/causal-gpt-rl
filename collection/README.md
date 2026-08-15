# collection

Turn your recorded typed-vector episodes into the
[Minari](https://minari.farama.org) dataset that declares the observation and
action spaces your model will use. You own the encoding and choose those spaces;
this directory records, checks, and packages that interface.

`build_minari.py` is **source-agnostic**: it packages per-episode `.npz` files
into an **env-less** Minari dataset: the observation and action spaces are
declared explicitly and no gym env is attached. Plain single-channel continuous
data stays a flat `Box`, the same convention the Gymnasium / MuJoCo Minari
datasets follow; `spec.json` preserves multi-sensor and discrete/hybrid
structure where you have it.

Whatever produced the episodes — a simulator, a game build, a logged control
system, a replayed production trace — if its fixed-shape numeric trajectories fit
the arrays below, you can package it. Raw pixels, audio, and text are encoded on
your side before this boundary.

## The input contract

A directory of `ep_*.npz` files, plus an optional `spec.json` beside them.

### Each `ep_*.npz` is one episode

| Array | Shape | Type |
|---|---|---|
| `observations` | `[T + 1, obs_width]` | float |
| `actions` | `[T, action_cols]` | float or int, see below |
| `rewards` | `[T]` | float |
| `terminations` | `[T]` | bool |
| `truncations` | `[T]` | bool |

`observations` is one longer than the rest: `T` transitions need `T + 1` states.
Files are read in sorted order and the episode id is their position.

`actions` is stored flat, and how it is read depends on `action_kind`:

| `action_kind` | `actions` columns | Becomes |
|---|---|---|
| `continuous` | `act_dim` floats | `Box(act_low, act_high, (act_dim,))` |
| `discrete` | one **index** per branch — not one-hot | `Discrete(n)` or `MultiDiscrete([...])` |
| `hybrid` | `continuous_size` floats, then one index per branch | `Tuple(Box, Discrete/MultiDiscrete)` |

### `spec.json` declares the spaces

**The spaces are mandatory; `spec.json` is optional.** An env-less dataset has no
environment to infer spaces from, so Minari refuses one without both:

```text
ValueError: Both observation space and action space must be provided, if env is None
```

`build_minari.py` therefore always declares them. Without `spec.json` it declares
the default — one continuous `Box` over the full observation width. The file is
how you say something else.


| Key | Applies to | Meaning |
|---|---|---|
| `action_kind` | all | `continuous` (default), `discrete`, or `hybrid` |
| `obs_channels` | all | Per-sensor widths, summing to `obs_width`. Omit for one channel |
| `act_dim` | continuous | Action width. Omit to take it from the array |
| `branches` | discrete, hybrid | Category count per branch, e.g. `[3, 3, 3]` |
| `continuous_size` | hybrid | Width of the continuous part |
| `act_low`, `act_high` | continuous, hybrid | Scalar or per-dimension continuous bounds. Default `-1`, `1` |

Omit the file entirely and you get a single continuous `Box` over the full
observation width — the common case for a plain sensor vector.

`obs_channels` is worth a moment: distinct sensors stay distinct leaves in a
`Tuple` rather than being concatenated, because distinct sensors carry distinct
meaning. `[126, 32]` becomes `Tuple(Box(126), Box(32))`; a consumer that wants
them joined does that itself. Any extra keys in `spec.json` are ignored, so it is
a fine place to keep provenance.

The helper covers the common flat-vector forms above: a bare observation `Box`,
a positional `Tuple` of observation channels, continuous/discrete/hybrid
actions, and the optional ego-agent `Dict` wrapper. The runtime supports broader
`Dict` / `Tuple` nesting; this packager does not invent a general schema language
for it.

## Writing episodes

Any loop that produces the five arrays will do:

```python
import numpy as np

obs, actions, rewards = [], [], []
state, _ = env.reset()
obs.append(state)

while True:
    action = policy(state)
    state, reward, terminated, truncated, _ = env.step(action)
    actions.append(action)
    rewards.append(reward)
    obs.append(state)
    if terminated or truncated:
        break

T = len(rewards)
term = np.zeros(T, dtype=bool); term[-1] = terminated
trunc = np.zeros(T, dtype=bool); trunc[-1] = truncated

np.savez(
    "raw/ep_000000.npz",
    observations=np.asarray(obs, dtype=np.float32),
    actions=np.asarray(actions, dtype=np.float32),
    rewards=np.asarray(rewards, dtype=np.float32),
    terminations=term,
    truncations=trunc,
)
```

For a `Discrete(7)` action, store the index instead and declare it:

```python
actions=np.asarray(actions, dtype=np.int64).reshape(T, 1)
# spec.json: {"action_kind": "discrete", "branches": [7]}
```

## Packaging

```bash
python collection/build_minari.py \
    --raw raw/ \
    --dataset-id <namespace>/<name>-v0
```

The final segment must end in `-v<integer>`; `review/badinput-v0` is valid while
`review/badinput/v0` is not. The id is also the path, so
`mujoco/humanoid/simple-v0` is written to
`~/.minari/datasets/mujoco/humanoid/simple-v0/`.

Before creating that directory, the command checks every episode for required
arrays, aligned lengths, finite observations/actions/rewards, consistent widths,
declared continuous bounds or discrete branches, and a final episode boundary.
The final step must set termination, truncation, or both. Only after the full
preflight succeeds does it write Minari, then it loads the result back and
verifies its counts and spaces.

Useful flags:

- `--description`, `--author`, `--author-email` — recorded in the dataset metadata.
- `--batch-episodes N` — episodes appended per HDF5 write (default 1000). Lower
  it if many short episodes strain memory.

Runs with `minari==0.5.3`.

### Multi-agent recordings

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

Load it back before you publish or upload it. The declared spaces are the part
worth reading closely — they become the model's interface, and a wrong
declaration can produce the wrong interface even when packaging succeeds. A bare
`Box`, a per-sensor `Tuple`, and an ego `Dict` are all walked down to the same
leaf specs, so no shape here costs you an adapter later; what it does decide is
which leaves the model sees.

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
which is the `T + 1` above surviving the round trip.

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

### Dropping and merging

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

## A worked source

[`../examples/unity_collection/`](../examples/unity_collection/) records episodes
from a Unity ML-Agents build and packages them with this tool, including quality
tiers synthesized by degrading a stock policy. The published datasets at
[ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets)
were made that way, from the builds at
[ccnets/causal-gpt-rl-unity-envs](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs).

A second worked source is planned from outside Gymnasium — a SUMO traffic-light
controller driven over TraCI — to walk the same path from a different engine.
Whatever the source, it only has to produce the typed-vector episode contract
described here.
