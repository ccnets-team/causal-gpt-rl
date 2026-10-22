# Collection

Packages recorded episodes — from any source — into an env-less
[Minari](https://minari.farama.org) dataset, and records them for the one policy
source this repository versions: a bundle of ours. Two functions and nothing
else.

What the episodes have to contain, where they can come from, and why you would
record a second time, are in [`docs/`](docs/README.md).

## What's here

| | |
|---|---|
| `build_minari.py` | CLI and `build_dataset()` — a raw directory becomes a verified Minari dataset. Any source. |
| `runner.py` | `CollectionRunner` — records the episodes one of our bundles drives. |
| `_internal/` | Implementation detail, no stability guarantee. |
| `docs/` | The input contract, the packaging path, and the collection cycle. |

## Package a raw directory

```bash
python collection/build_minari.py --raw raw/ --dataset-id <namespace>/<name>-v0
```

Every episode is checked before Minari creates anything, and the result is
loaded back and verified against the spaces that were declared. Flags,
multi-agent recordings, and how to read what came out are in
[Packaging](docs/03-packaging.md); what the directory must contain is
[The Input Contract](docs/01-the-input-contract.md).

For Unity recordings, `spec.json` may include each sensor's `obs_shapes` so a
flattened raw grid is restored as a spatial Minari `Box`. When an
`episode_metadata.jsonl` companion is present, the packager validates its
episode order and transition counts and copies it beside the Minari `data/`
directory with `collection_spec.json`. This keeps `match_id` and per-match
quality settings available after packaging.

For large Unity grids, use a small `--batch-episodes` value (for example 16)
when packaging. `python -m collection.compress_minari <dataset-id>` then
losslessly compresses the HDF5 arrays in bounded blocks and verifies a Minari
load-back. For cooperative scenes, run:

```bash
python -m collection.validate_unity_group --raw <raw-dir> --group-size 3 \
    --target 1000000 --dataset-id <dataset-id>
```

This checks that all agents in each match have complete, synchronized
trajectories and shared rewards.

For WallJump's changing behavior policies, use the specialized validator:

```bash
python -m collection.validate_walljump --raw <raw-dir> --target 1000000 \
    --dataset-id unity/walljump/expert-v0 --report validation.json
```

It checks both behavior identities, per-episode temperature bands, external step
deltas, outcome labels, and sampled Minari arrays against their raw episodes.

## Record with one of our bundles

Record straight from the command line:

```bash
python collection/record.py --env-id Humanoid-v5 --out raw/humanoid \
    --episodes 100 --context-length 128
```

`--context-length` is the history retained during rollout. The trained context
window remains fixed by the bundle. `--kv-cache-max-len` is an equivalent alias.

```python
from causal_gpt_rl.inference import load_runner
from collection import CollectionRunner

runner = CollectionRunner(load_runner("bundle/"), "raw/")
```

The wrapper keeps the runner's calls — `reset` / `act` / `observe` — and writes
what they drive. A runner loaded with `num_envs > 1` records a vectorized env
the same way, one episode file per row. The loop in full, what it records, and
its boundaries are in
[Improving the Next Dataset](docs/improving-the-next-dataset.md). Runnable
forms: [`examples/record_dataset.ipynb`](../examples/record_dataset.ipynb) and
[`examples/deploy/record.py`](../examples/deploy/record.py).

## Environments

| For | Install |
|---|---|
| Packaging | `minari==0.5.3` |
| Recording | `causal-gpt-rl` (torch) |

Neither pulls in the other. `import collection` costs nothing until you touch
`build_dataset` or `CollectionRunner`, so a packaging environment needs no torch
and a recording environment needs no Minari.
