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

## Record with one of our bundles

```python
from causal_gpt_rl.inference import load_runner
from collection import CollectionRunner

runner = CollectionRunner(load_runner("bundle/"), "raw/")
```

The wrapper keeps the runner's calls — `reset` / `act` / `observe` — and writes
what they drive. The loop in full, what it records, and its boundaries are in
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
