# collection

Turn recorded episodes into the [Minari](https://minari.farama.org) dataset that
declares the observation and action spaces your model will use. You own the
encoding and choose those spaces; this directory records, checks, and packages
that interface.

![From any source to an offline RL dataset — three entry points join a single path: with only an environment you first get a policy model, then record trajectories, then package; with an environment and a policy model you join at recording; with trajectories already recorded you join at packaging. All three end at the same env-less Minari dataset, and any offline-RL method can train on it as a separate, optional step](docs/assets/any-source-to-offline-rl-dataset.svg)

## Where you join

The three are not separate routes. They are one path with three entry points,
and each entry is the one above it with a stage already done.

| | You already have | Still ahead of you |
|---|---|---|
| **1** | an environment | get a policy model → record trajectories → package |
| **2** | an environment and a policy model | record trajectories → package |
| **3** | recorded trajectories | package |

A *policy model* here is whatever drives the environment while you record — a
network you trained (SB3's PPO, say), or any controller you can already call for
an action. It is not the model you get back from training; that one is
downstream of this whole picture.

`collection/` is the last stage, and only that stage. It is **source-agnostic**:
whatever produced the episodes — a simulator, a game build, a logged control
system, a replayed production trace — if its fixed-shape numeric trajectories fit
[the contract](docs/01-the-input-contract.md), you can package them. Raw pixels,
audio, and text are encoded on your side before this boundary.

## What ships for the earlier stages

Getting a policy model and recording trajectories are sibling concerns, not
this directory's. What this repository actually provides for them, by source:

| Source | 1 · get a policy model | 2 · record trajectories | 3 · package |
|---|---|---|---|
| Unity ML-Agents | stock policies in the [envs repo](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs) | [`examples/unity_collection/collect.py`](../examples/unity_collection/collect.py) | ✓ |
| Gymnasium | — | code sketch only, no runnable script | ✓ |
| Any other source | — | — | ✓ |

Packaging is the column that holds for every source. For stages 1 and 2 outside
Unity you bring your own policy model and your own loop — SB3's PPO, a scripted
controller, whatever already drives your system. Nothing here integrates with
those; what the loop has to produce is in
[Record Trajectories](docs/02-record-trajectories.md).

## Docs

[`docs/`](docs/README.md) is the packaging path in three parts — the input
contract, recording trajectories, then packaging and checking the result.

Diagrams live in [`docs/assets/`](docs/assets/).

## Not here

Environment stepping, policy-model training, quality-tier synthesis, and noise
calibration. Those live with the source that needs them —
[`examples/unity_collection/`](../examples/unity_collection/) is the worked one.
Training on the finished dataset is a separate, optional step and is not part of
this directory either.
