# Collection docs

Turn recorded episodes into the [Minari](https://minari.farama.org) dataset that
declares the observation and action spaces your model will use. You own the
encoding and choose those spaces; `collection/` records, checks, and packages
that interface.

![From any source to an offline RL dataset — three entry points join a single path: with only an environment you first get a policy model, then record trajectories, then package; with an environment and a policy model you join at recording; with trajectories already recorded you join at packaging. All three end at the same env-less Minari dataset, and any offline-RL method can train on it as a separate, optional step](assets/any-source-to-offline-rl-dataset.svg)

## Where you join

The three entry points are not separate routes: each one joins the same path
with an earlier stage already done.

| | You already have | Still ahead of you |
|---|---|---|
| **1** | an environment | get a policy model → record trajectories → package |
| **2** | an environment and a policy model | record trajectories → package |
| **3** | recorded trajectories | package |

A *policy model* here is whatever drives the environment while you record — a
network you trained (SB3's PPO, say), or any controller you can already call for
an action. It is not the model you get back from training; that one is
downstream of this whole picture — until a
[later cycle](improving-the-next-dataset.md) brings it back as one.

Packaging is what `collection/` does for every source, and it is
**source-agnostic**: whatever produced the episodes — a simulator, a game build,
a logged control system, a replayed production trace — if its fixed-shape numeric
trajectories fit [the contract](01-the-input-contract.md), you can package them.
Raw pixels, audio, and text are encoded on your side before this boundary.

Recording is the one earlier stage it also covers, and along a different axis:
when the policy driving the episode is a model of ours, this directory records
the episodes too — [below](#recording-with-a-policy-of-ours). That stage is
indexed by the *policy*, where everything else here is indexed by the
*environment*, so it is a section of its own rather than another entry in the
table.

## The path, in order

Everyone ends at part 3, whichever entry point they came in at.

1. [The Input Contract](01-the-input-contract.md) — the five arrays per episode,
   and the `spec.json` that declares your spaces
2. [Record Trajectories](02-record-trajectories.md) — producing those arrays
   from a loop, and the worked sources that already do
3. [Packaging](03-packaging.md) — running `build_minari.py`, its flags,
   multi-agent recordings, and reading back what came out

Part 3 is what runs here for every source; 1 and 2 define what your own recorder
has to produce. Read 1 and 3 if you only want to package something. Read
[Check what you built](03-packaging.md#check-what-you-built) before you publish or
upload — a dataset that loads can still declare the wrong interface.

## What ships for the earlier stages

Getting a policy model and recording trajectories are sibling concerns, and
mostly not this directory's. What this repository actually provides for them, by
source:

| Source | 1 · get a policy model | 2 · record trajectories | 3 · package |
|---|---|---|---|
| Unity ML-Agents | stock policies in the [envs repo](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs) | [`examples/unity_collection/collect.py`](../../examples/unity_collection/collect.py) | ✓ |
| Gymnasium | — | code sketch only, no runnable script | ✓ |
| Any other source | — | — | ✓ |

Packaging is the column that holds for every source. For stages 1 and 2 outside
Unity you bring your own policy model and your own loop — SB3's PPO, a scripted
controller, whatever already drives your system. Nothing here integrates with
those; what the loop has to produce is in
[Record Trajectories](02-record-trajectories.md).

## Recording with a policy of ours

Every row of that table assumes the policy is yours. When it is instead a model
this repository produced, stage 2 stops being your loop to write:
`CollectionRunner` wraps the runner the bundle loads into and records what it
drives, in whatever environment you point it at.

```python
runner = CollectionRunner(load_runner("bundle/"), "raw/")
```

That is the second collection cycle, and the only one where a policy source can
be integrated rather than contracted — SB3, CleanRL and RLlib each expose a
different API on a different release cadence, ours is the one this repository
versions. Why you would run it, the loop in full, and what it writes:
[Improving the Next Dataset](improving-the-next-dataset.md). It is optional, and
beside the path above rather than a step in it.

## Not here

Environment stepping, policy-model training, quality-tier synthesis, and noise
calibration. Those live with the source that needs them —
[`examples/unity_collection/`](../../examples/unity_collection/) is the worked
one. Training on the finished dataset is a separate, optional step and is not
part of this directory either.

Diagrams live in [`assets/`](assets/).
