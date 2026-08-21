# MuJoCo — tier collection recipe

Record a `simple` / `medium` / `expert` ladder from one published bundle and
package it in the layout the dataset repositories use.

[`../deploy/record.py`](../deploy/record.py) records **one** dataset at one
retention. This folder is the ladder: several tiers cut out of the same weights
by changing how much past the rollout keeps, all on the same seeds.

The bundles it records with are public:
[ccnets/causal-gpt-rl](https://huggingface.co/ccnets/causal-gpt-rl).

## Environments

Recording and packaging share no dependency, so they can be two environments or
one — the notebook does both in one kernel because it can, not because it must.

| For | Install |
|---|---|
| Calibrating and recording | `pip install "causal-gpt-rl[hub,mujoco]" "mujoco==3.2.3"` |
| Packaging | `minari==0.5.3` |

`mujoco` is pinned because a different simulator release is a different
measurement even with identical weights and seeds.

## The dial

A tier is a *worse policy*. With one bundle per environment there is no second
policy, so the tier has to be synthesized — and only one dial reaches the
recorder.

**Retention** (`kv_cache_max_len`) is how much past the rollout keeps. It is a
load-time argument, the same file runs at any of them, and it moves closed-loop
return. That is the dial here.

**Action noise** is the dial [`../unity_collection/`](../unity_collection/) uses,
and it does not transfer. `CollectionRunner` records the action `act()` returned,
and this policy conditions on its own past actions: perturbing the action after
it is emitted would leave the recorded trajectory and the policy's own context
disagreeing about what was taken. Recording a noised policy needs the noise
inside the policy, which a bundle does not offer.

That constraint is worth knowing before planning a ladder, because retention is
a **weak** dial in some environments. The published Humanoid bundle moves about
9% in mean return across its entire retention grid — nowhere near enough to cut
three tiers out of. Calibration is what tells you which case you are in.

## 1. Calibrate

A tier is defined by its normalized score, the same quantity the public tables
use:

```text
norm = 100 * (return - random_ref) / (expert_ref - random_ref)
```

so picking a retention needs both endpoints and the curve between them. One run
measures all three: `random_ref` is uniform action sampling under the same
seeds and the same batch width, `expert_ref` is the best retention in the grid,
and the curve is every level in it.

```bash
python -m examples.mujoco_collection.calibrate_retention \
    --env-id Hopper-v5 --target-simple 40 --target-medium 70
```

It prints the normalized score at each level, picks the level closest to each
target, and reports the **normalized spread** across the whole grid. A spread
under 20 means the levels are closer together than the seed draw, and the honest
answer there is one dataset at the best retention rather than a ladder wearing
three names. Refine `--grid` around a pick if no level is close enough.

## 2. Record and package

```bash
python -m examples.mujoco_collection.record_tiers \
    --env-id Hopper-v5 --out raw/hopper-v5 \
    --tier simple=16 --tier medium=32 --tier expert=128 \
    --episodes 200 --build
```

Each tier gets a fresh runner at its retention, its own raw directory of
`ep_%06d.npz` plus `spec.json`, and — with `--build` — a Minari dataset named
`<namespace>/<tier>-v0`, where the namespace defaults to the lowercased env id:

```text
hopper-v5/simple-v0/data/main_data.hdf5
hopper-v5/medium-v0/data/main_data.hdf5
hopper-v5/expert-v0/data/main_data.hdf5
hopper-v5/namespace_metadata.json
```

Every tier runs the same seeds, so the ladder is one policy at three retentions
rather than three unrelated draws. `spec.json` carries a provenance entry naming
the bundle and the `kv_cache_max_len` behind each one, so a tier stays readable
back to what produced it.

Leave `--build` off to record now and package later in a `minari==0.5.3`
environment; the script prints the
[`collection/build_minari.py`](../../collection/build_minari.py) commands either
way. Minari refuses to overwrite an existing dataset id — a re-recording needs
`--version 1`.

The recipe ends at the packaged datasets. Set `MINARI_DATASETS_PATH` before
packaging to keep them somewhere other than `~/.minari`.

## The grid behind the dial

The ladder above names one context length per tier. Which lengths those are is a
measurement, and
[`record_context_grid.py`](record_context_grid.py) is the recording form of it:
the same policy at several lengths, one dataset each, on the same seeds.

```bash
python -m examples.mujoco_collection.record_context_grid \
    --env-id Humanoid-v5 --out raw/humanoid \
    --context 8,16,32,64,128 --episodes 100 --build
```

```text
ccnets/humanoid/kv0008-v0
ccnets/humanoid/kv0016-v0
...
```

`--episodes` is both the episode count and the batch width — a level is one
batch of that many rows recording one episode each, so every episode carries a
seed the caller chose rather than an unseeded auto-reset, and every level runs
the same width. Both are what make the levels comparable at all, so neither is
a separate knob to get wrong.

The summary compares the grid against itself:

```text
 context  episodes  transitions                 return      worst   length  terminated
      16        12         2295      1333.02 +- 236.28     549.58    191.2           1
      64        12         2400      1406.46 +- 11.11     1378.29    200.0           0

spread across the grid: 73.45, against a within-level spread of up to 236.28
  [note] the levels differ by less than the episodes within one of them.
```

That note is the point. A grid whose levels sit closer together than the
episodes inside one of them has not separated anything, and the means are the
last column to trust — read the worst episode and the terminated count first.

Against [`calibrate_retention.py`](calibrate_retention.py), which measures a
curve and throws the rollouts away, this keeps them: every level is a packaged
dataset, so the one that wins can be trained on rather than re-recorded.

## Boundaries

**A batch is a width, not a shortcut.** `CollectionRunner` records a
vectorized env, one episode file per row, so a tier of 200 episodes can be 200
rows instead of 200 sequential rollouts. What it is not is free: the width is
part of the measurement, because rows reduce floating point in a different order
than a single env does. Compare levels recorded at one width, or do not compare
them.

**The ladder is measured, not assumed.** Retention is not monotonic — more
history helps some environments and hurts others, and the best level is often
not the largest. Nothing here reads a tier off the retention number itself.

**Fifty seeds is one draw.** Calibration defaults to the reproduction protocol's
width, and differences of a few percent in mean return do not survive a
reshuffle. Treat a pick whose `[note]` says it missed the target as a reason to
refine the grid, not as a tier.

What the recorded arrays have to contain, and how to check what came out, is the
[input contract](../../collection/docs/01-the-input-contract.md) and
[Packaging](../../collection/docs/03-packaging.md).
