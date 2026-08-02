# SageMaker Checkpoints

Causal GPT-RL training can save intermediate training state while a SageMaker training job is running. The SageMaker setting is a checkpoint S3 prefix, so this document uses the term checkpoint.

## What Is Saved Where

There are two related outputs:

- Checkpoint S3 prefix: resume/retraining state and intermediate inference bundles synced by SageMaker during training.
- SageMaker output artifact: final `model.tar.gz`, which contains the canonical inference bundle.

## Two Checkpoint Series

Set a SageMaker checkpoint S3 prefix when creating the training job.

```text
s3://my-bucket/cgrl/checkpoints/<training-job-name>/
```

Checkpoints are written into two series under that prefix. They differ in why a
checkpoint is written and how long it survives.

| Series | Written when | Identified by | Retention |
| --- | --- | --- | --- |
| `improvements/` | the evaluation metric reaches a new best | slot | 10 slots, rotating |
| `archive/` | a scheduled or requested step is reached | step | permanent |

```text
<checkpoint-prefix>/
  <namespace>/
    improvements/
      model_checkpoint_slot_000.pt
      ...
      model_checkpoint_slot_009.pt
      bundles/
        manifest.json
        slot_000/
          model.safetensors
          config.json
          metrics.json
        ...
        slot_009/
    archive/
      model_checkpoint_step_0020000.pt
      model_checkpoint_step_0025000.pt
      ...
      bundles/
        manifest.json
        step_0020000/
          model.safetensors
          config.json
          metrics.json
        ...
```

In both series the `*.pt` files hold training state for resume/retraining,
including model state and optimizer/scheduler state. `bundles/` holds a complete
inference bundle for the same point. Every `.pt` pairs with the bundle carrying
the same identity token:

```text
improvements/model_checkpoint_slot_003.pt   <->  improvements/bundles/slot_003/
archive/model_checkpoint_step_0025000.pt    <->  archive/bundles/step_0025000/
```

SageMaker live-syncs both series to the configured checkpoint S3 prefix while
training runs.

## `improvements/` — Best So Far

A checkpoint is written here whenever the evaluation metric reaches a new best.
That save also updates the canonical bundle.

At most 10 slots are kept. After `model_checkpoint_slot_009.pt`, training
rotates back to `model_checkpoint_slot_000.pt` and overwrites older slots. A
slot is overwritten after 10 further improvements, so **slot names are not
stable identities** — read `step` alongside the slot name.

There is no fixed interval between improvements and no minimum spacing. They
tend to be frequent early in a run and sparser later, as improvements become
harder to find.

## `archive/` — Preserved Points

Points here are never rotated away and are named by step, so `step_0020000/` is
step 20000 for the life of the prefix.

Two things put a point in the archive:

| Reason | When |
| --- | --- |
| `periodic` | 5 evenly spaced steps across the run, computed from `max_steps` at startup. |
| `requested` | The steps you list in `archive_steps`. Up to 10. |

A point is written the moment its step is reached, not at the end of the job.

Both schedules are reported at startup, so you know the steps before the run
produces anything:

```text
Archive periodic: 20000, 40000, 60000, 80000, 100000
Archive steps: 25000, 75000
Archive disk estimate: 700 MB for 7 point(s) (free: 24.3 GB)
```

Preserved points are never reclaimed, so the schedule costs disk for the whole
run. See `training/docs/aws/sagemaker-hyperparameters.md` for the sizing table.

### The Point a Run Ended On

The last point of a run always carries the reason `final`, so a single token
finds the model a run ended on.

| Situation | Reasons |
| --- | --- |
| Reached `max_steps` | `periodic`, `final` |
| Stopped with `StopTrainingJob` | `final`, `stopped` |
| Stopped on a scheduled step | `periodic`, `final`, `stopped` |

Stopping a job writes this point at the next step boundary. It is usually
present, but it is not guaranteed — the save and its sync have to finish inside
the stop grace period. **If you need the model a run ended on, copy the latest
archive point out before you stop the job.** A job killed outright, by an
out-of-memory condition or a host failure, has no `final` point at all.

## Checkpoint Selection Metric

The training job tracks an evaluation metric and direction. This metric is what
`improvements/` means by "better", and it is what selects the canonical bundle.
The startup log reports both:

```text
Checkpoint metric: offline_eval/checkpoint_score
Metric direction: max
```

`offline_eval/checkpoint_score` is Checkpoint Score, a bounded `[0, 1]` statistic
measured on a held-out split of your dataset — higher is better. Each
`bundles/*/metrics.json` records the evaluation metrics for its point, of which
this one is the selection metric. See `training/docs/aws/checkpoint-score.md` for
what it measures and how to read it, the eval metrics in
`training/docs/aws/aws-marketplace-training.md` for the full list, and
`training/docs/aws/sagemaker-realtime-policy-delivery.md` for what each key
means.

The metric is measured against your dataset, not against your environment. It
ranks checkpoints by how well the policy tracks your dataset while running on
its own rollout, which is not the same as how well the resulting policy performs
at your task.

## Canonical Bundle and Live Bundles

The final SageMaker output artifact is separate from the checkpoint prefix.
After training finishes, `model.tar.gz` contains the canonical bundle but does
not duplicate the live bundles:

```text
model.tar.gz
  reports/
    summary.json
  <namespace>/
    bundle/
      model.safetensors
      config.json
```

- `bundle/` is the canonical inference bundle to load by default.
- `<checkpoint-prefix>/<namespace>/improvements/bundles/` and
  `.../archive/bundles/` are policy bundles delivered while training runs. They
  load with the public inference runtime without restoring a training
  checkpoint. See `training/docs/aws/sagemaker-realtime-policy-delivery.md`.
- The `*.pt` files in both series are for resume/retraining, not normal
  inference. See `training/docs/aws/sagemaker-retraining.md`.

## Why Bundles Exist

The `.pt` files contain optimizer and scheduler state and are meant for training
resume. The bundles are a different thing: they are the product's real-time
policy delivery path. Each one is a complete inference bundle, synced out while
the job runs, so a policy can be evaluated in your own environment before
training finishes.

This document covers the checkpoint plumbing. For what the delivered bundles are
for and how to consume them — finding them, reading the manifests, polling for
new ones, and stopping a run early — see
`training/docs/aws/sagemaker-realtime-policy-delivery.md`.
