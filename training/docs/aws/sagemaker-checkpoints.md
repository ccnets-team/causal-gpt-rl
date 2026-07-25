# SageMaker Checkpoints

Causal GPT-RL training can save intermediate training state while a SageMaker training job is running. The SageMaker setting is a checkpoint S3 prefix, so this document uses the term checkpoint.

## What Is Saved Where

There are two related outputs:

- Checkpoint S3 prefix: resume/retraining state and intermediate inference snapshots synced by SageMaker during training.
- SageMaker output artifact: final `model.tar.gz`, which contains the canonical inference bundle.

## Checkpoint S3 Prefix

Set a SageMaker checkpoint S3 prefix when creating the training job.

```text
s3://my-bucket/cgrl/checkpoints/<training-job-name>/
```

The checkpoint prefix stores full training checkpoints and intermediate policy
snapshots under the same namespace:

```text
<checkpoint-prefix>/
  <namespace>/
    archive/
      model_checkpoint_slot_000.pt
      model_checkpoint_slot_001.pt
      ...
      model_checkpoint_slot_009.pt
    snapshots/
      manifest.json
      slot_000/
        model.safetensors
        config.json
        metrics.json
      ...
      slot_009/
```

The `archive/*.pt` files contain training state for resume/retraining, including
model state and optimizer/scheduler state. `snapshots/slot_NNN/` contains an
inference bundle aligned with each checkpoint slot. SageMaker live-syncs both
directories to the configured checkpoint S3 prefix while training runs.

## Slot Rotation

At most 10 checkpoint slots are kept. After `model_checkpoint_slot_009.pt`, training rotates back to `model_checkpoint_slot_000.pt` and overwrites older slots.

## Checkpoint Selection Metric

The training job tracks an evaluation metric and direction so checkpoints can be ranked by quality. The startup log reports both:

```text
Checkpoint metric: offline_eval/action_nll
Metric direction: min
```

`offline_eval/action_nll` is the held-out Action NLL (lower is better). Each `snapshots/slot_NNN/metrics.json` records this metric for its slot. See the eval metrics in `training/docs/aws/aws-marketplace-training.md` for details.

## Canonical Bundle and Live Snapshots

The final SageMaker output artifact is separate from the checkpoint prefix.
After training finishes, `model.tar.gz` contains the canonical bundle but does
not duplicate the live snapshots:

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
- `<checkpoint-prefix>/<namespace>/snapshots/slot_NNN/` are policy bundles delivered while training runs. They load with the public inference runtime without restoring a training checkpoint. See `training/docs/aws/sagemaker-realtime-policy-delivery.md`.
- `archive/*.pt` checkpoints are for resume/retraining, not normal inference.

## Why Snapshots Exist

Checkpoint `.pt` files contain optimizer and scheduler state and are meant for
training resume. Snapshot bundles are a different thing: they are the product's
real-time policy delivery path. Each one is a complete inference bundle, synced
out while the job runs, so a policy can be evaluated in your own environment
before training finishes.

This document covers the checkpoint plumbing. For what the delivered bundles are
for and how to consume them — finding them, reading the manifest, polling for new
ones, and stopping a run early — see
`training/docs/aws/sagemaker-realtime-policy-delivery.md`.
