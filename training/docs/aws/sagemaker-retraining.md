# SageMaker Retraining

Retraining resumes a Causal GPT-RL training job from saved checkpoints. Use this when you want a new training job to continue from previous training state.

## When a Checkpoint Prefix Is Needed

A clean training job does not need a previous checkpoint prefix. It starts fresh when no checkpoints are available under the SageMaker checkpoint local path.

For retraining/resume, the job must be connected to the checkpoint S3 prefix that contains the previous run's saved slots. A higher-level launcher may fill this in for you, but native SageMaker needs the checkpoint location in `CheckpointConfig`.

## How Resume Works

1. Start a new SageMaker training job.
2. Configure the job to use the checkpoint S3 prefix that contains the saved slots.
3. SageMaker syncs that prefix into the container checkpoint directory.
4. The training job loads the latest available checkpoint and continues training.
5. New checkpoints are written back to the configured checkpoint prefix.

Within that prefix, `<namespace>/archive/*.pt` contains the full training state
used for resume. `<namespace>/snapshots/` contains live-synced inference bundles
for inspecting intermediate policies; snapshots are not the source used to
restore optimizer and scheduler state.

## Recommended S3 Layout

Use a separate output prefix for each training job. Reuse a checkpoint prefix only when you intentionally want to resume from it.

```text
output path:
  s3://my-bucket/cgrl/output/<new-training-job-name>/

checkpoint path:
  s3://my-bucket/cgrl/checkpoints/<resume-series-name>/
```

## Clean Training vs. Retraining

- Clean training: use no previous checkpoints, or use a new/empty checkpoint prefix.
- Retraining/resume: use a checkpoint prefix that already contains saved checkpoint slots.

## Notes

- The input dataset layout must still match the requested `dataset_ids`.
- `max_steps` should be set for the total intended training run, not just the additional steps.
- Checkpoint archive slots are bounded to 10 files and rotate by overwriting older slots.
- Intermediate snapshot bundles are live-synced beside the archive under the same checkpoint namespace.
