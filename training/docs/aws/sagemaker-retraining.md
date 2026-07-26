# SageMaker Retraining

Retraining resumes a Causal GPT-RL training job from saved checkpoints. Use this when you want a new training job to continue from previous training state.

## When a Checkpoint Prefix Is Needed

A clean training job does not need a previous checkpoint prefix. It starts fresh when no checkpoints are available under the SageMaker checkpoint local path.

For retraining/resume, the job must be connected to the checkpoint S3 prefix that contains the previous run's saved checkpoints. A higher-level launcher may fill this in for you, but native SageMaker needs the checkpoint location in `CheckpointConfig`.

## How Resume Works

1. Start a new SageMaker training job.
2. Configure the job to use the checkpoint S3 prefix that contains the saved checkpoints.
3. SageMaker syncs that prefix into the container checkpoint directory.
4. The training job loads the latest available checkpoint and continues training.
5. New checkpoints are written back to the configured checkpoint prefix.

Within that prefix, both `<namespace>/archive/*.pt` and
`<namespace>/improvements/*.pt` contain full training state. Resume picks the
one with the largest training step across both series. The `bundles/`
directories beside them hold live-synced inference bundles for inspecting
intermediate policies; bundles are not the source used to restore optimizer and
scheduler state.

## Resuming and the Archive

Archive points are permanent, so a resumed run adds to what the previous run
left instead of replacing it.

- The archive manifest holds the union of both runs' points. Steps are
  cumulative, so the points order correctly across runs.
- The evenly spaced schedule is recomputed at startup from the new `max_steps`,
  so a resumed run's points do not line up with the previous run's. Both sets
  are kept.
- `final` therefore appears more than once. **The point with the largest step is
  the one the latest run ended on.**

That last rule holds because `max_steps` is the cumulative total for the whole
run, not the additional steps — see Notes below.

## Branching From a Preserved Step

Archive points are permanent and named by step, which makes any of them a stable
branch point. Copy a single `archive/model_checkpoint_step_NNNNNNN.pt` into a
fresh checkpoint prefix and start a job against that prefix: resume finds one
checkpoint and continues from exactly that step. Use `archive_steps` to preserve
the steps you expect to branch from.

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
- Retraining/resume: use a checkpoint prefix that already contains saved checkpoints.

## Notes

- The input dataset layout must still match the requested `dataset_ids`.
- `max_steps` should be set for the total intended training run, not just the additional steps.
- `improvements/` slots are bounded to 10 files and rotate by overwriting older slots. Archive points are permanent and are never overwritten.
- Inference bundles are live-synced beside each series' `.pt` files under the same checkpoint namespace.
