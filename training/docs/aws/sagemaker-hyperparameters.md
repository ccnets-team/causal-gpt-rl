# SageMaker Hyperparameters

This document lists only the minimum details needed to pass hyperparameters to a SageMaker training job. The canonical field list and defaults live in `training/hyperparameters.py`.

## Rules

- SageMaker hyperparameter values are passed as strings.
- `dataset_ids` is required.
- Values that are not provided use the training recipe defaults.
- Dataset-specific metadata is read from the Minari dataset itself.

## Required Value

| Name | Description | Example |
| --- | --- | --- |
| `dataset_ids` | Minari dataset ids to train on. Pass multiple ids as a comma-separated string. | `mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0` |

## Common Overrides

For the first Marketplace upload, these are the only fields most users need to see.

| Name | Use |
| --- | --- |
| `max_steps` | Number of training updates. Use a small value for smoke tests and a larger value for real training. |
| `batch_size` | Minibatch size. Start with the default. |
| `context_length` | Trajectory length visible to the policy. This also affects the exported bundle's inference behavior. |
| `seed` | Reproducibility seed. |
| `archive_steps` | Training steps to preserve permanently, on top of the ones the job already keeps. See below. |

## Archive Steps

Every job permanently preserves 5 evenly spaced checkpoints across the run. `archive_steps` adds steps of your own to that set. Preserved checkpoints are never rotated away, so you can score them in your own environment after the job ends. See `training/docs/aws/sagemaker-checkpoints.md`.

- Pass a comma-separated string of steps: `"25000, 75000"`.
- At most 10 steps, each within `0 < step <= max_steps`.
- The limit is applied before duplicates are removed, so listing the same step 11 times fails.
- A step outside the range fails the job at startup. It is not skipped silently, because a silently dropped step means waiting for an output that will never arrive.

### Disk Cost

Each preserved point costs about 5x the model parameter size — 4x training state (model, two AdamW moments, target network) plus 1x inference bundle — and is never reclaimed.

| Model | Per point | 5 evenly spaced + 10 preserved |
| --- | --- | --- |
| Default (~5M parameters) | ~100MB | ~1.5GB |
| 80M parameters | ~1.6GB | ~24GB |

The SageMaker `VolumeSizeInGB` default of 30GB also holds your dataset and the container. The startup log reports the estimate against actual free space:

```text
Archive disk estimate: 3.3 MB for 5 point(s) (free: 322.8 GB)
```

This is a report, not a limit. A schedule that does not fit produces a startup warning and the job still runs, then fails later when the volume fills. Size the volume for the schedule you asked for.

## Example

```python
hyperparameters = {
    "dataset_ids": "mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0",
    "max_steps": "100000",
    "batch_size": "128",
    "context_length": "32",
    "seed": "42",
    "archive_steps": "25000, 75000",
}
```

## Short Validation Run

For a product smoke test, reduce `max_steps`.

```python
hyperparameters = {
    "dataset_ids": "mujoco/humanoid/simple-v0",
    "max_steps": "1600",
}
```

This setting is for smoke testing only. It is not a final training configuration for quality evaluation.


