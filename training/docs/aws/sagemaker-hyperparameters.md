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

## Example

```python
hyperparameters = {
    "dataset_ids": "mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0",
    "max_steps": "100000",
    "batch_size": "128",
    "context_length": "32",
    "seed": "42",
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


