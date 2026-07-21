# SageMaker Input Datasets

Causal GPT-RL training input is Minari-based. Upload Minari dataset directories to S3, then select which datasets to train on with `dataset_ids`.

## Input Channel

The training job uses one SageMaker input channel named `training`.

```python
estimator.fit({
    "training": "s3://my-bucket/cgrl/datasets/minari/farama/"
})
```

This S3 prefix is the dataset root. `dataset_ids` are resolved relative to this root.

## Dataset Layout

Example S3 layout:

```text
s3://my-bucket/cgrl/datasets/minari/farama/
  mujoco/
    humanoid/
      simple-v0/
      medium-v0/
```

With this layout:

```text
training channel = s3://my-bucket/cgrl/datasets/minari/farama/
dataset_ids     = mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0
```

The training job resolves those ids to:

```text
s3://my-bucket/cgrl/datasets/minari/farama/mujoco/humanoid/simple-v0/
s3://my-bucket/cgrl/datasets/minari/farama/mujoco/humanoid/medium-v0/
```

## Required Inputs

- `training` channel: S3 root that contains the Minari dataset directories.
- `dataset_ids`: Dataset paths relative to that root.

## Validation Summary at Startup

When the job starts, it prints a validation summary to CloudWatch Logs confirming how each dataset was resolved and the observation/action schema the model will use, including the flattened observation/action shapes and the state/action specs. Use it to verify the data was read as intended before the run continues. See `training/docs/aws/aws-marketplace-training.md` for the full list of fields shown.



