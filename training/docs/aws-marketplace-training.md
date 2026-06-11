# AWS Marketplace Training

This document describes the minimum steps needed to run a Causal GPT-RL SageMaker training job after subscribing through AWS Marketplace.

## Purpose

A Causal GPT-RL training job takes user-provided offline trajectory datasets and produces a policy bundle from those datasets. SageMaker stores the training result as a model artifact in S3.

## Requirements

- AWS Marketplace subscription
- SageMaker execution role with permission to read the input S3 prefix and write to the output S3 prefix
- S3 prefix containing the training data
- S3 output prefix for the model artifact
- SageMaker Algorithm ARN provided through Marketplace

## Basic Flow

1. Upload the training data to S3.
2. Set `dataset_ids` to the dataset ids you want to train on.
3. Create a training job with the SageMaker Algorithm ARN.
4. After training finishes, download `model.tar.gz` from the S3 output path.
5. Extract the archive and load the canonical `bundle/` with the `causal_gpt_rl.inference` runtime.

## SageMaker SDK Example

```python
from sagemaker.algorithm import AlgorithmEstimator

algorithm_arn = "<marketplace-algorithm-arn>"
role_arn = "<your-sagemaker-execution-role-arn>"

estimator = AlgorithmEstimator(
    algorithm_arn=algorithm_arn,
    role=role_arn,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path="s3://my-bucket/cgrl/output/",
    hyperparameters={
        "dataset_ids": "mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0",
        "max_steps": "100000",
        "batch_size": "128",
    },
)

estimator.fit({
    "training": "s3://my-bucket/cgrl/datasets/minari/farama/"
})
```

## Recommended Instance

The current Marketplace training example uses a single training instance type:

- Training: `ml.m5.xlarge`

## Output Bundles

The final `model.tar.gz` contains a canonical `bundle/` for normal inference. It may also contain `snapshots/slot_NNN/` directories, which are intermediate policy bundles that can be loaded without restoring training checkpoints.

## More Details

- Input datasets: `training/docs/sagemaker-input-datasets.md`
- Hyperparameters: `training/docs/sagemaker-hyperparameters.md`
- Output artifact: `training/docs/sagemaker-output-artifacts.md`
- Checkpoints: `training/docs/sagemaker-checkpoints.md`
- Retraining: `training/docs/sagemaker-retraining.md`

