# Training

This directory contains hosted-training input definitions and notes for AWS
Marketplace/SageMaker training paths.

The local trainer implementation is not part of this repository.

## How it fits together

Your dataset's observation and action spaces define the model's I/O schema: the
same spaces are turned into a `DataSpec` schema at build-time, which fixes the
autoregressive token layout the model consumes and produces at inference.

![Causal GPT-RL — your dataset spaces define the model](docs/assets/dataset-spaces-define-the-model.svg)

## Hyperparameters

`hyperparameters.py` contains the training job payload schema. Hosted-training
quickstarts should import it instead of duplicating the field list:

```python
from training import Hyperparameters

hp = Hyperparameters()
hp.set_config(
    dataset_ids=["mujoco/humanoid/simple-v0"],
    max_steps=100_000,
)

training_hyperparameters = hp.to_dict()
```

## AWS/SageMaker Docs

AWS Marketplace/SageMaker training notes live under `training/docs/aws/`:

- `training/docs/aws/aws-marketplace-training.md` — running a job, logs, metrics
- `training/docs/aws/sagemaker-inputs.md` — datasets and hyperparameters
- `training/docs/aws/sagemaker-checkpoints.md` — delivered bundles and the output artifact
- `training/docs/aws/checkpoint-score.md` — the checkpoint-selection metric
- `training/docs/aws/sagemaker-retraining.md` — resuming a run

## Hosted Training

Hosted training is available on AWS Marketplace as the CCNets Causal GPT-RL
Training Algorithm:

https://aws.amazon.com/marketplace/pp/prodview-is6jt3bcwkq5c

Subscribe there, then follow `training/docs/aws/aws-marketplace-training.md`.
Take the Algorithm ARN from the listing rather than hard-coding one here, and
keep pricing and EULA text out of this repository — the listing is the only
place those are current.
