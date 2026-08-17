# Training

This directory holds the hosted-training input definitions, and the docs for
both what the product is and how to run it as a managed job.

The local trainer implementation is not part of this repository.

## How it fits together

Your dataset's observation and action spaces define the model's I/O schema: the
same spaces are turned into a `DataSpec` schema at build-time, which fixes the
autoregressive token layout the model consumes and produces at inference.

![Causal GPT-RL — your dataset spaces define the model](docs/assets/dataset-spaces-define-the-model.svg)

Our training service is designed to generalize across tasks by drawing on the
capabilities already present in transformer models, rather than relying on
task-specific tuning. We publish untuned bundle results, and they suggest that,
once that bridge is in place, the same training recipe can work across very
different control problems without task-by-task tuning.

## Hyperparameters

`hyperparameters.py` contains the training job payload schema. Hosted-training
quickstarts should import it instead of duplicating the field list:

```python
from training import Hyperparameters

hp = Hyperparameters()
hp.set_config(
    dataset_ids=["mujoco/walker2d/simple-v0"],
    max_steps=100_000,
)

training_hyperparameters = hp.to_dict()
```

`dataset_ids` is the only field you have to supply. Everything else arrives
already set, and those settings are what every published bundle was trained with
— see [Default Hyperparameters](default-hyperparameters.md).

We believe that is the more useful direction for the product: not eliminating
every reason to tune, but making tuning an optional optimization step rather
than a prerequisite for training a new task.

## Journey

[`training/docs/journey/`](docs/journey/README.md) is a four-part read on what
the product is and how you steer it — the data contract, how the model runs,
steering it through state, and steering it in natural language. Start there if
you are new to Causal GPT-RL.

## AWS/SageMaker Docs

[`training/docs/aws/`](docs/aws/README.md) covers the managed path end to end —
running a job, the dataset channel and hyperparameters, delivered bundles and
the output artifact, the checkpoint-selection metric, and resuming a run. Only
the dataset you upload crosses to hosted training; what comes back is yours to
run, and to continue training from.

## Hosted Training

Hosted training is available on AWS Marketplace as the CCNets Causal GPT-RL
Training Algorithm:

https://aws.amazon.com/marketplace/pp/prodview-is6jt3bcwkq5c

Subscribe there, then follow `training/docs/aws/aws-marketplace-training.md`.
Take the Algorithm ARN from the listing rather than hard-coding one here, and
keep pricing and EULA text out of this repository — the listing is the only
place those are current.
