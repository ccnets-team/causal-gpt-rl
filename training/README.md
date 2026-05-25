# Training

This directory contains training-job input definitions used while preparing AWS
Marketplace/SageMaker and Replicate hosted training paths.

The local trainer implementation is not part of this repository.

## Hyperparameters

`hyperparameters.py` contains the training job payload schema. Examples and
future hosted-training quickstarts should import it instead of duplicating the
field list:

```python
from training import Hyperparameters

hp = Hyperparameters()
hp.set_config(
    data_source="byo",
    dataset_ids=["my-org/hopper-medium-v5"],
    env_id="Hopper-v5",
)

training_hyperparameters = hp.to_dict()
```

## Hosted Training Status

AWS Marketplace/SageMaker and Replicate training entrypoints are being prepared
but are not live public products yet. Until they are published, keep listing
links, pricing, EULA text, default SageMaker Algorithm ARNs, and Replicate
model/version IDs out of public examples.

When either hosted path is ready, its quickstart should use this interface to
submit training hyperparameters, then load the exported bundle with the runtime
inference APIs.
