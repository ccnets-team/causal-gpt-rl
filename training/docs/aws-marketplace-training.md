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
4. Monitor training progress in CloudWatch Logs, including offline training metrics and optional Forecast metrics.
5. After training finishes, download `model.tar.gz` from the S3 output path.
6. Extract the archive and load the canonical `bundle/` with the `causal_gpt_rl.inference` runtime.

## SageMaker SDK Example

```python
from sagemaker.algorithm import AlgorithmEstimator

algorithm_arn = "<marketplace-algorithm-arn>"
role_arn = "<your-sagemaker-execution-role-arn>"

estimator = AlgorithmEstimator(
    algorithm_arn=algorithm_arn,
    role=role_arn,
    instance_count=1,
    instance_type="ml.g5.xlarge",
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

## Monitoring Training Logs

During training, users can monitor progress through Amazon CloudWatch Logs connected to the SageMaker training job.

In addition to standard offline training metrics, the training job may emit Forecast metrics that provide an approximate view of how the current policy may behave without directly running the target simulator, game engine, or environment inside the training container.

| Metric | Description |
| --- | --- |
| `Forecast/StepReward` | Estimated average reward per environment step. |
| `Forecast/EpisodeLength` | Estimated average episode length in environment steps. |
| `Forecast/EpisodeReturn` | Estimated average episode return, or total episode score. |

Forecast metrics are model-based estimates generated during training. They are not rollout scores measured from the actual simulator or game engine. They can help users roughly understand the training direction when live environment evaluation is unavailable, but the values may be inaccurate.

Final performance should be validated by running the exported `bundle/` in the customer’s actual simulator, game engine, or evaluation environment.

### Interpreting Forecast Metrics

`Forecast/StepReward` estimates the average reward the current policy may receive at each step, based on reward information in the training dataset. If the value trends upward, it may indicate that the policy is learning better actions. However, it should not be interpreted as an absolute environment score.

`Forecast/EpisodeLength` estimates how many steps the current policy may continue within an episode. This value is estimated using the model’s EOS, or episode termination, output. Depending on the task, a longer episode may or may not be better. For example, in control tasks with early failure conditions, longer episodes may be a positive signal. In tasks where shorter completion is preferred, the value should be interpreted differently.

`Forecast/EpisodeReturn` combines the estimated step reward and estimated episode length into a reference total score. It is useful for quickly checking the current model state during training, but it should not be used as a leaderboard score or guaranteed performance value.

### How to View Forecast Metrics

Open the training job in the SageMaker Console and choose `View logs`, or open the associated CloudWatch log stream directly. If metrics beginning with `Forecast/` appear in the log or dashboard, those entries are Forecast metrics.

Example:

```text
Forecast/StepReward: 1.24
Forecast/EpisodeLength: 730
Forecast/EpisodeReturn: 905.2
```

This example means that the model currently estimates an episode length of approximately 730 steps and an episode return of approximately 905 for the current policy. The actual result may differ when the policy is run in the real environment.

### Notes and Limitations

Forecast metrics can be unstable early in training. Before the model and dataset statistics become stable, the values may fluctuate sharply or may not appear.

Forecast metrics are difficult to compare directly across different datasets, reward scales, or environment settings. They are safest to use for comparing trends across repeated runs with the same configuration.

Improving Forecast metrics does not guarantee improved real environment performance. Final evaluation should be based on the exported `bundle/` running in the actual target environment.

### If Forecast Metrics Do Not Appear

Forecast metrics may not appear in the following cases:

- The training job has not yet initialized the required data statistics.
- The model does not provide the output needed for termination or episode length estimation.
- The current batch does not contain valid prediction positions.
- Invalid or non-finite values are detected and the metric is skipped from logging.

If Forecast metrics do not appear, continue training and check the training logs for warnings or errors.

## Recommended Instance

The current Marketplace training example uses a single training instance type:

- Training: `ml.g5.xlarge`

## Output Bundles

The final `model.tar.gz` contains a canonical `bundle/` for normal inference. It may also contain `snapshots/slot_NNN/` directories, which are intermediate policy bundles that can be loaded without restoring training checkpoints.

## More Details

- Input datasets: `training/docs/sagemaker-input-datasets.md`
- Hyperparameters: `training/docs/sagemaker-hyperparameters.md`
- Output artifact: `training/docs/sagemaker-output-artifacts.md`
- Checkpoints: `training/docs/sagemaker-checkpoints.md`
- Retraining: `training/docs/sagemaker-retraining.md`
