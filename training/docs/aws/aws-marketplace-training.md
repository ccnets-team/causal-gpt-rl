# AWS Marketplace Training

Product version: `0.0.10`

> **Not published yet.** This version is not live on AWS Marketplace. What
> follows describes what the `0.0.10` training image does, so the steps and
> contracts are usable once the listing is available. Until then there is no
> subscription to take and no Algorithm ARN to use. See "Hosted Training Status"
> in `training/README.md`.

This document describes the minimum steps needed to run a Causal GPT-RL SageMaker training job after subscribing through AWS Marketplace.

## Purpose

A Causal GPT-RL training job takes user-provided offline trajectory datasets and produces a policy bundle from those datasets. SageMaker stores the training result as a model artifact in S3.

## Requirements

- AWS Marketplace subscription
- SageMaker execution role with permission to read the input S3 prefix and write to the output S3 prefix
- S3 prefix containing the training data
- S3 output prefix for the model artifact
- SageMaker Algorithm ARN provided through Marketplace
- Checkpoint S3 prefix, if you want policy bundles delivered while the job runs. The execution role must be able to write to it.

## Basic Flow

1. Upload the training data to S3.
2. Set `dataset_ids` to the dataset ids you want to train on.
3. Create a training job with the SageMaker Algorithm ARN.
4. Monitor training progress in CloudWatch Logs, including the startup validation summary, eval metrics such as Action NLL, and optional Forecast metrics.
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
    # Live policy bundles are delivered through the checkpoint prefix. Without
    # it the job produces one model, at the end.
    checkpoint_s3_uri="s3://my-bucket/cgrl/checkpoints/my-job/",
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

### Startup Validation Summary

At the start of a training job, the log prints a validation summary so users can immediately confirm that the training data was read with the intended observation and action schema.

```text
Dataset validation: PASSED
Dataset IDs: unity/soccer-twos/medium-v0
Dataset variants: medium-v0
Datasets: 1
Episodes: 1,024
Transitions: 245,760
Observation space: Box(-inf, inf, (336,), float32)
Action space: MultiDiscrete([3 3 3])
Flattened observation shape: (336,)
Flattened action shape: (9,)
State specs: [continuous(size=336)]
Action specs: [multi_discrete(size=3), multi_discrete(size=3), multi_discrete(size=3)]
Evaluation mode: offline
Checkpoint metric: offline_eval/checkpoint_score
Metric direction: max
Archive periodic: 20000, 40000, 60000, 80000, 100000
Archive steps: none
Archive disk estimate: 500 MB for 5 point(s) (free: 24.3 GB)
```

Key items to confirm:

- Original observation/action space read from the dataset.
- Flattened observation/action shape the model actually consumes.
- Type, size, and order of the flattened state/action heads.
- Dataset, episode, and transition counts.
- Dataset validation result.
- Evaluation mode and the checkpoint-selection metric.
- The archive schedule: which steps are preserved for the whole run, including any you requested with `archive_steps`, and what they cost on disk.

The original environment action and the model's flattened action shape are shown together so that action-encoding mistakes are easy to spot. For example, a `MultiDiscrete([3, 3, 3])` action is three environment indices, but the model's flattened action shape is `(9,)` — the sum of the one-hot blocks. Seeing both values makes an incorrect encoding obvious.

### Training Progress

At the configured logging interval the job reports one progress line:

```text
Training: step=20000 learning_rate=8.7e-05 grad_norm=0.42 step_time_seconds=0.115
```

| SageMaker metric | Description |
| --- | --- |
| `training:learning_rate` | Current learning rate, after warmup and decay. |
| `training:grad_norm` | Gradient norm for the step, before clipping. |
| `training:step_time_seconds` | Average wall-clock seconds per training step over the logging interval. |

`training:step_time_seconds` is the one to watch for cost. Multiply it by the
steps remaining to `max_steps` to project how much longer the job will run, and
compare that against what the run has produced so far.

### Eval Metrics

The training job evaluates the policy on a held-out portion of the dataset. One of these metrics selects checkpoints — Checkpoint Score — and the rest are diagnostics. Unlike the Forecast metrics below, all of them are measured directly from held-out data rather than estimated by the model.

Action NLL is the negative log likelihood the model assigns to the dataset's ground-truth actions; lower means the model predicts the dataset actions better. It is reported as an overall value and per-context-length values.

| SageMaker metric | Description |
| --- | --- |
| `offline_eval:checkpoint_score` | Checkpoint-selection metric. Range `[0, 1]`, higher is better. |
| `offline_eval:rollout_action_prob` | The action term of the selection metric on its own. |
| `offline_eval:action_nll` | Representative Action NLL across eval positions within the training context length. |
| `offline_eval:short_context_action_nll` | Positions in the `0`–`0.5x` range of the training context length. |
| `offline_eval:standard_context_action_nll` | Positions in the `0.5`–`1.0x` range. |
| `offline_eval:long_context_action_nll` | Positions beyond the training context length, `1.0x` and above. |

The same metrics appear inside delivered bundles with a `/` separator instead of
`:` — `offline_eval/checkpoint_score`. The `:` form is the SageMaker metric name
you select in the console; the `/` form is the key inside `metrics.json` and
`manifest.json`. Same metric, different surface.

To keep results comparable across runs, the service evaluates at a standard Short `0.5x` and Long `2.0x` context; no user configuration is required.

`offline_eval:checkpoint_score` is the metric shown in the startup summary (`Checkpoint metric: offline_eval/checkpoint_score`, `Metric direction: max`): higher values rank as better checkpoints. It decides which checkpoints land in the `improvements/` series and which one becomes the canonical bundle. It does not affect the archive schedule, which is fixed by step. See `training/docs/aws/checkpoint-score.md` for what it measures and how to read it.

Delivered bundles also record `offline_eval/value_loss` and
`offline_eval/policy_loss` in `metrics.json`. These are internal training
diagnostics. They depend on your dataset and reward scale, so they are not
comparable across runs and must not be used to rank checkpoints or select a
model. Include them when contacting support.

### Forecast Metrics

Forecast metrics are not available in the current version yet; they will be enabled soon.

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

The final `model.tar.gz` contains a canonical `<namespace>/bundle/` for normal
inference.

Training does not make you wait for it. Policy bundles are exported while the
job runs and synced to the configured checkpoint S3 prefix, so you can load an
in-progress policy with the public runtime and roll it out in your own
environment. Because the job cannot measure episode return offline, this is the
only way to see real task performance before a run finishes — and the way to
stop a run that is not learning.

Bundles arrive in two series. `archive/` holds an even sample of the run plus
any steps you requested, kept permanently; it is the series to score when you
are deciding whether to let a run continue. `improvements/` holds the best-so-far
track by the offline metric, in 10 rotating slots.

See `training/docs/aws/sagemaker-realtime-policy-delivery.md`.

## More Details

- Input datasets: `training/docs/aws/sagemaker-input-datasets.md`
- Hyperparameters: `training/docs/aws/sagemaker-hyperparameters.md`
- Output artifact: `training/docs/aws/sagemaker-output-artifacts.md`
- Real-time policy delivery: `training/docs/aws/sagemaker-realtime-policy-delivery.md`
- Checkpoints: `training/docs/aws/sagemaker-checkpoints.md`
- Retraining: `training/docs/aws/sagemaker-retraining.md`
    
