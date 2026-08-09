# AWS Marketplace Training

This document describes the minimum steps needed to run a Causal GPT-RL SageMaker training job after subscribing through AWS Marketplace.

The product listing carries the current version and its Algorithm ARN:

https://aws.amazon.com/marketplace/pp/prodview-is6jt3bcwkq5c

## Purpose

A Causal GPT-RL training job takes user-provided offline trajectory datasets and produces a policy bundle from those datasets. SageMaker stores the training result as a model artifact in S3.

## Requirements

- AWS Marketplace subscription
- SageMaker execution role with permission to read the input S3 prefix and write to the output S3 prefix
- S3 prefix containing the training data
- S3 output prefix for the model artifact
- SageMaker Algorithm ARN provided through Marketplace
- Checkpoint S3 prefix, if you want policy bundles delivered while the job runs. It must be in the same region as the training job, and the execution role must be able to write to it.

## Basic Flow

1. Upload the training data to S3.
2. Set `dataset_ids` to the dataset ids you want to train on.
3. Create a training job with the SageMaker Algorithm ARN.
4. Monitor training progress in CloudWatch Logs, including the startup validation summary, eval metrics, and checkpoint save progress.
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

The summary is printed as three titled sections.

```text
================ Dataset Configuration ================
Dataset validation: PASSED
Dataset IDs: mujoco/humanoid/simple-v0
Dataset variants: simple-v0
Datasets: 1
Episodes: 1039
Transitions: 999269
Observation space: Box(-inf, inf, (348,), float64)
Action space: Box(-0.4, 0.4, (17,), float32)
Flattened observation shape: (348,)
Flattened action shape: (17,)
State specs: [continuous(size=348)]
Action specs: [continuous(size=17)]
=======================================================
================ Evaluation Configuration ==============
Evaluation mode: offline
Requested env ID: Humanoid-v5
Environment source: disabled
Checkpoint metric: eval_offline/checkpoint_score
Metric direction: max
========================================================
================ Checkpoint Schedule ===================
Archive periodic: 200, 400, 600, 800, 1000
Archive steps: none
Archive disk estimate: 410.1 MB for 5 point(s) (free: 867.3 GB)
========================================================
```

The line to check is the flattened action shape against your action space, because
that is where an encoding mistake shows. They match for a continuous `Box` action
like the one above. They do not match for categorical actions: `MultiDiscrete([3, 3, 3])`
is three environment indices, but the flattened shape is `(9,)` — the sum of the
one-hot blocks.

`Archive periodic` lists the steps preserved for the whole run, derived from
`max_steps` at 20, 40, 60, 80, and 100 percent, plus anything you requested with
`archive_steps`. `Environment source: disabled` is expected for offline training —
`Requested env ID` is metadata about how the dataset was recorded, not an
environment the job launches.

### Training Progress

At the configured logging interval the job reports one progress line:

```text
Training: step=20000 learning_rate=8.7e-05 grad_norm=0.42 raw_grad_norm=0.63 step_time_seconds=0.115
```

| SageMaker metric | Description |
| --- | --- |
| `training:learning_rate` | Current learning rate, after warmup and decay. |
| `training:grad_norm` | Gradient norm for the step, after clipping — the size of the update actually applied. |
| `training:raw_grad_norm` | The same norm before clipping. |
| `training:step_time_seconds` | Average wall-clock seconds per training step over the logging interval. |

`training:step_time_seconds` is the one to watch for cost: multiply it by the
steps remaining to `max_steps` to project how much longer the job will run.

### Checkpoint Progress

The job also reports how many checkpoints it has written so far, on its own line:

```text
Checkpoint: step=20000 checkpoints_saved=7
```

| SageMaker metric | Description |
| --- | --- |
| `eval_offline:checkpoints_saved` | Running count of completed checkpoint saves, across `improvements/` and `archive/` together. |

Graph it to confirm a long run is still producing points. This is job progress
rather than a property of a checkpoint, so it does not appear in `metrics.json`
or the bundle manifest.

### Eval Metrics

The training job evaluates the policy on a held-out portion of the dataset. One of these metrics selects checkpoints — Checkpoint Score — and the rest are diagnostics. All of them are measured directly from held-out data rather than estimated by the model.

Action NLL is the negative log likelihood the model assigns to the dataset's ground-truth actions; lower means the model predicts the dataset actions better. It is reported as an overall value and per-context-length values.

| SageMaker metric | Description |
| --- | --- |
| `eval_offline:checkpoint_score` | Checkpoint-selection metric. Range `[0, 1]`, higher is better. |
| `eval_offline:rollout_action_prob` | The action term of the selection metric on its own. |
| `eval_offline:action_nll` | Representative Action NLL across eval positions within the training context length. |
| `eval_offline:short_context_action_nll` | Positions in the `0`–`0.5x` range of the training context length. |
| `eval_offline:standard_context_action_nll` | Positions in the `0.5`–`1.0x` range. |

`eval_offline:checkpoint_score` is the metric shown in the startup summary. It
decides which checkpoints land in the `improvements/` series and which one
becomes the canonical bundle; it does not affect the archive schedule, which is
fixed by step. See `training/docs/aws/checkpoint-score.md`.

These are the same keys carried inside delivered bundles, written there with a
`/` separator instead of `:`.

None of these metrics is episode return — the job has no environment to measure
it in. Real task performance comes from running a delivered `bundle/` in your own
simulator or game engine, both during the run and to validate the final model.

## Output Bundles

The final `model.tar.gz` contains the canonical `bundle/` for normal inference,
alongside `archive/bundles/` — the run's preserved points, so its candidates can
be compared after the job ends.

Training does not make you wait for it. Policy bundles are exported while the job
runs and synced to the configured checkpoint S3 prefix, so you can load an
in-progress policy with the public runtime, roll it out in your own environment,
and stop a run that is not learning. See
`training/docs/aws/sagemaker-checkpoints.md`.

## More Details

- Datasets and hyperparameters: `training/docs/aws/sagemaker-inputs.md`
- Checkpoints, delivered bundles, and the output artifact: `training/docs/aws/sagemaker-checkpoints.md`
- Checkpoint Score: `training/docs/aws/checkpoint-score.md`
- Retraining: `training/docs/aws/sagemaker-retraining.md`

