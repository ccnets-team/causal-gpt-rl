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
- Checkpoint S3 prefix, if you want policy bundles delivered while the job runs. The execution role must be able to write to it.

## Basic Flow

1. Upload the training data to S3.
2. Set `dataset_ids` to the dataset ids you want to train on.
3. Create a training job with the SageMaker Algorithm ARN.
4. Monitor training progress in CloudWatch Logs, including the startup validation summary, eval metrics such as Action NLL, and checkpoint save progress.
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

Key items to confirm:

**Dataset Configuration**

- Original observation/action space read from the dataset.
- Flattened observation/action shape the model actually consumes.
- Type, size, and order of the flattened state/action heads.
- Dataset, episode, and transition counts.
- Dataset validation result.

**Evaluation Configuration**

- Evaluation mode and the checkpoint-selection metric.
- `Environment source` — `disabled` for offline training. `Requested env ID` records the environment the dataset was recorded against; it is metadata, not something the training job launches.

**Checkpoint Schedule**

- Which steps are preserved for the whole run, including any you requested with `archive_steps`, and what they cost on disk. `Archive periodic` is derived from `max_steps` at 20, 40, 60, 80, and 100 percent.

The original environment action and the model's flattened action shape are shown together so that action-encoding mistakes are easy to spot. They match for a continuous `Box` action like the one above. They do not match for categorical actions: a `MultiDiscrete([3, 3, 3])` action is three environment indices, but the model's flattened action shape is `(9,)` — the sum of the one-hot blocks. Seeing both values makes an incorrect encoding obvious.

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

The two gradient norms are read together: `training:raw_grad_norm` is what the
step asked for, `training:grad_norm` is what it got. While they track each other,
clipping is idle. A persistent gap means clipping is engaging on most steps and
is the mechanism holding the update size down.

`training:step_time_seconds` is the one to watch for cost. Multiply it by the
steps remaining to `max_steps` to project how much longer the job will run, and
compare that against what the run has produced so far.

### Checkpoint Progress

The job also reports how many checkpoints it has written so far, on its own line:

```text
Checkpoint: step=20000 checkpoints_saved=7
```

| SageMaker metric | Description |
| --- | --- |
| `eval_offline:checkpoints_saved` | Running count of completed checkpoint saves, across `improvements/` and `archive/` together. |

The count increments only after a save has been processed, and the line for a
step already includes that step's save. Unlike the eval metrics below, this is
job progress rather than a property of a checkpoint, so it does not appear in
`metrics.json` or the bundle manifest. Graph it to confirm a long run is still
producing points. See `training/docs/aws/sagemaker-checkpoints.md` for what the
two series are and when each is written.

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

The same metrics appear inside delivered bundles with a `/` separator instead of
`:` — `eval_offline/checkpoint_score`. The `:` form is the SageMaker metric name
you select in the console; the `/` form is the key inside `metrics.json` and
`manifest.json`. Same metric, different surface.

To keep results comparable across runs, the service evaluates at a standard Short `0.5x` and Long `2.0x` context; no user configuration is required.

`eval_offline:checkpoint_score` is the metric shown in the startup summary (`Checkpoint metric: eval_offline/checkpoint_score`, `Metric direction: max`): higher values rank as better checkpoints. It decides which checkpoints land in the `improvements/` series and which one becomes the canonical bundle. It does not affect the archive schedule, which is fixed by step. See `training/docs/aws/checkpoint-score.md` for what it measures and how to read it.

### Forecast Metrics (Planned)

Forecast metrics would give an approximate view of how the current policy may
behave without running the target simulator, game engine, or environment inside
the training container. Three are planned:

| Planned metric | Description |
| --- | --- |
| Estimated step reward | Average reward per environment step. |
| Estimated episode length | Average episode length in environment steps. |
| Estimated episode return | Average episode return, or total episode score. |

**None of them is emitted by the current version.** No `forecast:` metric is
registered with SageMaker, and no forecast line appears in a job's logs or
dashboard. They are experimental and are not exposed until validated; there is
nothing here to configure, watch for, or build tooling against.

When they do arrive, they will be model-based estimates rather than rollout
scores measured in your simulator or game engine, and improving one will not by
itself guarantee improved real-environment performance.

Until then, real task performance comes from running a delivered `bundle/` in
your own environment while the job runs — see
`training/docs/aws/sagemaker-checkpoints.md`. Final performance
should be validated the same way, in the customer’s actual simulator, game
engine, or evaluation environment.

## Recommended Instance

The current Marketplace training example uses a single training instance type:

- Training: `ml.g5.xlarge`

## Output Bundles

The final `model.tar.gz` contains the canonical `bundle/` for normal inference,
alongside `archive/bundles/` — the run's preserved points, so its candidates can
be compared after the job ends.

Training does not make you wait for it. Policy bundles are exported while the
job runs and synced to the configured checkpoint S3 prefix, so you can load an
in-progress policy with the public runtime and roll it out in your own
environment. Because the job cannot measure episode return offline, this is the
only way to see real task performance before a run finishes — and the way to
stop a run that is not learning.

Bundles arrive in two series. `archive/` holds an even sample of the run plus
any steps you requested, kept permanently; it is the series to score when you
are deciding whether to let a run continue. `improvements/` holds the best-so-far
track by the offline metric, in 5 rotating slots.

See `training/docs/aws/sagemaker-checkpoints.md`.

## More Details

- Datasets and hyperparameters: `training/docs/aws/sagemaker-inputs.md`
- Checkpoints, delivered bundles, and the output artifact: `training/docs/aws/sagemaker-checkpoints.md`
- Checkpoint Score: `training/docs/aws/checkpoint-score.md`
- Retraining: `training/docs/aws/sagemaker-retraining.md`

