# SageMaker Output Artifacts

A Causal GPT-RL SageMaker training job writes its final model artifact to the configured S3 output path.

## SageMaker Output

When training finishes, SageMaker stores the model artifact as `model.tar.gz`.

```text
s3://my-bucket/cgrl/output/<training-job-name>/output/model.tar.gz
```

## Artifact Layout

After extracting `model.tar.gz`, find the canonical `bundle/` directory. The
final artifact does not contain the intermediate bundles; those are live-synced
separately through the configured SageMaker checkpoint S3 prefix.

```text
model.tar.gz
  reports/
    summary.json
  <namespace>/
    bundle/
      model.safetensors
      config.json
```

## `reports/summary.json`

`summary.json` records how the canonical bundle was selected.

```json
{
  "evaluation": {
    "best_metric_name": "offline_eval/action_nll",
    "best_metric_value": 0.055177,
    "best_metric_direction": "min",
    "best_return": null
  }
}
```

- `best_metric_name` — the metric that selected the canonical bundle.
- `best_metric_value` — its value at the selected point.
- `best_metric_direction` — `min` or `max`. **Read it before ranking runs
  against each other.** `offline_eval/action_nll` is a negative log likelihood,
  so `min` wins: the run with the *lower* value scored better. Sorting the wrong
  way picks the worse model.
- `best_return` — filled in only when the selection metric is an actual episode
  return (`direction: max`). On an offline-selected run it is `null` — absent,
  not zero. Do not read `null` as a score of 0.

This block says which checkpoint predicted your dataset's actions best. It is
not episode return, and it is not a statement about task performance. To find
the point your own environment prefers, score the preserved archive bundles —
see `training/docs/aws/sagemaker-realtime-policy-delivery.md`.

## Bundle Files

- `model.safetensors`: Policy model weights.
- `config.json`: Model architecture, observation/action specs, and context settings.
- `state_normalizer.safetensors`: Optional legacy sidecar. Current bundle format
  v2 embeds state normalization statistics in `model.safetensors`.

Intermediate bundles and their `metrics.json` files live under the checkpoint
prefix, in `archive/bundles/step_NNNNNNN/` and `improvements/bundles/slot_NNN/`.
See `training/docs/aws/sagemaker-checkpoints.md`.

## Load Example

Use the canonical `bundle/` path for normal inference.

```python
from causal_gpt_rl.inference import load_runner

runner = load_runner("path/to/bundle")
```


