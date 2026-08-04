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
    "best_metric_name": "offline_eval/checkpoint_score",
    "best_metric_value": 0.418732,
    "best_metric_direction": "max",
    "best_return": null
  }
}
```

- `best_metric_name` — the metric that selected the canonical bundle.
- `best_metric_value` — its value at the selected point. It ranks points inside
  one run. The offline criterion is revised as it is tuned, so do not compare it
  across runs.
- `best_metric_direction` — `min` or `max`. It is carried as its own field
  because the direction is not part of the metric name. **Read it before ranking
  runs against each other.** `offline_eval/checkpoint_score` is a bounded
  `[0, 1]` score, so `max` wins: the run with the *higher* value scored better.
  Sorting the wrong way picks the worse model.
- `best_return` — filled in only when the selection metric is an actual episode
  return, which requires environment evaluation during training. On an
  offline-selected run it is `null` — absent, not zero. Do not read `null` as a
  score of 0; on that path the selected point is described by
  `best_metric_name` and `best_metric_value`.

This block says which checkpoint scored best on the held-out selection
criterion — see `training/docs/aws/checkpoint-score.md`. It is not episode
return, and it is not a statement about task performance. To find the point your
own environment prefers, score the preserved archive bundles — see
`training/docs/aws/sagemaker-realtime-policy-delivery.md`.

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


