# SageMaker Output Artifacts

A Causal GPT-RL SageMaker training job writes its final model artifact to the configured S3 output path.

## SageMaker Output

When training finishes, SageMaker stores the model artifact as `model.tar.gz`.

```text
s3://my-bucket/cgrl/output/<training-job-name>/output/model.tar.gz
```

## Artifact Layout

After extracting `model.tar.gz`, find the canonical `bundle/` directory. The
final artifact does not contain intermediate snapshots; those are live-synced
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

## Bundle Files

- `model.safetensors`: Policy model weights.
- `config.json`: Model architecture, observation/action specs, and context settings.
- `state_normalizer.safetensors`: Optional legacy sidecar. Current bundle format
  v2 embeds state normalization statistics in `model.safetensors`.

Intermediate `snapshots/slot_NNN/` bundles and their `metrics.json` files live
under the checkpoint prefix. See `training/docs/aws/sagemaker-checkpoints.md`.

## Load Example

Use the canonical `bundle/` path for normal inference.

```python
from causal_gpt_rl.inference import load_runner

runner = load_runner("path/to/bundle")
```


