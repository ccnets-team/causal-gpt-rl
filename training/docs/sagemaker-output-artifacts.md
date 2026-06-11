# SageMaker Output Artifacts

A Causal GPT-RL SageMaker training job writes its final model artifact to the configured S3 output path.

## SageMaker Output

When training finishes, SageMaker stores the model artifact as `model.tar.gz`.

```text
s3://my-bucket/cgrl/output/<training-job-name>/output/model.tar.gz
```

## Artifact Layout

After extracting `model.tar.gz`, find the canonical `bundle/` directory. Intermediate snapshot bundles may also be present under `snapshots/` so saved policies can be loaded without restoring training checkpoints.

```text
model.tar.gz
  reports/
    summary.json
  <run-name>/
    bundle/
      model.safetensors
      config.json
      state_normalizer.safetensors
    snapshots/
      manifest.json
      slot_000/
        model.safetensors
        config.json
        state_normalizer.safetensors
        metrics.json
      ...
```

## Bundle Files

- `model.safetensors`: Policy model weights.
- `config.json`: Model architecture, observation/action specs, and context settings.
- `state_normalizer.safetensors`: State normalization statistics required for inference.
- `metrics.json`: Snapshot metrics, present only inside `snapshots/slot_NNN/`.

## Load Example

Use the canonical `bundle/` path for normal inference.

```python
from causal_gpt_rl.inference import load_runner

runner = load_runner("path/to/bundle")
```


