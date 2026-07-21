# Export a delivered bundle to ONNX

Causal GPT-RL training delivers a complete inference bundle:

```text
bundle/
  config.json
  model.safetensors
  state_normalizer.safetensors  # legacy bundle formats only
```

Export the directory rather than passing `model.safetensors` alone. The config
contains the architecture, observation/action contracts, context length, and
normalization metadata required to reconstruct the policy safely.

## Install

```bash
python -m pip install "causal-gpt-rl[onnx]"
```

## Command line

```bash
causal-gpt-rl-export-onnx \
    --bundle path/to/bundle \
    --out policy.onnx \
    --batch-size 1
```

The equivalent module command is available when shell entry points are not on
`PATH`:

```bash
python -m causal_gpt_rl.export.onnx \
    --bundle path/to/bundle \
    --out policy.onnx \
    --batch-size 1
```

`--batch-size` is the number of independent agent contexts evaluated by one
ONNX call. It is fixed in the exported graph and means “agents controlled by
this policy,” not necessarily every agent present in the environment.

| Deployment | Batch size |
|---|---:|
| One general-purpose agent call | 1 |
| DungeonEscape, 12 arenas × 3 controlled agents | 36 |
| SoccerTwos, 8 fields × 2 Causal-controlled teammates | 16 |

The output is one self-contained ONNX file. Raw observations are inputs because
bundle state normalization is embedded in the graph. Continuous, discrete,
MultiDiscrete, structured observation, and hybrid-action bundles use the same
export command.

By default the exporter:

1. reconstructs the model from the bundle;
2. creates a real window using the bundle's declared observation structure;
3. exports a fixed batch and the bundle's context length;
4. folds external weight data into one ONNX file;
5. runs `onnx.checker`;
6. compares ONNX Runtime output with the PyTorch model and fails when the
   maximum absolute error is not below `1e-4`.

Use `--no-verify` only when ONNX Runtime cannot run on the export machine. The
checker still runs, but deployment should not proceed until the graph has been
validated in an ONNX runtime.

## Python API

```python
from causal_gpt_rl.export import export_onnx

result = export_onnx(
    "path/to/bundle",
    "policy.onnx",
    batch_size=16,
)
print(result)
```

## ONNX contract

The dimensions are derived from the bundle:

```text
states   [B, T, state_size]
actions  [B, T, action_size]
is_bos   [B, T, 1]
mask     [B, T]
action   [B, action_size]
```

`T` is the bundle context length. Discrete action heads produce logits; the
consumer selects one index per branch and feeds its one-hot representation back
into the action context. See the Unity evaluators for complete continuous,
Discrete, MultiDiscrete, and hybrid decoding examples.
