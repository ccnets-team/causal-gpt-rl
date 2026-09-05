# Embedded action normalization

Version 0.19.0 adds `action_normalization` to the public
`causal_gpt_rl.SUPPORTED_CAPABILITIES` set. Trainers require
`causal-gpt-rl>=0.19.0` for production exports using this capability.

## Trainer export

Validate the checkpoint coordinate before injecting its statistics. The model
API accepts statistics only; it cannot infer which coordinates they describe.

```python
from causal_gpt_rl import SUPPORTED_CAPABILITIES, export_bundle

coordinate = "bounds_atanh_then_standardize_v1"
if "action_normalization" not in SUPPORTED_CAPABILITIES:
    raise RuntimeError("Runtime lacks action_normalization")
if checkpoint["action_normalizer_info"]["coordinate_system"] != coordinate:
    raise ValueError("Unsupported action normalization coordinate")

model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.set_action_normalization_from_state_dict(
    checkpoint["action_normalizer_state"], enabled=True,
)
export_bundle(
    bundle_dir, model=model, model_config=model_config,
    state_specs=state_specs, action_specs=action_specs,
    context_length=context_length, state_normalizer=state_normalizer,
    obs_space=obs_space, action_space=action_space,
)
```

`model_state_dict` above denotes the trainer's model weights entry; adapt that
key to its checkpoint schema. Inject action statistics **after** loading weights.
Keep the existing state normalization export contract.

## Model API and state

- `model.has_embedded_action_normalizer() -> bool`
- `model.set_action_normalization(*, mean, std=None, var=None, enabled=True)`
- `model.set_action_normalization_from_state_dict(state_dict, *, enabled=True)`

Provide exactly one of `std` and `var` (`std = sqrt(var)`, without epsilon).
Statistics must have the entire flat mean-action feedback width in declared
action-head order. Continuous positions contain pre-tanh statistics; categorical,
MultiDiscrete and MultiBinary positions require `mean=0`, `std=1`.
All values must be finite and every standard deviation must be positive.
Enabled normalization requires finite, strictly increasing per-dimension bounds
from the action specs. Setters validate before changing state.

Three persistent float32 buffers hold the state: `action_normalization_enabled`
with shape `[1]`, and `action_normalization_mean` / `action_normalization_std`
with shape `[action_size]`. Their initial values are 0, 0, 1 respectively.
Direct `load_state_dict` also validates them. An old checkpoint without all three
gets disabled identity defaults, including when loaded into an enabled model;
partially present state is rejected.

## Inference coordinates

Every model input entry point accepts raw environment action history. Continuous
input uses `x = clamp(2*(a-low)/(high-low)-1, -1+1e-6, 1-1e-6)`,
`u = atanh(x)`, `z = (u-mean)/std`. At BOS the absent continuous action is zero
in model coordinates. An explicitly configured learned BOS prior still applies.

Raw projected continuous heads represent Gaussian mean and log standard deviation
in `z` coordinates. Deterministic means and completed stochastic samples use
`u = z*std+mean`, then `a = low+(tanh(u)+1)*(high-low)/2`.
Normalized paths select these transforms instead of composing them with the
legacy adapters. Discrete head representations stay unchanged. The runtime has
no NLL/log-probability API and therefore adds no density/Jacobian API.

The model returns flat environment-coordinate continuous heads. Runner decode
owns final clipping and Dict/Tuple reconstruction. With normalization enabled,
feedback stores the same clipped raw continuous action sent to the environment,
alongside the existing one-hot categorical and binary representations. Disabled
models keep the original decode/feedback behavior.

Windowed, incremental, cached-prefix inference and the ONNX wrapper share these
model transforms. Tensor selection by the enabled buffer avoids `Tensor.item()`
inside the exported graph and requires no mutable adapter flags. ONNX contains
the statistics and transforms, takes raw action histories and emits flat action
heads; the consuming environment integration retains final clip/container decode.

## Bundle contract

Enabled models automatically add this config metadata:

```json
{
  "requires_capabilities": ["action_normalization"],
  "action_normalization": {
    "embedded": true,
    "coordinate": "bounds_atanh_then_standardize_v1"
  }
}
```

Other required capabilities can coexist in the list. Statistics live in
`model.safetensors`; there is no action-normalizer sidecar or layout version bump.
Enabled state, required capability and metadata must agree. Missing statistics or
coordinate, invalid std and unsupported coordinates fail at load; the discarded
`bounds_then_standardize_v1` name is not an alias. Disabled bundles omit the
action-normalization metadata and requirement.

`tests/test_action_normalization.py` covers numeric references, exact bounds,
hybrid heads, BOS, sampling, corrupt bundles, legacy adapter parity, inference
paths and ONNX Runtime max-absolute error below `1e-4`.
