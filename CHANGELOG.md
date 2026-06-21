# Changelog

## 0.7.0

- Added structured action output support: `Dict` / `Tuple` action spaces are now
  reconstructed into their gym containers on decode via `gym.spaces.unflatten`
  (`ActionOutputAdapter` / `make_action_output_adapter`), with self-describing
  errors for unsupported spaces.
- Added the `action_container` capability for bundles whose declared action space
  is a structured container.
- Added `start` offset handling for `Discrete` and `MultiDiscrete` action spaces
  on decode, so bare and container forms agree.
- Fixed a buffer-aliasing bug in the input/output adapters that could make
  safetensors refuse to save shared-memory tensors (clone on bind; values and
  bytes unchanged).

## 0.2.0

- Added `std_scale` control for continuous action sampling.
- Updated default model architecture hyperparameters.
- Added optional `env_id` metadata to exported bundle configs.

## 0.1.0

- Initial inference runtime package.
- Added bundle loading, policy runner, MuJoCo deployment example, and focused
  runtime tests.
