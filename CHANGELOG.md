# Changelog

## 0.9.0

- Added `PolicyRunner.reset_rows(done_mask)` for per-env episode restarts in
  batched inference (`num_envs > 1`). When one env terminates mid-batch, its
  rolling context is wiped and re-seeded as a fresh episode on the next
  `observe`/`act`, while the other envs keep their history and continue
  uninterrupted — no full-batch restart. The shared KV cache is invalidated and
  recomputed from the buffer once, so surviving rows never lose context.
- `ContextBuffer.update_data` now accepts a per-agent `is_bos` vector (in
  addition to the scalar form, which stays byte-identical) so a subset of rows
  can start a fresh episode within a single batched step.

## 0.8.0

- Added MultiBinary action/observation support (independent Bernoulli per
  element). `MultiBinary(n)` spaces — bare or as Dict/Tuple leaves — now
  round-trip through the bundle and decode to their {0,1} n-vector (head
  logits thresholded at 0). This closes the last fixed-shape gymnasium space;
  Text, Sequence and Graph stay out of scope as variable-length / structural.

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
