# Changelog

## Unreleased

- **Breaking (customer tooling):** renamed the offline-evaluation metric
  namespace from `offline_eval` to `eval_offline` on every surface — the `/`
  form in `metrics.json`, `manifest.json`, and checkpoint metadata, and the `:`
  form registered as a SageMaker metric name. `offline_eval/checkpoint_score`
  becomes `eval_offline/checkpoint_score`, and so on for all keys in the
  contract. The old namespace is deliberately **not** kept as an alias:
  accepting both names would let the trainer and this repository drift and could
  silently select or display a stale metric. Tooling that reads these keys must
  be updated in the same release. No serving runtime code reads them, so this is
  a documentation and downstream-tooling change only.
- Documented `eval_offline/rollout_advantage_prob`, a new diagnostic key in the
  metric contract. It is the mean value-relative weight over the evaluated
  positions, bounded to `[0, 1]`, and distinguishes weight spread broadly across
  positions from weight concentrated on a few. It is a diagnostic, not a second
  selection criterion — `eval_offline/checkpoint_score` still selects.
- Documented `training:raw_grad_norm`, the step's gradient norm before clipping,
  now reported on the training progress line alongside `training:grad_norm`.
  **Corrected in the same pass:** `training:grad_norm` was described as the norm
  *before* clipping. It is the post-clip norm — the size of the update actually
  applied — and the pre-clip value is the new metric.
- Documented `eval_offline:checkpoints_saved`, a running count of completed
  checkpoint saves across `improvements/` and `archive/`, reported as
  `Checkpoint: step=<step> checkpoints_saved=<count>`. It is job progress rather
  than a property of a checkpoint, so it is registered as a SageMaker metric but
  is not written to `metrics.json` or the bundle manifest.
- Corrected the Forecast metrics documentation. `forecast:step_reward` is now
  emitted by Marketplace training jobs, on the progress-log cadence as
  `Forecast: step=<n> step_reward=<value>`; the docs previously said no forecast
  metric was available. Estimated episode length and episode return are **not**
  emitted by the current version and are now described as planned, replacing
  example log output and troubleshooting text that implied a running job
  produces them.

## 0.15.0

- Changed the default cached-inference KV retention from `4 × context_length`
  to `context_length` (1×). Omitting `kv_cache_max_len` now keeps a rollout
  inside the window the model was trained on — the setting the published
  benchmark table is measured at and the one the retention sweep names the safe
  default. The sweep covers 0.5×, 1×, and 2×, so retention past the trained
  window is measured, but it is environment-dependent there (at 2× `Walker2d-v5`
  drops from 4122 to 2659 return while `Humanoid-v5` is at its best and
  steadiest); the previous 4× default sat beyond the swept range. Larger values
  remain supported and within the backbone's position capacity (sized at 8× the
  context length), but retaining more history is now an explicit opt-in.
  **Migration:** callers that relied on the old behavior should pass
  `kv_cache_max_len=4 * context_length` (or set `KV_CACHE_MAX_LEN` for the
  serving container) to keep it.
- Added an optional `context_length` to `export_onnx`, and `--context-length` to
  the `causal-gpt-rl-export-onnx` console script, so one bundle can produce ONNX
  policies at several rolling-window lengths. A runtime that hosts a fixed graph
  picks the window at export time rather than at load time, which is what
  publishing per-context variants of the same policy requires. Omitting it keeps
  the previous behavior exactly — the bundle's own context length is used — so
  existing calls are unchanged. Lengths above the bundle's own context are not
  validated, and the backbones diverge there: a GPT-2 bundle raises `IndexError`
  past its baked position capacity, while a Llama bundle exports and verifies at
  any length because RoPE computes positions rather than looking them up — and
  verification compares ONNX against PyTorch, so it cannot flag a window the
  policy was never trained on. Treat the bundle's context length as the
  supported ceiling. Non-positive values are rejected before the bundle is
  loaded.

## 0.14.0

- Added `causal_gpt_rl.export` with `export_onnx` / `OnnxExportResult` and a
  `causal-gpt-rl-export-onnx` console script, turning a bundle into a
  fixed-batch, self-contained ONNX policy for runtimes that cannot host PyTorch
  (Unity ML-Agents / Barracuda / Sentis being the motivating case). The input is
  a complete bundle, not a bare safetensors file: `config.json` supplies the
  architecture, spaces, normalization contract, and context length, so the
  exported graph carries state normalization and the windowed observation →
  action path with it. Export verifies the graph against the source model and
  reports `max_abs_error`.
- Fixed ONNX export to honor the bundle's `bos_cache_mode` discard convention,
  which the exported graph previously ignored — the ONNX policy and the PyTorch
  runner could diverge on the episode-start token.
- Fixed the ONNX exporter's strategy output to be written as UTF-8 rather than
  the console default, which raised on non-UTF-8 consoles (cp949).
- `export_bundle` now stamps the `hybrid_state` capability on bundles whose
  declared observation space is a `Dict` or `Tuple`, not only on those with a
  non-continuous state spec. A structured container whose leaves are all `Box`
  produced uniformly continuous specs and so was gated by nothing, even though
  the caller still passes a container that `gym.flatten` has to collapse. A
  runtime without the input adapter would ignore `state_container`, expect a
  flat array, and fail obscurely instead of refusing with the "upgrade
  causal-gpt-rl" message. This mirrors `action_container` on the output side.
  Export-side only: already published bundles are unchanged and load exactly as
  before, and `bundle_format_version` stays at 2. Bundles exported from this
  release declare the capability, which raises their minimum runtime to one that
  advertises `hybrid_state`.

## 0.13.0

- Added sampling temperature (`std_scale`) to the discrete and multi-binary
  action heads, so `sample_action_from_heads` now scales categorical sampling
  the same way it already scaled continuous Gaussian noise. By the Gumbel-max
  identity, dividing logits by `std_scale` is equivalent to scaling the sampling
  noise, which makes the discrete and continuous knobs mean the same thing.
  `std_scale == 0` is the deterministic mode (argmax, no RNG draw); the default
  of `1.0` is unchanged behavior.

## 0.12.0

- Added `PolicyRunner.add_rows(initial_states)` to grow a live runner's batch:
  new agent rows are appended (each seeded as a fresh BOS episode) while every
  existing row keeps its full rolling context uninterrupted. The shared KV cache
  is invalidated and recomputed at the new batch size on the next step (the same
  warm-start discipline as `reset_rows`). Because the batch size is otherwise
  fixed at construction, this is the only way to raise `num_envs` on a live
  runner; use `reset()` to restart the whole batch instead.
- Fixed the warm-start on partial batch restarts (`reset_rows` / `add_rows`).
  These drop the shared KV cache, and the next `act()` previously sliced the
  warm-start input to each row's newest token — correct for a fresh `reset()`,
  but after a mid-episode restart it silently wiped the surviving / pre-existing
  rows' buffered history. Those rows are now re-primed over their full masked
  window, restoring the "surviving rows never lose context" guarantee. This
  supersedes the 0.11.0 note that the `reset_rows` path keeps legacy discard
  semantics: the partial-restart recompute is now a full-window (retain-flavored)
  pass that is kept. Same-phase restarts (lockstep `reset_rows`, `num_envs == 1`,
  and the `add_rows` recompute step) reach exact full-window parity; staggered
  (mixed-phase) restarts are improved but remain bounded by the shared cache's
  single position counter. The single-episode / full-`reset()` paths are
  byte-identical to before.

## 0.11.0

- Added the `bos_cache_mode` serving convention (`PolicyRunner` /
  `load_runner` / `load_runner_from_hub`), controlling whether the
  episode-start bos token's KV survives in the cached-inference KV cache.
  `"discard"` (default) reproduces the legacy behavior: the bos token's KV is
  dropped after the first `act()`, so the persisted cache carries only
  non-boundary (`is_bos == 0`) tokens. `"retain"` keeps the bos token's KV so
  it coexists with later tokens (matching full-window exposure). It is a
  runtime convention — no weights, architecture, or I/O schema change —
  resolved as: explicit argument > bundle `serving.bos_cache_mode` > `"discard"`.
  Bundles carry it under a new weight-independent `serving` namespace in
  `config.json`; absent (all existing bundles) resolves to `"discard"`, so
  behavior is byte-identical. Applies to the cached path only
  (`use_windowed=False`) and, in this version, to full `reset()` — the batched
  `reset_rows` partial-restart path keeps legacy discard semantics.
- `export_bundle` accepts an optional `bos_cache_mode` to bake that choice into
  the bundle's `serving` block at build time; omitting it writes no `serving`
  block, so existing bundles and older loaders are unaffected.

## 0.10.0

- Added the opt-in `use_bos_action_gate` capability. At an episode boundary
  (`is_bos == 1`) there is no genuine previous action, so the model neutralizes
  the previous-action input channel instead of consuming the placeholder value:
  the action columns are replaced by a per-head gate embedding before the input
  projection, while non-boundary steps (`is_bos == 0`) keep the real action
  feedback unchanged. The gate embedding ships in the bundle — zero by default
  (the action channel is simply emptied at the boundary), or a nonzero "null
  action" vector when the bundle provides one. Default-off and zeros-init, so
  bundles without the capability are byte-identical to before; older runtimes
  load newer bundles via `strict=False` (the extra weights are ignored).

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
