# Changelog

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
