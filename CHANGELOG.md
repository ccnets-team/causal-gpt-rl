# Changelog

## 0.18.0

- The `transformers` floor is 4.56, raised from 4.30. It was never a range this
  package ran on: `build_kv_cache` calls `DynamicCache(config=...)`
  unconditionally, and below 4.56 that raises `TypeError` while the cache is
  being built, so every cached rollout fails. The floor is measured rather than
  read off a signature — the suite passes whole on 4.56.0 and fails 41 tests on
  4.55.4 and 4.54.1, and 4.53 and below cannot take the keyword at all. Nothing
  about the package changed here; the range it declares now matches the range
  it works on, so a resolver refuses the install instead of pip succeeding and
  the first rollout failing.

- `examples/mujoco_collection/record_context_grid.py` records one dataset per
  rollout context length. The tier recipe names a length per tier and the
  calibration script measures a curve and discards the rollouts; this is the
  form that keeps them, so the level that wins is a packaged dataset rather than
  a number to re-record. `--context` takes the grid (`8,16,32` or `8-16`, sorted
  and deduplicated, since a grid is an axis), and `--episodes` is both the
  episode count and the batch width: a level is one batch of that many rows
  recording one episode each. Neither is optional in the sense that matters — a
  vector env is seeded once, so a narrower batch would leave most episodes to
  unseeded auto-resets, and two levels recorded at different widths reduce
  floating point in a different order. Every level runs the same seeds, which is
  what separates the grid from the seed draw. The summary reports the spread
  across the grid against the widest spread within a level, and says so when the
  first is smaller — a grid that has not separated anything should not be read
  off its means. No runtime change: the script drives `CollectionRunner` and
  `record_vector_episodes` as they are.

- `CollectionRunner` records a batch. A runner loaded with `num_envs > 1` now
  writes one episode file per row from a vectorized env, `observe` taking the
  arrays `step` returns; rows end at different steps, so each closes and is
  numbered as it finishes rather than by row. Gymnasium's vector auto-reset is
  `NEXT_STEP`, and the recorder owns the one-step wait it implies: the step a row
  ends on carries its true final observation and flags, the next carries the new
  episode's seed with a zero reward and an ignored action, and `reset_rows` is
  called between them so the ended episode leaves the row's context. A batched
  run therefore needs no `final_observation` handling from the caller.
  `num_envs == 1` keeps the single-env contract unchanged. The previous refusal
  justified itself with a parity claim that `tests/test_partial_restart_parity.py`
  measures at 0.0 for the rotary backbones every bundle uses; the boundary was
  the recorder tracking one episode, and that is what changed.

  `episodes_per_row` is each row's share, and it is what makes a batched count
  exact. A row that has written its share is retired: it keeps being driven,
  because the batch is one forward and a single row cannot leave it, but nothing
  it does after that reaches a file, `close()`'s flush included. Stopping on a
  total instead overshoots twice over — rows whose episodes are short churn
  through repeats while the long ones are still on their first, and whatever is
  in flight when the total lands is written as a truncation. Eight rows asked
  for eight episodes produced fourteen files that way, only four of them a
  seeded episode that ran to termination; with a share of one it is eight files,
  all seeded, none cut short.

  The share is also what keeps seeds meaningful. A vector env is seeded once, at
  `reset`, so only each row's first episode carries a seed the caller chose;
  every later one is the env restarting itself, and where it goes depends on how
  many steps the policy took before it — which is exactly what differs between
  two runs being compared. `episodes_per_row=1` is therefore the form to record
  with when seeds have to line up across runs, and it makes `num_envs` the
  episode count. `spec.json`'s provenance records both values.

  `examples/deploy/record.py` gains `--num-envs`: above 1 it builds the env with
  `gym.make_vec`, `--max-steps` becomes each sub-env's `max_episode_steps`
  because only the environment can restart a row it truncates, and each row
  records `ceil(--episodes / --num-envs)` episodes, which makes `--episodes` a
  floor the run clears by a known amount rather than an unknown one — the
  total is `--num-envs` times that share, and it is `--episodes` exactly when
  the two divide. `examples/mujoco_collection/record_tiers.py` gains it too,
  restricted to 1 or `--episodes`: that ladder is only one policy at several
  retentions if every tier draws the same initial states, and any value
  between the two would leave most episodes unseeded.

- The recorder is reachable from `collection/`, and its retention knob is named
  for what it retains. `collection/record.py` runs the same recording as
  `python -m examples.deploy.record`: it puts the checkout root on `sys.path`
  and calls that module's `main()`, so nothing about the run differs. What
  changes is that the directory owning the recorder is also where a run starts
  from. It stays a checkout-only entry point, because `collection/` is a
  directory of this repository rather than part of the installed
  `causal-gpt-rl` package.

  The knob is `--context-length` now, with `--kv-cache-max-len` kept as an
  alias, so command lines already written keep running. It has never been the
  model's window: it sets how much history the KV cache retains through a
  rollout, and the trained window is fixed in the bundle and not settable from
  the CLI at all. Because the two names now sit close together, a run prints
  both numbers on a `[context]` line before it starts, so the distinction is in
  the log kept beside the dataset. A value below 1 is refused while arguments
  are parsed, rather than after a bundle has loaded.

- A cached rollout no longer carries a rolling window it does not read. Once the
  KV cache holds history, `predict_incremental_cached` slices its input to the
  newest token, so the `context_length + 1` window the buffer allocated was
  staging rather than context — the context is the cache. The cached path now
  stages two tokens, which is the floor: an observation lands in the trailing
  slot with no action beside it and only the next roll pairs it into the
  `(state, action)` token the model reads. At Humanoid-v5 widths (348-wide
  observations, `context_length=32`, `kv_cache_max_len=1000`) `get_context()`
  drops from 464 to 4.8 microseconds at 64 rows and from 2.1 milliseconds to 19
  microseconds at 256; a whole policy step drops 19% at 64 rows and 25% at 256,
  because the window was also being converted to a tensor, moved to the device,
  and normalized before all but one token of it was discarded. Policy actions and
  `act_with_info`'s auxiliary output are bit-identical to the previous buffer
  across `reset_rows`, `add_rows`, both `bos_cache_mode` values, retention set
  above and below the trained window, and both backbone families — measured, not
  argued, at exactly zero rather than within a tolerance. Full-window runners
  (`use_windowed=True`) read the whole window and still allocate it.
  `context_length` is untouched: it is the bundle's trained window, and
  provenance records it. What does change shape is `runner.buffer` itself —
  `buffer.states` is now `(rows, 2, width)` and `buffer.context_length` reports
  the staging depth rather than the model's window — so code reaching into the
  buffer sees the smaller arrays.

  `add_rows()` now refuses a cache layout it cannot grow instead of rebuilding
  from the window, which is no longer there to rebuild from. It asks before it
  widens anything, so the refusal costs a caller neither its batch width nor its
  cached history. Only a cache exposing neither `layers` (transformers >= 4.46)
  nor `key_cache` (4.40-4.45) reaches that path; every supported `DynamicCache`
  grows in place and leaves the pre-existing rows untouched, which is what 0.17.0
  established. This is the one call that 0.17.0 accepted and this release does
  not — it accepted it by cutting every surviving row's history to
  `context_length`, which is the behavior 0.17.0 set out to remove.

- `use_windowed` is fixed at construction. It selects the inference path and the
  path sizes the buffer, so a live switch would read a window that is not
  allocated. Assigning it never worked anyway: switching to the cached path left
  the cache empty while the warm-start collapsed every row to its newest token
  (actions off by 5.5e-2), and switching back stacked new tokens onto a cache
  missing every windowed step (4.1e-3) — both silent. The attribute is now a
  property whose setter warns and leaves the runner in the mode it was built in,
  rather than raising, so callers that set it keep running and keep getting
  correct results for that mode; re-asserting the current mode stays silent. Two
  consequences of it being a property rather than an attribute: under `-W error`
  the assignment raises, and `use_windowed` no longer appears in
  `vars(runner)`. Under default filters the warning prints once per call site,
  so it is a notice rather than a guard. Reading `use_windowed` is unchanged, and
  constructing with either value is unchanged.

## 0.17.0

- A row finishing its episode no longer costs the other rows their history.
  `reset_rows` and `add_rows` dropped the batch's shared KV cache and rebuilt it
  from the rolling window, which holds `context_length` tokens and nothing more,
  so whatever a rollout had retained past that window was gone: one row falling
  over cut every row still running back to `context_length` of history. In a
  twelve-row Walker2d batch at `kv_cache_max_len=400`, two rows falling at steps
  254 and 266 moved the surviving rows' actions by up to 1.99 on a space bounded
  at ±1 and changed a scored episode's length; the same comparison on this
  release comes out at exactly zero. The cache is kept now — each row records how
  much of it is its own and the next forward masks the rest away, so its
  neighbours are untouched — exactly, on any backbone. On a rotary backbone
  (`Llama`, and every published bundle) the restarted row also starts exactly as
  a fresh runner does, `bos_cache_mode` included; with learned absolute positions
  (`GPT-2`) it still carries the position it restarted at and differs by ~1e-2,
  which is better than the ~4e-2 it was but not the same thing.

  Those two are the only functions whose behavior changes, and neither changes
  signature; `predict_incremental_cached` takes one new optional argument and is
  byte-identical without it. A caller that never restarts part of a batch cannot
  reach the new path at all: 2880 actions spanning both BOS modes, three
  retention settings and two batch widths are bit-identical to 0.16.0. While the
  mask is live it costs roughly 30% of a policy step, for as long as the
  restarted row takes to catch up with the cache length; against that, the
  recompute every partial restart used to pay is gone.

- An exported ONNX no longer carries the exporting machine's filesystem.
  `torch.onnx` stamps every node it emits with a `pkg.torch.onnx.stack_trace`
  entry holding absolute paths and source lines from wherever the export ran, so
  a delivered artifact named its author's home directory, conda prefix, and
  checkout path — 2898 occurrences of one user name in an 11-state bundle. The
  traces are export-time debugging aids that nothing reads back, and they are
  now dropped before the artifact is written. The sibling keys torch writes
  (`namespace`, `class_hierarchy`, `fx_node`, `name_scopes`) name graph
  structure rather than a filesystem, so they are kept. Weights and topology are
  untouched: the same bundle exports to byte-identical outputs before and after,
  and the file is roughly half a megabyte smaller.
- Corrected what the comment over `DEFAULT_KV_CACHE_CONTEXT_MULTIPLIER` claims
  about retention limits. It called 8x the context length the backbone's
  position "capacity"; 8x is only the Llama default, GPT-2 gets 2x, and every
  published bundle declares `max_position_embeddings` outright — where, on a
  Llama backbone, it is not a weight and does not bound sequence length at all.
  Nothing caps retention at the trained window; past it the extra history pays
  off only where the policy generalizes that far, which the README now says in
  place of a flat "larger values are supported". No behavior change; the default
  is still 1x, which is what the published scores were measured at. **This
  supersedes the 0.15.0 note** that larger values stay "within the backbone's
  position capacity (sized at 8x the context length)" — there is no such ceiling
  to stay within. All eight published MuJoCo bundles are Llama and declare
  `max_position_embeddings=256`, and the model card's retention sweep publishes a
  measured `kv=1000` column, four times that number, for Ant, HalfCheetah,
  Hopper, Walker2d and Humanoid.

## 0.16.0

- Loading a bundle no longer prints warnings. Two unrelated causes produced four
  lines between them before a gym environment even existed. The backbone built
  its transformers config with `vocab_size=1` while setting `bos_token_id=1` and
  `eos_token_id=2`, ids outside the only valid value; transformers 5.x
  range-checks these and logs one warning each. Nothing read them — there is no
  text vocabulary, and inputs arrive as continuous vectors through the adapters
  — so both are now unset. Separately, `deserialize_space` decoded Box bounds as
  float64 and handed them to a narrower `dtype`, so gymnasium down-cast them and
  warned once per array; the bounds are now built in the target dtype. Neither
  change affects inference: bundles produce byte-identical actions before and
  after.
- Kept `gym.spaces.Box`'s rejection of a finite bound past its dtype's range.
  The bounds down-cast above is skipped for those payloads (and for integer
  dtypes, which cannot hold ±inf), since casting first would overflow the bound
  to infinity and turn a clear error into a silently unbounded space.
- Added a `package` key to a bundle's `config.json`, naming what reads the file
  so the adjacent `package_version` says what it is the version of. It is
  descriptive only — the loader never reads it, because bundles written before
  this key exist and must keep loading. Bundles already delivered are
  unaffected; the key appears on the next export.
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
- Rewrote the Forecast metrics documentation as planned-only. No `forecast:`
  metric is registered with SageMaker and no forecast line appears in a job's
  logs — all three estimates (step reward, episode length, episode return) are
  held back from the customer surface until they are validated. The section
  previously carried example log output, interpretation guidance, and
  troubleshooting steps for metrics a running job does not produce; those are
  removed.

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
