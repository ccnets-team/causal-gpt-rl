# API Reference

`causal_gpt_rl.inference` is the Python API for loading and running trained
Causal-GPT-RL policy bundles. Most applications create a [`PolicyRunner`](#policyrunner)
with `load_runner`, then control the rollout through `reset`, `act`, and
`observe`. The module also provides episode evaluation and bundle export and
migration utilities.

For the rollout semantics, see [Transformer Model Integrating Environment
Dynamics for RL](environment-dynamics-in-transformer.md). For supported inputs
and outputs, see [Observation & Action Spaces](spaces.md).

```python
from causal_gpt_rl.inference import (
    BUNDLE_FORMAT_VERSION,
    PolicyRunner,
    convert_legacy_bundle_to_safetensors,
    export_bundle,
    load_runner,
    load_runner_from_hub,
    run_episodes,
)
```

## Basic rollout

Load a runner once, reset it at the start of each episode, and pass every new
observation back after the environment step:

```python
runner = load_runner("path/to/bundle")
obs, info = env.reset()
runner.reset(obs)

while True:
    action = runner.act()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    runner.observe(obs)
```

`act(obs)` is a convenience form of `observe(obs)` followed by `act()`. The same
loop can therefore carry the observation into the next call instead:

```python
runner.reset(obs)
action = runner.act()

while True:
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    action = runner.act(obs)
```

The runner owns the context and KV cache. Do not reuse it across episode
boundaries without calling `reset`.

## Loading

### `load_runner`

```python
load_runner(
    bundle_dir,
    *,
    device="cpu",
    num_envs=1,
    kv_cache_max_len=None,
    use_windowed=False,
    bos_cache_mode=None,
) -> PolicyRunner
```

Load a local policy bundle.

For ordinary inference, only `bundle_dir` and occasionally `device` need to be
set. The cache arguments are serving controls; leave them at their defaults
unless reproducing a specific deployment setup or measuring long rollouts.

| Parameter | Description |
|---|---|
| `bundle_dir` | Directory containing `config.json` and `model.safetensors` (or legacy `model.pt`). |
| `device` | Torch device used for inference. |
| `num_envs` | Number of independent environments evaluated in one batch. |
| `kv_cache_max_len` | Retained KV-cache length, in tokens. `None` uses the bundle's `context_length`; larger values run, but pay off only where the policy generalizes past it — see [The trained window is not a ceiling](environment-dynamics-in-transformer.md#the-trained-window-is-not-a-ceiling). |
| `use_windowed` | Recompute the full context window on each step instead of using the KV cache. Fixed for the runner's life — see the attribute below. |
| `bos_cache_mode` | Override the bundle's BOS cache mode with `"retain"` or `"discard"`. |

Returns a ready-to-use [`PolicyRunner`](#policyrunner).

Raises `FileNotFoundError` when the bundle files are missing and `ValueError`
when the bundle or loader arguments are incompatible with the runtime.

```python
runner = load_runner("path/to/bundle")
```

### `load_runner_from_hub`

```python
load_runner_from_hub(
    repo_id,
    *,
    repo_type="model",
    revision=None,
    subfolder="",
    cache_dir=None,
    token=None,
    local_files_only=False,
    device="cpu",
    num_envs=1,
    kv_cache_max_len=None,
    use_windowed=False,
    bos_cache_mode=None,
) -> PolicyRunner
```

Download a bundle with `huggingface_hub.snapshot_download`, then load it with
[`load_runner`](#load_runner).

| Parameter | Description |
|---|---|
| `repo_id` | Hugging Face repository ID. |
| `repo_type` | Hugging Face repository type. |
| `revision` | Branch, tag, or commit. |
| `subfolder` | Bundle directory inside the repository. |
| `cache_dir` | Hugging Face cache directory. |
| `token` | Authentication token for private repositories. |
| `local_files_only` | Use cached files without network access. |

The remaining parameters are passed to [`load_runner`](#load_runner). Requires
the `hub` extra:

```bash
pip install "causal-gpt-rl[hub]"
```

```python
runner = load_runner_from_hub(
    "ccnets/causal-gpt-rl",
    subfolder="ant-v5",
)
```

## `PolicyRunner`

A runner is a stateful rollout session. Create one with a loader, call `reset`
at an episode boundary, then alternate actions and observations.

```python
runner.reset(initial_obs)
action = runner.act()

next_obs, reward, terminated, truncated, info = env.step(action)
action = runner.act(next_obs)
```

The model predicts one step ahead: an observation passed to `act(state)` does not
change the action returned by that same call. See the [calling
contract](environment-dynamics-in-transformer.md).

The first two steps are:

| Call | What happens |
|---|---|
| `reset(s0)` | Clears the previous episode and seeds its first observation. |
| `act()` | Emits `a0` from the episode-start token. |
| `observe(s1)` | Pairs the emitted `a0` with the preceding state and stages `s1`. |
| `act()` | Emits the next action from the updated history. |

This delayed pairing is specific to this model. The high-level
[`run_episodes`](#run_episodes) helper handles it automatically.

### Attributes

| Attribute | Description |
|---|---|
| `num_envs` | Current number of batch rows. |
| `context_length` | Context window recorded in the bundle. |
| `kv_cache_max_len` | Retained KV-cache length, in tokens. |
| `state_size` | Flat model observation width. |
| `action_size` | Flat model action width. |
| `obs_space` | Declared Gymnasium observation space, or `None`. |
| `action_space` | Declared Gymnasium action space, or `None`. |
| `use_windowed` | Whether full-window inference is enabled. Read-only in effect: it selects the inference path and the path sizes the rolling buffer, so assigning it warns and leaves the mode unchanged. Build a new runner to switch. |
| `bos_cache_mode` | Resolved BOS cache mode. |

### `reset`

```python
reset(initial_state) -> None
```

Clear all rollout state and seed the next episode with its first observation.
For batched runners, `initial_state` contains one observation per row.

Use `reset` for a new single episode or when every batch row restarts together.
Use [`reset_rows`](#reset_rows) when only some rows have finished.

### `act`

```python
act(state=None) -> action
```

Return a deterministically decoded action in the bundle's declared action space.
Passing `state` is equivalent to calling `observe(state)` before `act()`.

With `num_envs=1`, the result has the structure of one environment action. With
a batched runner, it contains one action per row. Exact structures for Box,
Discrete, Dict, and Tuple spaces are listed in [Observation & Action
Spaces](spaces.md).

Calling `act()` without a state before `reset()` raises `RuntimeError`.

### `observe`

```python
observe(state) -> None
```

Record an observation with the previously emitted action. Before the first
`reset`, this is equivalent to `reset(state)`.

Use the explicit `observe(state)` form when the rollout loop naturally receives
an observation after `env.step`. Use `act(state)` when keeping the observation
and next action in one call is more convenient.

### `act_with_info`

```python
act_with_info(state=None) -> tuple[action, dict]
```

Like [`act`](#act), but also returns an info dict. Its `termination_prob` key
holds the model's termination estimate — one value per row for batched runners —
or `None` when the bundle has no termination head.

The estimate is model output only; it does not reset the runner or override the
environment's `terminated` and `truncated` values.

### `reset_rows`

```python
reset_rows(done_mask) -> None
```

Reset selected rows of a batched runner. `done_mask` has shape `(num_envs,)`;
true rows start a new episode on the next `observe(state)` or `act(state)` while
false rows retain their context.

```python
action = runner.act(state)
next_state, done = env.step(action)
runner.reset_rows(done)
runner.observe(next_state)
```

Surviving rows keep their cached history exactly, including whatever they
retained past `context_length`. The cached keys and values are one tensor shared
by every row, so a restarted row's columns cannot be cut out of it; it is
recorded as owning none of the cache instead and the next action masks its
previous episode away. Nothing is rebuilt, and no row's action changes because a
neighbour restarted.

On a rotary backbone (`Llama`, the default) a restarted row also starts exactly
as a freshly reset runner does, `bos_cache_mode` included, because masking its
past is indistinguishable from never having had one. A backbone with learned
absolute positions (`GPT-2`) still carries the position it restarted at, so its
first steps differ from a fresh runner by ~1e-2; the surviving rows are exact
either way.

Calling this method before the runner has been reset raises `RuntimeError`, and
a mask whose length is not `num_envs` raises `ValueError`.

### `add_rows`

```python
add_rows(initial_states) -> None
```

Append new episode rows to a live runner. Existing rows keep their context and
`num_envs` increases by the number of supplied observations.

`initial_states` has shape `(k, state_size)`, or contains `k` structured
observations when the bundle declares an observation space. The next action
contains the original and newly added rows. Calling this method before the
runner has been reset raises `RuntimeError`.

The shared KV cache grows with the batch: the new rows take slots that own no
history and are masked away, so the existing rows keep their cached context
untouched, as in [`reset_rows`](#reset_rows). A cache object this cannot grow —
not one this package builds — raises `RuntimeError`. That is checked before the
batch is widened, so a refused call leaves `num_envs`, the rows, and their cached
history exactly as they were. There is no rebuild from the rolling window to fall
back to: the cached path stages two tokens rather than carrying a window.

## Evaluation

### `run_episodes`

```python
run_episodes(
    env,
    runner,
    *,
    num_episodes,
    seed=None,
    max_steps=None,
) -> dict
```

Evaluate a single-environment runner using either the four- or five-value
`env.step` convention.

This is the shortest way to evaluate a policy when per-step control is not
needed. It resets both the environment and runner for every episode. A batched
runner or `num_episodes < 1` raises `ValueError`.

| Parameter | Description |
|---|---|
| `env` | Gymnasium-compatible single environment. |
| `runner` | A runner with `num_envs == 1`. |
| `num_episodes` | Number of episodes; must be at least 1. |
| `seed` | Seed applied to the first environment reset. |
| `max_steps` | Optional maximum steps per episode. |

The returned dict contains `num_episodes`, `returns`, `lengths`, `return_mean`,
`return_std`, `length_mean`, and `length_std`. Standard deviations are population
standard deviations.

`max_steps` stops an episode at the requested horizon even if the environment
has not terminated. Only the first `env.reset` receives `seed`; later resets
continue from the environment's RNG state — so `num_episodes=50, seed=0` is
fifty draws from the initial-state distribution, not seeds `0..49`. The two are
comparable in the mean and differ episode by episode. To measure a named seed
range, seed the resets yourself — [`examples/deploy/reproduce.py`](../examples/deploy/reproduce.py)
gives each seed its own environment and advances them as one batch, which is
the reproduction protocol.

```python
stats = run_episodes(env, runner, num_episodes=5, seed=0)
print(stats["return_mean"], stats["return_std"])
```

## Bundles

Most users only load bundles. `export_bundle` is for model publishers, while
`convert_legacy_bundle_to_safetensors` migrates artifacts created by older
versions.

### `export_bundle`

```python
export_bundle(
    bundle_dir,
    *,
    model,
    model_config,
    state_specs,
    action_specs,
    context_length,
    obs_space=None,
    action_space=None,
    state_normalizer=None,
    env_id=None,
    requires_capabilities=None,
    write_state_normalizer_sidecar=True,
    bos_cache_mode=None,
) -> Path
```

Write a loadable bundle and return its directory.

| Parameter | Description |
|---|---|
| `bundle_dir` | Destination directory; created if needed. |
| `model` | Model whose weights are exported. |
| `model_config` | Model architecture configuration. |
| `state_specs` / `action_specs` | Per-head tensor specifications. |
| `context_length` | Trained context window. |
| `obs_space` / `action_space` | Optional Gymnasium space declarations. |
| `state_normalizer` | State normalization statistics. Required unless embedded in the model. |
| `env_id` | Optional environment provenance label. |
| `requires_capabilities` | Runtime capabilities required to load the bundle. |
| `write_state_normalizer_sidecar` | Also write the compatibility normalizer sidecar. |
| `bos_cache_mode` | Optional default BOS cache mode. |

Writing a normalizer sidecar produces a compatibility v1 bundle. Without the
sidecar, the current format version is used.

The exported directory always contains `config.json` and
`model.safetensors`. Space declarations and required capabilities are stored in
the config so an incompatible runtime can reject the bundle during loading
instead of decoding observations or actions incorrectly.

### `convert_legacy_bundle_to_safetensors`

```python
convert_legacy_bundle_to_safetensors(
    bundle_dir,
    *,
    remove_legacy=False,
) -> Path
```

Convert legacy `model.pt` and `state_normalizer.pt` files to their safetensors
equivalents and return the bundle directory. With `remove_legacy=True`, the old
files are deleted after conversion.

Conversion is in place. Keep the default `remove_legacy=False` until the
converted bundle has been loaded successfully.

### `BUNDLE_FORMAT_VERSION`

```python
BUNDLE_FORMAT_VERSION = 2
```

The current bundle format version. Loaders reject unsupported versions;
`export_bundle` may emit compatibility v1 when it writes a normalizer sidecar.
