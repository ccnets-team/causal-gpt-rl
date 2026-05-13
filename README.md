# Causal GPT RL

Public inference runtime for Causal-GPT-RL policies.

This repository contains the code needed to load a policy bundle and run it in
an environment. It is intentionally focused on inference: model
construction, bundle loading, action decoding, rolling context state, optional
state normalization, and simple evaluation helpers.

Model creation workflows and experiment infrastructure are outside this runtime
boundary.

## What This Package Provides

- `causal_gpt_rl.model`: autoregressive policy model definitions and JSON-safe
  state/action specs.
- `causal_gpt_rl.inference.load_runner(...)`: load an exported bundle into a
  ready-to-run `PolicyRunner`.
- `PolicyRunner`: step-wise policy execution with `reset(...)`, `act(...)`, and
  `observe(...)`.
- `run_episodes(...)`: small single-environment evaluation helper.
- `export_bundle(...)`: write a public inference bundle from an in-memory model.
- `convert_legacy_bundle_to_safetensors(...)`: migrate old `.pt` bundle weights
  to `safetensors`.

## Bundle Format

A deployment bundle is a single directory:

```text
bundle/
  model.safetensors
  config.json
  state_normalizer.safetensors  # optional
```

`model.safetensors` contains only the model state dict needed for inference.
`config.json` contains public metadata required to reconstruct the runner:
model config, observation specs, action specs, and context length.
`state_normalizer.safetensors` is optional and stores state normalization
statistics when the policy expects normalized observations.

The bundle does not include experiment metadata or development-only state.

## Installation

Install the runtime dependencies in your environment:

```bash
pip install torch transformers safetensors numpy gymnasium
```

For MuJoCo environments, install the appropriate Gymnasium extras as well:

```bash
pip install "gymnasium[mujoco]"
```

To load bundles directly from Hugging Face Hub, install the Hub extra:

```bash
pip install "causal-gpt-rl[hub]"
```

If you are developing directly from this repository, install it editable:

```bash
pip install -e .
```

## Quick Start

```python
import gymnasium as gym

from causal_gpt_rl.inference import load_runner

env = gym.make("HalfCheetah-v5")
runner = load_runner(
    "path/to/bundle",
    device="cuda",          # or "cpu"
    kv_cache_max_len=None,  # default: 4 * context_length
    use_windowed=False,     # use cached incremental inference by default
)

obs, _ = env.reset()
runner.reset(obs)

done = False
while not done:
    action = runner.act()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if not done:
        runner.observe(obs)
```

The compatibility style `runner.act(obs)` is also supported. On the first call
after `reset(obs)`, the observation is already in the buffer, so `act()` is the
cleaner form.

## Evaluation Helper

For simple single-environment evaluation:

```python
import gymnasium as gym

from causal_gpt_rl.inference import load_runner, run_episodes

env = gym.make("HalfCheetah-v5")
runner = load_runner("path/to/bundle", device="cuda")

stats = run_episodes(env, runner, num_episodes=5, seed=0)
print(stats["return_mean"], stats["return_std"])
```

## Hugging Face Hub

Hub model repositories should contain a bundle at the repository root:

```text
model.safetensors
config.json
state_normalizer.safetensors  # optional
README.md
```

Then load it directly:

```python
from causal_gpt_rl.inference import load_runner_from_hub

runner = load_runner_from_hub(
    "ccnets/causal-gpt-rl-ant-v5",
    device="cuda",
)
```

`run_episodes(...)` is intentionally single-env only. For vectorized or
batched evaluation, drive `PolicyRunner` directly with `num_envs > 1`.

## Public API

The stable top-level inference surface is:

```python
from causal_gpt_rl.inference import (
    PolicyRunner,
    load_runner,
    run_episodes,
    export_bundle,
    convert_legacy_bundle_to_safetensors,
    load_runner_from_hub,
)
```

Lower-level components such as `ContextBuffer`, `ContextCache`, and
`StateNormalizer` remain available from their submodules for advanced use, but
they are not the preferred public entrypoint.

## Runtime Notes

- `load_runner(...)` accepts a local bundle path.
- `load_runner_from_hub(...)` downloads a Hugging Face Hub model repository and
  then loads the bundle.
- Continuous actions are clipped to the bounds stored in `action_specs`.
- Discrete actions are decoded to integer environment actions.
- Multi-discrete actions support batched decoding when `num_envs > 1`.
- Invalid runtime sizes such as non-positive `context_length`,
  `num_envs`, or `kv_cache_max_len` raise `ValueError`.
- When `use_windowed=False`, cached incremental inference is used. When
  `kv_cache_max_len` is omitted, the default cache cap is `4 * context_length`.
- When `use_windowed=True`, the full rolling window is passed each step and the
  KV cache is not used.

## Development Checks

Useful local checks:

```bash
python -m compileall -q causal_gpt_rl
python -m unittest discover -s tests
```

For package build checks:

```bash
python -m build
python -m twine check dist/*
```
