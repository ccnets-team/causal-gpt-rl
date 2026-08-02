---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
library_name: pytorch
tags:
  - reinforcement-learning
  - gymnasium
  - mujoco
  - unity
  - ml-agents
  - causal-gpt-rl
---

# Causal GPT-RL

GPT-style transformers (GPT-2, Llama) running as RL policies in continuous-control environments.

Both LLM generation and RL interaction are autoregressive:

```text
token           → next token                           (LLM generation)
(state, action) → (next state from env, next action)   (RL rollout)
```

Causal GPT-RL policies act stably under their own rollouts — long-horizon control without the drift that has historically kept transformers from being usable as RL agents.

A single autoregressive model drives full-episode rollouts via KV cache — no critic, no auxiliary networks at inference.

This repository is the public inference runtime. It loads policy bundles, runs Gymnasium/MuJoCo rollouts, and provides small evaluation helpers.

- **Code (GitHub):** [ccnets-team/causal-gpt-rl](https://github.com/ccnets-team/causal-gpt-rl)
- **Hugging Face org:** https://huggingface.co/ccnets
- Website: https://ccnets.org
- LinkedIn: https://www.linkedin.com/company/ccnets

Released under PolyForm Noncommercial 1.0.0. For commercial licensing, contact the maintainers via ccnets.org.

## Product Overview

Causal GPT-RL is a GPT-based reinforcement learning product that turns offline trajectory data into deployable decision-making agents.

The system is designed for users who have recorded interaction data, simulation logs, or control trajectories and want to train policies that can act in sequential decision-making environments.

At the public package level, causal-gpt-rl provides the inference runtime for loading and evaluating trained policy bundles. These bundles can be executed in Gymnasium / MuJoCo environments and used to reproduce rollout behavior, benchmark performance, and demonstrate GPT-style reinforcement learning agents.

For commercial use, Causal GPT-RL is intended to support custom training from private offline datasets, cloud-based training workflows, and deployment of trained policy bundles through managed infrastructure.

In short:

- Public PyPI package: provides the inference runtime for loading Hugging Face or local policy bundles
- Hugging Face Hub: provides public pretrained policy bundles for testing, evaluation, and demos
- Commercial product: trains custom GPT-style RL agents from user-provided offline datasets
- Future direction: managed cloud training and SaaS-based decision-agent deployment

Causal GPT-RL is positioned as a bridge between offline reinforcement learning research and deployable AI agents for real-world sequential decision-making.

## Install

For Hub loading and MuJoCo environments:

```bash
pip install "causal-gpt-rl[hub,mujoco]"
```

For local development:

```bash
git clone https://github.com/ccnets-team/causal-gpt-rl.git
cd causal-gpt-rl
python -m pip install -e ".[hub,mujoco]"
```

For private bundles, authenticate first:

```bash
hf auth login
```

To convert a delivered bundle (`config.json` + `model.safetensors`) into a
self-contained ONNX policy:

```bash
pip install "causal-gpt-rl[onnx]"
causal-gpt-rl-export-onnx --bundle ./bundle --out policy.onnx --batch-size 1
```

See [Export a delivered bundle to ONNX](docs/export-onnx.md) for fixed-batch
multi-agent examples and the Python API.

## Quick Start

```python
import gymnasium as gym

from causal_gpt_rl.inference import load_runner_from_hub, run_episodes

env = gym.make("Ant-v5")
runner = load_runner_from_hub(
    repo_id="ccnets/causal-gpt-rl",
    subfolder="ant-v5",
)

stats = run_episodes(env, runner, num_episodes=5, seed=0)
env.close()
print(stats["return_mean"], stats["return_std"])
```

Notebook version: [examples/hub_quickstart.ipynb](https://github.com/ccnets-team/causal-gpt-rl/blob/main/examples/hub_quickstart.ipynb)

## Observation & Action Spaces

A policy bundle carries its declared Gymnasium `observation_space` and
`action_space`; you interact with the runtime in those native spaces and it
adapts the rest. Supported: `Box` (1-D), `Discrete`, `MultiDiscrete`,
`MultiBinary`, and arbitrary `Dict` / `Tuple` nesting of them. Pass observations
exactly as your env produces them; the action you get back is always a valid
sample of the declared `action_space`.

See **[docs/spaces.md](docs/spaces.md)** for the full table, the rollout loop,
and a structured-space (`Dict` / `Tuple`) example.

## Available Policies

Policy bundles, the environments they run in, and the trajectory datasets are
published on the Hugging Face org:

| Repo | Contents |
|---|---|
| [ccnets/causal-gpt-rl](https://huggingface.co/ccnets/causal-gpt-rl) | MuJoCo continuous control — `Ant-v5`, `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Humanoid-v5` |
| [ccnets/causal-gpt-rl-unity](https://huggingface.co/ccnets/causal-gpt-rl-unity) | Unity ML-Agents — DungeonEscape, SoccerTwos (`model.safetensors` + per-context ONNX) |
| [ccnets/causal-gpt-rl-unity-envs](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs) | Model-removed Unity builds + stock policies where redistributable |
| [ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets) | Recorded Minari trajectories |

Per-bundle returns, the evaluation protocol, and the runtime versions each score
was measured on are on the corresponding model card. Unity download-and-measure
walkthroughs are in [examples/unity/](examples/unity/).

## Context Window and KV Cache

A bundle's `context_length` is the model's context window. It is fixed in the
bundle and is not changeable at inference.

`kv_cache_max_len` — how much past a rollout retains — *is* a load-time knob. It
defaults to the bundle's own `context_length`, which keeps a rollout inside the
window the policy was measured on:

```python
runner = load_runner("path/to/bundle", kv_cache_max_len=64)
```

Larger values are supported and stay within the backbone's position capacity, but
retaining more history than the context window is an extrapolation regime, and
its effect is environment-dependent — across the published MuJoCo bundles it
ranges from a modest improvement to a large regression. The default is the safe
choice; measure before raising it.

## Bundle Format

Public bundles use `bundle_format_version=2`:

```text
bundle/
  model.safetensors
  config.json
```

- `model.safetensors` — model state dict for inference, with state
  normalization statistics embedded in the weights.
- `config.json` — model config, observation specs, action specs, context length,
  a `state_normalization` block, and optional `env_id`.

Older bundles (`bundle_format_version=1`) shipped a separate
`state_normalizer.safetensors` sidecar. They still load with current releases.
If you are pinned to `causal-gpt-rl <= 0.2.x`, use the sidecar bundles preserved
at the `bundles-v1` tag:

```python
runner = load_runner_from_hub(
    repo_id="ccnets/causal-gpt-rl",
    subfolder="ant-v5",
    revision="bundles-v1",
)
```

## Hugging Face Layout

Recommended layout:

```text
ccnets/causal-gpt-rl/
  ant-v5/
    model.safetensors
    config.json
    README.md
```

For local bundles, use `load_runner("path/to/bundle")`.

## API

```python
from causal_gpt_rl.inference import (
    PolicyRunner,                          # step-wise rollout policy with KV cache
    load_runner,                           # load runner from a local bundle directory
    load_runner_from_hub,                  # load runner from a Hugging Face Hub repo
    run_episodes,                          # evaluate over N episodes; returns stats dict
    export_bundle,                         # write a bundle directory from a runner
    convert_legacy_bundle_to_safetensors,  # migrate legacy bundles to the safetensors format
)
```

## Development Checks

```bash
python -m compileall -q causal_gpt_rl
python -m unittest discover -s tests
python -m build
python -m twine check dist/*
```

## License

Released under PolyForm Noncommercial License 1.0.0. See `LICENSE` for details. For commercial licensing, contact the maintainers via ccnets.org.
