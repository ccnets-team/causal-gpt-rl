---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
library_name: pytorch
tags:
  - reinforcement-learning
  - gymnasium
  - mujoco
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
- **Run logs (W&B, public):** [wandb.ai/junhopark/Causal GPT-RL](https://wandb.ai/junhopark/Causal%20GPT-RL)
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

## Supported Environments

| Env | Bundle | Ctx | Return | Norm. | Medium Ref. |
|---|---|---:|---:|---:|---:|
| `Ant-v5` | `ant-v5` | 32 | 3339.51±1115.40 | 50.56±16.54 | 86.54 |
| `HalfCheetah-v5` | `halfcheetah-v5` | 32 | 6865.15±2657.69 | 43.17±16.11 | 74.83 |
| `Hopper-v5` | `hopper-v5` | 32 | 2836.28±987.67 | 73.40±25.72 | 72.91 |
| `Walker2d-v5` | `walker2d-v5` | 32 | 3883.30±684.09 | 56.69±9.99 | 83.26 |
| `Humanoid-v5` | `humanoid-v5` | 32 | 6511.87±2855.54 | 75.38±33.62 | 81.30 |

Training data is expert-free: bundles are trained using Minari simple and medium datasets only; expert trajectories are not used for training.

`Return` and `Norm.` are mean±std over 50 episodes with seeds `0..49`. `Ctx` is context length. `max_steps=1000`, and KV cache max length is capped to `Ctx`.

Normalized scores use random=0 and expert=100:

```text
100 * (return - random_ref) / (expert_ref - random_ref)
```

Medium reference scores are shown for context and are not the normalization baseline.

Evaluation runtime:

```text
causal-gpt-rl 0.2.1
torch 2.12.0+cu132
gymnasium 1.2.2
mujoco 3.8.1
minari 0.5.3
```

## Bundle Format

All public bundles include:

```text
bundle/
  model.safetensors
  config.json
  state_normalizer.safetensors
```

- `model.safetensors` — model state dict for inference.
- `config.json` — model config, observation specs, action specs, context length,
  and optional `env_id`.
- `state_normalizer.safetensors` — state normalization statistics used by the policy.

## Hugging Face Layout

Recommended layout:

```text
ccnets/causal-gpt-rl/
  ant-v5/
    model.safetensors
    config.json
    state_normalizer.safetensors
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
