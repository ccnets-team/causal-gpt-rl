# Causal GPT-RL

GPT-style transformers (GPT-2, Llama) running as RL policies in continuous-control environments.

The autoregressive structure is the same on both sides:

```text
action → next state → next action      (RL rollouts)
token  → next token  → next token      (LLM generation)
```

Causal GPT-RL policies act stably under their own rollouts — long-horizon control without the drift that has historically kept transformers from being usable as RL agents.

A single autoregressive model drives full-episode rollouts via KV cache — no critic, no auxiliary networks at inference.

This repository is the public inference runtime. It loads policy bundles, runs Gymnasium/MuJoCo rollouts, and provides small evaluation helpers.

- **Run logs (W&B, public):** [wandb.ai/junhopark/Causal GPT-RL](https://wandb.ai/junhopark/Causal%20GPT-RL)
- **Models (Hugging Face):** https://huggingface.co/ccnets
- Website: https://ccnets.org
- LinkedIn: https://www.linkedin.com/company/ccnets

Released under PolyForm Noncommercial 1.0.0. For commercial licensing, contact the maintainers via ccnets.org.

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

Notebook version: [examples/hub_quickstart.ipynb](examples/hub_quickstart.ipynb)

## Supported Environments

| Environment | Gymnasium ID | Hub subfolder |
|---|---|---|
| Ant | `Ant-v5` | `ant-v5` |
| HalfCheetah | `HalfCheetah-v5` | `halfcheetah-v5` |
| Walker2d | `Walker2d-v5` | `walker2d-v5` |
| Humanoid | `Humanoid-v5` | `humanoid-v5` |

## Bundle Format

All public bundles include:

```text
bundle/
  model.safetensors
  config.json
  state_normalizer.safetensors
```

- `model.safetensors` — model state dict for inference.
- `config.json` — model config, observation specs, action specs, and context length.
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

Root-level bundles are supported by omitting `subfolder`:

```python
load_runner_from_hub(repo_id="ccnets/causal-gpt-rl-ant")
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

PolyForm Noncommercial License 1.0.0. See `LICENSE` for details.
