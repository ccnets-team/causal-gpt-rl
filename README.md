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

## Supported Environments

| Env | Bundle | Ctx | Return | Norm. | Simple Ref. | Medium Ref. |
|---|---|---:|---:|---:|---:|---:|
| `Ant-v5` | `ant-v5` | 32 | 5262.53±1400.17 | 79.08±20.76 | 59.99 ✓ | 86.54 ✗ |
| `HalfCheetah-v5` | `halfcheetah-v5` | 32 | 6816.48±3135.53 | 42.87±19.01 | 43.54 ✗ | 74.83 ✗ |
| `Hopper-v5` | `hopper-v5` | 32 | 2713.66±1075.57 | 70.21±28.01 | 42.65 ✓ | 72.91 ✗ |
| `Walker2d-v5` | `walker2d-v5` | 32 | 3899.88±706.57 | 56.93±10.32 | 59.51 ✗ | 83.26 ✗ |
| `Humanoid-v5` | `humanoid-v5` | 32 | 7892.65±1018.11 | 91.63±11.99 | 63.29 ✓ | 81.30 ✓ |

Training data is expert-free: bundles are trained using Minari simple and medium datasets only; expert trajectories are not used for training.

`Return` and `Norm.` are mean±std over 50 episodes with seeds `0..49`. `Ctx` is context length. `max_steps=1000`, and KV cache max length is capped to `Ctx`.

Normalized scores use random=0 and expert=100:

```text
100 * (return - random_ref) / (expert_ref - random_ref)
```

`Simple Ref.` and `Medium Ref.` are the normalized means of the Minari `simple-v0`
and `medium-v0` datasets. They are shown for context and are not the normalization
baseline. `✓` marks a reference the bundle's `Norm.` exceeds, `✗` one it does not.

### KV cache retention sweep

`Ctx` above is the bundle's `context_length` — the model's context window, fixed
at **32** and not changeable at inference. `kv_cache_max_len` (how much past the
rollout retains) *is* a load-time knob; the main table caps it to `Ctx` (1×).
Sweeping it to 0.5×, 1×, and 2× the window, with the same protocol (50 episodes,
seeds `0..49`, `max_steps=1000`):

| Env | `kv=16` (0.5×) | `kv=32` (1×) | `kv=64` (2×) | Trend |
|---|---:|---:|---:|---|
| `Ant-v5` | 5163.35±1608.02 | 5262.53±1400.17 | 4516.35±1773.93 | ≈, -14% at 2× |
| `HalfCheetah-v5` | 6793.17±2939.17 | 6816.48±3135.53 | 6468.21±3234.51 | ≈ flat |
| `Hopper-v5` | 3361.71±103.69 | 2713.66±1075.57 | 992.92±445.63 | shorter ↑, 2× collapses |
| `Walker2d-v5` | 3950.09±459.01 | 3899.88±706.57 | 3842.19±718.60 | ≈ flat |
| `Humanoid-v5` | 7431.52±2024.95 | 7892.65±1018.11 | 8040.41±38.02 | longer ↑, steadiest |

The `kv=32` column matches the main table. At `kv=64` the rollout attends over
more history than the model's 32-token window — positions outside its native
range. This stays within the backbone's position capacity (Llama/RoPE,
`max_position_embeddings=256`), so it is an extrapolation regime, not a hard cap.

Best retention is **environment-dependent**, but for most envs the difference
across 0.5×/1×/2× is within run-to-run noise (`Trend` marks these `≈`). The real
exceptions: `Hopper-v5` clearly prefers a shorter window (0.5× is +24% with much
lower variance; 2× collapses to roughly a third of return, episodes ending early),
while `Humanoid-v5` is best at 2× (highest return and its steadiest — std 38
across all 50 episodes). The context window (1×) is a safe default; deviating from
it helps only in specific environments.

Evaluation runtime — every row above is measured on this one:

```text
causal-gpt-rl 0.13.0
torch 2.8.0+cu129
gymnasium 1.2.3
mujoco 3.2.3
minari 0.5.3
```

`mujoco` is pinned to `3.2.3` because that is the version the Minari datasets
were recorded with (`requirements: ['mujoco==3.2.3', 'gymnasium>=1.0.0']`). The
`Norm.` and `Medium Ref.` columns are derived from those recorded trajectories,
so returns are only comparable to them when measured on the same physics.

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
