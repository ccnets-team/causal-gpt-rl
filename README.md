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

[![PyPI](https://img.shields.io/pypi/v/causal-gpt-rl)](https://pypi.org/project/causal-gpt-rl/)
[![Python](https://img.shields.io/pypi/pyversions/causal-gpt-rl)](https://pypi.org/project/causal-gpt-rl/)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm%20NC%201.0.0-blue)](https://polyformproject.org/licenses/noncommercial/1.0.0)

GPT-style transformers (Llama) running as RL policies in continuous and discrete control environments.

**Here to run a policy?** → [Quick Start](#quick-start).
**Evaluating the product?** → [Available Policies](#available-policies) for what
is published and how it scores.

Both LLM generation and RL interaction are autoregressive:

```text
token           → next token                           (LLM generation)
(state, action) → (next state from env, next action)   (RL rollout)
```

![A rollout unrolled — each token pairs the current state with the action for it, that action moves the environment, and the environment returns the next state; meanwhile the model generates the next action from completed state-action pairs, not from the state it just received](docs/assets/action-generation-in-a-rollout.svg)

Causal GPT-RL policies act stably under their own rollouts — long-horizon control without the drift that has historically kept transformers from being usable as RL agents.

A single autoregressive model drives full-episode rollouts via KV cache — no separate critic, no auxiliary networks at inference. The model carries a value head and computes it on every forward pass, but a rollout does not read it: the action alone carries the loop.

The calling contract that follows from this — why the output is one step ahead of the observation you just passed, and why actions keep coming with no environment attached — is [Transformer Model Integrating Environment Dynamics for RL](docs/environment-dynamics-in-transformer.md).

- **Code (GitHub):** [ccnets-team/causal-gpt-rl](https://github.com/ccnets-team/causal-gpt-rl)
- **Hugging Face org:** https://huggingface.co/ccnets
- **MuJoCo runs (W&B):** https://wandb.ai/causal-gpt-rl/mujoco
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

On Windows, the Hub cache uses symlinks, which fail with
`OSError: [WinError 1314]` unless Developer Mode is on or Python runs as
administrator. Either enable Developer Mode or disable symlinks for the cache:

```bash
set HF_HUB_DISABLE_SYMLINKS=1
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

Five episodes off one seed is a smoke test, not the protocol the published scores
were measured under. To measure a bundle that way, see
[Reproduce a published score](examples/README.md#reproduce-a-published-score).

## Available Policies

Policy bundles, the environments they run in, and the trajectory datasets are
published on the Hugging Face org:

| Repo | Contents |
|---|---|
| [ccnets/causal-gpt-rl](https://huggingface.co/ccnets/causal-gpt-rl) | MuJoCo continuous control — `Ant-v5`, `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Humanoid-v5` |
| [ccnets/causal-gpt-rl-unity](https://huggingface.co/ccnets/causal-gpt-rl-unity) | Unity ML-Agents — Crawler, DungeonEscape, Pyramids, SoccerTwos (`model.safetensors` + per-context ONNX) |
| [ccnets/causal-gpt-rl-unity-envs](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs) | Model-removed Unity builds + stock policies where redistributable |
| [ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets) | Recorded Minari trajectories |

Per-bundle returns, the evaluation protocol, and the runtime versions each score
was measured on are on the corresponding model card. Worked runs — including the
Unity download-and-measure walkthroughs — are in
[examples/](examples/README.md).

The runs behind the MuJoCo bundles are public at
[wandb.ai/causal-gpt-rl/mujoco](https://wandb.ai/causal-gpt-rl/mujoco) — the
learning curves and per-run configuration, alongside the reported returns.

This repository is the public inference runtime: it loads policy bundles, runs Gymnasium/MuJoCo rollouts, and provides small evaluation helpers. Training is not here — it runs on AWS Marketplace as the [CCNets Causal GPT-RL Training Algorithm](training/docs/aws/README.md).

## Observation & Action Spaces

**Every fixed-shape Gymnasium space, and any `Dict` / `Tuple` nesting of them.**
Variable-length and structural spaces — raw images, `Text`, `Sequence`, `Graph`,
`OneOf` — are out of scope; encode those into vectors on your side. A bundle
carries its declared `observation_space` and `action_space`, so you work in your
environment's own spaces: pass observations exactly as your env produces them,
and the action you get back is always a valid sample of the action space.

See **[docs/](docs/README.md)** for the per-space contract, the calling contract,
the API reference, and ONNX export.

## Context Window and KV Cache

A bundle's `context_length` is its trained context window. It is fixed in the
bundle and is not changeable at inference.

`kv_cache_max_len` — how much past a rollout retains — *is* a load-time knob. It
defaults to the bundle's own `context_length`, which keeps a rollout inside the
window the policy was measured on:

```python
runner = load_runner("path/to/bundle", kv_cache_max_len=64)
```

![Why long-context extrapolation can be stable in RL — two measures leave the same first step, a short arrow to the trained window of 32 and a long one to a 1000-step retention, over a single unbroken bar of tokens that runs on past the shorter one](docs/assets/trained-window-is-not-a-ceiling.svg)

Larger values run: weights trained on a 32-token window carry an episode to the
end with a 1000-step KV cache. Whether that extra history helps is
environment-dependent — across the Hugging Face bundles it gains in some
environments and costs in others.

So the knob's purpose is not performance tuning but **drift control over long
rollouts**. A policy conditioning on its own outputs rides on the history it has
been building, and how much of that history it carries is what governs the ride.
See [Transformer Model Integrating Environment Dynamics for RL](docs/environment-dynamics-in-transformer.md).

On long rollouts a return mean can average early failures and full-length runs
into a number that describes neither — what to report instead is
[Measuring a Long Horizon](docs/long-horizon.md).

## Bundle Format

A bundle is a directory of two files, and it is the same directory whether it
came from Hugging Face, from local export, or from a training run. On the Hub
each bundle is one subfolder of a repo, which is what `subfolder=` selects;
locally, pass the directory to `load_runner("path/to/bundle")`.

```text
bundle/
  model.safetensors    # inference weights, state normalization embedded
  config.json          # model config, observation/action specs, context length
```

Public bundles are `bundle_format_version=2`.

## Development Checks

```bash
python -m compileall -q causal_gpt_rl
python -m unittest discover -s tests
python -m build
python -m twine check dist/*
```

## License

Released under PolyForm Noncommercial License 1.0.0. See `LICENSE` for details. For commercial licensing, contact the maintainers via ccnets.org.
