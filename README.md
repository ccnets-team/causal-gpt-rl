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

GPT-style transformers (GPT-2, Llama) running as RL policies in continuous-control environments.

**Here to run a policy?** → [Quick Start](#quick-start).
**Evaluating the product?** → [Available Policies](#available-policies) for what
is published and how it scores.

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
- **MuJoCo runs (W&B):** https://wandb.ai/causal-gpt-rl/mujoco
- Website: https://ccnets.org
- LinkedIn: https://www.linkedin.com/company/ccnets

Released under PolyForm Noncommercial 1.0.0. For commercial licensing, contact the maintainers via ccnets.org.

## What Is Here

- **This package** — the inference runtime. Load a policy bundle from Hugging
  Face or disk, roll it out, evaluate it.
- **Hugging Face** — pretrained bundles to run, with their environments and
  recorded datasets.
- **Not here** — training. Causal GPT-RL trains custom agents from private
  offline datasets as a commercial product; the trainer is not part of this
  repository.

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

## Observation & Action Spaces

**Every fixed-shape Gymnasium space, and any `Dict` / `Tuple` nesting of them.**
Variable-length and structural spaces — raw images, `Text`, `Sequence`, `Graph`,
`OneOf` — are out of scope; encode those into vectors on your side.

A bundle carries its declared `observation_space` and `action_space`, so you work
in your environment's own spaces: pass observations exactly as your env produces
them, and the action you get back is always a valid sample of the action space.

See **[docs/spaces.md](docs/spaces.md)** for the per-space contract, the rollout
loop, and a structured-space example. `docs/` holds the runtime references for
this package — that contract and [ONNX export](docs/export-onnx.md).

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

A bundle is a directory of two files, and it is the same directory whether it
came from Hugging Face, from local export, or from a training run. On the Hub
each bundle is one subfolder of a repo, which is what `subfolder=` selects;
locally, pass the directory to `load_runner("path/to/bundle")`.

```text
bundle/
  model.safetensors    # inference weights, state normalization embedded
  config.json          # model config, observation/action specs, context length
```

Public bundles are `bundle_format_version=2`. Version 1 shipped a separate
`state_normalizer.safetensors` sidecar and still loads with current releases; if
you are pinned to `causal-gpt-rl <= 0.2.x`, the sidecar bundles are preserved at
the `bundles-v1` tag, loadable with `revision="bundles-v1"`.

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
