# Unity ML-Agents — collection & measurement recipe

Reproduce the Causal GPT-RL Unity artifacts end to end, from public materials.

- **Measure** the policy's closed-loop return in a Unity build — see
  [`../deploy/mlagents.py`](../deploy/mlagents.py).
- **Collect** trajectories and package them as a Minari dataset — this folder.

All inputs are public Hugging Face repos:

| Repo | Contents |
|---|---|
| [ccnets/causal-gpt-rl-unity](https://huggingface.co/ccnets/causal-gpt-rl-unity) | policy `crawler.onnx` |
| [ccnets/causal-gpt-rl-unity-envs](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs) | model-removed Crawler build + stock `Crawler.onnx` |
| [ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets) | recorded trajectories `unity/crawler/expert-v0` |

## Environments

`mlagents_envs` 1.x pins an older numpy/gymnasium than Minari, so use two envs:

- **Collection** (`collect.py`): `mlagents_envs==1.1.0` + `onnxruntime` — see
  [`requirements-collect.txt`](requirements-collect.txt).
- **Packaging** ([`collection/build_minari.py`](../../collection/build_minari.py)): `minari==0.5.3`.

The measurement runner (`../deploy/mlagents.py`) uses the same collection env
(`onnxruntime` + `mlagents_envs`, no PyTorch).

## Collect → Minari

1. Get the model-removed Crawler build and the stock `Crawler.onnx` from the
   [envs repo](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs).
2. Record rollouts (the stock policy driving the build):

   ```bash
   python collect.py \
       --build path/to/Crawler.exe \
       --onnx  path/to/Crawler.onnx \
       --out   raw/ \
       --target 1000000
   ```
3. Package the raw episodes into a Minari dataset — the source-agnostic packager
   lives in [`collection/`](../../collection/); run it in a `minari==0.5.3` env:

   ```bash
   python ../../collection/build_minari.py \
       --raw raw/ \
       --dataset-id unity/crawler/expert-v0 \
       --description "ML-Agents Crawler, baked ONNX policy (model-removed build)."
   ```

The recipe ends at the Minari dataset — a portable, env-less trajectory set
(observation `Box(158)`, action `Box(20, [-1, 1])`), the same flat convention as
the Gymnasium / MuJoCo Minari datasets.

## Measure return

[`../deploy/mlagents.py`](../deploy/mlagents.py) drives the `crawler.onnx` policy
(from the model repo) in the build and reports closed-loop return per agent:

```bash
python ../deploy/mlagents.py --build path/to/Crawler.exe --onnx path/to/crawler.onnx
```

## Files

| File | Role |
|---|---|
| `unity_env.py` | ML-Agents → gymnasium stepping wrapper |
| `onnx_policy.py` | runs a stock ML-Agents ONNX policy in `onnxruntime` |
| `collect.py` | record per-episode transitions to `.npz` |
| `requirements-collect.txt` | collection env pins |

Packaging (`.npz` → Minari) uses the source-agnostic
[`collection/build_minari.py`](../../collection/build_minari.py).
