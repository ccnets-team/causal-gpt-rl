# Examples

Worked runs of the inference runtime, from a five-line quickstart to full
environment walkthroughs. Pick by what you came to do.

## Run a policy

Load a bundle, roll it out, read the return.

| | |
|---|---|
| [`hub_quickstart.ipynb`](hub_quickstart.ipynb) | Load from the Hugging Face Hub and evaluate over N episodes. Start here. |
| [`deploy/mujoco.py`](deploy/mujoco.py) | The same rollout as a script, against a MuJoCo Gymnasium env. |
| [`deploy/mlagents.py`](deploy/mlagents.py) | The Crawler-specific ML-Agents variant. |

```bash
python -m examples.deploy.mujoco --env-id Hopper-v5 --bundle path/to/bundle --episodes 5
```

## Reproduce a published score

[`unity/`](unity/) walks the whole path: download a policy and its
model-removed Unity build from Hugging Face, then measure the policy
closed-loop against the number on the model card.

| | |
|---|---|
| [`unity/evaluate_onnx.py`](unity/evaluate_onnx.py) | Continuous, discrete, MultiDiscrete, hybrid, and cooperative multi-agent policies. |
| [`unity/evaluate_matchup.py`](unity/evaluate_matchup.py) | Side-swapped team-vs-team evaluation with a symmetry baseline. |
| [`unity/dungeon_escape_hf.ipynb`](unity/dungeon_escape_hf.ipynb) | Cooperative three-agent walkthrough, end to end. |
| [`unity/soccer_twos_hf.ipynb`](unity/soccer_twos_hf.ipynb) | Adversarial 2-vs-2 walkthrough, end to end. |

A batch dimension equal to the scene's agent count runs one call per decision
tick; batch rows stay independent temporal contexts and never attend to one
another. [`unity/README.md`](unity/README.md) has the details and the published
numbers.

## Build a dataset in the same format

[`unity_collection/`](unity_collection/) is the worked recipe — record rollouts
from a Unity build, synthesize quality tiers, and package the result. The
datasets at
[ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets)
were made this way.

[`../collection/`](../collection/) is the packager underneath it, and it is
source-agnostic: any source whose episodes fit five typed-vector arrays becomes
a Minari dataset the same way, Unity or not. Its README is the input contract
and how to check what came out.

## Environments

`mlagents_envs` 1.x pins older NumPy and Gymnasium than Minari and the PyTorch
runtime, so the Unity work wants its own environment:

| For | Install |
|---|---|
| Running a policy | `pip install "causal-gpt-rl[hub,mujoco]"` |
| Unity evaluation and collection | [`unity_collection/requirements-collect.txt`](unity_collection/requirements-collect.txt) + `huggingface_hub>=0.23` |
| Packaging to Minari | `minari==0.5.3` |
