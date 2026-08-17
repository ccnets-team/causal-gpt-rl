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
| [`deploy/survival.py`](deploy/survival.py) | Survival, per-interval hazard, and completer return/step — what to report once a horizon is long enough that a return mean stops describing any real episode. |
| [`deploy/checkup.py`](deploy/checkup.py) | Pre-flight check of a delivered bundle against the dataset it was trained on — no environment needed. Observation fit, action agreement, action spread, value and termination heads. |

```bash
python -m examples.deploy.mujoco --env-id Hopper-v5 --bundle path/to/bundle --episodes 5
```

## Reproduce a published score

[`deploy/reproduce.py`](deploy/reproduce.py) measures a MuJoCo bundle under the
reproduction protocol: 50 episodes, seeds 0..49, run together as one 50-row
batch, the KV cache left at the bundle's context length.

```bash
python -m examples.deploy.reproduce --env-id Ant-v5          # one bundle
python -m examples.deploy.reproduce --env-id all --json out.json   # every bundle
```

Three things fix the number, and dropping any one of them changes it.

**The seeds.** `run_episodes` seeds only its first reset and lets the
environment's RNG carry the rest, so fifty of its episodes are fifty draws from
the same distribution rather than seeds 0..49 — comparable in the mean, never
equal episode by episode.

**The batch width.** A batch-of-fifty forward and a batch-of-one forward do not
reduce in the same order, and in a closed autoregressive loop that last-bit
difference compounds over a thousand steps until the trajectories separate.
Fifty sequential rollouts are a different measurement, not a slower one.

**The runtime.** The script prints the installed torch / gymnasium / mujoco
beside the versions the protocol is defined on, because a different simulator
release is a different measurement even with identical weights and seeds.

[`unity/`](unity/) walks the whole path for Unity: download a policy and its
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

[`unity_collection/`](unity_collection/) is the worked recipe — record
trajectories from a Unity build, synthesize quality tiers, and package the
result. The datasets at
[ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets)
were made this way.

[`../collection/`](../collection/) is the packager underneath it, and it is
source-agnostic: any source whose episodes fit five typed-vector arrays becomes
a Minari dataset the same way, Unity or not. Its
[docs](../collection/docs/README.md) are the input contract and how to check
what came out.

## Environments

`mlagents_envs` 1.x pins older NumPy and Gymnasium than Minari and the PyTorch
runtime, so the Unity work wants its own environment:

| For | Install |
|---|---|
| Running a policy | `pip install "causal-gpt-rl[hub,mujoco]"` |
| Unity evaluation and collection | [`unity_collection/requirements-collect.txt`](unity_collection/requirements-collect.txt) + `huggingface_hub>=0.23` |
| Packaging to Minari | `minari==0.5.3` |
