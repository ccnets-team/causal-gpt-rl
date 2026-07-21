# Unity policy evaluation

This directory shows how to download a trained Causal GPT-RL ONNX policy and a
matching model-removed Unity ML-Agents build from Hugging Face, then measure the
policy closed-loop. ONNX conversion is intentionally out of scope: published
models are ready to run.

## Install

Use Python 3.10 and the lightweight Unity evaluation dependencies:

```bash
python -m pip install -r examples/unity_collection/requirements-collect.txt
python -m pip install "huggingface_hub>=0.23"
```

`mlagents_envs==1.1.0` pins older NumPy/Gymnasium versions, so this is best kept
in a separate environment from the PyTorch inference package.

## Download from Hugging Face

The policy repository is
[`ccnets/causal-gpt-rl-unity`](https://huggingface.co/ccnets/causal-gpt-rl-unity),
and matching builds live in the companion
[`ccnets/causal-gpt-rl-unity-envs`](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs)
dataset repository.

```python
from huggingface_hub import hf_hub_download, snapshot_download

policy = hf_hub_download(
    repo_id="ccnets/causal-gpt-rl-unity",
    filename="dungeon-escape/dungeonescape-b36.onnx",
    local_dir="hf_unity/model",
)

snapshot_download(
    repo_id="ccnets/causal-gpt-rl-unity-envs",
    repo_type="dataset",
    allow_patterns="DungeonEscape/**",
    local_dir="hf_unity/envs",
)
```

The DungeonEscape build is being published separately. Until its folder appears
in the environment repository, use a compatible local release-23 model-removed
build. The model card records the required observation/action signature.

The published DungeonEscape ONNX is an **intermediate training checkpoint**
provided to validate this end-to-end download and evaluation workflow. Its
reported performance is provisional and does not represent the final trained
policy. The artifact and metrics will be updated after training completes.

## Measure return

[`evaluate_onnx.py`](evaluate_onnx.py) reads the ONNX contract, validates it
against the live ML-Agents behavior spec, maintains one autoregressive context
per agent, and reports both individual and cooperative-group returns:

```bash
python examples/unity/evaluate_onnx.py \
    --build hf_unity/envs/DungeonEscape/UnityEnvironment.exe \
    --onnx hf_unity/model/dungeon-escape/dungeonescape-b36.onnx
```

It supports continuous, discrete, MultiDiscrete, and hybrid policies. A model
with batch size equal to the scene agent count runs one ONNX call per decision;
a batch-1 model is called once per agent.

## Decentralized cooperative policies

DungeonEscape is a decentralized cooperative environment: three agents share a
goal and group reward, but each acts from its own ego observation. The shared
policy is evaluated independently for every agent. Batching makes those
independent evaluations efficient; it does not create attention or hidden-state
communication between batch rows.

The recorded three-agent match is converted into three linked agent episodes:

```text
one cooperative match
  -> ego trajectory for agent A
  -> ego trajectory for agent B
  -> ego trajectory for agent C
```

`match_id` and `group_id` retain the relationship. An agent can terminate with
return zero before teammates complete the task, so individual zero return is not
equivalent to group failure. Group success is `any(agent_return > 0)` for the
three linked episodes.

For the published dataset `unity/dungeon-escape/expert-poca-v0`:

- 35,787 agent episodes and 1,003,523 transitions
- mean agent return: 0.606589
- 11,929 linked three-agent matches
- mean summed group return: 1.819767
- group success rate: 95.32%

The policy input uses 36 independent rows because the public evaluation scene
contains 12 arenas with three agents each:

```text
states  [36, 32, 371]  # 36 independent temporal contexts
actions [36, 32, 7]    # one-hot previous Discrete(7) action
is_bos  [36, 32, 1]
mask    [36, 32]
action  [36, 7]         # logits; argmax per row
```

Rows from the same match do not attend to one another. Cooperation comes from
the ego observations, shared group-reward demonstrations, and a shared policy.
For continuing multi-episode serving, reset only a terminated agent's context
row while preserving surviving teammates' histories; also track the group
boundary separately.

## DungeonEscape notebook

[`dungeon_escape_hf.ipynb`](dungeon_escape_hf.ipynb) is the environment-specific
walkthrough: download artifacts, inspect the model contract, run the evaluator,
and compare its agent/group returns with the dataset reference.

Dataset collection and Minari packaging remain in
[`examples/unity_collection`](../unity_collection/README.md).

## SoccerTwos: decentralized adversarial evaluation

SoccerTwos demonstrates the same shared decentralized policy in a competitive
setting. One scene contains eight independent 2-vs-2 fields (32 agents total).
The Causal policy controls both agents on one team, so its fixed ONNX batch is
`8 fields * 2 controlled agents = 16`; the stock ML-Agents ONNX controls the
other 16 agents. Both policies are evaluated before their actions are routed
into the same Unity step.

```bash
python examples/unity/evaluate_matchup.py \
    --build hf_unity/envs/SoccerTwos/UnityEnvironment.exe \
    --causal-onnx hf_unity/model/soccer-twos/soccertwos-b16.onnx \
    --stock-onnx hf_unity/envs/SoccerTwos.onnx \
    --causal-team both \
    --stock-baseline
```

`--causal-team both` runs Causal-team-0 vs stock-team-1 and then swaps sides.
The evaluator reports W/D/L, win rate, controlled-agent return, team return, and
a stock-vs-stock symmetry baseline. Batch rows remain independent temporal
contexts: batching does not permit cross-agent attention or communication.

The currently published SoccerTwos policy is an **approximately 50%-trained
intermediate checkpoint** provided to complete and validate the public example.
Its evaluation numbers are provisional and will be replaced after training.

[`soccer_twos_hf.ipynb`](soccer_twos_hf.ipynb) is the worked download and
side-swapped matchup tutorial. Its dataset representation is one ego-centric
episode per agent, with `match_id`, `field_id`, `team_id`, and `group_id` kept in
the collection manifest for match-level analysis. The shared model is trained
over those decentralized agent episodes; it does not jointly attend across the
four agents in a match.
