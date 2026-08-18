# Record Trajectories

This is entry point 2 on the [path](../README.md#where-you-join) — you have an
environment and a policy model that drives it, and you need the five arrays.

## What runs, and what is only a sketch

Nothing here steps an environment for you — the loop is always yours. What can
be shipped is the recording around it, and the two that are cover different
halves of this stage:

| | covers | |
|---|---|---|
| [`examples/unity_collection/collect.py`](../../examples/unity_collection/collect.py) | one **environment** source — a Unity ML-Agents build, driven by an ONNX policy | a script to run |
| `CollectionRunner` | one **policy** source — a bundle of ours, in whatever environment you point it at | [Improving the Next Dataset](improving-the-next-dataset.md) |

Your own policy in your own environment is neither, and it is the ordinary case:
for it the loop below is a specification, not a script you can run. Write the
five arrays however suits your source and the packager takes it from there.

## The loop

Any loop that produces the five arrays will do:

```python
import numpy as np

obs, actions, rewards = [], [], []
state, _ = env.reset()
obs.append(state)

while True:
    action = policy(state)
    state, reward, terminated, truncated, _ = env.step(action)
    actions.append(action)
    rewards.append(reward)
    obs.append(state)
    if terminated or truncated:
        break

T = len(rewards)
term = np.zeros(T, dtype=bool); term[-1] = terminated
trunc = np.zeros(T, dtype=bool); trunc[-1] = truncated

np.savez(
    "raw/ep_000000.npz",
    observations=np.asarray(obs, dtype=np.float32),
    actions=np.asarray(actions, dtype=np.float32),
    rewards=np.asarray(rewards, dtype=np.float32),
    terminations=term,
    truncations=trunc,
)
```

`policy(state)` is the only line that knows what is driving the episode. An SB3
model goes in as `policy = lambda s: model.predict(s, deterministic=True)[0]`, a
scripted controller as itself — the packager never sees the difference.

For a `Discrete(7)` action, store the index instead and declare it:

```python
actions=np.asarray(actions, dtype=np.int64).reshape(T, 1)
# spec.json: {"action_kind": "discrete", "branches": [7]}
```

## Worked sources

[`examples/unity_collection/`](../../examples/unity_collection/) records episodes
from a Unity ML-Agents build and packages them with this tool, including quality
tiers synthesized by degrading a stock policy. The published datasets at
[ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets)
were made that way, from the builds at
[ccnets/causal-gpt-rl-unity-envs](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs).
That recorder also writes its own `spec.json` from the live ML-Agents behavior
spec, so a build with different sensors needs no code change.

A second worked source is planned from outside Gymnasium — a SUMO traffic-light
controller driven over TraCI — to walk the same path from a different engine.
Whatever the source, it only has to produce the typed-vector episode contract.

---

Next: [Packaging](03-packaging.md) — turning the raw directory into a dataset.
