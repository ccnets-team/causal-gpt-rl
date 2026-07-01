# Bring Your Own Data

**Causal GPT-RL is a GPT-style policy you can train small.** It has the shape people
know from LLMs — autoregressive, promptable — but it emits *actions*, not words,
learned from your own recorded data. Generating open-ended language is what forces an
LLM to be huge; a policy that only outputs a structured action over typed-vector
tokens carries none of that weight, so it stays light.

*Part 1 of 4 — the data contract. You start by handing over recorded data; by the
end you'll have declared the interface that becomes your model. Part 2 covers how
that model runs; Part 3, how you steer it through state; Part 4, how to steer it in
plain words.*

---

## Not imitation — offline RL

Record ordinary trajectories — imperfect actions, each with a reward — and the
model learns to **improve on them**. It predicts the action you recorded **and**,
because every step carries a score, learns to do better: mediocre-but-scored data
can train a better-than-demonstrated policy. That is the "RL" in the product, and
the reason this is more than a data dump — **it is offline RL, not imitation.**

---

## The contract

For that to stay clean, the whole relationship rests on a single boundary:

> **You hand over typed vectors. Encoding is yours; decision-making is ours.**

You never send us pixels, audio, or text, and you never hand us a language model.
Raw perception becomes numbers *on your side*; we learn the decisions. The four
parts of this journey are that one boundary, unfolded:

1. **Bring your own data** *(this part)* — the typed-vector data contract.
2. **[The acting policy](02-the-acting-policy.md)** — how the model runs.
3. **[Shaping behavior through state](03-shaping-behavior-through-state.md)** — steer
   it by what you put in its state.
4. **[A policy you can prompt](04-a-policy-you-can-prompt.md)** — that steering, in
   natural language, where *encoding is yours* pays off.

---

## What you provide

Offline trajectories — for each timestep, a tuple of
`(observation, action, reward, terminated, truncated)` — packaged as a
[Minari](https://minari.farama.org/) dataset that carries a declared Gymnasium
`observation_space` and `action_space`.

```python
import gymnasium as gym
import numpy as np

observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(16,))  # your sensors
action_space      = gym.spaces.Discrete(7)                        # seven moves
# record aligned (observation, action, reward, terminated, truncated) into a Minari dataset
```

The supported building blocks are the fixed-shape Gymnasium spaces — `Box` (1-D),
`Discrete`, `MultiDiscrete`, `MultiBinary` — and any `Dict` / `Tuple` nesting of
them. (Variable-length or structural spaces — raw images, `Text`, `Sequence`,
`Graph`, `OneOf` — are out of scope, which is exactly why you pre-encode.)

---

## Your spaces are the model

Here is the part that surprises people: **your declared spaces don't just describe
the data — they build the model.** Together, the `observation_space` and
`action_space` form the model's input — it reads `(state, action)` pairs, its own
actions feeding back as history — and the `action_space` *also* becomes its output
heads. From them follow the total input width, the number and type of output heads
(continuous / categorical / binary), how each field is normalized, and how actions
decode. You never configure a network separately — the **spaces *are* the
configuration**, frozen into the trained bundle.

The practical consequence: a space is a **commitment, not a knob**. Adding a sensor
dimension or swapping a leaf's type produces a *different model* — you cannot tune
it after the fact. So decide your dimensions and action structure **deliberately,
up front**. This is where the design actually lives.

---

**Next → [The Acting Policy](02-the-acting-policy.md)** — you've declared the
interface; now see how the model built from it runs, and how you shape its behavior
through state.
