# A Policy You Can Prompt

*Part 4 of 4. In [Part 3](03-shaping-behavior-through-state.md) you learned to steer
the policy by adding channels to its state — a mask for what's allowed, a goal for
what to pursue. This part is the richest of those channels: the goal expressed in
natural language — the promptable, LLM-shaped promise from Part 1, delivered.*

---

## The goal, in natural language

Part 3 added **raw** channels — a binary mask, a task id. Language is the same
skill with one new move: the channel you add is an **encoding**.

The most expressive instruction is **words**, and words aren't numbers — so you
**encode them on your side**: run the instruction through a frozen sentence
encoder and add the resulting vector as one more channel on your state. Same
boundary as Part 1, now applied to language — you never hand us the text, only
the vector. (Whatever the modality — sentence embeddings, token encoders, your
own discrete-embedding scheme — the encoding is yours to bring.) And because the
model reads only that vector — never the words — prompting adds no language-scale
weight: it stays as small as the policy from Part 1.

```python
observation_space = gym.spaces.Dict({
    "sensors":     gym.spaces.Box(-np.inf, np.inf, shape=(16,)),
    "action_mask": gym.spaces.MultiBinary(7),
    "instruction": gym.spaces.Box(-1.0, 1.0, shape=(384,)),   # L2-normed embedding
})

def encode_instruction(text):
    v = frozen_sentence_encoder(text)                 # your frozen encoder → (384,)
    return (v / np.linalg.norm(v)).astype(np.float32) # L2-norm → components in [-1, 1]
```

> **Why declare it `Box(-1, 1, d)`?** An L2-normalized vector already sits in
> `[-1, 1]` on every axis, so that declaration is *honest by construction* — it
> states the vector's true range rather than an invented one. It also keeps the
> channel **isotropic**: the embedding is handled as one geometric whole, not
> rescaled axis by axis, so the meaning your encoder put into it survives intact.

---

## What steering looks like

A working promptable policy shows one thing: from the *same* starting state,
instruction *A* and instruction *B* produce *different* behavior, each aimed at its
own goal — the policy is reading the instruction, not averaging it away.

The architecture makes this *possible* — a natural consequence of *action
generated, state (instruction included) given*. Whether a *particular* policy
steers **well** is up to your **data**: it has to actually show instructions
changing behavior. The shape is guaranteed; the steering is yours to teach.

---

To run a trained policy in your environment — the exact per-space input/output
contract at inference — see [spaces.md](../../../docs/spaces.md).

---

*The durable contract is the first line of Part 1: typed vectors in, decisions
out. The container (Minari) and the space set may grow, but the boundary — you
own the encoder, the model owns the policy — does not move.*
