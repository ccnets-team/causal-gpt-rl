# Shaping Behavior Through State

*Part 3 of 4. [Part 2](02-the-acting-policy.md) gave you a running agent — it reacts
to whatever state you supply. This part is the skill the rest of the product is
built on: you shape what the policy *does* by what you put in its state.
[Part 4](04-a-policy-you-can-prompt.md) takes that skill to natural language.*

---

## The skill: behavior is state

Recall the one fact from Part 2 — **the state is simply *given* to the model each
step.** That fact is a design tool: anything you add to the observation, the model
learns to condition on. Two moves cover most needs — tell it *what's allowed*, and
tell it *what to pursue*. Both are just channels on your `observation_space`.

---

## Action masking — what's allowed

A common instinct from game RL: *"invalid actions should never come out of the
model."* So people reach for output masking.

That doesn't fit here. Because the action feeds back into the model (Part 2), **the
AR loop is inviolable: the model always emits an action.** You cannot mask its
output without breaking the sequence.

Instead, you inform the model **through its state**: add an action-mask channel, and
the model learns to condition on what's currently valid. Take a turn-based game,
where only some moves are legal each turn — expose that as a channel:

```python
observation_space = gym.spaces.Dict({
    "board":       gym.spaces.Box(-np.inf, np.inf, shape=(64,)),  # your encoded board
    "legal_moves": gym.spaces.MultiBinary(32),                    # 1 = playable this turn
})
```

The hard guarantee at the environment boundary — never *acting* on an illegal move
— is the **adapter's** job, not the model's: it filters the emitted action into a
legal one for `env.step(...)`. Model informed by state, boundary enforced by
adapter.

---

## Instructions — what to pursue

Masking tells the model *what's allowed*. The same mechanism tells it *what to
pursue*: add a **goal channel** to the state.

Because the state is simply *given*, an **instruction is just more state** — no new
mechanism. It's fixed for the whole episode (it *is* the goal for that run), so you
re-supply the same value every step. It needs no encoder — a task id or a target
vector is enough. In the same game, condition the policy on which objective to
chase:

```python
observation_space = gym.spaces.Dict({
    "board":       gym.spaces.Box(-np.inf, np.inf, shape=(64,)),
    "legal_moves": gym.spaces.MultiBinary(32),
    "objective":   gym.spaces.Discrete(4),   # e.g. rush / defend / economy / explore
})
```

The same two channels carry straight over to non-game apps. A task-routing agent
decides which queue a request goes to — masking the queues that are currently full,
conditioned on the operating mode you want:

```python
observation_space = gym.spaces.Dict({
    "request":     gym.spaces.Box(-np.inf, np.inf, shape=(32,)),  # your encoded request
    "open_queues": gym.spaces.MultiBinary(8),                     # 1 = queue can take work now
    "mode":        gym.spaces.Discrete(3),   # e.g. speed / cost / balance
})
```

Masking constrains *possibility*; an instruction sets *intent*. Different purposes,
same skill: **you're just defining your state space.**

---

## Run it

You now have a complete, steerable agent: it reacts to state, respects what's
currently valid, and pursues a goal. You run it exactly as in
[Part 2](02-the-acting-policy.md) — `reset` / `act` / `observe` — now passing your
full `Dict` observation each step. The loop doesn't change; the richer state is the
only difference. (For the exact per-space input/output contract, see [spaces.md](../../../docs/spaces.md).)

> 🚉 **You can stop here.** With Parts 1–3 you can prepare data, train, and run a
> policy that reacts to state, respects valid actions, and pursues a goal — enough
> for most control, game, and app problems.
> [Part 4](04-a-policy-you-can-prompt.md) takes the *goal* one step further:
> expressing it in natural language.

---

**Next → [A Policy You Can Prompt](04-a-policy-you-can-prompt.md)** — the goal, in
words.
