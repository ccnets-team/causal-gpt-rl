# Journey

A four-part read on what Causal GPT-RL is and how you steer it, in order. Start
here if you are new.

![The reinforcement learning loop — an agent acts, the environment returns an observation and a reward, and trajectories of those steps make the dataset you hand over](../assets/reinforcement-learning-loop.svg)

That loop is where your data comes from. In offline RL the agent in it is yours
— a scripted policy, a prior model, a human operator — and ours is trained
afterwards, from the dataset it produced.

1. [Bring Your Own Data](01-bring-your-own-data.md) — a GPT-shaped policy
   trained small on your recorded data, and the spaces you declare for it
2. [The Acting Policy](02-the-acting-policy.md) — how the model runs: load a
   bundle and roll it out
3. [Shaping Behavior Through State](03-shaping-behavior-through-state.md) —
   steering what the policy does by what you put in its state
4. [A Policy You Can Prompt](04-a-policy-you-can-prompt.md) — that same channel
   expressed in natural language

Parts 1–3 are enough for most control, game, and app problems.
