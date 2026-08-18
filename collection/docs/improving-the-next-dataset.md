# Improving the Next Dataset

*Optional, and outside the packaging path.* [The path](README.md) ends at a
dataset. This is what the dataset you just built can be used for, if you intend
to collect again.

## The cycle

A packaged dataset trains a policy. That policy, run with more history retained
than it was trained on, acts in the environment and records the experience that
becomes the next dataset. A seed policy — SB3's PPO, a scripted controller,
whatever already drives your system — is needed only for the first turn of the
cycle.

![Collecting better RL data with transformer context extrapolation — train a transformer policy on the current dataset, deploy it with a longer context, collect new experience, and use that experience as the next dataset](assets/collecting-better-rl-data-with-context-extrapolation.svg)

This closes the one gap the [entry table](../README.md#where-you-join) leaves
open. Entry point 1 has an environment and still owes you a policy model; after
a cycle you have one, and it is what the previous dataset produced.

## Where the gain is

The cycle turns on running the trained policy with more retained history than
its trained window — `kv_cache_max_len` above `context_length`. What that buys
is a step up in the *tier* of what you record: simple trajectories train a
policy that, deployed at a longer context, records the next dataset at medium.
The simple-to-medium crossing is where extrapolation pays, and it is what makes
a second collection run worth doing at all.

It is not a knob that keeps paying, though, and the closest published evidence
is a different measurement — an inference-time retention sweep over bundles
already trained across simple and medium, where doubling the window helps some
environments and hurts others (`Humanoid-v5` improves, `HumanoidStandup-v5`
falls, on the [model card](https://huggingface.co/ccnets/causal-gpt-rl)). So
measure the retention you mean to record at, in your own environment, before
spending a collection run on it.
