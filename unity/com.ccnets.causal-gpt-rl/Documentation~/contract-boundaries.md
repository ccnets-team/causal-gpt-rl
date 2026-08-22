# Integration contracts not validated at runtime

Read this before deployment.

The runtime validates **sizes**. The properties below are invisible to a size
check. A mismatch may therefore produce valid inference with incorrect actions.
Integration code must enforce these contracts between the scene and the bundle.

| Contract | Runtime behaviour or integration requirement |
|---|---|
| Branch split when different layouts have the same total width | Not validated. `[3,3,3]` and `[2,2,5]` are both width 9 |
| Observation packing when different layouts have the same total width | Not validated |
| Order, scaling, and range of continuous components | Not validated. The decoder only clips to the declared bounds; it cannot tell whether a column means what the environment expects |
| Stable agent-to-row mapping | The integration adapter must preserve the mapping |
| Active agents exceeding the fixed batch size | Not defined by this API; integration code must keep active rows within the fixed batch |
| Inputs, masks, and ignored outputs for inactive or padded rows | Not defined by this API; integration code must define and apply a consistent policy |
| First-decision offset and ordering around terminal and reset events | Not defined by this API; integration code must define the timing |
| Decision interval, action repeat, `fixedDeltaTime`, and observation sampling time | Not defined by this API; integration code must keep these consistent with training |
| Order and meaning of discrete heads | Not validated. The decoder emits an argmax index but cannot determine its environment semantics |
| Where a decoded action lands in the host's action structure | Not validated. Nothing this runtime returns says which of the host's fields each value belongs in — see below |
| Units, normalisation, and clipping of observations | Not validated. Only tensor sizes are compared |
| Decision cadence across active rows, and the batch barrier | **Lockstep is required.** Every row must report before the next batched action. Staggered per-agent decisions, or a different action repeat per row, cannot be expressed through this API — your adapter must synchronise them or hold the previous action |
| Pairing an ONNX model with a different config of the same shape | Not validated |

## Delivering a decoded action

Verification stops at `DecodedAction`. How its values reach the environment is
the adapter's, and a bundle that carries both kinds of head fills **two
destinations, not one**: continuous columns and branch indices are separate
fields in most engines, including ML-Agents' `ActionBuffers`.

Filling only the continuous half is silent for as long as no branch is read.
It works for every continuous-only bundle, and the first bundle with a discrete
head indexes an empty segment.

A size check cannot see this. The decoded action is the right width either way —
the values simply never arrive.

## When to sample the observation

Two timings are not the runtime's to enforce, and both produce a full-width,
correctly shaped, wrong observation.

**The first one.** An environment that randomises at episode start has not done
so while it is still being wired: sensors read serialized defaults then, so an
observation taken during setup describes an episode that never ran. Sample the
first observation only once the environment reports that its episodes have
begun.

**Every one after that.** Engines that run all fixed-step callbacks before
stepping physics make "read the observation right after applying the action"
predate that action's effect entirely. The pair `(state, action)` then goes into
the window misaligned — not delayed, but describing an action the environment
had not yet taken. Sample instead at the start of the next decision, after a
full decision interval of physics.

Both are invisible in a size check and in a replay against recorded inputs; only
a live environment shows them.

## Why model and config identity is not validated

Recording a hash of the `.onnx` file in the bundle is not enough. That hash
belongs to the original file; what the runtime receives is a `ModelAsset` that
Unity has already imported — not the original bytes. Addressing this requires
one of:

1. a companion asset written at import time carrying the original hash, or
2. an export id embedded in the ONNX metadata and still readable after import.

A future deployment manifest could carry this identity metadata.

## What acceptance guarantees for observations

An all-continuous state spec is accepted. **That is not a statement that your
packing is correct.** Swapping two `Box` leaves keeps the total size identical.
Integration code must concatenate fields in declared order — the same order used
when the trajectories behind the bundle were recorded.

An artifact format with ordered channel metadata and a schema fingerprint would
allow the validator to check a **schema ID instead of a size**. Until then,
integration code must preserve the packing order.

## Summary

The runtime validates sizes, not order, meaning, timing, or model/config
identity. Integration tests must validate changes to observation packing and
model/config pairing because the runtime may continue without reporting an
error.
