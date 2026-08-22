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
| How exactly an observation must be reproduced | Not validated, and not a matter of physical significance — see below |
| Decision cadence across active rows, and the batch barrier | **Lockstep is required.** Every row must report before the next batched action. Staggered per-agent decisions, or a different action repeat per row, cannot be expressed through this API — your adapter must synchronise them or hold the previous action |
| Retiring an ended row on a pipelined turn | Not expressible: run that turn in the serial order — see below |
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

## How close an observation has to be

"Close enough" is not decided by what the number means. It is decided by how much that
channel varied while the bundle was recorded, because the bundle normalises each channel
by its own standard deviation before the model sees it.

A channel that never varied is the dangerous case. Its recorded deviation is not zero but
the small constant the recorder adds to avoid dividing by it, so the divisor can land near
1e-8. A difference of 1e-5 in such a channel — far below anything a physical argument would
call significant, and well inside float32 precision — then reaches the model as **hundreds
of standard deviations**, on every step, for as long as the policy runs.

Nothing else reports it. Sizes match, the environment behaves the same, and a comparison
against a recorded observation shows a difference of 1e-5. Only the actions are wrong.

Two consequences for integration code:

- **Do not re-derive a value the recording took as constant.** Masses, inertias, link
  lengths, gear ratios: carry them through exactly. Anything that rebuilds them — an editor,
  an importer, a level tool — can move them by float32 precision, and that is enough.
- **A tolerance justified by physical significance is the wrong tolerance.** If a comparison
  needs one at all, it belongs per channel, scaled by that channel's recorded deviation.

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

## Retiring a row while an action is in flight

Overlapping the forward pass with the environment step means stepping while an
action is in flight, so **every termination happens inside that window**, not
between turns. `ResetRows` runs only between an action and the observations that
follow it, and the pipelined turn never enters that state: staging the
observation and reading the action leaves nothing outstanding, which is exactly
what makes the next schedule immediate.

**A whole-batch restart is fine.** Cancel the pending action — the visible window
has not moved, so nothing is lost — and `Reset` is accepted from `Ready`. A scene
running one agent restarts this way and never meets the limit below.

**Retiring one row of several means running that turn serially.** Do not stage
the observation on a turn where a row ended: read the action from `InFlight`
instead, which lands in `AwaitingObservations`, then `ResetRows` the rows that
finished and `Observe` the new observations. That turn pays the wait the pipeline
exists to remove, once per termination.

Do not reach for `Cancel` to escape this. It returns the runner to `Ready`, where
`ResetRows` is refused just the same, and the row stays un-retired. Cancel is for
abandoning a step nobody will finish — a scene tearing down, a read that failed.

Nothing here is validated, and nothing about a mishandled turn looks wrong until
a respawning agent is served the row it inherited.

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
