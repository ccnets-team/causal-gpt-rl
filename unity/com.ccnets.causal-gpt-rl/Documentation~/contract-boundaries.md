# Integration contracts not validated at runtime

Read this before deployment.

The runtime validates **sizes**. The properties below are invisible to a size
check. A mismatch may therefore produce valid inference with incorrect actions.
Integration code must enforce these contracts between the scene and the bundle.

| Contract | Runtime behaviour or integration requirement |
|---|---|
| Branch split when different layouts have the same total width | Not validated. `[3,3,3]` and `[2,2,5]` are both width 9 |
| Observation packing when different layouts have the same total width | Not validated |
| Order, scaling, and range of continuous components | Not validated. The decoder only clips to `[-1, 1]` |
| Stable agent-to-row mapping | The integration adapter must preserve the mapping |
| Active agents exceeding the fixed batch size | Not defined by this API; integration code must keep active rows within the fixed batch |
| Inputs, masks, and ignored outputs for inactive or padded rows | Not defined by this API; integration code must define and apply a consistent policy |
| First-decision offset and ordering around terminal and reset events | Not defined by this API; integration code must define the timing |
| Decision interval, action repeat, `fixedDeltaTime`, and observation sampling time | Not defined by this API; integration code must keep these consistent with training |
| Order and meaning of discrete heads | Not validated. The decoder emits an argmax index but cannot determine its environment semantics |
| Units, normalisation, and clipping of observations | Not validated. Only tensor sizes are compared |
| Decision cadence across active rows, and the batch barrier | **Lockstep is required.** Every row must report before the next batched action. Staggered per-agent decisions, or a different action repeat per row, cannot be expressed through this API — your adapter must synchronise them or hold the previous action |
| Pairing an ONNX model with a different config of the same shape | Not validated |

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
