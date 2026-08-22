# What the runtime cannot check for you

Read this before shipping.

The runtime validates **sizes**. The properties below are invisible to a size
check, so nothing here will fail loudly — a mismatch produces a policy that runs
happily and acts wrongly. Each one is held by contract between the code that
packs your data and the bundle you are running.

| Boundary | Status |
|---|---|
| A wrong branch split with the same total width | Undetected. `[3,3,3]` and `[2,2,5]` are both width 9 |
| A wrong observation packing with the same total width | Undetected |
| The **order** of continuous components, and the environment's own scaling and range | Undetected. The decode knows only "clip to `[-1, 1]`" |
| Stability of the agent → row mapping | Outside the runtime. Your adapter owns it |
| What happens when active agents outnumber the fixed batch | Undefined so far |
| Inputs, mask, and output-ignoring rules for inactive or padded rows | Undefined so far |
| First decision offset, and ordering around terminal and reset | Undefined so far |
| Decision interval, action repeat, `fixedDeltaTime`, when observations are sampled | Undefined so far |
| The **order** of discrete heads, and what an index means in your environment | Undetected. We emit an argmax index; we do not know that "2" means turn left |
| Units, normalisation, clipping — the **value** semantics of observations | Undetected. Only sizes are compared |
| Decision cadence across active rows, and the batch barrier | **Lockstep is required.** Every row must report before the next batched action. Staggered per-agent decisions, or a different action repeat per row, cannot be expressed through this API — your adapter must synchronise them or hold the previous action |
| Pairing a different ONNX with a config of the same shape | Undetected |

## Why the last one is hard to fix

Recording a hash of the `.onnx` file in the bundle is not enough. That hash
belongs to the original file; what the runtime receives is a `ModelAsset` that
Unity has already imported — not the original bytes. Closing it needs one of:

1. a companion asset written at import time carrying the original hash, or
2. an export id embedded in the ONNX metadata and still readable after import.

Both belong with a deployment manifest, which does not exist yet.

## What "accepted" means for observations

An all-continuous state spec is accepted. **That is not a statement that your
packing is correct.** Swapping two `Box` leaves keeps the total size identical.
Acceptance rests on the contract that you concatenate in declared order — the
same order used when the trajectories behind the bundle were recorded.

The long-term fix is ordered channel metadata and a schema fingerprint in the
artifact, so the validator can check a **schema id instead of a size**. Until
then, this is yours.

## The honest summary

Sizes are checked. Order, meaning, timing, and identity are not. If you change
how you pack observations, or which ONNX you pair with which config, no error
will tell you — the agent will simply get worse.
