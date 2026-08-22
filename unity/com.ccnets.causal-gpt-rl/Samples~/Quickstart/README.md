# Quickstart

One component, `QuickstartAgent`, that drives a bundle through a full step.

## Setting it up

1. Import this sample from the Package Manager's **Samples** tab.
2. Put `QuickstartAgent` on a GameObject.
3. Assign **Policy** — your `.onnx`, imported as a `ModelAsset`.
4. Assign **Config** — the bundle's `config.json`. Unity imports `.json` as a
   `TextAsset`, so drop it in as-is.
5. Pick a **Backend**. `GPUCompute` is the usual choice; `CPU` is there for
   determinism and for machines without a usable compute device.

The sample does not include a model. Assign the model and config from your own
bundle.

## Required integration points

Two stubs, both marked in the file:

- `PackObservations(float[] destination)` — write `BatchSize * StateSize` values,
  row by row. **This must match the packing behind your bundle.** The runtime
  checks the length but cannot validate field order; an incorrect order can
  produce valid inference with incorrect actions.
- `ApplyAction(row, continuous, branches)` — apply one row's action and return
  `true` if that row's episode just ended. `branches` contains one chosen index
  per declared branch and is empty for a purely continuous policy.

## Asynchronous request flow

The split request. `RequestAction()` schedules and returns immediately;
`IsDone` is polled and `GetAction()` reads back. In `Update` that means a
decision costs at least two frames, which is the point — the frame that
schedules does not wait for the GPU. For synchronous execution, call
`runner.Act()` and remove the polling.

The step order also matters and is enforced: apply the action, then
`ResetRows` for the rows that finished, then `Observe`. Getting that order
wrong throws with a reason rather than corrupting the context window.

## Errors

Most API misuse throws an exception with a specific message. If `Load` throws,
the bundle is outside the runtime's supported set. See
`Documentation~/bundle-gate.md` for the validation rules.
