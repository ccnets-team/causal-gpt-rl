# Bundle validation

`BundleValidator` runs during `Load`. It rejects unsupported bundles and reports
the reason. The accepted set is intentionally narrow:

> **The accepted set equals what `ActionCodec` can express and a fixture has
> actually exercised.** Implemented but untested configurations remain
> unsupported.

This gate is narrower than the Python loader's. Rejection does not necessarily
mean that the bundle is invalid; it may indicate that the configuration has not
been verified in the Unity runtime.

## Rejected configurations

| Configuration | Reason |
|---|---|
| `bundle_format_version` outside {1, 2} | A newer layout may move the fields being read |
| `hybrid_action` or an unknown token in `requires_capabilities` | Not implemented here |
| `bos_cache_mode` other than `discard` | Only the discard path is implemented; no `retain` fixture exists |
| A non-continuous state spec | One-hot and continuous-first ordering cannot be verified by size |
| `multi_binary` actions, or an unknown action type | Needs a per-leaf Bernoulli threshold, not argmax |
| Continuous action bounds other than `[-1, 1]` | The decode clips to `[-1, 1]`, which would silently ignore the declaration |
| A non-zero `start`, **including on nested leaves** | The decode emits 0-based indices and adds no offset, so a non-zero start silently shifts every action |
| An unknown container type, or a malformed `Dict` pair | Nothing to infer from |
| A continuous head declared after a branch | The decode reads continuous columns first and would slice the wrong span |
| Mixed continuous + branch schedules | Same code path as the parts, but no fixture covers it |
| `config` and graph disagreeing on context / state / action size | A mismatched pair builds the window to the wrong width |

`discrete` and `multi_discrete` are **accepted** — implemented and exercised by
fixtures. A single-branch schedule declares no capability at all, so the gate
inspects `action_specs` directly rather than trusting `requires_capabilities`.

## `action_container` handling

The ONNX path does not reconstruct `action_container`; it returns a flat
environment action containing clipped continuous values and one argmax index per
branch. Container structure is therefore outside this path rather than a
supported runtime feature.

Applying the PyTorch loader's container checks here would incorrectly reject
compatible multi-agent bundles that declare a nested
`{"agents": {"agent_0": ...}}` container. Of the container's action semantics,
this path uses only the `start` offset. Validation checks that field recursively
and requires known, well-formed space declarations, but it does not reconstruct
the container.

## What the graph check can and cannot do

`ValidateGraph` compares context length, state size, and action size between the
config and the ONNX file, requires all four inputs (`states`, `actions`,
`is_bos`, `mask`) to be float32 with static shapes, and rejects any extra input
— an unexpected input would never be set and the graph would silently run on a
default.

> The dtype and extra-input checks are **implemented but not covered by a
> negative test.** Reaching either needs a malformed ONNX that the Inference
> Engine still imports, and we have not produced one. Every other refusal on this
> page is exercised by a test.

It only requires batch ≥ 1. **`config.json` does not declare a batch**; it is
baked into the graph. Integration code must verify that the graph's batch size
matches the scene because the runner cannot inspect the scene.

Branch *structure* has the same problem: the graph carries only the total action
width, so `[3,3,3]` and `[2,2,5]` both pass as width 9. See
[contract-boundaries.md](contract-boundaries.md).
