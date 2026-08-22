# Why a bundle was refused

`BundleValidator` runs when you `Load`. It refuses anything this runtime cannot
serve and names the reason. The rule behind the list is narrow on purpose:

> **The accepted set equals what `ActionCodec` can express and a fixture has
> actually exercised.** Implemented but unexercised is still refused.

That makes this gate narrower than the Python loader's. A refusal here is not a
statement that your bundle is broken — often it means we have not verified that
shape yet.

## Refusals

| Refused | Why |
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

## What is deliberately *not* checked

`action_container` is classified as irrelevant to this path rather than
supported. That is a real distinction: the question does not arise here, because
the ONNX path never reconstructs the container. It returns a flat environment
action — continuous values clipped, plus one argmax index per branch.

Porting the container refusal from the PyTorch loader was a category error. It
rejected published multi-agent bundles that run correctly, because those declare
a nested `{"agents": {"agent_0": ...}}` container that this path never reads.
The only thing a container genuinely determines here is the `start` offset,
which is why that check recurses into nested leaves and the rest was removed.

## What the graph check can and cannot do

`ValidateGraph` compares context length, state size, and action size between the
config and the ONNX file, requires all four inputs (`states`, `actions`,
`is_bos`, `mask`) to be float32 with static shapes, and refuses any extra input
— an unexpected input would never be set and the graph would silently run on a
default.

> The dtype and extra-input refusals are **implemented but not covered by a
> negative test.** Reaching either needs a malformed ONNX that the Inference
> Engine still imports, and we have not produced one. Every other refusal on this
> page is exercised by a test.

It only requires batch ≥ 1. **`config.json` does not declare a batch**; it is
baked into the graph. Whether that batch matches your scene is a contract you
hold, because the runner never sees your scene.

Branch *structure* has the same problem: the graph carries only the total action
width, so `[3,3,3]` and `[2,2,5]` both pass as width 9. See
[contract-boundaries.md](contract-boundaries.md).
