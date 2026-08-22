# Causal GPT-RL for Unity — documentation

Use the following guides according to the question you need to answer.

| If you are asking | Read |
|---|---|
| "In what order do I call this, and what happens if I get it wrong?" | [lifecycle.md](lifecycle.md) |
| "Why was my bundle refused?" | [bundle-gate.md](bundle-gate.md) |
| "What can go wrong that the runtime will *not* tell me about?" | [contract-boundaries.md](contract-boundaries.md) |

Read the third one before shipping. The runtime checks sizes; several
correctness properties are invisible to a size check and are held by contract
between your packing code and the bundle.

## The shape of it

```text
your game                    this package                  Inference Engine
─────────                    ────────────                  ────────────────
pack observations  ──────▶   WindowContext (rolling)
                             BundleConfig + Validator
                             PolicyRunner            ──────▶  ONNX graph
apply action       ◀──────   ActionCodec (decode)    ◀──────  raw output
```

The package handles the middle column. Integration code is responsible for
packing observations, mapping agents to batch rows, and choosing when decisions
occur.

## Versions

Verified against Unity 6000.0 and `com.unity.ai.inference` 2.6.1. The package
declares that exact Inference Engine version — UPM has no range syntax, and a
wider claim would be a guess until it is measured.

## Where the behaviour is defined

The reference implementation for this runtime is the ONNX evaluation path in
this repository (`examples/unity/evaluate_onnx.py`). It takes its contract from
two places: the ONNX graph's shapes, and a live environment's action spec. This
package mirrors that path. Where a live environment would supply the branch
layout, the bundle's declared `config.action_specs` stands in.

This differs from the PyTorch loading path in the same repository, which reads
the complete `config.json` and reconstructs declared containers. Applying those
container checks to the ONNX path would incorrectly reject compatible bundles.
See [bundle-gate.md](bundle-gate.md).
