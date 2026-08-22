# Causal GPT-RL for Unity — documentation

Three documents, split by the question you are asking.

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

The package owns the middle column only. Packing observations, mapping agents to
batch rows, and deciding *when* a decision happens are all yours.

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

That distinction matters, and getting it wrong is the most expensive mistake
made while building this package. The PyTorch loading path in the same
repository reads the whole `config.json` and reconstructs declared containers;
this path does not, and porting that path's refusals here rejects bundles that
in fact run correctly. See [bundle-gate.md](bundle-gate.md).
