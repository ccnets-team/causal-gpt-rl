# Causal GPT-RL for Unity

Run a Causal GPT-RL policy bundle inside Unity, in-process, with no Python at
runtime. The package owns the rolling context window, calls the Inference
Engine, and decodes the result into an action your game can apply.

Verified against **Unity 6000.0** and **`com.unity.ai.inference` 2.6.1**. Those
are the only versions the package declares; wider support is not claimed until
it is measured.

> **Pre-1.0.** The API may change before 1.0. UPM git dependencies pin a
> revision, so an existing install does not break when it does.

## Install

Package Manager → **Install package from git URL**:

    https://github.com/ccnets-team/causal-gpt-rl.git?path=/unity/com.ccnets.causal-gpt-rl

Pin a revision by appending `#<tag-or-commit>`. The `?path=` and `#revision`
order matters — path first.

## Use

```csharp
using CCNets.CausalGPTRL;

var runner = PolicyRunner.Load(policyAsset, configAsset.text, BackendType.GPUCompute);
runner.Reset(observations);                      // batch * StateSize, flattened

// each decision tick
var request = runner.RequestAction();            // schedules; does not wait on readback
// ... your game logic ...                       // spend the gap, then read back below;
var action = request.GetAction();                // or poll IsDone across frames instead

var branchCount = runner.ActionLayout.BranchSizes.Count;
for (var row = 0; row < runner.BatchSize; row++) {
    var steering = action.Continuous(row);       // view, no allocation
    for (var branch = 0; branch < branchCount; branch++) {
        var choice = action.Discrete(row, branch);   // this branch's index, int
    }
}

runner.ResetRows(finishedRows);                  // only rows whose episode ended
runner.Observe(nextObservations);                // or ObserveRow(row, obs) per row
```

`ActionLayout` says which outputs a bundle has — a purely continuous policy has
no branches, so indexing one throws. Drive the loop from `BranchSizes.Count`
rather than assuming a branch exists. Bundles that mix continuous and branch
outputs are **refused at load** for now: the decode path exists but no fixture
covers it, so the gate would rather say why than act on an unverified decode.

`Documentation~/` covers the call order, what the bundle gate accepts, and the
contract boundaries the runtime cannot check for you. Read
`Documentation~/contract-boundaries.md` before shipping — several correctness
properties are your side of the contract, not ours.

## What this package is not

- **It does not train.** Training is not part of this repository's Unity
  surface; the package only runs a bundle that already exists.
- **It does not pack your observations.** You hand it a flat `float[]`, and it
  must be packed the same way the trajectories behind your bundle were.
- **It does not know your scene.** Agent-to-row mapping, spawn/despawn, and
  decision timing are the adapter's job.

## Samples

**Quickstart** — a single component that drives one bundle through the full
loop. Import it from the Package Manager's Samples tab. It does not embed a
model; point it at your own bundle.

## License

Released under PolyForm Noncommercial License 1.0.0. See `LICENSE.md` for details.
Commercial use is licensed separately — through a CCNets Causal GPT-RL Training
Algorithm subscription on AWS Marketplace, or by contacting the maintainers via
ccnets.org.
