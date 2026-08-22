# Changelog

All notable changes to this package are documented here. This package versions
independently of the Python package in this repository; its tags are namespaced
`unity-v*`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

First promotion out of local development. Verified on Unity 6000.0.40f1 with
`com.unity.ai.inference` 2.6.1: **95 EditMode tests, 0 failures**.

### Added

- `PolicyRunner` — the entry point. `Load`, `Reset`, `Observe`, `ObserveRow`,
  `ResetRows`, `RequestAction`/`ActionRequest.GetAction`, `Act`, `Dispose`.
  `RequestAction`/`GetAction` split so a caller can spend the gap on its own
  work instead of stalling on readback.
- `BundleConfig` and `BundleValidator` — parse `config.json` and reject bundles
  this runtime cannot serve, naming the reason.
- `WindowContext` — the rolling context window, including per-row episode
  restart (`ResetRows`) and the BOS convention that goes with it.
- `ActionCodec` and `ActionLayout` — continuous, `discrete`, and
  `multi_discrete` decoding. Two widths are distinguished: the model width
  (continuous + every branch's logits, fed back into the window) and the
  environment width (continuous + one index per branch).
- `DecodedAction` — allocation-free `Continuous(row)` / `Discrete(row, branch)`
  accessors, plus `CopyDiscrete`.
- A state machine that rejects invalid call sequences instead of serving a stale
  observation: acting twice without observing, acting while a row has not
  reported, resetting a row that already reported, and requesting while a
  request is outstanding.
- `LICENSE.md` — PolyForm Noncommercial License 1.0.0, matching the repository.
  Commercial use is licensed separately through a CCNets Causal GPT-RL Training
  Algorithm subscription on AWS Marketplace, or by contacting the maintainers
  via ccnets.org.

### Known limits

- **Hybrid actions (continuous + branches) are not supported.** The decode path
  exists, but no fixture exercises it, so validation rejects mixed bundles.
- **`bos_cache_mode: retain` is not supported.** Only `discard` is implemented.
- **`multi_binary` actions are not supported.** They need a per-leaf Bernoulli
  threshold rather than argmax.
- **`AddRows` does not exist.** The batch is baked into the graph, so rows
  cannot grow.
- Batch is fixed by the exported graph. Integration code must ensure that
  `batch ∈ {1, num_agents}` because the runtime cannot inspect the scene.
- Several correctness properties cannot be validated from tensor sizes and must
  be enforced by integration code. They are listed in
  `Documentation~/contract-boundaries.md`.

[0.1.0]: https://github.com/ccnets-team/causal-gpt-rl/releases/tag/unity-v0.1.0
