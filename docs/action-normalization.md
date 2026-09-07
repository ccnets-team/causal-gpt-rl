# Bundle compatibility

Normalization and rollout feedback settings are supplied with the delivered
bundle. Load it normally with `load_runner`; no manual configuration is needed.

| Bundle requirement | Minimum runtime |
|---|---|
| `action_normalization` | `causal-gpt-rl>=0.19.0` |
| `pre_tanh_rollout_context` | `causal-gpt-rl>=0.20.0` |

Existing bundles retain their behavior on the newer runtime. An unsupported or
inconsistent bundle is rejected during loading. Use the runtime version required
by the delivered bundle rather than editing its settings.

The runner returns actions in the environment's declared action space. See the
[API reference](api.md) for usage and the [ONNX export guide](export-onnx.md) for
deployment outside Python.
