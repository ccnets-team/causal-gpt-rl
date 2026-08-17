# The Input Contract

What `build_minari.py` reads: a directory of `ep_*.npz` files, plus an optional
`spec.json` beside them.

## Each `ep_*.npz` is one episode

| Array | Shape | Type |
|---|---|---|
| `observations` | `[T + 1, obs_width]` | float |
| `actions` | `[T, action_cols]` | float or int, see below |
| `rewards` | `[T]` | float |
| `terminations` | `[T]` | bool |
| `truncations` | `[T]` | bool |

`observations` is one longer than the rest: `T` transitions need `T + 1` states.
Exactly one episode boundary per file, at the end: the final step must set
`terminations` or `truncations`, and no step before it may. Files are read in
sorted order and the episode id is their position.

`actions` is stored flat, and how it is read depends on `action_kind`:

| `action_kind` | `actions` columns | Becomes |
|---|---|---|
| `continuous` | `act_dim` floats | `Box(act_low, act_high, (act_dim,))` |
| `discrete` | one **index** per branch — not one-hot | `Discrete(n)` or `MultiDiscrete([...])` |
| `hybrid` | `continuous_size` floats, then one index per branch | `Tuple(Box, Discrete/MultiDiscrete)` |

## `spec.json` declares the spaces

**The spaces are mandatory; `spec.json` is optional.** The dataset is env-less —
no gym env is attached — so Minari has nothing to infer spaces from and refuses
one without both:

```text
ValueError: Both observation space and action space must be provided, if env is None
```

`build_minari.py` therefore always declares them, defaulting to one continuous
`Box` over the full observation width — the same flat convention the Gymnasium /
MuJoCo Minari datasets follow. `spec.json` is how you say something else.

| Key | Applies to | Meaning |
|---|---|---|
| `action_kind` | all | `continuous` (default), `discrete`, or `hybrid` |
| `obs_channels` | all | Per-sensor widths summing to `obs_width`; `[126, 32]` → `Tuple(Box(126), Box(32))`. Omit for one channel |
| `act_dim` | continuous | Action width. Omit to take it from the array |
| `branches` | discrete, hybrid | Category count per branch, e.g. `[3, 3, 3]` |
| `continuous_size` | hybrid | Width of the continuous part |
| `act_low`, `act_high` | continuous, hybrid | Scalar or per-dimension continuous bounds. Default `-1`, `1` |

Splitting is the safer default: a consumer can always concatenate leaves, but it
cannot recover boundaries you never recorded.

Any extra keys are ignored, so `spec.json` is a fine place to keep provenance.

Beyond these flat-vector forms — and the optional ego-agent `Dict` wrapper in
[Packaging](03-packaging.md) — the runtime supports wider `Dict` / `Tuple`
nesting; this packager does not.

---

Next: [Record Trajectories](02-record-trajectories.md) — producing these arrays.
