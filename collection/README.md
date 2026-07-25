# collection

Turn recorded episodes into a [Minari](https://minari.farama.org) dataset.

`build_minari.py` packages raw per-episode `.npz` files — `observations`
(length T+1) and `actions` / `rewards` / `terminations` / `truncations` (length
T), with an optional sibling `spec.json` declaring the action kind — into an
**env-less** Minari dataset: the observation and action spaces are declared
explicitly and no gym env is attached, so `minari.load_dataset(id)` returns a
dataset with no `recover_environment()`, following the same flat-`Box`
convention as the Gymnasium / MuJoCo Minari datasets.

It is **source-agnostic** — the episodes can come from any environment.

```bash
python collection/build_minari.py \
    --raw raw/ \
    --dataset-id <namespace>/<name>/<version>
```

For a multi-agent recording — one episode per physical agent — add
`--ego-agent` to nest both spaces under an ego key, so a consumer reads
`observations["agents"]["agent_0"]`:

```bash
python collection/build_minari.py \
    --raw raw/ \
    --dataset-id <namespace>/<name>/<version> \
    --ego-agent agent_0
```

The leaf spaces are identical either way; the wrapper only names whose
trajectory the episode is.

Runs with `minari==0.5.3`.

## Example source

A worked example that records the episodes first (drive a Unity ML-Agents build
with a policy, write per-episode `.npz`) lives at
[`../examples/unity_collection/`](../examples/unity_collection/).
