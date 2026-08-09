# Checkpoints and Policy Bundles

Everything a Causal GPT-RL training job produces: the policy bundles it exports
while running, and the model artifact it writes when it finishes.

## Real-Time Policy Delivery

Training runs offline. There is no simulator or game engine inside the training
container, so the job cannot measure episode return — the thing you actually care
about. What it can measure is `eval_offline/checkpoint_score`, which says how
well the policy tracks your dataset, not how well it performs at your task.

That leaves one trustworthy check: **run the policy in your own environment.**

So the job does not make you wait for the final artifact. It exports runnable
policy bundles as it goes and SageMaker syncs them to the checkpoint S3 prefix
continuously, so you can load an in-progress policy with the public runtime,
score it yourself, and stop a run that is not learning.

## Two Destinations

| | Set by | Holds |
| --- | --- | --- |
| Checkpoint S3 prefix | `checkpoint_s3_uri` on the training job | Bundles synced while the job runs, plus `.pt` training state for resume |
| Output artifact | `output_path` on the training job | Final `model.tar.gz` |

Without a checkpoint prefix the job still produces the final artifact — you just
do not see anything until it finishes.

## During the Run

```text
s3://my-bucket/cgrl/checkpoints/<training-job-name>/
  <namespace>/
    archive/
      model_checkpoint_step_0020000.pt
      bundles/
        manifest.json
        step_0020000/
          config.json
          metrics.json
          model.safetensors
    improvements/
      model_checkpoint_slot_000.pt
      bundles/
        manifest.json
        slot_000/
```

Any `bundles/*/` directory is a complete inference bundle and loads with the
public `causal-gpt-rl` runtime directly. The `*.pt` files beside them are
training state for resume — see `training/docs/aws/sagemaker-retraining.md`.

`<namespace>` is derived from the run's dataset identity. Read it from
CloudWatch Logs; every export logs its full path.

```text
Checkpoint artifact exported: artifact=archive_bundle path=<ckpt>/<ns>/archive/bundles/step_0020000 step=20000 reasons=periodic
```

| Series | Written when | Named by | Retention |
| --- | --- | --- | --- |
| `archive/` | a scheduled or requested step is reached | step | permanent |
| `improvements/` | when a selected checkpoint is published | slot | 5 slots, rotating |

### `archive/` — the series to score

Points here are never rotated away and are named by step, so `step_0020000/` is
step 20000 for the life of the prefix. They are evenly spaced across the run and
announced before it starts, which is what makes them a basis for a decision.

Two things put a point in the archive: the periodic schedule — 5 evenly spaced
steps computed from `max_steps` at startup, at 20/40/60/80/100% — and the steps
you list in `archive_steps`. Both are reported in the startup log before the run
produces anything. A point is written the moment its step is reached.

The last archive point a run saves carries the reason `final`.

### `improvements/` — best so far

Slots rotate from `slot_000` through `slot_004`, so a slot name is a reusable
location rather than a model identity — read the associated `step`, and copy
anything you intend to keep out of the prefix.

This series answers "what is the best the offline metric has seen so far". It is
not a trend signal and not the series to drive early stopping from.

### Manifests

Each series has a `bundles/manifest.json`. Read the list from it rather than
hardcoding a count — early stopping and requested steps both change it.

```json
{
  "points": [
    { "step": 20000, "reasons": ["periodic"],
      "metrics": { "eval_offline/checkpoint_score": 0.3812 } }
  ]
}
```

- `archive/` lists `points`, keyed by `step`, with `reasons` — `periodic`,
  `requested`, `final`, `stopped`. The largest `final` step is the run's final
  saved weights.
- `improvements/` lists `slots`, each with its `step`, plus `canonical_source`:
  the slot and step whose weights match the canonical bundle. It can be `null`
  once rotation overwrites that slot; the canonical bundle is still valid at its
  own location.

S3 synchronization order is not guaranteed. Treat the manifest as an index,
verify the referenced directory, and retry entries that are not yet complete.

## Loading a Bundle

```bash
aws s3 cp --recursive \
  s3://my-bucket/cgrl/checkpoints/<job>/<ns>/archive/bundles/step_0020000/ \
  ./step_0020000/
```

```python
from causal_gpt_rl.inference import load_runner

runner = load_runner("./step_0020000")
```

From here it is the same `PolicyRunner` you would get from the final artifact or
from Hugging Face: `reset(obs)`, then `act()` and `observe(obs)` each step, in
your environment's own observation and action spaces. `run_episodes(env, runner,
num_episodes=...)` does that loop for a Gymnasium env and returns return
statistics.

See the [Quick Start](../../../README.md#quick-start) and
[docs/spaces.md](../../../docs/spaces.md) for the rollout loop and structured
spaces.

Treat a bundle that does not load as a skip: it may still be syncing, so retry it,
and the next delivery is unaffected either way. A bundle needing a newer runtime
than you have installed is refused by name rather than mis-decoded — upgrade the
package and the same bundle loads.

## Early Stopping

Poll `archive/bundles/manifest.json`, diff `points` against the steps you have
already scored, roll each new bundle out in your environment, and track your own
score against `step`. If the score is flat or falling, stop the job — you stop
paying at that point rather than at `max_steps`. A point carrying `final` is the
run ending on its own, not a reason to call `StopTrainingJob`.

The default schedule gives 4 opportunities to stop before the run ends. For a
finer trend, add up to 10 steps with `archive_steps`, chosen before the run
starts; each added point costs permanent disk, so see the sizing table in
`training/docs/aws/sagemaker-inputs.md`.

## The Final Artifact

```text
s3://my-bucket/cgrl/output/<training-job-name>/output/model.tar.gz
```

```text
model.tar.gz
  bundle/
    config.json
    metrics.json
    model.safetensors
  archive/bundles/
    manifest.json
    step_NNNNNNN/
  reports/
    summary.json
```

- `bundle/` is the canonical bundle, selected by the checkpoint metric. Load this
  by default.
- `archive/bundles/` is the same preserved series as above, included so the run's
  candidates can be compared after the job ends without going back to the
  checkpoint prefix.
- `improvements/` and the `.pt` files are not included.
- There is no namespace level and no root `config.json`. The namespace applies to
  the checkpoint prefix only.

### Bundle files

- `model.safetensors`: Policy model weights.
- `config.json`: Model architecture, observation/action specs, and context settings.
- `metrics.json`: The evaluation metrics for that point.

The first two are the bundle format itself, which the public runtime defines —
see [Bundle Format](../../../README.md#bundle-format). `metrics.json` is added by
training and is not read at inference.

```python
runner = load_runner("path/to/bundle")
```

To try an archive candidate instead, point the runtime at its directory. The path
and the bundle format are the same runtime contract, so nothing else changes.

```text
MODEL_PATH=/opt/ml/model/archive/bundles/step_NNNNNNN
```

### `reports/summary.json`

Records how the canonical bundle was selected.

```json
{
  "evaluation": {
    "best_metric_name": "eval_offline/checkpoint_score",
    "best_metric_value": 0.418732,
    "best_metric_direction": "max",
    "best_return": null
  }
}
```

- `best_metric_value` ranks points inside one run. Do not compare it across runs.
- `best_metric_direction` is a separate field because the direction is not part
  of the metric name. Use it when comparing checkpoints selected under the same
  run and metric definition — sorting the wrong way picks the worse model.
- `best_return` is `null` on an offline-selected run — absent, not zero. It is
  filled in only when the selection metric is an actual episode return.

## Choosing What to Deploy

The selection metric is measured on a held-out split of your dataset, not against
your environment. It does not tell you how the policy performs at your task, and
the gap can be large. See `training/docs/aws/checkpoint-score.md`.

So **treat the canonical bundle as a default, not as a verdict.** The archive
points are preserved precisely so you can score them yourself and pick the one
your environment prefers.
