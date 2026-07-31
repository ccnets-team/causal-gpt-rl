# Real-Time Policy Delivery

Causal GPT-RL training exports runnable policy bundles while the job is running.
Policies are exported to the checkpoint directory as soon as they are produced,
and SageMaker syncs that directory to S3 continuously. When a bundle becomes
visible in S3 depends on that sync completing. Nothing waits for the final model
artifact.

## Why this exists

Training runs offline. There is no simulator or game engine inside the training
container, so the job cannot measure episode return — the thing you actually
care about. What it can measure is `offline_eval/action_nll`, which tells you how well
the model predicts your dataset's actions, not how well the resulting policy
performs at your task.

That leaves exactly one trustworthy check: **run the policy in your own
environment.** Your environment is the only place it can happen.

Real-time delivery is what makes that possible while the job is still running.
Instead of paying for a full training run and evaluating once at the end, you
can evaluate updated policies as the run progresses, watch the trend in your own
metrics, and stop a run that is not learning.

## What gets delivered

Bundles arrive in two series with different jobs. SageMaker syncs both to the
checkpoint S3 prefix configured on the training job.

```text
<checkpoint-prefix>/
  <run-namespace>/
    archive/                              # preserved points, named by step
      model_checkpoint_step_0020000.pt    # training state, for resume only
      ...
      bundles/
        manifest.json
        step_0020000/
          config.json                     # loadable inference bundle
          model.safetensors
          metrics.json
        ...
    improvements/                         # best-so-far track, rotating slots
      model_checkpoint_slot_000.pt
      ...
      bundles/
        manifest.json
        slot_000/
        ...
        slot_009/
```

Any `bundles/*/` directory is a complete inference bundle. It loads with the
public `causal-gpt-rl` runtime directly — no training stack, no checkpoint
restore, no waiting for the final artifact.

| Series | Arrives | Retention | What it is |
| --- | --- | --- | --- |
| `archive/` | at scheduled and requested steps | permanent | An even sample of the run, plus any steps you asked for. |
| `improvements/` | whenever `offline_eval/action_nll` reaches a new minimum | 10 rotating slots | The best-so-far track by the offline metric. Also updates the canonical bundle. |

`step` is the training-loop counter used by `max_steps`. Do not interpret it as
an episode count.

### Which one to score

**Score `archive/`.** Its points are evenly spaced across the run, announced
before the run starts, and never overwritten. That makes it the series you can
build a decision on.

`improvements/` answers a different question — "what is the best the offline
metric has seen so far" — and it is worth loading when you want that specific
model. It is not a good trend signal, for one reason worth stating plainly:

**Silence in `improvements/` does not mean the policy stopped improving.**
Deliveries stop when the offline metric stops reaching new minimums, and that
metric is not your task performance. In one measured run the offline metric
froze at step 54,300 and no further bundle was delivered for 46,000 steps —
while measured episode return over that same stretch rose from 2,045 to 5,391.
A polling loop watching `improvements/` would have seen a flat line through the
most productive part of the run.

## Finding your bundles

You need two things: the checkpoint prefix and the run namespace.

The **checkpoint prefix** is what you set on the training job:

```text
s3://my-bucket/cgrl/checkpoints/<training-job-name>/
```

The **run namespace** is derived from the run's environment or dataset identity.
Read it from CloudWatch Logs — every export logs its full path:

```text
Checkpoint artifact exported: artifact=archive_bundle     path=<ckpt>/<ns>/archive/bundles/step_0020000 step=20000 reasons=periodic
Checkpoint artifact exported: artifact=improvement_bundle path=<ckpt>/<ns>/improvements/bundles/slot_003 step=98000 slot=003
```

The `.pt` files log the same way, as `artifact=archive_checkpoint` and
`artifact=improvement_checkpoint`, so `artifact=archive_` and
`artifact=improvement_` each select one series.

Open the training job in the SageMaker Console and choose `View logs`, or open
the CloudWatch log stream directly.

## The manifests

Each series has its own manifest with its own schema. **`metrics` is abbreviated
to one key in both examples below**; every point carries all six. The full list
is under "What `metrics` contains".

### `archive/bundles/manifest.json`

```json
{
  "periodic_planned": 5,
  "points": [
    { "step": 20000, "reasons": ["periodic"],
      "metrics": { "offline_eval/action_nll": 0.91 }, "metrics_step": 19800 },
    { "step": 25000, "reasons": ["periodic", "requested"],
      "metrics": { "offline_eval/action_nll": 0.87 }, "metrics_step": 24800 }
  ]
}
```

- `points` — one entry per preserved point, keyed by `step`.
- `reasons` — why the point was kept: `periodic`, `requested`, `final`,
  `stopped`. A point can have several.
- `metrics_step` — the eval step the metrics were actually measured at. An
  archive point fires at its exact target step and carries the most recent eval
  result, which can be up to one eval interval old. The field exists so that lag
  is visible rather than hidden.

**The point with the largest `step` is the one the run ended on**, and it
carries `final`. On a resumed run the manifest holds the union of both runs'
points; steps are cumulative, so they order correctly and `final` may appear
more than once. See `training/docs/aws/sagemaker-retraining.md`.

### `improvements/bundles/manifest.json`

```json
{
  "latest_slot": "slot_003",
  "canonical_source": { "slot": "slot_002", "step": 98000 },
  "slots": {
    "slot_002": { "step": 98000, "metrics": { "offline_eval/action_nll": 0.8421 } },
    "slot_003": { "step": 100000, "metrics": { "offline_eval/action_nll": 0.8390 } }
  }
}
```

- `latest_slot` — most recently written slot. This is write order, **not** model
  quality.
- `canonical_source` — the slot and step whose weights match the canonical
  bundle in the final model artifact. May be `null`.
- `slots` — per-slot `step` and metrics.

### What `metrics` contains

Six keys per point, in both manifests and in each bundle's `metrics.json`:

| Key | Role |
| --- | --- |
| `offline_eval/action_nll` | Selection metric. Lower is better. |
| `offline_eval/short_context_action_nll` | Positions in the `0`–`0.5x` context range. |
| `offline_eval/standard_context_action_nll` | Positions in the `0.5`–`1.0x` range. |
| `offline_eval/long_context_action_nll` | Positions at `1.0x` and above. |
| `offline_eval/value_loss` | Diagnostic. |
| `offline_eval/policy_loss` | Diagnostic. |

The two diagnostics are internal training values. They depend on your dataset and
reward scale, so they are not comparable across runs and have no direction to
sort by — do not rank checkpoints with them. They are worth including when you
contact support.

Rank and select with `offline_eval/action_nll`, and score the bundles in your own
environment for the decision that actually matters.

### Reading them safely

**Slot names are not identities.** Slots cycle through `slot_000`–`slot_009` and
are reused. `slot_003` today is not `slot_003` an hour from now. Always check
`step` alongside the slot name. Archive steps have the opposite property:
`step_0020000` is step 20000 permanently.

**The directories are authoritative; a manifest is a hint.** A manifest can lag
the files next to it.

**`canonical_source` can be `null`.** That happens when rotation overwrote the
slot the canonical bundle came from. The canonical bundle is still valid at its
own location. Do not fall back to `latest_slot` — it is unrelated.

**`metrics` can be empty, and the two metric fields move together.** `metrics`
is `{}` if and only if `metrics_step` is `null`, so testing either one is
enough. Improvement slots are selected by the metric, so they always carry one.
Archive points carry the most recent eval result, but a `requested` step early
enough to precede the first evaluation has nothing to carry. A point that fires
between evaluations carries the previous evaluation's values along with its
older `metrics_step` — that lag is what the field exists to show, not a missing
metric. Do not require the field.

It is all six keys or none. A point never carries a partial set, so testing for
one key is enough to know the rest are there.

## Loading a delivered bundle

Download one point and point the runtime at it:

```bash
aws s3 cp --recursive \
  s3://my-bucket/cgrl/checkpoints/<job>/<run-namespace>/archive/bundles/step_0020000/ \
  ./step_0020000/
```

```python
from causal_gpt_rl.inference import load_runner

runner = load_runner("./step_0020000")
```

From here it is the same `PolicyRunner` you would get from the final artifact or
from Hugging Face. Roll it out in your environment and score it however you
already score policies.

## Verify on load, then continue

Delivered bundles are **not** individually verified before delivery. A save can
fail, and a failure is logged to CloudWatch while training continues to the next
checkpoint.

Treat a failed load as a skip, not an error:

- A bundle that does not load is skipped. The next delivery is unaffected.
- If **every** bundle fails to load the same way, that is not a transient
  problem — check the reported reason before waiting for more.

One failure is worth recognizing on sight. If a bundle needs a newer runtime
than you have installed, `load_runner` refuses it by name rather than
mis-decoding:

```text
Bundle requires capabilities this causal-gpt-rl <version> build does not support:
  - <capability>
Upgrade causal-gpt-rl to a build that advertises them.
```

Upgrade the package and the same bundle loads.

## Retention

The two series retain differently, and it is the one difference worth
internalizing.

**`archive/` is permanent.** Points are never rotated away. You can take your
time, and a bundle that was still syncing when you first read the manifest is
worth retrying on the next pass.

**`improvements/` rotates.** At most 10 slots are kept; after `slot_009`, saves
return to `slot_000` and overwrite it. A slot is overwritten after 10 further
improvements, and improvements are frequent early in a run. If you intend to
keep or deploy an improvement slot, copy it out of the checkpoint prefix first.
Do not serve directly from a slot path.

One caveat applies to stopping a run. The point a run ended on is written at the
next step boundary after the stop signal, and it is usually there — but the save
and its sync have to finish inside the stop grace period. **If you need the
model the run ended on, copy the latest archive point out before you stop the
job.**

## Early stopping

The workflow this is built for:

1. Poll `archive/` for new points while the job runs.
2. Load each new bundle and roll it out in your environment.
3. Track your own score against `step`.
4. If the score is flat or falling, stop the job. You stop paying for training
   time at that point rather than at `max_steps`.

By default the archive schedule gives 5 evenly spaced points, at 20%, 40%, 60%,
80% and 100% of `max_steps`. That is 4 opportunities to stop before the run ends
on its own. Stopping at the third point ends the job at 60% of `max_steps`.

If you want a finer trend, add steps with `archive_steps` — up to 10 more,
chosen before the run starts. That is the lever for resolution; each added point
also costs permanent disk, so see the sizing table in
`training/docs/aws/sagemaker-hyperparameters.md`.

### Polling loop

```python
import json
import shutil
import time
from pathlib import Path

import boto3

from causal_gpt_rl.inference import load_runner

BUCKET = "my-bucket"
PREFIX = "cgrl/checkpoints/<job>/<run-namespace>/archive/bundles"
JOB_NAME = "<training-job-name>"
POLL_SECONDS = 300
WORKDIR = Path("./deliveries")

s3 = boto3.client("s3")


def read_manifest() -> dict:
    body = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
    return json.loads(body)


def download_point(step: int) -> Path:
    name = f"step_{step:07d}"
    dest = WORKDIR / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    point_prefix = f"{PREFIX}/{name}/"
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=point_prefix
    ):
        for obj in page.get("Contents", []):
            relative = obj["Key"].removeprefix(point_prefix)
            if relative:
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, obj["Key"], str(target))
    return dest


def evaluate(runner) -> float:
    """Roll the policy out in your environment and return a score.

    `runner` is a PolicyRunner: reset(obs) -> act() -> observe(obs) -> act() ...
    """
    raise NotImplementedError


def should_stop(history, patience=2) -> bool:
    """Stop when the last `patience` points did not beat anything before them."""
    if len(history) <= patience:
        return False
    best_before = max(score for _, score in history[:-patience])
    return all(score <= best_before for _, score in history[-patience:])


done = set()
attempts = {}
history = []

while True:
    manifest = read_manifest()
    points = sorted(manifest.get("points", []), key=lambda p: p["step"])
    fresh = [p for p in points if p["step"] not in done]

    if not fresh:
        time.sleep(POLL_SECONDS)
        continue

    for point in fresh:
        step = point["step"]
        local = download_point(step)

        try:
            runner = load_runner(str(local))
        except Exception as exc:
            # A point that is still syncing will load on a later pass, so retry
            # before giving up on it. Archive points do not disappear.
            attempts[step] = attempts.get(step, 0) + 1
            if attempts[step] >= 3:
                done.add(step)
                print(f"skip step {step}: {exc}")
            continue

        done.add(step)
        score = evaluate(runner)
        history.append((step, score))
        print(f"step {step}: {score:.3f}")

        if "final" in point.get("reasons", []):
            print(f"run ended at step {step}")
            raise SystemExit

    if should_stop(history):
        boto3.client("sagemaker").stop_training_job(TrainingJobName=JOB_NAME)
        print(f"stopped {JOB_NAME} at step {history[-1][0]}")
        break
```

Only `evaluate` is yours to write. Everything else is the delivery mechanics.

What the loop does deliberately:

- **Diffs the manifest against the steps it has already scored.** Archive points
  are permanent and step-keyed, so a set of steps is enough state. There is no
  need to re-check anything after downloading, and no delivery is skipped
  because two of them landed between polls.
- **Retries a bundle that fails to load.** A point listed in the manifest can
  still be syncing. Because it will not be overwritten, retrying is worthwhile —
  three attempts, then move on.
- **Stops on the `final` point.** That is the run ending on its own, not a
  reason to call `StopTrainingJob`.
- **Uses `patience=2`, and no minimum-step guard.** The first archive point is
  already at 20% of the run, so the noisy start is excluded by the schedule
  itself. A decision becomes possible at the third point.

`POLL_SECONDS = 300` is a reasonable default. Reading a manifest is a single
small object fetch, so polling costs little; the useful lower bound is set by
how long your own rollout takes, not by S3.

### Watching `improvements/` as well

If you also want the best-by-metric track, poll
`improvements/bundles/manifest.json` — but a slot is overwritten after 10
further improvements, so a slow download can genuinely race a rotation. Select
by slot, then verify the step did not change:

```python
def fetch_latest_improvement():
    """Download the latest improvement slot, or None if rotation raced us."""
    manifest = read_improvements_manifest()
    slot = manifest["latest_slot"]
    step = manifest["slots"][slot]["step"]

    local = download_slot(slot)

    # Slot names are reused. If rotation overwrote this slot mid-download,
    # the bytes on disk are not the ones the manifest described. Discard them
    # and pick up the next delivery instead.
    if read_improvements_manifest()["slots"].get(slot, {}).get("step") != step:
        return None
    return local
```

The re-check costs one request and is what makes the result trustworthy.

Do not drive early stopping from this feed. See "Which one to score" above.

## Choosing what to deploy

The canonical bundle is selected by `offline_eval/action_nll` — the best the
model scored at predicting your dataset's actions. That is not the same as the
best policy for your task, and the gap can be large. In the measured run cited
earlier, the canonical checkpoint scored 2,045 average return while the point
the run ended on scored 5,391.

So: **treat the canonical bundle as a default, not as a verdict.** The archive
points are preserved precisely so you can score them yourself and pick the one
your environment prefers.

## Relationship to the final artifact

The final `model.tar.gz` contains the canonical bundle selected by the
checkpoint metric. If no earlier best bundle exists, managed training creates a
canonical fallback from the final weights. The model artifact is stored
separately and is not affected by slot rotation.

To check whether a delivered improvement slot is the one that will ship, compare
it against `canonical_source`: when **both** its `slot` and `step` match the
slot you are holding, that slot's weights are the canonical bundle's weights.

See `training/docs/aws/sagemaker-output-artifacts.md` for the final artifact
layout, and `training/docs/aws/sagemaker-checkpoints.md` for checkpoint and
resume mechanics.
