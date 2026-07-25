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

Every successful checkpoint save writes a complete inference bundle under the
checkpoint directory. SageMaker syncs that directory to the checkpoint S3 prefix
configured on the training job.

```text
<checkpoint-prefix>/
  <run-namespace>/
    archive/
      model_checkpoint_slot_000.pt      # training state, for resume only
      ...
    snapshots/
      manifest.json                     # what is in each slot
      slot_000/
        config.json                     # loadable inference bundle
        model.safetensors
        metrics.json
      ...
      slot_009/
```

A `snapshots/slot_NNN/` directory is a complete inference bundle. It loads with
the public `causal-gpt-rl` runtime directly — no training stack, no checkpoint
restore, no waiting for the final artifact.

Two things cause a bundle to be written:

| Trigger | When |
| --- | --- |
| `best` | `offline_eval/action_nll` reaches a new minimum. This save also updates the canonical bundle. |
| `final` | End of the job. Final-step weights, which do not overwrite an existing canonical best bundle. |

`step` is the training-loop counter used by `max_steps`. Do not interpret it as
an episode count.

### How often deliveries arrive

There is no fixed delivery period and no minimum spacing between saves. A bundle
is written whenever action NLL reaches a new minimum, so **deliveries tend to be
frequent early in a run and sparser later**, as improvements become harder to
find. Export time and SageMaker checkpoint-sync latency also affect when a
bundle becomes visible in S3.

The delivery rate matters for a second reason beyond polling: only 10 slots
exist, so **a slot is overwritten after 10 further successful saves.** The faster
deliveries arrive, the shorter the window you have to fetch or copy one out.

## Finding your bundles

You need two things: the checkpoint prefix and the run namespace.

The **checkpoint prefix** is what you set on the training job:

```text
s3://my-bucket/cgrl/checkpoints/<training-job-name>/
```

The **run namespace** is derived from the run's environment or dataset identity.
Read it from CloudWatch Logs — every export logs its full path:

```text
Checkpoint artifact exported: artifact=snapshot_bundle path=... step=... slot=... role=best
```

Open the training job in the SageMaker Console and choose `View logs`, or open
the CloudWatch log stream directly.

## The manifest

`snapshots/manifest.json` tells you what is in each slot.

```json
{
  "latest_slot": "slot_003",
  "canonical_source": {
    "slot": "slot_002",
    "step": 98000
  },
  "slots": {
    "slot_002": {
      "step": 98000,
      "slot_role": "best",
      "metrics": { "offline_eval/action_nll": 0.8421 }
    },
    "slot_003": {
      "step": 100000,
      "slot_role": "final",
      "metrics": {}
    }
  }
}
```

- `latest_slot` — most recently written slot. This is write order, **not** model
  quality.
- `canonical_source` — the slot and step whose weights match the canonical
  bundle in the final model artifact. May be `null`.
- `slots` — per-slot `step`, `slot_role`, and available metrics.

`slot_role` records why a slot was written. In this managed configuration you
will see `best` and `final`. `interval` is a supported manifest value but does
not appear in this configuration.

### Four rules for reading it

**Slot names are not identities.** Slots cycle through `slot_000`–`slot_009` and
are reused. `slot_003` today is not `slot_003` an hour from now. Always check
`step` alongside the slot name.

**The slot directories are authoritative; the manifest is a hint.** If a job was
resumed, the manifest may have been rebuilt from post-resume saves only, so it
does not necessarily describe every slot present on disk.

**`canonical_source` can be `null`.** That happens when rotation overwrote the
slot the canonical bundle came from. The canonical bundle is still valid at its
own location. Do not fall back to `latest_slot` — it is unrelated.

**`metrics` can be empty.** Not every save has evaluation results available at
that step.

## Loading a delivered bundle

Download one slot and point the runtime at it:

```bash
aws s3 cp --recursive \
  s3://my-bucket/cgrl/checkpoints/<job>/<run-namespace>/snapshots/slot_002/ \
  ./slot_002/
```

```python
from causal_gpt_rl.inference import load_runner

runner = load_runner("./slot_002")
```

From here it is the same `PolicyRunner` you would get from the final artifact or
from Hugging Face. Roll it out in your environment and score it however you
already score policies.

## Verify on load, then continue

Delivered snapshots are **not** individually verified before delivery. A save
can fail, and a failure is logged to CloudWatch while training continues to the
next checkpoint.

Treat a failed load as a skip, not an error:

- A slot that does not load is skipped. The next delivery is unaffected.
- If **every** slot fails to load the same way, that is not a transient problem
  — check the reported reason before waiting for more.

One failure is worth recognizing on sight. If a bundle needs a newer runtime
than you have installed, `load_runner` refuses it by name rather than
mis-decoding:

```text
Bundle requires capabilities this causal-gpt-rl <version> build does not support:
  - <capability>
Upgrade causal-gpt-rl to a build that advertises them.
```

Upgrade the package and the same slot loads.

## Retention

At most 10 slots are kept. After `slot_009`, saves rotate back to `slot_000` and
overwrite it.

A slot's `slot_role` describes why it was written. It does **not** reserve the
slot, prevent rotation, or guarantee retention — a slot marked `best` will be
overwritten in turn like any other.

**If you intend to keep or deploy a delivered policy, copy it out of the
checkpoint prefix first.** Do not serve directly from a slot path.

## Early stopping

The workflow this is built for:

1. Poll for new deliveries while the job runs.
2. Load each new bundle and roll it out in your environment.
3. Track your own score against `step`.
4. If the score is flat or falling well into the run, stop the job. You stop
   paying for training time at that point rather than at `max_steps`.

### Polling loop

```python
import json
import shutil
import time
from pathlib import Path

import boto3

from causal_gpt_rl.inference import load_runner

BUCKET = "my-bucket"
PREFIX = "cgrl/checkpoints/<job>/<run-namespace>/snapshots"
JOB_NAME = "<training-job-name>"
POLL_SECONDS = 300
WORKDIR = Path("./deliveries")

s3 = boto3.client("s3")


def read_manifest() -> dict:
    body = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")["Body"].read()
    return json.loads(body)


def slot_step(manifest: dict, slot: str):
    """Step currently reported for `slot`, or None if it is not listed."""
    return manifest.get("slots", {}).get(slot, {}).get("step")


def download_slot(slot: str) -> Path:
    dest = WORKDIR / slot
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    slot_prefix = f"{PREFIX}/{slot}/"
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=slot_prefix
    ):
        for obj in page.get("Contents", []):
            relative = obj["Key"].removeprefix(slot_prefix)
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


def should_stop(history, patience=5, min_step=20_000) -> bool:
    """Stop when the score has not improved over the last `patience` deliveries."""
    if len(history) <= patience or history[-1][0] < min_step:
        return False
    best_before = max(score for _, score in history[:-patience])
    return all(score <= best_before for _, score in history[-patience:])


seen_step = -1
history = []

while True:
    manifest = read_manifest()
    slot = manifest.get("latest_slot")
    step = slot_step(manifest, slot)

    # Nothing new delivered yet.
    if step is None or step <= seen_step:
        time.sleep(POLL_SECONDS)
        continue

    local = download_slot(slot)

    # Slot names are reused. If rotation overwrote this slot mid-download,
    # discard it and pick up the next delivery instead.
    if slot_step(read_manifest(), slot) != step:
        continue

    seen_step = step

    try:
        runner = load_runner(str(local))
    except Exception as exc:
        # Not fatal — a slot that fails to load is skipped, and the next
        # delivery is unaffected.
        print(f"skip {slot} @ step {step}: {exc}")
        continue

    score = evaluate(runner)
    history.append((step, score))
    print(f"step {step}: {score:.3f}")

    if should_stop(history):
        boto3.client("sagemaker").stop_training_job(TrainingJobName=JOB_NAME)
        print(f"stopped {JOB_NAME} at step {step}")
        break
```

Only `evaluate` is yours to write. Everything else is the delivery mechanics.

What the loop does deliberately:

- **Tracks `step`, not slot name.** `latest_slot` alone cannot tell you whether
  a delivery is new, because slot names are reused.
- **Re-checks the step after downloading.** This is the "select by slot, verify
  step" pattern. A slot is overwritten after 10 further saves, so a slow
  download can genuinely race a rotation. The re-check costs one request.
- **Treats a failed load as a skip.** `seen_step` is advanced before the load
  attempt, so a bad slot is not retried forever.
- **Guards early stopping with `min_step`.** Scores are noisy at the start of a
  run; stopping on the first flat stretch would end good runs.

`POLL_SECONDS = 300` is a reasonable default. Reading the manifest is a single
small object fetch, so polling costs little; the useful lower bound is set by
how long your own rollout takes, not by S3.

**The loop takes only the latest slot on each pass.** If several saves happen
between two polls, the intermediate bundles are skipped — you will see the trend
but not every checkpoint. Poll more often if you want a denser sample, and copy
slots out promptly if you want to keep them.

The manifest's per-slot `metrics` is useful context alongside your own score,
but it may be empty for a given slot — do not require it.

## Relationship to the final artifact

The final `model.tar.gz` contains the canonical bundle selected by the
checkpoint metric. If no earlier best bundle exists, managed training creates a
canonical fallback from the final weights. The model artifact is stored
separately and is not affected by slot rotation.

To check whether a delivered slot is the one that will ship, compare it against
`canonical_source`: when **both** its `slot` and `step` match the slot you are
holding, that slot's weights are the canonical bundle's weights.

See `training/docs/aws/sagemaker-output-artifacts.md` for the final artifact
layout, and `training/docs/aws/sagemaker-checkpoints.md` for checkpoint and
resume mechanics.
