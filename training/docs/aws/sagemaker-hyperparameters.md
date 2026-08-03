# SageMaker Hyperparameters

This document is the contract for passing hyperparameters to a SageMaker training job: what you can set, what the job adjusts for you, and what it refuses to start with.

`training/hyperparameters.py` is the canonical field list and carries the defaults. Submitting it is the supported path. A managed job also accepts a number of older names so that existing payloads keep working; those are listed under "Other Accepted Keys" below.

## Rules

- SageMaker hyperparameter values are passed as strings.
- `dataset_ids` is required.
- Values that are not provided use the training recipe defaults.
- Dataset-specific metadata is read from the Minari dataset itself.
- Some values are **adjusted** to the nearest supported value rather than rejected. Others **fail the job at startup**. Which is which is fixed and listed below.

## Required Value

| Name | Description | Example |
| --- | --- | --- |
| `dataset_ids` | Minari dataset ids to train on. Pass multiple ids as a comma-separated string. | `mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0` |

## What You Can Set

### Training

| Name | Default | Effective value |
| --- | --- | --- |
| `max_steps` | `100000` | As given. `0` is valid for a smoke run. |
| `batch_size` | `128` | Nearest of `32, 64, 128, 256, 512`, then possibly lowered for dataset capacity. |
| `context_length` | `32` | Nearest of `16, 24, 32, 48, 64`, then possibly lowered for dataset capacity. |
| `gamma` | `0.99` | Clamped to `[0.98, 0.995]`. |
| `td_lambda` | `0.95` | Clamped to `[0.90, 0.975]`. |
| `seed` | `42` | As given. Must not be negative. |

`context_length` is the trajectory length the policy is trained on. Its effective value is recorded in the exported bundle, which is the context serving reads as the policy's default.

### Optimization

| Name | Default | Effective value |
| --- | --- | --- |
| `learning_rate` | `1e-4` | Clamped to `[1e-6, 1e-3]`. |
| `min_lr` | `1e-6` | Clamped to `[1e-8, 1e-3]`, then capped at the effective `learning_rate`. |
| `lr_scheduler_type` | `cosine` | `linear` or `cosine`, case-insensitive. Anything else uses `cosine`. |

Setting `min_lr` equal to `learning_rate` gives a constant learning rate after warmup.

### Network

These are yours. When valid they are used exactly as given — never rounded, clamped, downgraded, or replaced by a recipe. When invalid the job fails rather than adjusting them.

| Name | Default | Constraint |
| --- | --- | --- |
| `d_model` | `256` | `> 0`, and divisible by `num_heads`. |
| `num_layers` | `4` | `> 0`. |
| `num_heads` | `8` | `> 0`. |
| `dropout` | `0.05` | Within `[0.0, 1.0]`. |

The resulting backbone must also stay within 20M parameters; see "Values That Fail the Job" below.

### Checkpointing

| Name | Default | Constraint |
| --- | --- | --- |
| `archive_steps` | none | Up to 10 steps are preserved. Entries outside `0 < step <= max_steps` are dropped rather than kept. See "Archive Steps" below. |

## Automatic Adjustment

The job resolves adjustable values in two stages. First, it normalizes the
requested value without looking at the dataset. It then checks the normalized
batch and context against the dataset capacity described in the next section.

`batch_size` and `context_length` are first moved to the nearest supported
value:

```text
context_length = 20   -> 16    # tie between 16 and 24
context_length = 33   -> 32
batch_size     = 48   -> 32    # tie between 32 and 64
batch_size     = 100  -> 128
```

The nearest value can be above or below the request. When a request is exactly
halfway between two supported values, the smaller value wins. Only positive
requests enter this nearest-value step; zero and negative `batch_size` or
`context_length` values fail the job instead of rounding upward.

After this static step, dataset capacity can only lower batch or context. It
never raises either value and never selects a value outside the supported sets.
The next section explains that check in terms of episodes and transitions.

Bounded training and optimization preferences use clamping instead of nearest
values. Finite requests outside their supported range, including zero and
negative learning-rate preferences, move to the closest boundary:

```text
learning_rate = 0    -> 1e-6
min_lr        = -1   -> 1e-8
gamma         = 0.5  -> 0.98
```

`min_lr` is also capped at the effective `learning_rate`. An unsupported
scheduler name is not a failure; it falls back to `cosine`. A non-finite number
(NaN or infinity) cannot be adjusted and fails the job.

Training and inference use the effective values, not the original request.

## Dataset Capacity Can Lower Batch and Context

After the nearest-value adjustment, the job checks whether the episodes in your
dataset contain enough consecutive transitions for the requested batch and
context.

Let:

```text
B = batch_size
C = context_length
```

Think of the job as cutting each episode into chunks of `2C` consecutive
transitions. `C` of each chunk is the context the policy trains on; the rest is
what that context is scored against while the run is in progress. `2C` is
therefore a requirement on your data, not the policy's context — the bundle
records `C`.

Training can start when the dataset can supply at least `B` such chunks from its
episodes, enough to form one batch with shape:

```text
[B, 2C, x]
```

At minimum, that is:

```text
B × 2C transitions in complete episode chunks
```

With the defaults `batch_size=128` and `context_length=32`, that is:

```text
128 × 64 = 8,192 usable transitions
```

The word *usable* matters. Chunks never cross episode boundaries. Transitions
left over at the end of an episode do not combine with the next episode, so the
raw dataset may need more transitions than the formula suggests. A large
dataset made mostly of short episodes may therefore support a smaller context
than a dataset with fewer but longer episodes.

Short episodes are not discarded from training. They contribute where their
length allows; they simply do not count as complete `2C` chunks for this
startup capacity check.

If the requested shape does not fit, the job lowers batch and context one
supported value at a time. It never selects a value outside the documented
sets. The search stops at the first shape for which the episodes provide `B`
complete `2C` chunks, or bottoms out at `batch_size=32` and
`context_length=16`.

If even the smallest supported shape does not fit, the job fails before it
builds the model. It does not silently change the training objective or join
transitions from different episodes.

The effective context is committed once and written to the exported bundle.
Inference reads it from the bundle, so a bundle produced after a capacity
step-down defaults to its own context with no action from you.

## Values That Fail the Job

The job writes a single line to `/opt/ml/output/failure`, which SageMaker surfaces as the `FailureReason`. It is prefixed with a stable code — `CGRL_MISSING_HYPERPARAMETER` or `CGRL_INVALID_HYPERPARAMETER` for everything in this section — so a failure maps to a known cause without reading the log. The full traceback still goes to CloudWatch.

Payload:

- an unknown key that is not in the compatibility surface below;
- missing or empty `dataset_ids`;
- `data_source` set to anything other than `"byo"`;
- a non-numeric value where a number is required;
- NaN or infinity for any numeric field;
- negative `seed`;
- negative `max_steps` (`0` is valid);
- non-positive `batch_size` or `context_length`;
- an `archive_steps` entry that is not a number. Numeric entries are never a startup failure — see "Archive Steps" for how they are filtered.

Network:

- `d_model`, `num_layers`, or `num_heads` not positive;
- `d_model` not divisible by `num_heads`;
- `dropout` outside `[0.0, 1.0]`;
- a backbone over 20,000,000 parameters.

The backbone count is exact and depends only on `d_model` and `num_layers`:

```text
backbone_parameters = num_layers * (16 * d_model^2 + 2 * d_model) + 2 * d_model
```

The default `d_model=256, num_layers=4` is about 4.2M. At `d_model=512` the limit is reached at 5 layers (~21.0M), so 4 layers (~16.8M) is the largest that starts. This is an admission limit for the managed training instance; it is checked before your data is loaded, so an oversized request fails in seconds rather than after a long download.

## Product-Owned Values

These were customer controls in earlier versions and are now fixed by the product:

```text
warmup_ratio  = 0.05
max_grad_norm = 1.0
```

Payloads that still carry them are accepted and the values ignored, so an older submission script keeps working. They are not written into the bundle or the serving metadata.

## Other Accepted Keys

Beyond the fields above, a managed job accepts the following. These exist for compatibility; new payloads should use the canonical names.

Aliases — if both are present, the canonical field wins:

| Alias | Canonical |
| --- | --- |
| `lambda`, `lambd` | `td_lambda` |
| `max_iters` | `max_steps` |
| `scheduler_type` | `lr_scheduler_type` |
| `hidden_size` | `d_model` |

`lambda` is the natural name for the TD coefficient but is a Python keyword, so `training/hyperparameters.py` spells the field `td_lambda` and the job accepts either over the wire.

Accepted and ignored — fields that were customer controls in earlier versions and are now owned by the recipe:

```text
env_id   tau   init_log_std   extra_td_ratio   lr_decay_rate
warmup_ratio   max_grad_norm   export_bos_cache_mode
```

Accepted but pinned:

```text
data_source = "byo"
```

## Archive Steps

Every job permanently preserves 5 evenly spaced checkpoints across the run. `archive_steps` adds up to 10 steps of your own to that set, for a maximum of 15 preserved points. A requested step that coincides with a periodic one is stored once, not twice. Preserved checkpoints are never rotated away, so you can score them in your own environment after the job ends. See `training/docs/aws/sagemaker-checkpoints.md`.

- Pass a comma-separated string of steps: `"25000, 75000"`.
- Entries outside `0 < step <= max_steps` are dropped, and duplicates are removed.
- If more than 10 steps survive that, the **10 largest** are kept.
- A token that cannot be read as a number is a malformed payload and fails the job at startup.

Only that last case stops a run. The effective schedule is:

```python
effective_steps = sorted({
    step for step in archive_steps if 0 < step <= max_steps
})[-10:]
```

The order you write the steps in does not matter, and when more than 10 survive it is the **earliest** ones that are dropped — the schedule keeps the end of the run, where the policy is furthest along. A schedule can therefore come back smaller, or start later, than you wrote it. Check the archive schedule in the startup log if a step you expected never arrives.

### Disk Cost

Each preserved point costs about 5x the model parameter size — 4x training state (model, two AdamW moments, target network) plus 1x inference bundle — and is never reclaimed.

| Model | Per point | 5 evenly spaced + 10 preserved |
| --- | --- | --- |
| Default (~5M parameters) | ~100MB | ~1.5GB |
| 80M parameters | ~1.6GB | ~24GB |

The SageMaker `VolumeSizeInGB` default of 30GB also holds your dataset and the container. The startup log reports the estimate against actual free space:

```text
Archive disk estimate: 3.3 MB for 5 point(s) (free: 322.8 GB)
```

This is a report, not a limit. A schedule that does not fit produces a startup warning and the job still runs, then fails later when the volume fills. Size the volume for the schedule you asked for.

## Example

```python
hyperparameters = {
    "dataset_ids": "mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0",
    "max_steps": "100000",
    "batch_size": "128",
    "context_length": "32",
    "seed": "42",
    "archive_steps": "25000, 75000",
}
```

`batch_size` and `context_length` here are already supported values, so they are used as-is.

## Short Validation Run

For a product smoke test, reduce `max_steps`.

```python
hyperparameters = {
    "dataset_ids": "mujoco/humanoid/simple-v0",
    "max_steps": "1600",
}
```

This setting is for smoke testing only. It is not a final training configuration for quality evaluation.
