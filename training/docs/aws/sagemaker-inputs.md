# SageMaker Training Inputs

Everything you configure before a training job starts: the dataset channel and
the hyperparameters — what you can set, what the job adjusts for you, and what it
refuses to start with.

`training/hyperparameters.py` is the canonical field list and carries the defaults. Submitting it is the supported path. A managed job also accepts a number of older names so that existing payloads keep working; those are listed under "Other Accepted Keys" below.

## Dataset Input

You bring the data. A training job reads only the datasets you upload — there is
no vendor-hosted corpus and no download path out of the container, which is what
`data_source` being pinned to `"byo"` means.

Input is Minari-based. Upload Minari dataset directories to S3 and pass that root
as the single input channel, named `training`.

```python
estimator.fit({
    "training": "s3://my-bucket/cgrl/datasets/minari/farama/"
})
```

That prefix is the dataset root, and `dataset_ids` are resolved relative to it.
The tree under it is the one Minari itself writes, so uploading a local
`~/.minari/datasets/` gives the right shape:

```text
s3://my-bucket/cgrl/datasets/minari/farama/
  mujoco/
    humanoid/
      simple-v0/
        data/
          main_data.hdf5
          metadata.json
      medium-v0/
```

`dataset_ids = mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0` resolves to
those two directories. An id is a Minari dataset id —
`<namespace>/<name>/<version>` — and it is also the path, so the two always match.

### Bringing your own episodes

Any recorded trajectories can be packaged, whatever produced them. `collection/`
in this repository turns per-episode `.npz` files into an env-less Minari
dataset:

```bash
python collection/build_minari.py --raw raw/ --dataset-id mujoco/humanoid/simple-v0
```

Each `.npz` holds one episode: `observations` of length `T+1`, and `actions`,
`rewards`, `terminations`, `truncations` of length `T`. See
[`collection/README.md`](../../../collection/README.md) for the observation and
action space declarations, including multi-agent recordings.

Datasets published on Hugging Face or the Farama registry can be downloaded and
uploaded as they are — no repackaging.

At startup the job prints a validation summary to CloudWatch Logs showing how
each dataset resolved and the observation/action schema the model will use. Check
it before letting the run continue — see
`training/docs/aws/aws-marketplace-training.md`.

## Rules

- SageMaker hyperparameter values are passed as strings.
- `dataset_ids` is required.
- Values that are not provided use the training recipe defaults.
- Dataset-specific metadata is read from the Minari dataset itself.
- Some values are **adjusted** to the nearest supported value rather than rejected. Others **fail the job at startup**. Which is which is fixed and listed below.

## Required Value

| Name | Description | Example |
| --- | --- | --- |
| `dataset_ids` | Minari dataset ids to train on, relative to the `training` channel root. Pass multiple ids as a comma-separated string. | `mujoco/humanoid/simple-v0,mujoco/humanoid/medium-v0` |

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

`batch_size` and `context_length` move to the nearest supported value, above or
below the request; an exact tie takes the smaller one. Bounded values clamp to
the nearest boundary instead, `min_lr` is capped at the effective
`learning_rate`, and an unsupported scheduler name falls back to `cosine`.

```text
context_length = 33   -> 32
batch_size     = 48   -> 32     # tie between 32 and 64
learning_rate  = 0    -> 1e-6
```

Training and inference use the effective values, not the original request. The
effective context is written to the exported bundle, so inference reads it from
there with no action from you.

## Dataset Capacity Can Lower Batch and Context

Your dataset must supply `batch_size × 2 × context_length` transitions in
complete chunks that do not cross episode boundaries — 8,192 at the defaults.
The doubling is a requirement on your data, not the policy's context: the bundle
records `context_length`.

If the dataset cannot supply that, the job lowers batch and context one supported
value at a time, never below `batch_size=32` and `context_length=16`, and fails
before building the model if even that does not fit. Short episodes still
contribute to training; they just do not count toward this startup check, so a
dataset of many short episodes supports a smaller context than one with fewer,
longer episodes.

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

The backbone count depends only on `d_model` and `num_layers`. The default
`d_model=256, num_layers=4` is about 4.2M; at `d_model=512`, 4 layers (~16.8M) is
the largest that starts. The limit is checked before your data is loaded, so an
oversized request fails in seconds rather than after a long download.

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
- Entries outside `0 < step <= max_steps` are dropped, and duplicates are removed. If more than 10 survive, the **10 largest** are kept.
- A token that cannot be read as a number fails the job at startup. Numeric entries never do.

A schedule can therefore come back smaller, or start later, than you wrote it.
The startup log reports the one the job settled on.

### Disk Cost

Each preserved point costs about 5x the model parameter size and is never reclaimed.

| Model | Per point | 5 evenly spaced + 10 preserved |
| --- | --- | --- |
| Default (~5M parameters) | ~100MB | ~1.5GB |
| 80M parameters | ~1.6GB | ~24GB |

The SageMaker `VolumeSizeInGB` default of 30GB also holds your dataset and the container. A schedule that does not fit produces a startup warning and the job still runs, then fails later when the volume fills — size the volume for the schedule you asked for.

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
