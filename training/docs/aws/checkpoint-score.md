# Checkpoint Score

An environment-free criterion for choosing which checkpoint of an offline
training run to keep.

Offline training has no simulator in the loop, so checkpoint selection has to
run on a held-out statistic. Checkpoint Score is that statistic. It is computed
from quantities the evaluation forward already produces, so it costs no extra
forward passes and introduces no separate evaluation model.

This document describes the metric itself. For where checkpoints are written and
how they rotate, see `training/docs/aws/sagemaker-checkpoints.md`; for consuming
delivered bundles while a job runs, see
`training/docs/aws/sagemaker-realtime-policy-delivery.md`.

## Which metric selects

Every key below goes to the training dashboard, which also carries internal
diagnostics not listed here. A subset reaches what a training job ships — the
checkpoint metadata, each snapshot's `metrics.json`, the bundle manifest, and
the job's log stream. **Where** says which. Only one key selects.

| Key | Role | Where |
|---|---|---|
| `OfflineEval/CheckpointScore` | **Selection criterion.** Direction `max` | dashboard + artifacts |
| `OfflineEval/RolloutActionProb` | Component: the action term on its own | dashboard + artifacts |
| `OfflineEval/ActionNll` | **Diagnostic, not the criterion.** Held-out negative log-likelihood of the dataset action | dashboard + artifacts |
| `OfflineEval/{Short,Standard,Long}ContextActionNll` | Position-bucket diagnostics, including behaviour past the training window | dashboard + artifacts |
| `OfflineEval/ValueLoss`, `PolicyLoss` | Held-out value and policy error terms | dashboard + artifacts |
| `OfflineEval/AdvantageMean`, `AdvantageStd` | Distribution of the raw value gap. **Not an input to the score** — the score reads the gap per position, never its batch mean or spread | dashboard |
| `OfflineEval/ActionNll/head_*` | Per-action-head breakdown | dashboard |
| `OfflineEval/CheckpointsSaved` | Cumulative count of saved checkpoints | dashboard |

### Key forms

The names above are how the dashboard renders them. The artifact-facing keys
carry the same metric under two more notations:

| Surface | Notation |
| --- | --- |
| SageMaker metric name (console, CloudWatch) | `offline_eval:checkpoint_score` |
| `metrics.json`, `manifest.json`, checkpoint metadata | `offline_eval/checkpoint_score` |
| Training dashboard | `OfflineEval/CheckpointScore` |

The mapping is mechanical — PascalCase becomes snake_case, and the separator is
`:` for the SageMaker metric name you select in the console and `/` everywhere
else. Same metric, different surface.

Direction is **not** part of the name. It travels as its own field: artifacts
record `best_metric_name` and `best_metric_direction` separately, and the startup
summary prints them on separate lines. Searching an artifact for a fused
`checkpoint_score/max` finds nothing.

Of the artifact-facing keys, six are also published as SageMaker metric names a
training job can graph: `offline_eval:checkpoint_score`,
`offline_eval:rollout_action_prob`, `offline_eval:action_nll`, and the three
context-band NLLs. `ValueLoss` / `PolicyLoss` reach the log stream and the
artifacts but are not among the scraped scalars.

Held-out action NLL is kept as a diagnostic rather than promoted to the
criterion for two reasons, both properties of the statistic itself. It is
unbounded per position, so a single position can dominate the mean and one
evaluation can set a record that later evaluations cannot beat. And minimizing
it selects the checkpoint that best reproduces the behaviour policy, which is
the right target only when the training objective is behaviour cloning.

## What the evaluation compares

One forward over a held-out split. The distinguishing choice is what fills the
context.

- **Context**: the model's own generated actions, so the model is running on its
  own trajectory rather than being fed the dataset's actions.
- **Target**: the dataset's next action at each position.

So the quantity is *how much probability the model assigns to what the dataset
actually did, while the model is drifting on its own rollout*. This is not a
distance between generated actions and dataset actions — no such comparison is
computed. It is a likelihood of the dataset action under a self-rollout context,
which is a strictly harder condition than teacher forcing and closer to how the
model is used at serving time.

## Rollout action probability

Each action head's raw log-score is mapped into `[0, 1]` against an anchor:

```text
rollout_action_prob = exp(min(raw_log_prob - kind_anchor, 0))
```

"Rollout" is the load-bearing part of the name: the log-score is taken under the
model's own generated context, so this is not a teacher-forced likelihood.

Below the anchor the value falls off exponentially with the log-score; at or
above it the head saturates at `1` and earns no further credit.

The anchor depends on the action kind because the raw quantity does.

| Action kind | Raw score | Anchor |
|---|---|---|
| Continuous | log **density** — unbounded above | An internal constant |
| Categorical / multi-discrete / binary | log probability **mass** — always `<= 0` | `0`, so `rollout_action_prob` is the demonstrated action's own probability |

Neither anchor is a training-job setting. The continuous one is not even
evaluation-specific: it is the same anchor the training objective centers the
continuous log-density on — the point past which a sharper policy stops being
pushed is the point past which it stops earning credit. One constant sets both,
which is why it is not exposed as a knob on either side.

Heads are then averaged with equal weight.

Two properties follow, and they are the reason for the construction:

- **Bounded before aggregation.** A single catastrophic position contributes a
  value near zero instead of an arbitrarily large one, so it cannot dominate the
  mean. The aggregate stays usable when one position of one row goes wrong.
- **Comparable across action kinds.** A density and a probability mass are not on
  the same scale, and averaging them raw would let one head's units set the
  metric. Anchoring each kind puts every head on a `[0, 1]` scale first.

## Advantage

The second term is a value gap: how far the model's own rollout falls **behind**
the dataset's continuation from the same point.

The name invites the wrong intuition. Unlike a standard RL advantage, **smaller
is better here.** It is not how much an action beats a baseline; it is how far
the model's rollout sits below the reference. Offline, the dataset is the
ceiling, so this is a deficit to drive toward zero, not a margin to grow.

The gap becomes a per-position weight through the same exponential family used
for the action score, with the sign that makes a small gap good:

```text
advantage_weight = exp(-relu(gap))
```

| Gap | Meaning | `advantage_weight` |
|---|---|---|
| Zero or below | the model has caught up, or is ahead | `1` |
| Above zero | the reference is still ahead | decays as `e^{-gap}` |

Two properties matter.

- **Beating the reference earns no extra credit.** Everything at or below zero
  weighs exactly `1`. Offline the dataset is the ceiling, so positions where the
  rollout appears to beat the reference are estimation noise rather than skill.
- **No position is discarded.** `advantage_weight` decays but never reaches zero,
  so a lagging position is down-weighted, not deleted. The anchor is an absolute
  point — the gap being zero — not a statistic of whichever positions happened to
  be drawn together, so a position's weight is the same in every evaluation.
  There is nothing to configure here: the offline ceiling is what defines "caught
  up", and moving the anchor off zero would move that definition arbitrarily.

The score is the mean of `advantage_weight * rollout_action_prob` over the
positions an evaluation scores. A checkpoint should close as much of the gap to
the dataset's continuation as it can while learning the dataset's action.

Both terms are logged, so `CheckpointScore / RolloutActionProb` recovers the mean
`advantage_weight` without a third metric — exactly when the two channels are
uncorrelated, and close enough when they nearly are.

Read that mean as what it is: the **average per-position weight**, `1` wherever
the model has caught up and less wherever it lags. It is *not* the fraction of
positions that caught up. A position lagging by a little still weighs nearly `1`,
so the average sits well above that fraction whenever gaps are small — which is
the normal case. It is a level that falls as gaps widen, not a percentage.

## How to read it

- Range `[0, 1]`, higher is better. It discriminates between checkpoints within
  one run; comparing absolute values across datasets, or across revisions of the
  metric, is not meaningful.
- **It measures how often the model falls behind, not how badly.** Both terms
  decay exponentially at their bad end, so past a few nats they are already near
  zero and stop separating degrees: a position that diverged badly and one that
  diverged far worse contribute the same almost-nothing. That is the same
  bounding that keeps one position from dominating the mean, and it means a
  checkpoint that tracks the reference on most positions and diverges badly on a
  few will still score well. Severity is not what this number reports.

## Scope

Checkpoint Score is a selection criterion, not a performance result. It ranks
checkpoints of one run against each other; it does not tell you how the policy
performs at your task. That answer only comes from running a delivered bundle in
your own environment — see
`training/docs/aws/sagemaker-realtime-policy-delivery.md`.

When a simulator is available during training, environment return is the better
criterion and takes precedence; this exists for the case where the dataset
arrives without one.
