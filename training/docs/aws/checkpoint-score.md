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

A small set of keys travels with what a training job ships — the checkpoint
metadata, each snapshot's `metrics.json`, the bundle manifest, and the job's log
stream. That set is the contract, and only one key in it selects.

| Key | Role |
|---|---|
| `eval_offline/checkpoint_score` | **Selection criterion.** Direction `max` |
| `eval_offline/rollout_action_prob` | Component: the action term on its own |
| `eval_offline/action_nll` | **Diagnostic, not the criterion.** Held-out negative log-likelihood of the dataset action |
| `eval_offline/short_context_action_nll`, `eval_offline/standard_context_action_nll` | The same NLL split by position bucket within the training context length |

The training dashboard renders these alongside internal instrumentation, under
display names of its own. Those names are for reading a run, not an interface —
they are regrouped and relabelled between versions. Key your own tooling off the
table above.

### Key forms

The keys above appear under two notations, one per surface:

| Surface | Notation |
| --- | --- |
| `metrics.json`, `manifest.json`, checkpoint metadata | `eval_offline/checkpoint_score` |
| SageMaker metric name (console, CloudWatch) | `eval_offline:checkpoint_score` |

The only difference is the separator: `:` for the SageMaker metric name you
select in the console, `/` everywhere else. Same metric, different surface. The
two sets coincide, so a key you can read in `metrics.json` is a key you can
graph.

Direction is **not** part of the name. It travels as its own field: artifacts
record `best_metric_name` and `best_metric_direction` separately, and the startup
summary prints them on separate lines. Searching an artifact for a fused
`checkpoint_score/max` finds nothing.

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

## How to read it

Range `[0, 1]`, higher is better. It ranks checkpoints within one run and nothing
wider — the criterion is revised as it is tuned, so values are not comparable
across runs or datasets. A good score requires fitting the dataset's action *and*
closing the gap to its continuation; how the two are combined is an
implementation detail.

## Scope

Checkpoint Score is a selection criterion, not a performance result. It ranks
checkpoints of one run against each other; it does not tell you how the policy
performs at your task. That answer only comes from running a delivered bundle in
your own environment — see
`training/docs/aws/sagemaker-realtime-policy-delivery.md`.

When a simulator is available during training, environment return is the better
criterion and takes precedence; this exists for the case where the dataset
arrives without one.
