# Checkpoint Score

Automatic checkpoint selection, with no simulator required.

Offline training has no environment in the loop, so there is nothing to roll a
policy out against while the job runs. Checkpoint Score closes that gap: a single
bounded score, measured on data your dataset already holds back, that the job
uses to decide which checkpoint to keep. You get a canonical bundle chosen for
you and a ranked view of the run — no simulator, no extra evaluation setup, no
configuration.

Read it as a within-run ranking: higher is better across the checkpoints of one
training job. The criterion is refined between versions, so it is not a number to
carry between runs, and it is not a measure of how the policy performs in your
environment.

## Which metric selects

A small set of keys travels with what a training job ships — the checkpoint
metadata, each bundle's `metrics.json`, the bundle manifest, and the job's log
stream. That set is the contract, and only one key in it selects.

| Key | Role |
|---|---|
| `eval_offline/checkpoint_score` | **Selection criterion.** Range `[0, 1]`, direction `max` |
| `eval_offline/rollout_action_prob` | A component of the selection criterion, reported on its own |
| `eval_offline/action_nll` | **Diagnostic, not the criterion.** Held-out negative log-likelihood of the dataset action |
| `eval_offline/short_context_action_nll`, `eval_offline/standard_context_action_nll` | The same NLL split by position bucket within the training context length |

The keys appear under two notations, one per surface. The only difference is the
separator: `:` for the SageMaker metric name you select in the console,
`/` in `metrics.json`, `manifest.json`, and checkpoint metadata. The two sets
coincide, so a key you can read in `metrics.json` is a key you can graph.

Direction is **not** part of the name. It travels as its own field: artifacts
record `best_metric_name` and `best_metric_direction` separately. Searching an
artifact for a fused `checkpoint_score/max` finds nothing.

The training dashboard renders these alongside internal instrumentation under
display names of its own. Those names are for reading a run, not an interface —
key your own tooling off the table above.

Held-out action NLL is kept as a diagnostic rather than promoted to the criterion
because it is unbounded per position, so a single position can dominate the mean
and one evaluation can set a record that later evaluations cannot beat.

## Scope

It is a selection criterion, not a performance result. For how the policy
performs at your task, roll a delivered bundle out in your own environment — see
`training/docs/aws/sagemaker-checkpoints.md`.

When a simulator is available during training, environment return is the better
criterion and takes precedence; this exists for the case where the dataset
arrives without one.
