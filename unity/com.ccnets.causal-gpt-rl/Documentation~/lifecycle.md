# Call order and lifecycle

A step is:

```text
Observe → RequestAction → GetAction → apply → ResetRows (whoever finished) → Observe
```

`Reset` starts it. Every public call names the states it accepts, so a wrong
order fails with a reason instead of quietly serving the policy a stale
observation or reviving the last action of an agent that already left.

## States

```text
NeedsReset ──Reset──▶ Ready ──RequestAction──▶ InFlight ──GetAction──▶ AwaitingObservations
                        ▲                                                       │
                        └──────────────── every row has reported ───────────────┘
```

| Call | Accepted in | Leaves you in | Also refuses |
|---|---|---|---|
| `Reset(obs)` | NeedsReset · Ready · AwaitingObservations | Ready | Not while in flight: that execution would be decoded against a window it never saw, and its output buffer is about to be reused |
| `Observe` / `ObserveRow` | AwaitingObservations | Ready once every row has reported | Reporting the same row twice in one step, including a partial report followed by a whole-batch one |
| `ResetRows(rows)` | AwaitingObservations **only** | unchanged | A row that already reported — resetting it would clear the report and let one action be answered by two observations |
| `RequestAction()` | Ready | InFlight | Any row unobserved since its last action; a request already outstanding |
| `GetAction()` | InFlight | AwaitingObservations | Calling twice does not re-run; it returns the same action |
| `Act()` | Ready | AwaitingObservations | The two above, composed |
| `Dispose()` | any | Disposed | See below |

This table is **this package's design**, not a port. The Python runner has no
equivalent — the split into `RequestAction`/`GetAction` exists so the readback
does not stall your frame, and the states follow from that split.

## After `Dispose`

Three cases, not one.

| Case | Behaviour |
|---|---|
| An **unread** request, then the runner is disposed | `IsDone` and `GetAction()` both throw `ObjectDisposedException` |
| An **already-read** request, then the runner is disposed | `IsDone` stays `true`, `GetAction()` returns the same action. A result you already read is still valid; it does not touch the runner |
| Any runner method after disposal | `ObjectDisposedException`, even with a bad argument — the disposed check runs before argument validation, so the failure is consistent |

## Per-row episode restart

Order is the contract:

1. `GetAction()` — the episode-start token is used for this action and then
   dropped from every later attention mask.
2. `ResetRows(finished)` — those rows' context is wiped and their carried action
   cleared, so whoever occupies the row next is not steered by the previous
   agent's last move.
3. `Observe` — the episode-start marker attaches at the *next* observation, per
   row, not at the moment of the reset.

`ResetRows` deliberately accepts only the window between the action and the
observations that follow it. Despawn that is not tied to an action is not
supported; if you need it, ask for an explicit rebind API rather than widening
this one.

## Threading and allocation

`RequestAction` schedules without waiting. Input tensors are owned by the
pending execution rather than disposed at the end of the call — disposing a CPU
tensor completes the job it belongs to, which would block on the very inference
the split exists to avoid.

`DecodedAction.Continuous(row)` returns a view, and `Discrete(row, branch)`
returns an `int`; neither allocates. `CopyDiscrete` is there for when your code
wants its own array.
