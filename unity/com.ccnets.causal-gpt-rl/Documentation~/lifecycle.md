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
                        ▲   ▲                     │                             │
                        │   └────── Cancel ───────┘                             │
                        └──────────────── every row has reported ───────────────┘
```

| Call | Accepted in | Leaves you in | Rejected when |
|---|---|---|---|
| `Reset(obs)` | NeedsReset · Ready · AwaitingObservations | Ready | Not while in flight: that execution would be decoded against a window it never saw, and its output buffer is about to be reused |
| `Observe` / `ObserveRow` | AwaitingObservations | Ready once every row has reported | Reporting the same row twice in one step, including a partial report followed by a whole-batch one |
| `ResetRows(rows)` | AwaitingObservations **only** | unchanged | A row that already reported — resetting it would clear the report and let one action be answered by two observations |
| `RequestAction()` | Ready | InFlight | Any row unobserved since its last action; a request already outstanding |
| `GetAction()` | InFlight | AwaitingObservations | Calling twice does not re-run; it returns the same action |
| `ActionRequest.Cancel()` | InFlight | Ready | Cancelling an action that was already collected does nothing — nothing is in flight then. Reading a cancelled request throws `ObjectDisposedException` |
| `Act()` | Ready | AwaitingObservations | The two above, composed |
| `Dispose()` | any | Disposed | See below |

This table is **this package's design**, not a port. The Python runner has no
equivalent — the split into `RequestAction`/`GetAction` exists so the readback
does not stall your frame, and the states follow from that split.

`InFlight` has two exits rather than one. A runner that could only leave it
through a successful read had no way back from a failed one: every other call
refuses while an action is in flight, so one failure ended that runner. `Cancel`
is that way back, and a failed `GetAction()` takes it on the caller's behalf
before rethrowing — the window has not moved and the rows still hold the
observations the request was scheduled from, so the state to return to is the
one the request was made in.

## What the split is worth

Scheduling early removes the inference from the calling thread only for as long
as the caller stays away. The pass has to fit in the gap: collect one fixed step
later and a pass shorter than that step costs nothing to collect, while a longer
one is waited on for the difference. Widening the gap trades against how late
the action reaches the environment, which is the caller's contract to keep — see
"When to sample the observation" in `contract-boundaries.md`.

Measured on a 16-row discrete bundle collected one 20 ms fixed step after being
scheduled: 1.83 ms to schedule and 0.52 ms to collect, against 18–25 ms for the
blocking `Act()` doing the same work. The 0.5 ms is the tell — the pass had
already finished, so nothing was waited on. A batch large enough to run past the
step would show up in that number and nowhere else, which is the number to watch
when sizing a scene.

## A pipelined turn

The gap the split is worth something in is the environment step, and filling it
takes one arrangement:

```
Ready(oₜ) ─ RequestAction()      schedules π(oₜ)
            apply aₜ₋₁            the action collected last turn
            step the environment  the pass runs against this
          ─ GetAction() = aₜ      held for next turn
          ─ Observe(oₜ₊₁)         Ready again
```

The action is applied one turn after the observation it was computed from.
Nothing else is available: π(oₜ) cannot exist before oₜ does, so an adapter that
wants aₜ inside turn t has to wait for it, which is `Act()`. What this order buys
is that no turn ever waits — there is always an action in hand when the step
begins.

The runner cannot tell the two orders apart; frames simply pass between the two
calls. Verified as such: replaying the same observations through both orders
gives **identical actions**, exactly and not within a tolerance, across every
staged bundle on both backends.

One rule comes with it. Stepping the environment between the two calls puts
every termination inside the in-flight window, and an ended row is retired by
**collecting** the pending action, not by cancelling it — see "Retiring a row
while an action is in flight" in `contract-boundaries.md`.

## Collecting late

`IsDone` says the inference has finished. It carries no deadline: collect the
action in that frame, ten frames later, or not at all.

That is worth stating because the engine underneath does attach one. A GPU
readback request is only valid inside the frame it completes in, and reading it
afterwards fails while its own "done" flag still reads true — so a caller that
schedules on a fixed decision period, and therefore lands on that frame roughly
never, cannot use it. `GetAction()` re-reads the output when that happens. The
result is identical, the cost is a readback rather than another forward pass,
and the pass still overlapped whatever the caller did in between, which is the
whole point of scheduling early.

## Behaviour after `Dispose`

| Case | Behaviour |
|---|---|
| Runner disposed before a request is read | `IsDone` and `GetAction()` both throw `ObjectDisposedException` |
| Runner disposed after a request is read | `IsDone` stays `true`, and `GetAction()` returns the same action. The cached result remains valid because it no longer accesses the runner |
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
supported; that use case requires a dedicated row-rebinding API.

## Threading and allocation

`RequestAction` schedules without waiting. Input tensors are owned by the
pending execution rather than disposed at the end of the call — disposing a CPU
tensor completes the job it belongs to, which would block on the very inference
the split exists to avoid.

`DecodedAction.Continuous(row)` returns a view, and `Discrete(row, branch)`
returns an `int`; neither allocates. `CopyDiscrete` writes branch values into a
caller-provided array.
