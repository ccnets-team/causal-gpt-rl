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
                        ┌───── StageObservation ─────▶ Staged ──GetAction──┐
                        │                                                  │
NeedsReset ──Reset──▶ Ready ──RequestAction──▶ InFlight ──GetAction──▶ AwaitingObservations
                        ▲   ▲                     │                             │
                        │   └────── Cancel ───────┘                             │
                        └──────────────── every row has reported ───────────────┘
```

The upper path is the pipelined turn and the lower one is the serial turn. They
differ in which observation is reported: staging reports the one the in-flight
action is being taken **at**, so reading closes the pair and leaves nothing
outstanding; `Observe` reports the one that **followed** an action already taken.

| Call | Accepted in | Leaves you in | Rejected when |
|---|---|---|---|
| `Reset(obs)` | NeedsReset · Ready · AwaitingObservations | Ready | Not while in flight, staged or not: that execution would be decoded against a window it never saw, and its output buffer is about to be reused |
| `Observe` / `ObserveRow` | AwaitingObservations | Ready once every row has reported | Reporting the same row twice in one step, including a partial report followed by a whole-batch one |
| `StageObservation(obs)` | InFlight | Staged | There is no action in flight to pair the observation with |
| `ResetRows(rows)` | AwaitingObservations **only** | unchanged | A row that already reported — resetting it would clear the report and let one action be answered by two observations. Not reachable from the pipelined turn; see below |
| `RequestAction()` | Ready | InFlight | Any row unobserved since its last action; a request already outstanding |
| `GetAction()` | InFlight | AwaitingObservations | Calling twice does not re-run; it returns the same action |
| `GetAction()` | Staged | Ready | As above |
| `ActionRequest.Cancel()` | InFlight · Staged | Ready | Cancelling an action that was already collected does nothing — nothing is in flight then. Reading a cancelled request throws `ObjectDisposedException` |
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

The action this policy emits is not a function of the newest observation. The
window keeps one slot more than the model reads: `Observe(oₜ₊₁)` rolls the pair
`(oₜ, aₜ)` into the last slot the model sees and **stages `oₜ₊₁` where the model
does not see it**. The pass that follows therefore reads a context ending at
`(oₜ, aₜ)` and emits `aₜ₊₁` — the action to take at an observation it has not
been given.

That is what makes the step period `max(world, model)` rather than
`world + model`: the pass does not need the state the engine is still computing.

```
boundary t ─ StageObservation(oₜ)   the observation aₜ is being taken at
             GetAction() = aₜ       scheduled last turn; closes the pair (oₜ, aₜ)
             apply aₜ               no wait, no staleness
             RequestAction()        emits aₜ₊₁ from the trajectory so far
             step the environment   the pass runs under it
boundary t+1 ─ StageObservation(oₜ₊₁) …
```

The first action of an episode is the exception: nothing was scheduled under a
previous step, so it is waited for once. After that the pipeline stays full.

`aₜ` is applied at the boundary `oₜ` arrived at, exactly as the serial order
applies it, and the window records `(oₜ, aₜ)` — which is true, because `aₜ` is
the action that was executed at `oₜ`. Nothing is stale and nothing is
misreported. What changed is only *when* the pass ran.

Do not instead apply the previously collected action while scheduling against
the observation that just arrived. That order pairs `(oₜ, aₜ)` in the window
while the environment executes `aₜ₋₁`, so the trajectory the model conditions on
records an action that was never taken. Measured on the Humanoid bundle: it
falls at the same decision every episode.

One limit comes with the overlap. Stepping the environment while a pass is in
flight puts every termination inside that window, and the pipelined turn has no
state in which `ResetRows` runs — it goes from the staged read straight back to
ready. A whole-batch restart is fine; retiring one row of several means running
that turn in the serial order. See "Retiring a row while an action is in flight"
in `contract-boundaries.md`.

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
