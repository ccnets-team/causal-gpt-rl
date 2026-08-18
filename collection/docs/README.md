# Collection docs

The packaging path in order. Everyone ends here, whichever
[entry point](../README.md#where-you-join) they came in at.

1. [The Input Contract](01-the-input-contract.md) — the five arrays per episode,
   and the `spec.json` that declares your spaces
2. [Record Trajectories](02-record-trajectories.md) — producing those arrays
   from a loop, and the worked sources that already do
3. [Packaging](03-packaging.md) — running `build_minari.py`, its flags,
   multi-agent recordings, and reading back what came out

Part 3 is what runs here for every source; 1 and 2 define what your own recorder
has to produce. Read 1 and 3 if you only want to package something. Read
[Check what you built](03-packaging.md#check-what-you-built) before you publish or
upload — a dataset that loads can still declare the wrong interface.

Once you have packaged one:
[Improving the Next Dataset](improving-the-next-dataset.md) — where the policy
model for the next cycle can come from, what to measure before recording with
it, and `CollectionRunner`, which records part 2 for you when that policy is one
of ours. Optional, and beside the path above rather than a step in it: it is
indexed by the policy driving the episode, where the path is indexed by the
environment it runs in.

Diagrams live in [`assets/`](assets/).
