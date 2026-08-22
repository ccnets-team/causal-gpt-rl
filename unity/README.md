# Unity

Everything Unity-related in this repository lives here. The Python project at
the repository root is unaffected: `pyproject.toml` discovers only
`causal_gpt_rl*`, so nothing under `unity/` becomes an import package.

![Causal GPT-RL — the game engine does not wait for the model: two lanes of equal turn width, each with a world row and a model row, so the same work can be read against itself. In the reactive lane the two alternate, because the action is a function of the state that just arrived, and one state-action pair is spread over two turns; in the lower lane the action is generated from the trajectory so far, so it never waits on the state being computed beside it and each pair closes a turn sooner. The filled cells are equally many in both — what changes is the ordering the dependency imposes](docs/assets/game-engine-does-not-wait-for-the-model.svg)

That property belongs to the model, not to this package: an action is generated
from the trajectory so far, so it is already there when the engine needs it. What
this package adds on top is that reading it back does not stall a frame either —
`RequestAction` schedules and `GetAction` collects. The two are separate claims
that happen to share a word.

| Path | What it is | Ships to customers |
|---|---|---|
| [com.ccnets.causal-gpt-rl/](com.ccnets.causal-gpt-rl/) | The UPM package | **Yes** — this is the only path a customer installs |
| [test-project/](test-project/) | Unity project used to run the tests | No |
| [tools/](tools/) | Fixture generation, staging, batch test and build helpers | No |

## Installing the package

    https://github.com/ccnets-team/causal-gpt-rl.git?path=/unity/com.ccnets.causal-gpt-rl

Unity registers the subfolder containing `package.json` as the package root, so
the repository being a Python project does not matter. Append `#<revision>` to
pin; path comes first, revision second.

> **A `?path=` URL installs from a commit even with no tag.** Anything merged
> here is installable, which is why promotion and release are effectively the
> same event.

## Running the tests

The test project does not carry fixtures — they are large and regenerated
rather than committed. Generate, stage, then run:

```powershell
# 1. deterministic fixtures, using ONNX Runtime as the reference
python.exe tools\generate_fixtures.py `
  --onnx   <policy>.onnx `
  --config <bundle>\config.json `
  --out    fixtures\<model-id>          # one directory per model

# 2. stage into the project (the Asset Database will not import from outside it)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\stage_fixtures.ps1 `
  -Source fixtures -Destination test-project\Assets\CGRLTests\Fixtures

# 3. EditMode parity
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\run-unity-tests.ps1
```

`FixtureModels.Required` names the models the suite expects; a missing one fails
rather than silently shrinking coverage.

> **A compile error makes batchmode Unity exit without writing the results XML.**
> If `logs/editmode-results.xml` is absent, the run died rather than hung — look
> for `error CS` in `logs/editmode.log`.

When counting results, count `<test-case>` elements only. Counting
`<test-suite>` as well inflates the number.

## Performance harness

Three steps, in order:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\stage-performance-models.ps1 `
  -Model 'crawler-b1-ctx32=<crawler>.onnx', 'soccertwos-b16-ctx32=<soccer>.onnx'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\build-performance-player.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\run-performance-player.ps1
```

The staging label becomes the file name under `Models/`, and
`PerformanceBuild.cs` loads exactly those two names — so the labels above are
fixed, not examples. A missing model fails the build with a named error rather
than measuring nothing.

**Neither the models nor the harness scene are committed.** `PerformanceBuild`
builds the scene from an empty one on every run, wiring the models by asset
path, and saves over `Performance.unity`. Asset GUIDs play no part, so the
models' `.meta` files carry nothing worth keeping and Unity is free to drop them
whenever `Models/` is empty. Committing the scene would only produce a diff
after every build.

The harness reports p50/p95. Its allocation column is **not measured** — the
Mono counter reports 0 B, which is false — and tensor upload cost is excluded
because tensors are reused outside the loop.

## What is not here

Bundles, policy `.onnx` files, and generated fixtures. They come from the model
repository or from a release artifact, never from this repository's history.
