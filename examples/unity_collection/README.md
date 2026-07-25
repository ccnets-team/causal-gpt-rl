# Unity ML-Agents — collection & measurement recipe

Reproduce the Causal GPT-RL Unity artifacts end to end, from public materials.

- **Measure** a downloaded policy's closed-loop return in a Unity build — use
  [`../unity/evaluate_onnx.py`](../unity/evaluate_onnx.py) for continuous, discrete,
  hybrid, and cooperative multi-agent policies. The older
  [`../deploy/mlagents.py`](../deploy/mlagents.py) is the Crawler-specific example.
- **Collect** trajectories and package them as a Minari dataset — this folder.

All inputs are public Hugging Face repos:

| Repo | Contents |
|---|---|
| [ccnets/causal-gpt-rl-unity](https://huggingface.co/ccnets/causal-gpt-rl-unity) | trained ONNX policies |
| [ccnets/causal-gpt-rl-unity-envs](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs) | model-removed Unity builds + stock policies where redistributable |
| [ccnets/causal-gpt-rl-unity-datasets](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-datasets) | recorded Minari trajectories |

## Environments

`mlagents_envs` 1.x pins an older numpy/gymnasium than Minari, so use two envs:

- **Collection** (`collect.py`): `mlagents_envs==1.1.0` + `onnxruntime` — see
  [`requirements-collect.txt`](requirements-collect.txt).
- **Packaging** ([`collection/build_minari.py`](../../collection/build_minari.py)): `minari==0.5.3`.

The measurement runner (`../unity/evaluate_onnx.py`) uses the same collection env
(`onnxruntime` + `mlagents_envs`, no PyTorch).

## Collect → Minari

1. Get the model-removed Crawler build and the stock `Crawler.onnx` from the
   [envs repo](https://huggingface.co/datasets/ccnets/causal-gpt-rl-unity-envs).
2. Record rollouts (the stock policy driving the build):

   ```bash
   python collect.py \
       --build path/to/Crawler.exe \
       --onnx  path/to/Crawler.onnx \
       --out   raw/ \
       --target 1000000
   ```
3. Package the raw episodes into a Minari dataset — the source-agnostic packager
   lives in [`collection/`](../../collection/); run it in a `minari==0.5.3` env:

   ```bash
   python ../../collection/build_minari.py \
       --raw raw/ \
       --dataset-id unity/crawler/expert-v0 \
       --description "ML-Agents Crawler, baked ONNX policy (model-removed build)."
   ```

The recipe ends at the Minari dataset — a portable, env-less trajectory set whose
observation and action spaces mirror the build's sensors and action spec (see
[Observation & action spaces](#observation--action-spaces) below). Single-`Box`
spaces follow the same convention as the Gymnasium / MuJoCo Minari datasets.

The same two commands collect **any** build — point `--build`/`--onnx` at it and
pick a `--dataset-id`. For example the discrete **PushBlock** build (a single
`Discrete(7)` move/turn action):

```bash
python collect.py \
    --build path/to/PushBlock.exe \
    --onnx  path/to/PushBlock.onnx \
    --out   raw_pushblock/ \
    --target 1000000
python ../../collection/build_minari.py \
    --raw raw_pushblock/ \
    --dataset-id unity/pushblock/expert-v0 \
    --description "ML-Agents PushBlock, stock discrete ONNX policy (model-removed build)."
```

For multi-agent matches, add `--complete-matches` so the transition target does
not cut through an in-flight field. SoccerTwos has eight 2-vs-2 fields per build
instance (32 agents):

```bash
python collect.py \
    --build path/to/SoccerTwos/UnityEnvironment.exe \
    --onnx path/to/SoccerTwos.onnx \
    --out raw_soccer/ \
    --target 1000000 \
    --complete-matches \
    --env-id soccer-twos
python ../../collection/build_minari.py \
    --raw raw_soccer/ \
    --dataset-id unity/soccer-twos/expert-v0 \
    --ego-agent agent_0 \
    --description "ML-Agents SoccerTwos release-23 stock self-play trajectories."
```

The collector writes each ego-agent trajectory as one episode and writes match
relationships (`match_id`, `field_id`, `team_id`, `group_id`) to the adjacent
`manifest.jsonl`. `--ego-agent` nests both spaces under
`Dict{"agents": {"agent_0": ...}}`, so a consumer reads
`observations["agents"]["agent_0"]` and the episode says whose trajectory it is —
this is the schema the published SoccerTwos and DungeonEscape datasets use. The
leaf spaces are unchanged by the wrapper.

The packager consumes the episode arrays, not the `manifest.jsonl` sidecar;
decentralized shared-policy training therefore sees independent per-agent
episodes. Retain the manifest for W/D/L, team-return, provenance, and future
group-aware processing.

## Observation & action spaces

The Minari spaces are derived from the build's ML-Agents behavior spec, so a build
with different sensors or actions produces the matching dataset with **no code
change**:

- **Observation** — one `Box` per sensor, kept distinct in a `Tuple` (a
  single-sensor build stays a bare `Box`). Distinct sensors carry distinct
  meaning, so they remain separate leaves rather than being flattened into one
  vector; a consumer that wants them concatenated does so itself.
- **Action** — `Box[-1, 1]` (continuous), `Discrete` / `MultiDiscrete`
  (discrete), or `Tuple(Box, Discrete/MultiDiscrete)` (hybrid — continuous and
  discrete together, e.g. move + jump).
- **Multi-agent** — `--ego-agent KEY` nests either of the above under
  `Dict{"agents": {KEY: ...}}`. The leaf spaces are unchanged; the wrapper only
  names whose trajectory the episode holds.

Whichever shape a build produces, the declared space *is* the interface —
`causal_gpt_rl` walks `Tuple` / `Dict` nesting down to the same leaf specs, so a
bare `Box`, a per-sensor `Tuple`, and an ego `Dict` are all ingested the same
way with no adapter.

Two worked builds:

| Build | Observation | Action |
|---|---|---|
| Crawler | `Tuple(Box(126), Box(32))` | `Box(20, [-1, 1])` |
| PushBlock | `Tuple(Box(105), Box(105))` | `Discrete(7)` |

`collect.py` records the raw obs channels and actions flat plus a `spec.json`
(obs channel dims + action kind); `build_minari.py` reads it and rebuilds the
declared spaces above, storing each leaf as its own array.

## Quality tiers (simple / medium)

The MuJoCo Minari datasets ship a `simple` / `medium` / `expert` ladder where each
tier is a *separate policy*. Crawler has only one policy — the stock
`.onnx` — so the lower tiers are synthesized from it by injecting a calibrated
amount of **action noise**: more noise → lower closed-loop return → a lower tier.
The dataset records the noised action that was actually taken, so it stays a valid
`Box[-1, 1]` trajectory set (see [`noisy_policy.py`](noisy_policy.py)).

A tier is defined by its **normalized score**, the same quantity the public table
uses:

```text
norm = 100 * (return - random_ref) / (expert_ref - random_ref)
```

1. Find the noise level for each tier — one build launch measures both endpoints
   (`expert_ref` = no noise, `random_ref` = a uniform-random policy) and the
   return-vs-noise curve, then reports the normalized score at each level:

   ```bash
   python calibrate_noise.py \
       --build path/to/Crawler.exe \
       --onnx  path/to/Crawler.onnx \
       --target-simple 40 --target-medium 70
   ```

   It prints the grid level closest to each target as a ready-to-use
   `--noise-std <value>`. Refine `--grid` around a pick if no level is close
   enough.

2. Record each tier with the chosen noise level (`--noise-seed` makes it
   reproducible), then package as before with a tier-specific id:

   ```bash
   python collect.py --build ... --onnx ... --out raw_medium/ --noise-std 0.20
   python ../../collection/build_minari.py --raw raw_medium/ \
       --dataset-id unity/crawler/medium-v0 \
       --description "ML-Agents Crawler, stock ONNX policy + Gaussian action noise (medium tier)."
   ```

`--noise-std 0` (the default) records the `expert-v0` tier — the plain recipe
above. For a discrete behavior, Gaussian noise is meaningless; use `--epsilon`
(random-action probability) instead, which is also how `random_ref` is measured
(`--epsilon 1`).

### Degradation dials

A scalar dial gives every episode the same skill. A **range** dial instead draws
a value i.i.d. per agent per episode — keyed by `(--noise-seed, agent_index,
episode_index)`, so it is reproducible and independent of traversal order — which
records a *spread* of skill levels whose mean lands the tier, closer to a span of
early-training checkpoints than to one uniformly-degraded policy.

| Dial | Action space | Scope | Flag |
|---|---|---|---|
| Gaussian noise | continuous | scalar | `--noise-std` |
| Gaussian noise | continuous | per episode | `--noise-std-range lo,hi` |
| Random-action ε | discrete | scalar | `--epsilon` |
| Random-action ε | discrete | per episode | `--epsilon-range lo,hi` |
| Softmax temperature | discrete | scalar | `--temperature` |
| Softmax temperature | discrete | per episode | `--temperature-range lo,hi` |
| Random-action ε | discrete | per team, per match | `--team-epsilon-values` (+ `--team0/1-epsilon-values`) |
| Random-action ε | discrete | per cooperative group, per match | `--group-epsilon-values` |

The three range dials are mutually exclusive. Temperature samples
`softmax(logits / T)` and so needs a policy with a `logits` output (the patched
`tier_policies` ONNX); `T = 1.0` is expert and larger `T` degrades smoothly,
sampling plausible near-expert actions rather than uniformly random ones — which
is why it is preferred over `--epsilon` where a `logits` output is available.

Team and group pools share the sampled strength across a whole match, so a
degraded team is degraded *together* rather than per agent. They are epsilon-only
and mutually exclusive with each other.

Record the tier label and policy identity as provenance with `--dataset-quality
{simple,medium,expert,random}`, `--policy-id`, and `--opponent-policy-id`.

## Measure return

Download the ONNX policy from its Hugging Face model repository and the matching
model-removed build from the companion environment repository. Then run:

```bash
python ../unity/evaluate_onnx.py \
    --build path/to/UnityEnvironment.exe \
    --onnx path/to/policy.onnx
```

The script reads the ONNX context, observation, action, and batch dimensions;
validates them against the live ML-Agents behavior spec; maintains a separate
autoregressive window per scene agent; and reports both agent return and, when
ML-Agents group IDs are present, cooperative group return and success rate.

An ONNX exported with batch size equal to the number of agents uses one runtime
call per decision tick. A batch-1 model also works, but is invoked once per agent.

## Files

| File | Role |
|---|---|
| `unity_env.py` | ML-Agents → gymnasium stepping wrapper |
| `onnx_policy.py` | runs a stock ML-Agents ONNX policy in `onnxruntime` |
| `noisy_policy.py` | wraps a policy with action noise to synthesize lower tiers |
| `collect.py` | record per-episode transitions to `.npz` (`--noise-std` for tiers) |
| `calibrate_noise.py` | measure return vs noise; pick `--noise-std` for a target tier |
| `requirements-collect.txt` | collection env pins |
| `../unity/evaluate_matchup.py` | side-swapped Causal-vs-stock team evaluation |

Packaging (`.npz` → Minari) uses the source-agnostic
[`collection/build_minari.py`](../../collection/build_minari.py).
