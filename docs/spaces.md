# Observation & Action Spaces

Causal-GPT-RL policies are **autoregressive**: at each step the model reads the
running `(state, action)` context and emits the next action, which is fed
straight back as the next input (KV-cached, no critic or auxiliary networks at
inference). A policy is therefore tied to the **shapes it acts in** — so every
bundle carries its declared Gymnasium `observation_space` and `action_space`,
and the runtime adapts your environment's observations and the emitted actions
to and from those spaces for you.

You interact with the runtime entirely in your env's **native Gymnasium
spaces**: pass observations exactly as your env produces them, and you receive
actions that are valid samples of the declared `action_space` — ready to drop
into `env.step(...)`.

## Supported spaces

Fixed-shape Gymnasium spaces, and any `Dict` / `Tuple` nesting of them:

| Space | You pass (observation) | You get back (action) |
|---|---|---|
| `Box(n)` (1-D) | `np.ndarray`, shape `(n,)` | `np.ndarray`, clipped to bounds |
| `Discrete(n, start=s)` | `int` | `int` in `[s, s+n)` |
| `MultiDiscrete([...])` | `np.ndarray` of ints | `np.ndarray` of ints (per-dim `start` applied) |
| `MultiBinary(n)` | `np.ndarray` of `{0,1}` | `np.ndarray` of `{0,1}` (int8) |
| `Dict({...})` | `dict` (key → subspace value) | `dict` |
| `Tuple((...))` | `tuple` | `tuple` |

`Dict` / `Tuple` nest the leaf spaces arbitrarily. The runtime flattens
observations and restructures actions to match the declared container —
including Gymnasium's alphabetical key-sorting for `Dict`, positional order for
`Tuple`, and per-dimension `start` offsets.

**Out of scope:** image (n-D `Box`), `Text`, `Sequence`, `Graph`, `OneOf` —
variable-length or structural spaces that don't map to fixed policy heads.
Loading or running such a space raises a self-describing error that names the
offending type and the supported set, so the boundary is explicit rather than a
silent mis-decode.

## Loading a policy

```python
from causal_gpt_rl.inference import load_runner, load_runner_from_hub

# From a local bundle directory
runner = load_runner("path/to/bundle")

# Or from the Hugging Face Hub
runner = load_runner_from_hub(repo_id="ccnets/causal-gpt-rl", subfolder="ant-v5")
```

If a bundle declares spaces this runtime cannot serve, it is refused at load
time with a clear message instead of mis-decoding later.

## Running a rollout

The runner keeps the autoregressive context internally. Seed it with the first
observation, then alternate `act()` and `observe()`:

```python
import gymnasium as gym

env = gym.make("Ant-v5")
obs, _ = env.reset(seed=0)

runner.reset(obs)                  # seed the context with the first observation
done = False
while not done:
    action = runner.act()          # next action — a valid sample of action_space
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    if not done:
        runner.observe(obs)        # feed the next observation back into the context
```

`act()` decodes deterministically (the policy's mean action). For batch
evaluation over several episodes, use the helper:

```python
from causal_gpt_rl.inference import run_episodes

stats = run_episodes(env, runner, num_episodes=5, seed=0)
print(stats["return_mean"], stats["return_std"])
```

## Structured spaces — a worked example

Suppose your environment declares:

```python
import gymnasium as gym

observation_space = gym.spaces.Dict({
    "pos":  gym.spaces.Box(-1.0, 1.0, shape=(3,)),
    "kind": gym.spaces.Discrete(4),
})
action_space = gym.spaces.Tuple((
    gym.spaces.Box(-1.0, 1.0, shape=(2,)),   # e.g. a move vector
    gym.spaces.MultiBinary(3),               # e.g. three independent toggles
))
```

You pass observations as the env emits them and receive actions in the same
shape — no flattening on your side:

```python
runner.reset({"pos": pos_array, "kind": 2})
action = runner.act()
# action == (np.array([...], dtype=float32), np.array([1, 0, 1], dtype=int8))
move, toggles = action
```

The runtime maps the structured observation into the model and restructures the
emitted action back into the declared `Tuple`.

## What the decode guarantees

- **Box** — clipped to the declared bounds.
- **Discrete / MultiDiscrete** — the most-likely class index, with the declared
  `start` offset applied.
- **MultiBinary** — each of the `n` bits independently on or off.
- **Dict / Tuple** — the leaf values above, reassembled into the declared
  container (`Dict` keys sorted alphabetically, `Tuple` order preserved).

The action you get is always a structurally valid sample of `action_space`, so
it goes straight into `env.step(...)`.
