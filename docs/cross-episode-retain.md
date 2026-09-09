# Cross-episode context retention

Cross-episode retention and its lifecycle APIs require `causal-gpt-rl>=0.21.0`.

Select `bos_cache_mode="retain"` to keep BOS and history across natural episode
boundaries. The default `discard` preserves the original execution behavior,
including for existing bundles with no serving field. There is no second
episode-context option. Model weights, heads and feedback coordinates are unchanged.

```python
runner = load_runner(bundle_path, bos_cache_mode="retain")
observation, info = env.reset()
runner.reset(observation)  # new session
for step in range(total_steps):
    action = runner.act()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        terminal = observation
        observation, info = env.reset()
        runner.restart_episode(observation, terminal_state=terminal)
    else:
        runner.observe(observation)
```

The terminal action uses the state where it was selected. The next BOS uses the
new initial state and a zero action. The next real action token uses that same
initial state. Direct pre-tanh feedback preserves the original model sample;
it is never reconstructed from a saturated environment action.

`reset()` and `reset_rows()` always erase session history. Use them for a new
user, agent replacement, or model/normalizer changes. `restart_episode()` retains
history only under retain. Under discard it accepts an all-row reset; partial
restarts are rejected without changing state. Use `advance` for partial discard
transitions so survivors and restarted rows are each consumed once.

For same-step autoreset or explicit partial resets, use
`runner.advance(observations, done_mask=done, terminal_states=final_observations)`.
It consumes survivors and restarted rows in one call. Missing final observations
may be omitted; do not substitute reset observations for terminal observations.
Data recording still needs actual final observations.

`run_episodes` selects the lifecycle from the runner's mode, including environment
truncation and its own `max_steps`. Each episode keeps separate return/length
statistics. `CollectionRunner` automatically handles vector NEXT_STEP: the ignored
autoreset action adds no history and no recorded transition. For single-env
collection, call `collector.restart_episode(initial_state)` after the terminal
`collector.observe(...)`; `collector.reset(...)` starts a new session. The
`examples/deploy/record.py` episode loop selects this automatically.

The Python runner supports cached and windowed retention. Cached ordinary steps
still process one new token per active row. Asynchronous rows split into groups
with independent cache lengths and positions; compatible groups rejoin. A boundary
adds an action-free terminal ingest. No full history is re-encoded, and paused
rows receive no dummy KV positions. Row splitting/merging copies KV tensors, so
boundary cost depends on cache length and the distribution of episode endings.
The cache adapter supports layers-based caches used by the supported Transformers
runtime; other cache layouts fail explicitly at the transition.

Exporting with `bos_cache_mode="retain"` automatically adds
`requires_capabilities: ["cross_episode_context", ...]`. Loaders validate the mode
and capability together. Old BOS-only retain artifacts need re-export; no legacy
retain compatibility mode is provided. Explicit load-time mode overrides take
precedence after artifact validation. Checkpoints can declare the same serving
metadata or select the mode through `PolicyRunner.from_checkpoint`.

ONNX graphs keep their tensor I/O and carry retain/capability metadata. The host
must implement the lifecycle. The shipped Unity C# runtime and Python Unity ONNX
evaluators currently reject cross-episode retain; they do not silently run the
bundle with discard semantics. Python windowed PolicyRunner is supported.

Applied-action overrides and history-preserving model hot-swaps are unsupported.
