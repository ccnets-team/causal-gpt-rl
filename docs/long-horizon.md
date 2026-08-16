# Measuring a Long Horizon

Past the horizon where episodes start splitting into "died early" and "ran the
full length", a return mean stops describing any real episode — it lands between
the two modes, and its standard deviation reports the split rather than
run-to-run noise. Report survival there:

```python
from causal_gpt_rl.inference import load_runner, run_episodes
from examples.deploy.survival import format_survival_table, survival_stats

for kv in (32, 64, 128):
    runner = load_runner("path/to/bundle", kv_cache_max_len=kv)
    stats = run_episodes(env, runner, num_episodes=50, seed=0, max_steps=5000)
    survival = survival_stats(
        stats["lengths"], stats["returns"], horizon=5000, bucket=1000
    )
    print(f"kv={kv}")
    print(format_survival_table(survival))
```

Two columns carry most of the signal:

- **`conditional`** — the share of episodes entering an interval that also leave
  it. Flat across intervals means failure is a constant per-step risk; falling
  means it compounds with rollout depth.
- **`return_per_step_completers`** — return per step over the episodes that
  reached the horizon. Episodes that died early drag the all-episode figure down,
  so this is the one that separates "still performing" from "alive but no longer
  doing the task".

[`survival_stats`](../examples/deploy/survival.py) is a pure function of lengths
and returns — analysis over results rather than runtime behaviour, so it works
equally on episodes collected by a loop of your own. It lives in the repository's
`examples/` rather than in the installed package, so working from a `pip install`
alone means copying the file or cloning the repository.
