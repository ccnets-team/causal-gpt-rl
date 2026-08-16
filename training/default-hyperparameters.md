# Default Hyperparameters

[`hyperparameters.py`](hyperparameters.py) is the payload schema for a training
job. Fifteen fields, and exactly one of them you have to supply:

```python
dataset_ids: Optional[list[str]] = None   # required
```

Your datasets. Everything else arrives already set, and those settings are the
recipe every published bundle was trained with.

Their ranges, what fails a job, and which of them the product owns are in
[docs/aws/sagemaker-inputs.md](docs/aws/sagemaker-inputs.md).

## Every published bundle came out of those defaults

The eight MuJoCo bundles on
[ccnets/causal-gpt-rl](https://huggingface.co/ccnets/causal-gpt-rl) — Ant,
HalfCheetah, Hopper, Walker2d, Humanoid, HumanoidStandup, Pusher and Swimmer —
and the five Unity ML-Agents bundles on
[ccnets/causal-gpt-rl-unity](https://huggingface.co/ccnets/causal-gpt-rl-unity) —
Crawler, DungeonEscape, PushBlock, Pyramids and SoccerTwos — were each trained by
handing over a dataset and changing nothing else. Nothing was selected per task.

That is a wider spread of tasks than the sameness suggests. Swimmer's observation
is 8 numbers and Humanoid's is 348, a factor of forty-three, and both are read by
the same 256-wide, four-layer, eight-head backbone over the same 32-step window.
The action side varies as much: Crawler emits 20 continuous values, Pyramids
picks one of five, and SoccerTwos drives three MultiDiscrete heads. The model
does not grow to meet the task, and it does not change shape to meet the
interface.

There are still RL-specific differences that the training system has to handle.
Continuous, discrete, MultiDiscrete, multi-head, and hybrid action spaces do not
produce the same learning signals. A continuous head predicts a mean and a
log-standard-deviation and is scored by a Gaussian likelihood; a categorical head
predicts logits and is scored by a cross-entropy. Those two are not on the same
scale to begin with, and the gap moves during a run rather than staying put: the
categorical gradient shrinks as the policy sharpens and its entropy falls, while
the continuous one grows as the predicted spread narrows, since the error term is
divided by a variance that is itself being learned. A bundle with several heads
carries several of these at once, so their objectives have to be balanced
accordingly. But those are properties of the RL interface, not reasons to
redesign the transformer or invent a new recipe for every environment.

That is also why we publish untuned bundle results. Their purpose is not to show
that the defaults are optimal. They are a test of whether the training service
can behave like a general BYOD system: provide a compatible offline trajectory
dataset, let the dataset define the observation and action interface, and train
without first searching for environment-specific hyperparameters.

The dataset still matters. A general training recipe cannot create information
that the trajectories do not contain, and harder control problems may demand
better coverage, richer state, or more informative behavior in the data. The
current results therefore validate generalization across the tasks we have
tested; they do not establish a ceiling. Problems substantially more complex
than the published Humanoid setting, especially those with longer-horizon
interaction or more difficult contact dynamics, remain a stronger test.

### Checking the model half yourself

Five of the defaults are recorded in every bundle, so you do not have to take the
paragraph above on trust:

```python
import json, urllib.request

BUNDLES = [
    ("ccnets/causal-gpt-rl",
     ["ant-v5", "halfcheetah-v5", "hopper-v5", "humanoid-v5",
      "humanoidstandup-v5", "pusher-v5", "swimmer-v5", "walker2d-v5"]),
    ("ccnets/causal-gpt-rl-unity",
     ["crawler", "dungeon-escape", "pushblock", "pyramids", "soccer-twos"]),
]

for repo, envs in BUNDLES:
    for env in envs:
        url = f"https://huggingface.co/{repo}/raw/main/{env}/config.json"
        with urllib.request.urlopen(url) as r:
            c = json.loads(r.read())
        m = c["model_config"]
        print(env, c["context_length"], m["d_model"], m["num_layers"],
              m["num_heads"], m["dropout"])
```

Every row comes back `32 256 4 8 0.05` — `context_length`, `d_model`,
`num_layers`, `num_heads` and `dropout`, each still at the value the schema
ships. What differs across the thirteen is only what the environment dictates:
the observation and action shapes, their bounds and dtypes. Nothing that was
*chosen* varies.

The remaining eight fields are training-time only and are not written into a
bundle, so that check covers the model, not the optimizer.
