# Causal-GPT-RL serving container

A minimal SageMaker "bring your own container" image that serves a trained
Causal-GPT-RL policy bundle. It wraps the public inference surface
(`causal_gpt_rl.inference`) with the two HTTP endpoints SageMaker requires and
nothing else — model packaging, training, and deployment glue live outside this
directory.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile` | CPU inference image; installs the package from the current source tree. |
| `predictor.py` | Flask app exposing `GET /ping` and `POST /invocations`. |
| `serve.py` | Container entrypoint; launches gunicorn on port 8080. |
| `requirements.txt` | Serving-only dependencies (Flask, gunicorn). |

## Build

Build from the **repository root** so the package source is in the build context:

```bash
docker build -f serving/Dockerfile -t causal-gpt-rl-serving .
```

## Run locally

Mount an exported bundle directory at `/opt/ml/model`, the path SageMaker uses.
A current (v2) bundle is two files — `config.json` and `model.safetensors`, which
carries the state-normalization statistics inside it. Older v1 bundles also ship
a `state_normalizer.safetensors` sidecar and still load.

```bash
docker run --rm -p 8080:8080 \
  -v /path/to/export-bundle:/opt/ml/model \
  causal-gpt-rl-serving serve
```

Health check and a sample inference call:

```bash
curl http://localhost:8080/ping

curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"observations": [[0.0, 0.0, 0.0], [0.1, 0.0, -0.1]]}'
```

## Request / response contract

The container is **stateless**: each request carries the observation history
for one episode, and the handler returns the action for the latest observation.

Single episode:

```json
{"observations": [[o0...], [o1...], ..., [oT...]]}
```
```json
{"action": [...]}
```

Batch of independent episodes:

```json
{"instances": [{"observations": [...]}, {"observations": [...]}]}
```
```json
{"predictions": [{"action": [...]}, {"action": [...]}]}
```

A bare JSON list is also accepted and treated as `observations`.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/opt/ml/model` | Where to find the bundle (see below). |
| `INFERENCE_DEVICE` | `cpu` | Torch device for inference. |
| `LOG_LEVEL` | `INFO` | Container log level. |
| `KV_CACHE_MAX_LEN` | _(bundle default)_ | Override the KV cache cap. |
| `USE_WINDOWED` | `0` | Use windowed prediction instead of cached KV. |
| `GUNICORN_WORKERS` | `1` | Worker processes. |
| `GUNICORN_THREADS` | `4` | Threads per worker. |
| `GUNICORN_TIMEOUT` | `120` | Worker timeout (seconds). |

### Bundle location

`MODEL_PATH` does not have to be the bundle directory itself. The container
looks, in order, for:

1. `MODEL_PATH/config.json` — a mounted export, a Hugging Face snapshot, or a
   single downloaded checkpoint slot.
2. `MODEL_PATH/bundle/config.json`.
3. exactly one `MODEL_PATH/*/bundle/config.json` — the layout SageMaker produces
   when it extracts a training artifact, where the bundle sits under a run
   namespace the deployer cannot know in advance.

If nothing matches, or more than one candidate does, startup fails with the
paths it considered rather than guessing. Set `MODEL_PATH` to the exact bundle
directory to override.
