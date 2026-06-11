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

Mount an exported bundle directory (containing `config.json`,
`model.safetensors`, and `state_normalizer.safetensors`) at `/opt/ml/model`,
the path SageMaker uses:

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
| `MODEL_PATH` | `/opt/ml/model` | Bundle directory to load. |
| `INFERENCE_DEVICE` | `cpu` | Torch device for inference. |
| `KV_CACHE_MAX_LEN` | _(bundle default)_ | Override the KV cache cap. |
| `USE_WINDOWED` | `0` | Use windowed prediction instead of cached KV. |
| `GUNICORN_WORKERS` | `1` | Worker processes. |
| `GUNICORN_THREADS` | `4` | Threads per worker. |
| `GUNICORN_TIMEOUT` | `120` | Worker timeout (seconds). |
