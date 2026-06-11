"""SageMaker inference handler for Causal-GPT-RL policy bundles.

This module is a thin serving layer around the public inference surface
(``causal_gpt_rl.inference.load_runner`` / ``PolicyRunner``). It exposes the
two endpoints SageMaker requires from a "bring your own container" image:

    GET  /ping          -> 200 when the model bundle is loaded and healthy.
    POST /invocations   -> run the policy on the request body and return actions.

The container is stateless: each ``/invocations`` request carries the
observation history for an episode, and the handler replays that history
through a fresh policy context to produce the action for the latest
observation. No training-time logic is involved.

Request body (Content-Type: application/json), single episode::

    {"observations": [[o0...], [o1...], ..., [oT...]]}

    -> {"action": [...]}            # action for the latest observation oT

Or a batch of independent episodes::

    {"instances": [{"observations": [...]}, {"observations": [...]}]}

    -> {"predictions": [{"action": [...]}, {"action": [...]}]}

A bare list payload is also accepted and treated as ``observations``.

The model bundle is read from ``MODEL_PATH`` (default ``/opt/ml/model``),
which is where SageMaker mounts the model artifact.

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

import numpy as np
from flask import Flask, Response, request

from causal_gpt_rl.inference import PolicyRunner, load_runner

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model")
DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")
KV_CACHE_MAX_LEN = os.environ.get("KV_CACHE_MAX_LEN")
USE_WINDOWED = os.environ.get("USE_WINDOWED", "0").lower() in ("1", "true", "yes")

app = Flask(__name__)

_runner: PolicyRunner | None = None
_runner_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_runner() -> PolicyRunner:
    """Load the policy runner once and reuse it for every request."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                kv = int(KV_CACHE_MAX_LEN) if KV_CACHE_MAX_LEN else None
                _runner = load_runner(
                    MODEL_PATH,
                    device=DEVICE,
                    num_envs=1,
                    kv_cache_max_len=kv,
                    use_windowed=USE_WINDOWED,
                )
    return _runner


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _predict_action(observations: list) -> Any:
    """Replay an episode's observation history and return the latest action.

    ``observations`` is an ordered list ``[o0, o1, ..., oT]`` of states seen so
    far. The returned action is the one the policy would take at ``oT``.
    """
    if not observations:
        raise ValueError("'observations' must contain at least one observation.")

    runner = get_runner()
    with _inference_lock:
        runner.reset(np.asarray(observations[0], dtype=np.float32))
        action = runner.act()
        for obs in observations[1:]:
            runner.observe(np.asarray(obs, dtype=np.float32))
            action = runner.act()
    return _to_jsonable(action)


def _extract_observations(instance: Any) -> list:
    if isinstance(instance, dict):
        if "observations" not in instance:
            raise ValueError("Each instance must contain an 'observations' list.")
        return list(instance["observations"])
    if isinstance(instance, list):
        return instance
    raise ValueError(f"Unsupported instance type: {type(instance).__name__}")


@app.route("/ping", methods=["GET"])
def ping() -> Response:
    try:
        get_runner()
        status = 200
    except Exception:  # noqa: BLE001 - report any load failure as unhealthy.
        status = 503
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations() -> Response:
    if request.content_type and "json" not in request.content_type:
        return Response(
            response=json.dumps({"error": "Content-Type must be application/json."}),
            status=415,
            mimetype="application/json",
        )

    try:
        payload = request.get_json(force=True)
    except Exception:  # noqa: BLE001
        return Response(
            response=json.dumps({"error": "Request body is not valid JSON."}),
            status=400,
            mimetype="application/json",
        )

    try:
        if isinstance(payload, dict) and "instances" in payload:
            predictions = [
                {"action": _predict_action(_extract_observations(inst))}
                for inst in payload["instances"]
            ]
            body = {"predictions": predictions}
        else:
            observations = _extract_observations(payload)
            body = {"action": _predict_action(observations)}
    except ValueError as exc:
        return Response(
            response=json.dumps({"error": str(exc)}),
            status=400,
            mimetype="application/json",
        )

    return Response(
        response=json.dumps(body),
        status=200,
        mimetype="application/json",
    )
