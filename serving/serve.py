"""Container entrypoint for SageMaker model serving.

SageMaker starts an inference container with ``docker run <image> serve``;
the literal ``serve`` argument is passed through and ignored here. This script
launches gunicorn bound to port 8080 (the port SageMaker probes for /ping and
/invocations) with the Flask app defined in ``predictor:app``.

Tunables via environment variables:
    SAGEMAKER_BIND     bind address (default 0.0.0.0:8080)
    GUNICORN_WORKERS   worker processes (default 1; the model loads per worker)
    GUNICORN_THREADS   threads per worker (default 4)
    GUNICORN_TIMEOUT   worker timeout in seconds (default 120)

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    bind = os.environ.get("SAGEMAKER_BIND", "0.0.0.0:8080")
    workers = os.environ.get("GUNICORN_WORKERS", "1")
    threads = os.environ.get("GUNICORN_THREADS", "4")
    timeout = os.environ.get("GUNICORN_TIMEOUT", "120")

    cmd = [
        "gunicorn",
        "--bind", bind,
        "--workers", workers,
        "--threads", threads,
        "--timeout", timeout,
        "--preload",
        "predictor:app",
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
