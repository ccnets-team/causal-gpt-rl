# SageMaker "bring your own container" inference image for Causal-GPT-RL.
#
# Build from the repository root so the build context contains the package
# source (this wraps the *current* project, not a pinned PyPI release):
#
#     docker build -f serving/Dockerfile -t causal-gpt-rl-serving .
#
# The resulting image responds to GET /ping and POST /invocations on port 8080.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/opt/ml/model \
    INFERENCE_DEVICE=cpu

WORKDIR /opt/program

# Install a CPU-only torch first so `pip install .` does not pull the much
# larger default CUDA build to satisfy the torch dependency.
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the causal-gpt-rl package from the current source tree.
COPY pyproject.toml README.md LICENSE ./
COPY causal_gpt_rl ./causal_gpt_rl
RUN pip install .

# Serving-only dependencies and the inference handler.
COPY serving/requirements.txt ./serving-requirements.txt
RUN pip install -r serving-requirements.txt
COPY serving/predictor.py serving/serve.py ./

EXPOSE 8080

# SageMaker runs the image as `docker run <image> serve`; serve.py ignores the
# passed-through "serve" argument and launches gunicorn on port 8080.
ENTRYPOINT ["python", "serve.py"]
