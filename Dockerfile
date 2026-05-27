FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY pipelines/ pipelines/
COPY README.md ./

RUN uv sync --frozen --no-dev


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_FORMAT=json
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns

RUN useradd --create-home appuser \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "iris.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
