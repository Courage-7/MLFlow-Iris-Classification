# 🌸 Iris Flower Classification (MLflow)

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker" />
</div>

A machine learning pipeline for classifying Iris flowers into three species based on sepal and petal dimensions. This project demonstrates model training, inference, REST API serving, and robust experiment tracking using scikit-learn, MLflow, and FastAPI.

## 📖 Project Overview

This repository contains an end-to-end machine learning workflow built around the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). The model attempts to classify an Iris flower into one of three species (*setosa*, *versicolor*, or *virginica*) based on four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The primary goal of this repository is to demonstrate how to track models, parameters, and metrics systematically using **MLflow** alongside a clean, modular project structure.

## 🏗️ Project Structure

```text
src/iris/
├── config.py                # Environment-driven configuration (dotenv)
├── data/                    # Data pipeline
│   └── loader.py            # Dataset loading & preprocessing
├── training/                # Training pipelines
│   ├── autolog.py           # MLflow automatic logging implementation
│   └── manual.py            # Custom, manual metric and parameter logging
├── inference/               # Prediction engine
│   └── predictor.py         # Model loading and batch inference
├── api/                     # REST API serving
│   └── app.py               # FastAPI application with versioned endpoints
└── utils/                   # Shared utilities
    └── logging.py           # Logging with text/JSON output support

pipelines/                   # CLI execution entry points
├── train_autolog.py         # Script to run autolog training
├── train_manual.py          # Script to run manual training
├── predict.py               # Script to run predictions
└── serve.py                 # Script to start the FastAPI server

tests/                       # Test suite
├── conftest.py              # Shared fixtures
├── test_data_loader.py      # Data loader unit tests
└── test_api.py              # API endpoint tests
```

## ⚙️ Configuration

All settings are managed via environment variables. Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking backend URI |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | `./mlruns` | Relative artifact root for local runs |
| `MLFLOW_EXPERIMENT_NAME` | `MLflow Quickstart` | Experiment name for runs |
| `MLFLOW_REGISTERED_MODEL_NAME` | `iris-logistic-regression` | Registered model name in MLflow Model Registry |
| `MODEL_URI` | *(empty)* | Model to auto-load on API startup |
| `LOG_FORMAT` | `text` | `text` or `json` for structured logging |
| `LOG_LEVEL` | `INFO` | Python logging level |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

### 1. Installation

Clone the repository and install the required dependencies:

```bash
# Using uv (recommended - uses the locked dependencies)
uv sync

# With dev dependencies (pytest, black, httpx)
uv sync --all-extras

# Or using pip (installs latest compatible versions)
pip install -e .
```

### 2. Train the Model

You can choose between two training pipelines to see different ways MLflow can track your experiments:

Training writes to `sqlite:///mlflow.db` and stores artifacts under `./mlruns`
by default so the run metadata stays compatible with the Docker volume mounts.
Each training run also registers a new model version in the MLflow Model
Registry using `MLFLOW_REGISTERED_MODEL_NAME` (`iris-logistic-regression` by
default).

**Option A: Manual Logging**
Manually logs specific metrics (like accuracy) and parameters.
```bash
uv run pipelines/train_manual.py
```

**Option B: Autologging**
Automatically logs the model, parameters, and training metrics without explicit logging code.
```bash
uv run pipelines/train_autolog.py
```

### 3. Track Experiments in MLflow

Once you have trained the model, launch the MLflow UI to inspect the runs, compare parameters, and view the logged models:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```
Open your browser and navigate to `http://127.0.0.1:5000` to view the MLflow dashboard.

### 4. Make Predictions

To run inference on new data, use the `predict.py` pipeline. You will need to provide the MLflow model URI (you can find this in the MLflow UI after running a training script, or use the latest run).

```bash
uv run pipelines/predict.py --model-uri "models:/iris-logistic-regression/1"
```
*(Note: Replace the version number with the exact registered model URI printed by training or returned by `/v1/model/discover`.)*

### 5. Serve the Model via REST API

Start the FastAPI server to serve predictions over HTTP:

```bash
# Start with a pre-loaded model
uv run pipelines/serve.py --model-uri "models:/iris-logistic-regression/1"

# Or start without a model and load one dynamically via the API
uv run pipelines/serve.py
```

Open the interactive Swagger UI at `http://127.0.0.1:8000/docs` to explore and test the API.

#### Available Endpoints

All endpoints are versioned under `/v1/`:

| Endpoint | Method | Tag | Description |
|---|---|---|---|
| `/v1/health` | `GET` | Health | Check API health and model loading status |
| `/v1/model/discover` | `GET` | Model | Auto-discover available MLflow model URIs |
| `/v1/model/load` | `POST` | Model | Load an MLflow model into memory |
| `/v1/data` | `GET` | Data | Retrieve the full Iris dataset (`?limit=N` to cap rows) |
| `/v1/predict` | `POST` | Inference | Predict Iris species from feature values |
| `/v1/evaluate` | `POST` | Inference | Evaluate model accuracy against labeled test data |

Every response includes an `X-Request-ID` header for request tracing.

#### Model Registry API Workflow

After training, discover registered model versions:

```bash
curl http://127.0.0.1:8000/v1/model/discover
```

Load a registered model version into the API:

```bash
curl -X POST http://127.0.0.1:8000/v1/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_uri":"models:/iris-logistic-regression/1"}'
```

Check which model is loaded:

```bash
curl http://127.0.0.1:8000/v1/health
```

Run a prediction:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      }
    ]
  }'
```

The server runs with **hot-reload** enabled — code changes in `src/` are picked up automatically.

### 6. Run with Docker

Build and run the API as a container:

```bash
# Build the image
docker build -t iris-api .

# Run the container (mount local MLflow data for model discovery)
docker run -p 8000:8000 \
  -v "${PWD}/mlruns:/app/mlruns" \
  -v "${PWD}/mlflow.db:/app/mlflow.db" \
  iris-api

# Run with a model pre-loaded
docker run -p 8000:8000 \
  -v "${PWD}/mlruns:/app/mlruns" \
  -v "${PWD}/mlflow.db:/app/mlflow.db" \
  -e MODEL_URI="models:/iris-logistic-regression/1" \
  iris-api
```

## 🧪 Testing

Run the full test suite with:

```bash
uv run pytest tests/ -v
```

Tests cover:
- **Data loader**: Shape validation, split proportions, return types
- **API endpoints**: Health check, data retrieval, prediction error handling, evaluation, input validation, request ID headers

## 📊 Tech Stack

- **Machine Learning Framework**: [scikit-learn](https://scikit-learn.org/) (Logistic Regression)
- **Experiment Tracking & Registry**: [MLflow](https://mlflow.org/)
- **REST API**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Containerization**: [Docker](https://www.docker.com/) (multi-stage build)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/)
- **Testing**: [pytest](https://docs.pytest.org/)
- **Package Management**: `uv` or `pip`
