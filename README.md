# 🌸 MLflow Iris Flower Classification

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
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
├── config.py                # Centralized configuration (paths, hyperparameters)
├── data/                    # Data pipeline
│   └── loader.py            # Dataset loading & preprocessing
├── training/                # Training pipelines
│   ├── autolog.py           # MLflow automatic logging implementation
│   └── manual.py            # Custom, manual metric and parameter logging
├── inference/               # Prediction engine
│   └── predictor.py         # Model loading and batch inference
├── api/                     # REST API serving
│   └── app.py               # FastAPI application with model serving endpoints
└── utils/                   # Shared utilities
    └── logging.py           # Standardized Python logging

pipelines/                   # CLI execution entry points
├── train_autolog.py         # Script to run autolog training
├── train_manual.py          # Script to run manual training
├── predict.py               # Script to run predictions
└── serve.py                 # Script to start the FastAPI server

tests/                       # Unit tests
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- `uv` (recommended) or `pip`

### 1. Installation

Clone the repository and install the required dependencies:

```bash
# Using uv (recommended - uses the locked dependencies)
uv sync

# Or using pip (installs latest compatible versions)
pip install -e .
```

### 2. Train the Model

You can choose between two training pipelines to see different ways MLflow can track your experiments:

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
uv run mlflow ui
```
Open your browser and navigate to `http://127.0.0.1:5000` to view the MLflow dashboard.

### 4. Make Predictions

To run inference on new data, use the `predict.py` pipeline. You will need to provide the MLflow model URI (you can find this in the MLflow UI after running a training script, or use the latest run).

```bash
uv run pipelines/predict.py --model-uri "models:/iris_model"
```
*(Note: Replace the URI with the exact model path if not using the model registry).*

### 5. Serve the Model via REST API

Start the FastAPI server to serve predictions over HTTP:

```bash
# Start with a pre-loaded model
uv run pipelines/serve.py --model-uri "runs:/<run_id>/model"

# Or start without a model and load one dynamically via the API
uv run pipelines/serve.py
```

Open the interactive Swagger UI at `http://127.0.0.1:8000/docs` to explore and test the API.

#### Available Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/model/discover` | `GET` | Auto-discover available MLflow model URIs from local experiment runs |
| `/model/load` | `POST` | Load an MLflow model into memory by providing its URI |
| `/data` | `GET` | Retrieve the full Iris dataset (use `?limit=N` to cap rows) |
| `/predict` | `POST` | Predict Iris species from feature values using the loaded model |

The server runs with **hot-reload** enabled — code changes in `src/` are picked up automatically.

## 📊 Tech Stack

- **Machine Learning Framework**: [scikit-learn](https://scikit-learn.org/) (Logistic Regression)
- **Experiment Tracking & Registry**: [MLflow](https://mlflow.org/)
- **REST API**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/)
- **Package Management**: `uv` or `pip`
