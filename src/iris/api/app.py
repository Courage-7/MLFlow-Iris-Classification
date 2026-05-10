import os
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from contextlib import asynccontextmanager  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import mlflow  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from iris.utils.logging import setup_logging  # noqa: E402

logger = setup_logging(__name__)

# Global variable to hold the loaded MLflow model
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the MLflow model on startup."""
    global model
    model_uri = os.environ.get("MODEL_URI")
    if not model_uri:
        logger.warning(
            "MODEL_URI environment variable not set. "
            "Endpoint will fail unless model is loaded manually."
        )
    else:
        logger.info(f"Loading MLflow model from {model_uri}...")
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load MLflow model: {e}")

    yield

    # Cleanup on shutdown (if any)
    logger.info("Shutting down model serving API.")
    model = None


app = FastAPI(
    title="Iris Classification API",
    description="API for predicting Iris flower species using an MLflow tracked model.",
    version="0.1.0",
    lifespan=lifespan,
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")


class ModelDiscoverResponse(BaseModel):
    available_models: List[Dict[str, Any]]


class ModelLoadRequest(BaseModel):
    model_uri: str = Field(
        ..., description="MLflow model URI to load (e.g., runs:/<id>/model)"
    )


class ModelLoadResponse(BaseModel):
    message: str


class DatasetResponse(BaseModel):
    data: List[IrisFeatures]
    labels: List[str]
    total: int


class PredictionRequest(BaseModel):
    instances: List[IrisFeatures]


class PredictionResponse(BaseModel):
    predictions: List[int]
    class_names: List[str]


# Mapping from integer class to string class
CLASS_MAPPING = {0: "setosa", 1: "versicolor", 2: "virginica"}


# ── 1. Model Discovery ──────────────────────────────────────────────


@app.get("/model/discover", response_model=ModelDiscoverResponse)
def discover_models():
    """Auto-discover available MLflow model URIs from local runs."""
    from iris.config import EXPERIMENT_NAME, TRACKING_URI

    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)

    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return ModelDiscoverResponse(available_models=[])

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10,
        )

        models = []
        for run in runs:
            model_uri = f"runs:/{run.info.run_id}/model"
            models.append(
                {
                    "run_id": run.info.run_id,
                    "model_uri": model_uri,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "metrics": dict(run.data.metrics),
                }
            )

        return ModelDiscoverResponse(available_models=models)
    except Exception as e:
        logger.error(f"Error discovering models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. Model Loading ────────────────────────────────────────────────


@app.post("/model/load", response_model=ModelLoadResponse)
def load_model_endpoint(request: ModelLoadRequest):
    """Dynamically load an MLflow model into memory."""
    global model
    logger.info(f"Attempting to load model from {request.model_uri}...")
    try:
        model = mlflow.pyfunc.load_model(request.model_uri)
        msg = f"Model loaded successfully from {request.model_uri}"
        logger.info(msg)
        return ModelLoadResponse(message=msg)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


# ── 3. Dataset ───────────────────────────────────────────────────────


@app.get("/data", response_model=DatasetResponse)
def get_data(limit: int | None = None):
    """Get the full Iris dataset. Set `limit` to cap the number of rows returned."""
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data, iris.target

    n = len(X) if limit is None else min(limit, len(X))

    features = [
        IrisFeatures(
            sepal_length=float(X[i][0]),
            sepal_width=float(X[i][1]),
            petal_length=float(X[i][2]),
            petal_width=float(X[i][3]),
        )
        for i in range(n)
    ]
    labels = [CLASS_MAPPING.get(int(y[i]), "unknown") for i in range(n)]

    return DatasetResponse(data=features, labels=labels, total=len(X))


# ── 4. Prediction ───────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict the class of iris flowers."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        import pandas as pd

        data = [
            {
                "sepal length (cm)": instance.sepal_length,
                "sepal width (cm)": instance.sepal_width,
                "petal length (cm)": instance.petal_length,
                "petal width (cm)": instance.petal_width,
            }
            for instance in request.instances
        ]
        df = pd.DataFrame(data)

        preds = model.predict(df)
        pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        names = [CLASS_MAPPING.get(p, "unknown") for p in pred_list]

        return PredictionResponse(predictions=pred_list, class_names=names)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start():
    """Entrypoint for the `serve-api` script defined in pyproject.toml."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Start the Iris FastAPI Server")
    parser.add_argument(
        "--model-uri",
        type=str,
        required=False,
        default="",
        help="MLflow model URI (optional)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    if args.model_uri:
        os.environ["MODEL_URI"] = args.model_uri

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "iris.api.app:app",
        host=args.host,
        port=args.port,
        reload=True,
        reload_dirs=["src"],
    )


if __name__ == "__main__":
    start()
