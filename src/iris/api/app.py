import os
import sys
import uuid
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from contextlib import asynccontextmanager  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import mlflow  # noqa: E402
from fastapi import FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.routing import APIRouter  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from iris.config import (  # noqa: E402
    LOG_FORMAT,
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
    configure_mlflow,
)
from iris.utils.logging import setup_logging  # noqa: E402

logger = setup_logging(__name__, log_format=LOG_FORMAT)

# Global variable to hold the loaded MLflow model
model = None
loaded_model_uri: str | None = None


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the MLflow model on startup if MODEL_URI is set."""
    global loaded_model_uri, model
    model_uri = os.environ.get("MODEL_URI")
    if not model_uri:
        logger.warning(
            "MODEL_URI environment variable not set. "
            "Use POST /v1/model/load to load a model."
        )
    else:
        logger.info(f"Loading MLflow model from {model_uri}...")
        try:
            logger.info(f"Setting tracking URI to: {TRACKING_URI}")
            configure_mlflow()
            model = mlflow.pyfunc.load_model(model_uri)
            loaded_model_uri = model_uri
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load MLflow model: {e}")

    yield

    logger.info("Shutting down model serving API.")
    model = None
    loaded_model_uri = None


# ── App & Middleware ─────────────────────────────────────────────────

app = FastAPI(
    title="Iris Classification API",
    description=(
        "REST API for predicting Iris flower species using an "
        "MLflow tracked model. All endpoints are versioned under /v1/."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    """Inject a unique request ID into every request/response."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Pydantic Models ──────────────────────────────────────────────────

CLASS_MAPPING: Dict[int, str] = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_uri: Optional[str] = None
    version: str


class ModelDiscoverResponse(BaseModel):
    available_models: List[Dict[str, Any]]


class ModelLoadRequest(BaseModel):
    model_uri: str = Field(
        ...,
        description="MLflow model URI (e.g., models:/iris-logistic-regression/1)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_uri": "models:/iris-logistic-regression/1",
                }
            ]
        }
    }


class ModelLoadResponse(BaseModel):
    message: str
    model_uri: str


class DatasetResponse(BaseModel):
    data: List[IrisFeatures]
    labels: List[str]
    total: int


class PredictionRequest(BaseModel):
    instances: List[IrisFeatures]


class PredictionResponse(BaseModel):
    predictions: List[int]
    class_names: List[str]


class EvaluationRequest(BaseModel):
    instances: List[IrisFeatures]
    labels: List[int] = Field(
        ..., description="True class labels (0=setosa, 1=versicolor, 2=virginica)"
    )


class EvaluationResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int


# ── Versioned Router ─────────────────────────────────────────────────

v1 = APIRouter(prefix="/v1")


# ── 1. Health ────────────────────────────────────────────────────────


@v1.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Check API health and model loading status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_uri=loaded_model_uri,
        version="0.1.0",
    )


# ── 2. Model Discovery ──────────────────────────────────────────────


@v1.get(
    "/model/discover",
    response_model=ModelDiscoverResponse,
    tags=["Model"],
)
def discover_models() -> ModelDiscoverResponse:
    """Auto-discover available MLflow model URIs from registry and local runs."""
    from iris.config import EXPERIMENT_NAME

    logger.info(f"Setting tracking URI to: {TRACKING_URI}")
    configure_mlflow()

    try:
        client = mlflow.tracking.MlflowClient()
        models = []

        if hasattr(client, "search_model_versions"):
            model_versions = client.search_model_versions(
                filter_string=f"name = '{REGISTERED_MODEL_NAME}'",
                max_results=10,
                order_by=["last_updated_timestamp DESC"],
            )
            for model_version in model_versions:
                metrics = {}
                start_time = None
                run_status = None
                if model_version.run_id:
                    run = client.get_run(model_version.run_id)
                    metrics = dict(run.data.metrics)
                    start_time = run.info.start_time
                    run_status = run.info.status

                models.append(
                    {
                        "run_id": model_version.run_id,
                        "registered_model_name": model_version.name,
                        "registered_model_version": model_version.version,
                        "model_uri": (
                            f"models:/{model_version.name}/{model_version.version}"
                        ),
                        "status": run_status or model_version.status,
                        "start_time": start_time,
                        "metrics": metrics,
                    }
                )

        if models:
            return ModelDiscoverResponse(available_models=models)

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return ModelDiscoverResponse(available_models=[])

        if hasattr(client, "search_logged_models"):
            logged_models = client.search_logged_models(
                experiment_ids=[experiment.experiment_id],
                max_results=10,
            )
            for logged_model in logged_models:
                metrics = {}
                start_time = None
                run_status = None
                if logged_model.source_run_id:
                    run = client.get_run(logged_model.source_run_id)
                    metrics = dict(run.data.metrics)
                    start_time = run.info.start_time
                    run_status = run.info.status

                model_status = getattr(
                    logged_model.status,
                    "value",
                    str(logged_model.status),
                )
                models.append(
                    {
                        "run_id": logged_model.source_run_id,
                        "model_id": logged_model.model_id,
                        "model_name": logged_model.name,
                        "model_uri": f"models:/{logged_model.model_id}",
                        "status": run_status or model_status,
                        "start_time": start_time,
                        "metrics": metrics,
                    }
                )

        if models:
            return ModelDiscoverResponse(available_models=models)

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10,
        )

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


# ── 3. Model Loading ────────────────────────────────────────────────


@v1.post(
    "/model/load",
    response_model=ModelLoadResponse,
    tags=["Model"],
)
def load_model_endpoint(request: ModelLoadRequest) -> ModelLoadResponse:
    """Dynamically load an MLflow model into memory."""
    global loaded_model_uri, model
    logger.info(f"Attempting to load model from {request.model_uri}...")
    try:
        logger.info(f"Setting tracking URI to: {TRACKING_URI}")
        configure_mlflow()
        model = mlflow.pyfunc.load_model(request.model_uri)
        loaded_model_uri = request.model_uri
        msg = f"Model loaded successfully from {request.model_uri}"
        logger.info(msg)
        return ModelLoadResponse(message=msg, model_uri=request.model_uri)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {e}",
        )


# ── 4. Dataset ───────────────────────────────────────────────────────


@v1.get("/data", response_model=DatasetResponse, tags=["Data"])
def get_data(limit: int | None = None) -> DatasetResponse:
    """Get the full Iris dataset.

    Set `limit` to cap the number of rows returned.
    Omit `limit` to return all 150 samples.
    """
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


# ── 5. Prediction ───────────────────────────────────────────────────


@v1.post(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["Inference"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict the class of iris flowers using the loaded model."""
    global model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Use POST /v1/model/load first.",
        )

    try:
        import pandas as pd

        data = [
            {
                "sepal length (cm)": inst.sepal_length,
                "sepal width (cm)": inst.sepal_width,
                "petal length (cm)": inst.petal_length,
                "petal width (cm)": inst.petal_width,
            }
            for inst in request.instances
        ]
        df = pd.DataFrame(data)

        preds = model.predict(df)
        pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        names = [CLASS_MAPPING.get(p, "unknown") for p in pred_list]

        return PredictionResponse(predictions=pred_list, class_names=names)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Evaluation ───────────────────────────────────────────────────


@v1.post(
    "/evaluate",
    response_model=EvaluationResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["Inference"],
)
def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate the loaded model against labeled test data.

    Provide instances with their true labels and receive
    accuracy, precision, recall, and F1 score.
    """
    global model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Use POST /v1/model/load first.",
        )

    if len(request.instances) != len(request.labels):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Mismatch: {len(request.instances)} instances "
                f"but {len(request.labels)} labels."
            ),
        )

    try:
        import pandas as pd

        data = [
            {
                "sepal length (cm)": inst.sepal_length,
                "sepal width (cm)": inst.sepal_width,
                "petal length (cm)": inst.petal_length,
                "petal width (cm)": inst.petal_width,
            }
            for inst in request.instances
        ]
        df = pd.DataFrame(data)

        preds = model.predict(df)
        pred_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        y_true = request.labels
        return EvaluationResponse(
            accuracy=accuracy_score(y_true, pred_list),
            precision=precision_score(
                y_true, pred_list, average="weighted", zero_division=0
            ),
            recall=recall_score(y_true, pred_list, average="weighted", zero_division=0),
            f1_score=f1_score(y_true, pred_list, average="weighted", zero_division=0),
            total_samples=len(y_true),
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Mount router ─────────────────────────────────────────────────────

app.include_router(v1)


# ── CLI Entrypoint ───────────────────────────────────────────────────


def start() -> None:
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
