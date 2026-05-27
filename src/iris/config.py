# config.py — Shared configuration for Iris classification

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# MLflow experiment name
EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "MLflow Quickstart")

# MLflow Model Registry name
REGISTERED_MODEL_NAME: str = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "iris-logistic-regression"
)

# MLflow tracking configuration
TRACKING_URI: str = (
    os.getenv("MLFLOW_TRACKING_URI", "").strip() or "sqlite:///mlflow.db"
)
ARTIFACT_ROOT: str = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT", "").strip() or "./mlruns"


def _sqlite_db_path(tracking_uri: str) -> Path | None:
    """Return the local SQLite DB path for sqlite:/// URIs."""
    if not tracking_uri.startswith("sqlite:///"):
        return None

    db_path = tracking_uri.removeprefix("sqlite:///")
    if db_path == ":memory:":
        return None

    return Path(db_path)


def _ensure_relative_artifact_root() -> None:
    """Keep local SQLite experiment artifact roots container-portable."""
    if Path(ARTIFACT_ROOT).is_absolute() or "://" in ARTIFACT_ROOT:
        return

    db_path = _sqlite_db_path(TRACKING_URI)
    if db_path is None or not db_path.exists():
        return

    # MLflow expands relative artifact locations for direct SQLite stores.
    # Store the configured relative root so runs stay portable across mounts.
    default_artifact_root = f"{ARTIFACT_ROOT.rstrip('/')}/0"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
            (default_artifact_root, 0),
        )
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE name = ?",
            (ARTIFACT_ROOT, EXPERIMENT_NAME),
        )


def configure_mlflow() -> None:
    """Configure MLflow tracking and ensure the experiment exists."""
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACT_ROOT,
        )

    _ensure_relative_artifact_root()
    mlflow.set_experiment(EXPERIMENT_NAME)


# Model hyperparameters
PARAMS: dict = {
    "solver": os.getenv("LR_SOLVER", "lbfgs"),
    "max_iter": int(os.getenv("LR_MAX_ITER", "1000")),
    "random_state": int(os.getenv("LR_RANDOM_STATE", "8888")),
}

# Train/test split settings
TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "text")
