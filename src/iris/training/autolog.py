# autolog.py — Train with MLflow autologging

import mlflow
from sklearn.linear_model import LogisticRegression

from iris.config import (
    EXPERIMENT_NAME,
    PARAMS,
    REGISTERED_MODEL_NAME,
    TRACKING_URI,
    configure_mlflow,
)
from iris.data.loader import load_data
from iris.utils.logging import setup_logging

logger = setup_logging(__name__)


def train_with_autolog() -> LogisticRegression:
    """Train model using MLflow autologging with skops serialization.

    Returns:
        The trained LogisticRegression model.
    """
    logger.info("Starting autolog training...")

    logger.info(f"Setting tracking URI to: {TRACKING_URI}")
    logger.info(f"Setting experiment: {EXPERIMENT_NAME}")
    configure_mlflow()

    logger.info(
        "Enabling MLflow autologging with skops format "
        f"and registry name {REGISTERED_MODEL_NAME}..."
    )
    mlflow.sklearn.autolog(
        serialization_format="skops",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    logger.info("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    logger.info(f"Training LogisticRegression with params: {PARAMS}")
    lr = LogisticRegression(**PARAMS)
    lr.fit(X_train, y_train)

    logger.info("Training complete!")
    logger.info(f"Registered model name: {REGISTERED_MODEL_NAME}")
    logger.info("Check the MLflow server for run details: mlflow server")
    return lr


if __name__ == "__main__":
    train_with_autolog()
