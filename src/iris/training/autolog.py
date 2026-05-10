# autolog.py — Train with MLflow autologging

import mlflow
from sklearn.linear_model import LogisticRegression

from iris.config import EXPERIMENT_NAME, PARAMS, TRACKING_URI
from iris.data.loader import load_data
from iris.utils.logging import setup_logging

logger = setup_logging(__name__)


def train_with_autolog():
    """Train model using MLflow autologging with skops serialization."""
    logger.info("Starting autolog training...")

    if TRACKING_URI:
        logger.info(f"Setting tracking URI to: {TRACKING_URI}")
        mlflow.set_tracking_uri(TRACKING_URI)
    else:
        logger.info("Using local tracking (mlruns/)")

    logger.info(f"Setting experiment: {EXPERIMENT_NAME}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Enabling MLflow autologging with skops format...")
    mlflow.sklearn.autolog(serialization_format="skops")

    logger.info("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    logger.info(f"Training LogisticRegression with params: {PARAMS}")
    lr = LogisticRegression(**PARAMS)
    lr.fit(X_train, y_train)

    logger.info("Training complete!")
    logger.info("Check the MLflow UI for run details: mlflow ui")
    return lr


if __name__ == "__main__":
    train_with_autolog()
