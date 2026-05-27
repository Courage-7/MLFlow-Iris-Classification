# manual.py — Train with manual MLflow logging

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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


def train_with_manual_logging() -> None:
    """Train model with explicit MLflow logging and skops serialization."""
    logger.info("Starting manual training with MLflow logging...")

    logger.info(f"Setting tracking URI to: {TRACKING_URI}")
    logger.info(f"Setting experiment: {EXPERIMENT_NAME}")
    configure_mlflow()

    logger.info("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        logger.info(f"Logging hyperparameters: {PARAMS}")
        mlflow.log_params(PARAMS)

        logger.info("Training LogisticRegression...")
        lr = LogisticRegression(**PARAMS)
        lr.fit(X_train, y_train)
        logger.info("Training complete!")

        logger.info(
            "Logging and registering model with skops format "
            f"as {REGISTERED_MODEL_NAME}..."
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            name="iris_model",
            serialization_format="skops",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        logger.info(f"Model logged: {model_info.model_uri}")
        if model_info.registered_model_version is not None:
            logger.info(
                "Registered model URI: "
                f"models:/{REGISTERED_MODEL_NAME}/{model_info.registered_model_version}"
            )

        logger.info("Evaluating model on test set...")
        y_pred = lr.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }
        logger.info(f"Logging metrics: {metrics}")
        mlflow.log_metrics(metrics)

        mlflow.set_tag("Training Info", "Basic LR model for iris data")
        logger.info("Run tagged")

        logger.info("\n=== RUN SUMMARY ===")
        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Model URI: {model_info.model_uri}")
        if model_info.registered_model_version is not None:
            logger.info(
                "Registered Model URI: "
                f"models:/{REGISTERED_MODEL_NAME}/{model_info.registered_model_version}"
            )
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    train_with_manual_logging()
