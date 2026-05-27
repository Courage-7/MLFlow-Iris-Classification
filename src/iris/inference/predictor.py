# predictor.py — Load model and run inference with scoring

import mlflow.pyfunc
import pandas as pd
from sklearn import datasets

from iris.config import TRACKING_URI, configure_mlflow
from iris.data.loader import load_data
from iris.utils.logging import setup_logging

logger = setup_logging(__name__)


def run_inference(model_uri: str) -> pd.DataFrame:
    """Load model and run predictions with accuracy scoring.

    Args:
        model_uri: MLflow model URI to load.

    Returns:
        DataFrame with features, actual classes, and predicted classes.
    """
    logger.info("Starting inference...")

    logger.info(f"Setting tracking URI to: {TRACKING_URI}")
    configure_mlflow()

    logger.info(f"Loading model from: {model_uri}")
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully!")

    logger.info("Loading test data...")
    _, X_test, _, y_test = load_data()

    logger.info(f"Running predictions on {len(X_test)} test samples...")
    predictions = loaded_model.predict(X_test)
    logger.info("Predictions complete!")

    # Build results dataframe
    feature_names = datasets.load_iris().feature_names
    result = pd.DataFrame(X_test, columns=feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    logger.info("\n=== SAMPLE PREDICTIONS (first 10 rows) ===")
    logger.info("\n" + result.head(10).to_string(index=False))
    logger.info(f"\nTotal predictions: {len(result)}")

    # Calculate accuracy
    accuracy = (result["actual_class"] == result["predicted_class"]).mean()
    logger.info(f"Accuracy: {accuracy:.4f}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference with a logged MLflow model."
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI, e.g. models:/iris-logistic-regression/1",
    )
    args = parser.parse_args()
    run_inference(args.model_uri)
