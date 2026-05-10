# loader.py — Load and split the Iris dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from iris.config import RANDOM_STATE, TEST_SIZE
from iris.utils.logging import setup_logging

logger = setup_logging(__name__)


def load_data():
    """Load the Iris dataset and return train/test splits."""
    logger.info("Loading Iris dataset...")
    X, y = load_iris(return_X_y=True)
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    logger.info(
        f"Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE})..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        f"Data split complete: {X_train.shape[0]} training samples, "
        f"{X_test.shape[0]} test samples"
    )
    return X_train, X_test, y_train, y_test
