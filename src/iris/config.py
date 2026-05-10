# config.py — Shared configuration for Iris classification

# MLflow experiment name
EXPERIMENT_NAME = "MLflow Quickstart"

# Model hyperparameters
PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

# Train/test split settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# MLflow tracking URI (local by default)
# Change to a remote URI if using a tracking server
TRACKING_URI = None  # e.g. "http://localhost:5000"
