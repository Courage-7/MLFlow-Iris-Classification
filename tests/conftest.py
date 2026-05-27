"""Shared test fixtures."""

import pytest
from fastapi.testclient import TestClient

from iris.api.app import app


@pytest.fixture()
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture()
def sample_features() -> dict:
    """A valid set of Iris features (setosa)."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
