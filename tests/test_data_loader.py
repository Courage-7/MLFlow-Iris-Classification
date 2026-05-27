"""Tests for the data loader module."""

import numpy as np

from iris.data.loader import load_data


def test_load_data_returns_four_arrays() -> None:
    """load_data() should return exactly four numpy arrays."""
    result = load_data()
    assert len(result) == 4
    for arr in result:
        assert isinstance(arr, np.ndarray)


def test_load_data_shapes_match() -> None:
    """Training and test arrays should have consistent dimensions."""
    X_train, X_test, y_train, y_test = load_data()

    # Feature arrays should have 4 columns (iris features)
    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4

    # Label arrays should match their corresponding feature arrays
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_load_data_test_size_proportion() -> None:
    """Test split should be approximately 20% of the full dataset."""
    X_train, X_test, _, _ = load_data()
    total = X_train.shape[0] + X_test.shape[0]

    # Iris dataset has 150 samples; 20% = 30 test samples
    assert total == 150
    assert X_test.shape[0] == 30
    assert X_train.shape[0] == 120
