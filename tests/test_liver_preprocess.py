import numpy as np
import pytest

from src.liver.preprocess import DATA_PATH, NO_LAB_COLUMNS, load_and_split

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="Liver CSV not available")

EXPECTED = {"full": 10, "no_lab": 9}


@pytest.mark.parametrize("feature_set", list(EXPECTED))
def test_shapes(feature_set):
    X_train, X_test, y_train, y_test, _ = load_and_split(feature_set)
    n_features = EXPECTED[feature_set]
    assert X_train.shape == (1360, n_features)
    assert X_test.shape == (340, n_features)
    assert y_train.shape == (1360,)
    assert y_test.shape == (340,)
    assert list(X_train.columns) == (
        NO_LAB_COLUMNS if feature_set == "no_lab" else NO_LAB_COLUMNS + ["LiverFunctionTest"]
    )


@pytest.mark.parametrize("feature_set", list(EXPECTED))
def test_stratified_positive_rates(feature_set):
    X_train, X_test, y_train, y_test, _ = load_and_split(feature_set)
    assert abs(y_train.mean() - y_test.mean()) <= 0.01
    assert X_train.mean().abs().max() < 1e-9


def test_scaler_fitted_on_train_only():
    X_train, X_test, _, _, scaler = load_and_split("full")
    assert scaler.n_samples_seen_ == len(X_train)
    assert np.allclose(scaler.inverse_transform(X_train).mean(axis=0), scaler.mean_)
    assert not np.allclose(scaler.inverse_transform(X_test).mean(axis=0), scaler.mean_)
