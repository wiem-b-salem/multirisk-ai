import numpy as np
import pytest

from src.alzheimers.preprocess import DATA_PATH, load_and_split

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="Alzheimer's CSV not available")

EXPECTED = {
    "full": (1719, 34),
    "risk_only": (1719, 24),
    "compact": (1719, 12),
}


@pytest.mark.parametrize("feature_set", list(EXPECTED))
def test_shapes(feature_set):
    X_train, X_test, y_train, y_test, _ = load_and_split(feature_set)
    n_features = EXPECTED[feature_set][1]
    assert X_train.shape == (1719, n_features)
    assert X_test.shape == (430, n_features)
    assert y_train.shape == (1719,)
    assert y_test.shape == (430,)


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
