import numpy as np
import pytest

from houseprice.metrics import mae, mse, r2, rmse
from houseprice.models import LinearRegression, Ridge


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 4.0
    return X, y


def test_normal_equation_recovers_coefficients(linear_data):
    X, y = linear_data
    m = LinearRegression().fit(X, y)
    np.testing.assert_allclose(m.coef_, [2.0, -1.0, 0.5], atol=1e-10)
    assert m.intercept_ == pytest.approx(4.0)


def test_gradient_descent_matches_normal_equation(linear_data):
    X, y = linear_data
    exact = LinearRegression().fit(X, y)
    gd = LinearRegression(method="gd", learning_rate=0.1, epochs=5000).fit(X, y)
    np.testing.assert_allclose(gd.coef_, exact.coef_, atol=1e-4)
    assert gd.loss_history_[-1] < gd.loss_history_[0]


def test_ridge_shrinks_coefficients(linear_data):
    X, y = linear_data
    plain = np.abs(LinearRegression().fit(X, y).coef_).sum()
    shrunk = np.abs(Ridge(alpha=100).fit(X, y).coef_).sum()
    assert shrunk < plain


def test_ridge_gd_matches_normal(linear_data):
    X, y = linear_data
    exact = Ridge(alpha=10).fit(X, y)
    gd = Ridge(alpha=10, method="gd", learning_rate=0.1, epochs=5000).fit(X, y)
    np.testing.assert_allclose(gd.coef_, exact.coef_, atol=1e-4)


def test_model_roundtrip(linear_data):
    X, y = linear_data
    m = Ridge(alpha=3).fit(X, y)
    again = LinearRegression.from_dict(m.to_dict())
    assert isinstance(again, Ridge) and again.alpha == 3
    np.testing.assert_allclose(again.predict(X), m.predict(X))


def test_errors():
    with pytest.raises(RuntimeError):
        LinearRegression().predict(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        Ridge(alpha=-1)
    with pytest.raises(ValueError):
        LinearRegression(method="magic")


def test_metrics():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])
    assert mse(y, p) == pytest.approx(4 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(4 / 3))
    assert mae(y, p) == pytest.approx(2 / 3)
    assert r2(y, y) == 1.0
    assert r2(y, np.full(3, 2.0)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        mse(y, p[:2])
