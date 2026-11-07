import numpy as np
import pytest

from houseprice.data_io import Table, TableError
from houseprice.preprocessing import Preprocessor, target_vector, train_test_indices


def test_detects_numeric_and_categorical(table):
    pre = Preprocessor(["x1", "x2", "city"]).fit(table)
    assert pre.numeric == ["x1", "x2"]
    assert pre.categories == {"city": ["A", "B", "C"]}
    assert pre.output_names() == ["x1", "x2", "city=B", "city=C"]


def test_transformed_columns_are_standardized(table):
    X = Preprocessor(["x1", "x2", "city"]).fit_transform(table)
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-12)


def test_polynomial_columns(table):
    pre = Preprocessor(["x1", "x2"], degree=3).fit(table)
    assert pre.output_names() == ["x1", "x1^2", "x1^3", "x2", "x2^2", "x2^3"]
    assert pre.transform(table).shape == (len(table), 6)


def test_unknown_category_at_predict_time(table):
    pre = Preprocessor(["city"]).fit(table)
    new = Table(["city"], [["Z"]])
    with pytest.raises(TableError, match="unknown categories"):
        pre.transform(new)


def test_roundtrip_dict(table):
    pre = Preprocessor(["x1", "city"], degree=2).fit(table)
    again = Preprocessor.from_dict(pre.to_dict())
    np.testing.assert_allclose(pre.transform(table), again.transform(table))


def test_target_vector_rejects_missing():
    t = Table(["y"], [[1.0], [None]])
    with pytest.raises(TableError):
        target_vector(t, "y")


def test_train_test_indices():
    train, test = train_test_indices(50, 0.2, seed=0)
    assert len(test) == 10 and len(train) == 40
    assert set(train).isdisjoint(test)
    assert sorted(np.concatenate([train, test])) == list(range(50))
    with pytest.raises(ValueError):
        train_test_indices(50, 1.5)
