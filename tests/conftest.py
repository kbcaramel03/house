import numpy as np
import pytest

from houseprice.data_io import Table


def make_table(n=200, seed=0, with_city=True):
    """Clean synthetic table: price = 1000 + 50*x1 - 20*x2 (+ city bonus) + small noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 100, n)
    x2 = rng.uniform(0, 50, n)
    cities = rng.choice(["A", "B", "C"], n)
    bonus = {"A": 0.0, "B": 300.0, "C": -200.0}
    y = 1000 + 50 * x1 - 20 * x2 + rng.normal(0, 1, n)
    columns = ["x1", "x2", "price"]
    rows = []
    for i in range(n):
        if with_city:
            rows.append([float(x1[i]), float(x2[i]), str(cities[i]), float(y[i] + bonus[cities[i]])])
        else:
            rows.append([float(x1[i]), float(x2[i]), float(y[i])])
    if with_city:
        columns = ["x1", "x2", "city", "price"]
    return Table(columns, rows)


@pytest.fixture
def table():
    return make_table()
