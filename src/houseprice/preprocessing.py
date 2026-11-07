"""Turning a Table into a numeric matrix the models can use.

The Preprocessor:
1. detects which feature columns are numeric and which are categorical (text),
2. one-hot encodes categorical columns (the first category is dropped and
   serves as the reference level),
3. optionally adds polynomial terms (squares, cubes...) of numeric columns,
4. standardizes every resulting column to mean 0 and standard deviation 1.

It is fitted on training data only, saved with the model, and reapplied
unchanged at prediction time.
"""

from __future__ import annotations

import numpy as np

from .data_io import Table, TableError


class Preprocessor:
    def __init__(self, features: list[str], degree: int = 1):
        if not features:
            raise ValueError("at least one feature is required")
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.features = list(features)
        self.degree = degree
        self.numeric: list[str] = []
        self.categories: dict[str, list[str]] = {}
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ---- fitting --------------------------------------------------------
    def fit(self, table: Table) -> "Preprocessor":
        self.numeric, self.categories = [], {}
        for name in self.features:
            values = table.column(name)
            if any(isinstance(v, str) for v in values):
                levels = sorted({v for v in values if isinstance(v, str)})
                if len(levels) < 2:
                    raise TableError(f"categorical column {name!r} needs at least two distinct values")
                self.categories[name] = levels
            else:
                self.numeric.append(name)
        raw = self._raw_matrix(table)
        self.mean_ = raw.mean(axis=0)
        std = raw.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, table: Table) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return (self._raw_matrix(table) - self.mean_) / self.std_

    def fit_transform(self, table: Table) -> np.ndarray:
        return self.fit(table).transform(table)

    # ---- building columns -----------------------------------------------
    def output_names(self) -> list[str]:
        names = []
        for name in self.numeric:
            names.append(name)
            names += [f"{name}^{p}" for p in range(2, self.degree + 1)]
        for name, levels in self.categories.items():
            names += [f"{name}={lvl}" for lvl in levels[1:]]
        return names

    def _numeric_column(self, table: Table, name: str) -> np.ndarray:
        out = []
        for i, v in enumerate(table.column(name)):
            if v is None:
                out.append(0.0)
            elif isinstance(v, str):
                raise TableError(f"column {name!r}, row {i + 2}: {v!r} is not a number")
            else:
                out.append(float(v))
        return np.array(out)

    def _raw_matrix(self, table: Table) -> np.ndarray:
        cols = []
        for name in self.numeric:
            x = self._numeric_column(table, name)
            cols += [x ** p for p in range(1, self.degree + 1)]
        for name, levels in self.categories.items():
            values = table.column(name)
            unknown = sorted({v for v in values if v is not None and v not in levels})
            if unknown:
                raise TableError(f"column {name!r}: unknown categories {unknown} (known: {levels})")
            for lvl in levels[1:]:
                cols.append(np.array([1.0 if v == lvl else 0.0 for v in values]))
        return np.column_stack(cols)

    # ---- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"features": self.features, "degree": self.degree, "numeric": self.numeric,
                "categories": self.categories, "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        p = cls(d["features"], d["degree"])
        p.numeric = list(d["numeric"])
        p.categories = {k: list(v) for k, v in d["categories"].items()}
        p.mean_ = np.array(d["mean"], dtype=float)
        p.std_ = np.array(d["std"], dtype=float)
        return p


def target_vector(table: Table, target: str) -> np.ndarray:
    values = table.column(target)
    bad = [i + 2 for i, v in enumerate(values) if not isinstance(v, float)]
    if bad:
        raise TableError(f"target column {target!r} has missing or non-numeric values on rows {bad[:10]}")
    return np.array(values, dtype=float)


def train_test_indices(n: int, test_size: float = 0.2, seed: int | None = None):
    """Shuffled row indices for a train/test split."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    n_test = int(round(n * test_size))
    if n_test == 0 or n_test == n:
        raise ValueError("test_size leaves an empty train or test set")
    idx = np.random.default_rng(seed).permutation(n)
    return idx[n_test:], idx[:n_test]
