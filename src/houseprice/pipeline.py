"""A Pipeline bundles a Preprocessor and a regression model.

It is what gets trained, saved to JSON, and loaded again to predict.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_io import Table
from .models import LinearRegression, Ridge
from .preprocessing import Preprocessor, target_vector


class Pipeline:
    def __init__(self, target: str, features: list[str], model: str = "linear",
                 alpha: float = 1.0, degree: int = 1, method: str = "normal"):
        self.target = target
        self.preprocessor = Preprocessor(features, degree)
        if model == "linear":
            self.model = LinearRegression(method=method)
        elif model == "ridge":
            self.model = Ridge(alpha=alpha, method=method)
        else:
            raise ValueError("model must be 'linear' or 'ridge'")
        self.metrics: dict = {}

    @property
    def features(self) -> list[str]:
        return self.preprocessor.features

    def fit(self, table: Table) -> "Pipeline":
        X = self.preprocessor.fit_transform(table)
        self.model.fit(X, target_vector(table, self.target))
        return self

    def predict(self, table: Table) -> np.ndarray:
        return self.model.predict(self.preprocessor.transform(table))

    def coefficients(self) -> list[tuple[str, float, float]]:
        """(column, coefficient on scaled data, coefficient in original units).

        The original-unit coefficient is the change in the target for +1 unit
        of that column (for a one-hot column: being in that category rather
        than the reference category).
        """
        names = self.preprocessor.output_names()
        coefs = self.model.coef_
        return [(n, float(c), float(c / s)) for n, c, s in zip(names, coefs, self.preprocessor.std_)]

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "target": self.target,
            "preprocessor": self.preprocessor.to_dict(),
            "model": self.model.to_dict(),
            "metrics": self.metrics,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        d = json.loads(Path(path).read_text())
        pipe = cls.__new__(cls)
        pipe.target = d["target"]
        pipe.preprocessor = Preprocessor.from_dict(d["preprocessor"])
        pipe.model = LinearRegression.from_dict(d["model"])
        pipe.metrics = d.get("metrics", {})
        return pipe
