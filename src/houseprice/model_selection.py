"""K-fold cross-validation and a simple grid search."""

from __future__ import annotations

import itertools

import numpy as np

from .data_io import Table
from .metrics import report
from .preprocessing import Preprocessor, target_vector
from .models import LinearRegression, Ridge


def kfold_indices(n: int, k: int = 5, seed: int | None = None):
    """Yield (train_idx, val_idx) for k shuffled folds of roughly equal size."""
    if k < 2 or k > n:
        raise ValueError(f"k must be between 2 and the number of rows ({n})")
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        yield train, val


def cross_validate(table: Table, target: str, features: list[str], model: str = "linear",
                   alpha: float = 1.0, degree: int = 1, k: int = 5, seed: int | None = 0) -> dict:
    """Return the mean and standard deviation of R2, RMSE and MAE over k folds."""
    pre = Preprocessor(features, degree)
    X = pre.fit_transform(table)
    y = target_vector(table, target)

    scores = {"r2": [], "rmse": [], "mae": []}
    for train, val in kfold_indices(len(table), k, seed):
        reg = Ridge(alpha=alpha) if model == "ridge" else LinearRegression()
        reg.fit(X[train], y[train])
        for name, value in report(y[val], reg.predict(X[val])).items():
            scores[name].append(value)

    return {name: {"mean": float(np.mean(v)), "std": float(np.std(v))} for name, v in scores.items()}


def grid_search(table: Table, target: str, features: list[str], alphas: list[float],
                degrees: list[int], k: int = 5, seed: int | None = 0) -> list[dict]:
    """Cross-validate every (alpha, degree) pair with Ridge. Best (highest mean R2) first."""
    results = []
    for alpha, degree in itertools.product(alphas, degrees):
        cv = cross_validate(table, target, features, "ridge", alpha, degree, k, seed)
        results.append({"alpha": alpha, "degree": degree, **{f"{m}_mean": s["mean"] for m, s in cv.items()},
                        **{f"{m}_std": s["std"] for m, s in cv.items()}})
    results.sort(key=lambda r: r["r2_mean"], reverse=True)
    return results
