"""Linear models: y = X @ coef + intercept.

Two ways to find the coefficients:
- "normal": solve the least-squares problem exactly (np.linalg.lstsq)
- "gd":     batch gradient descent on the mean squared error

Ridge adds an L2 penalty alpha * ||coef||^2 (the intercept is not penalized).
"""

from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, method: str = "normal", learning_rate: float = 0.01,
                 epochs: int = 1000, tol: float = 1e-9):
        if method not in ("normal", "gd"):
            raise ValueError("method must be 'normal' or 'gd'")
        if learning_rate <= 0 or epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    alpha = 0.0  # no regularization for plain linear regression

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D, y 1D, with the same number of rows")
        if self.method == "normal":
            self._fit_normal(X, y)
        else:
            self._fit_gd(X, y)
        return self

    def _fit_normal(self, X, y):
        n, p = X.shape
        Xb = np.column_stack([np.ones(n), X])
        if self.alpha > 0:
            # Ridge: append sqrt(alpha) * I rows (skipping the intercept column)
            penalty = np.sqrt(self.alpha) * np.eye(p + 1)[1:]
            Xb = np.vstack([Xb, penalty])
            y = np.concatenate([y, np.zeros(p)])
        theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]

    def _fit_gd(self, X, y):
        n, p = X.shape
        self.coef_ = np.zeros(p)
        self.intercept_ = float(y.mean())
        self.loss_history_ = []
        prev = np.inf
        for _ in range(self.epochs):
            err = X @ self.coef_ + self.intercept_ - y
            loss = float(np.mean(err ** 2) + self.alpha * np.sum(self.coef_ ** 2) / n)
            self.loss_history_.append(loss)
            if abs(prev - loss) < self.tol * max(1.0, abs(prev)):
                break
            prev = loss
            grad_coef = 2 * X.T @ err / n + 2 * self.alpha * self.coef_ / n
            grad_intercept = 2 * err.mean()
            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model must be fitted before predict")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(f"expected {self.coef_.shape[0]} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_

    def to_dict(self) -> dict:
        return {"type": type(self).__name__, "alpha": self.alpha, "method": self.method,
                "coef": self.coef_.tolist(), "intercept": self.intercept_}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearRegression":
        model = Ridge(alpha=d["alpha"]) if d["type"] == "Ridge" else LinearRegression()
        model.method = d.get("method", "normal")
        model.coef_ = np.array(d["coef"], dtype=float)
        model.intercept_ = float(d["intercept"])
        return model


class Ridge(LinearRegression):
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = alpha
