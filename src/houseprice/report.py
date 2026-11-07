"""Excel report for a trained pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_io import Table, write_excel
from .pipeline import Pipeline
from .preprocessing import target_vector


def write_report(path: str | Path, pipe: Pipeline, test: Table) -> None:
    """Write a workbook with three sheets: metrics, coefficients, test_predictions."""
    m = pipe.metrics
    metric_rows = [[split, m[split]["r2"], m[split]["rmse"], m[split]["mae"]] for split in ("train", "test")]

    coef_rows = [[name, round(scaled, 4), round(original, 4)] for name, scaled, original in pipe.coefficients()]
    coef_rows.append(["(intercept)", round(pipe.model.intercept_, 4), None])

    y = target_vector(test, pipe.target)
    pred = pipe.predict(test)
    pred_rows = [row + [round(float(p), 2), round(float(p - t), 2)]
                 for row, p, t in zip(test.rows, pred, y)]

    write_excel(path, {
        "metrics": (["split", "r2", "rmse", "mae"], metric_rows),
        "coefficients": (["column", "coef_scaled", "coef_original_units"], coef_rows),
        "test_predictions": (test.columns + ["predicted", "error"], pred_rows),
    })
