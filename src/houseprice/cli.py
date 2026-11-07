"""Command-line interface.

    houseprice train data/houses.xlsx --target price --out model.json --report report.xlsx
    houseprice predict model.json data/new_houses.csv --out predictions.xlsx
    houseprice cv data/houses.csv --target price --model ridge --alpha 10
    houseprice tune data/houses.csv --target price --alphas 0.1 1 10 --degrees 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data_io import TableError, read_table, write_csv, write_excel
from .metrics import report
from .model_selection import cross_validate, grid_search
from .pipeline import Pipeline
from .preprocessing import target_vector, train_test_indices
from .report import write_report


def _features(table, target, explicit):
    if explicit:
        return explicit
    return [c for c in table.columns if c != target]


def train(path, target: str, features: list[str] | None = None, model: str = "linear",
          alpha: float = 1.0, degree: int = 1, method: str = "normal",
          test_size: float = 0.2, seed: int = 0, sheet: str | None = None):
    """Train on a split of the file. Returns (pipeline, test_table)."""
    table = read_table(path, sheet)
    feats = _features(table, target, features)
    train_idx, test_idx = train_test_indices(len(table), test_size, seed)
    train_t, test_t = table.take(train_idx), table.take(test_idx)

    pipe = Pipeline(target, feats, model, alpha, degree, method).fit(train_t)
    pipe.metrics = {
        "train": report(target_vector(train_t, target), pipe.predict(train_t)),
        "test": report(target_vector(test_t, target), pipe.predict(test_t)),
        "n_train": len(train_t),
        "n_test": len(test_t),
    }
    return pipe, test_t


def _print_metrics(metrics):
    print(f"{'':8}{'R2':>8}{'RMSE':>14}{'MAE':>14}")
    for split in ("train", "test"):
        r = metrics[split]
        print(f"{split:8}{r['r2']:>8.3f}{r['rmse']:>14,.1f}{r['mae']:>14,.1f}")


def _cmd_train(args) -> int:
    pipe, test_t = train(args.data, args.target, args.features, args.model, args.alpha,
                         args.degree, args.method, args.test_size, args.seed, args.sheet)
    pipe.save(args.out)
    m = pipe.metrics
    print(f"Trained on {m['n_train']} rows, tested on {m['n_test']} rows\n")
    _print_metrics(m)
    print(f"\nEffect of +1 unit on {pipe.target}:")
    for name, _, original in pipe.coefficients():
        print(f"  {name:<28}{original:>+14,.2f}")
    print(f"\nModel saved to {args.out}")
    if args.report:
        write_report(args.report, pipe, test_t)
        print(f"Report written to {args.report}")
    return 0


def _cmd_predict(args) -> int:
    pipe = Pipeline.load(args.model)
    table = read_table(args.data, args.sheet)
    preds = pipe.predict(table)
    columns = table.columns + [f"predicted_{pipe.target}"]
    rows = [row + [round(float(p), 2)] for row, p in zip(table.rows, preds)]
    out = Path(args.out)
    if out.suffix.lower() == ".csv":
        write_csv(out, columns, rows)
    else:
        write_excel(out, {"predictions": (columns, rows)})
    print(f"Wrote {len(preds)} predictions to {out}")
    return 0


def _cmd_cv(args) -> int:
    table = read_table(args.data, args.sheet)
    feats = _features(table, args.target, args.features)
    res = cross_validate(table, args.target, feats, args.model, args.alpha, args.degree, args.k, args.seed)
    print(f"{args.k}-fold cross-validation ({args.model}, alpha={args.alpha}, degree={args.degree})")
    for name, s in res.items():
        print(f"  {name:<6}{s['mean']:>14,.3f}  +/- {s['std']:,.3f}")
    return 0


def _cmd_tune(args) -> int:
    table = read_table(args.data, args.sheet)
    feats = _features(table, args.target, args.features)
    results = grid_search(table, args.target, feats, args.alphas, args.degrees, args.k, args.seed)
    print(f"{'alpha':>10}{'degree':>8}{'R2 mean':>10}{'R2 std':>9}{'RMSE mean':>14}")
    for r in results:
        print(f"{r['alpha']:>10g}{r['degree']:>8}{r['r2_mean']:>10.3f}{r['r2_std']:>9.3f}{r['rmse_mean']:>14,.1f}")
    best = results[0]
    print(f"\nBest: alpha={best['alpha']:g}, degree={best['degree']}")
    if args.out:
        cols = list(results[0].keys())
        write_excel(args.out, {"grid_search": (cols, [[r[c] for c in cols] for r in results])})
        print(f"Results written to {args.out}")
    return 0


def _add_common(p, with_model=True):
    p.add_argument("data", help="Path to a .xlsx or .csv file")
    p.add_argument("--target", required=True, help="Name of the column to predict")
    p.add_argument("--features", nargs="+", help="Columns to use (default: all except the target)")
    p.add_argument("--sheet", help="Excel sheet name (default: first sheet)")
    p.add_argument("--seed", type=int, default=0)
    if with_model:
        p.add_argument("--model", choices=["linear", "ridge"], default="linear")
        p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
        p.add_argument("--degree", type=int, default=1, help="Polynomial degree for numeric columns")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="houseprice", description="Linear regression on Excel and CSV files.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Train a model and evaluate it on a held-out test split")
    _add_common(p)
    p.add_argument("--method", choices=["normal", "gd"], default="normal",
                   help="Exact least squares or gradient descent")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--out", default="model.json", help="Where to save the model")
    p.add_argument("--report", help="Also write an Excel report to this path")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="Predict with a saved model")
    p.add_argument("model", help="Path to model.json")
    p.add_argument("data", help="A .xlsx or .csv file with the same feature columns")
    p.add_argument("--sheet")
    p.add_argument("--out", default="predictions.xlsx", help="Output .xlsx or .csv")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("cv", help="K-fold cross-validation")
    _add_common(p)
    p.add_argument("-k", type=int, default=5, help="Number of folds")
    p.set_defaults(func=_cmd_cv)

    p = sub.add_parser("tune", help="Grid search over Ridge alpha and polynomial degree")
    _add_common(p, with_model=False)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("-k", type=int, default=5, help="Number of folds")
    p.add_argument("--out", help="Write all results to this Excel file")
    p.set_defaults(func=_cmd_tune)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (TableError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
