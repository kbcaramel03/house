import csv

import pytest
from openpyxl import load_workbook

from conftest import make_table
from houseprice.cli import main, train
from houseprice.data_io import write_csv, write_excel
from houseprice.model_selection import cross_validate, grid_search, kfold_indices
from houseprice.pipeline import Pipeline


def test_pipeline_learns_true_effects(table):
    pipe = Pipeline("price", ["x1", "x2", "city"]).fit(table)
    effects = {name: orig for name, _, orig in pipe.coefficients()}
    assert effects["x1"] == pytest.approx(50, abs=0.5)
    assert effects["x2"] == pytest.approx(-20, abs=0.5)
    assert effects["city=B"] == pytest.approx(300, abs=5)
    assert effects["city=C"] == pytest.approx(-200, abs=5)


def test_pipeline_save_load(tmp_path, table):
    pipe = Pipeline("price", ["x1", "x2", "city"], model="ridge", alpha=2, degree=2).fit(table)
    pipe.save(tmp_path / "m.json")
    again = Pipeline.load(tmp_path / "m.json")
    assert list(again.predict(table)) == pytest.approx(list(pipe.predict(table)))


def test_kfold_covers_each_row_once():
    seen = [i for _, val in kfold_indices(23, 4, seed=0) for i in val]
    assert sorted(seen) == list(range(23))


def test_cross_validate_and_grid_search(table):
    cv = cross_validate(table, "price", ["x1", "x2", "city"], k=5)
    assert cv["r2"]["mean"] > 0.99
    results = grid_search(table, "price", ["x1", "x2", "city"], [0.01, 1000.0], [1], k=3)
    assert results[0]["alpha"] == 0.01


@pytest.fixture
def data_files(tmp_path):
    t = make_table(150, seed=1)
    write_excel(tmp_path / "train.xlsx", {"houses": (t.columns, t.rows)})
    new = [row[:-1] for row in make_table(10, seed=2).rows]
    write_csv(tmp_path / "new.csv", t.columns[:-1], new)
    return tmp_path


def test_cli_full_workflow(data_files):
    d = data_files
    assert main(["train", str(d / "train.xlsx"), "--target", "price",
                 "--out", str(d / "m.json"), "--report", str(d / "r.xlsx")]) == 0
    assert load_workbook(d / "r.xlsx").sheetnames == ["metrics", "coefficients", "test_predictions"]

    assert main(["predict", str(d / "m.json"), str(d / "new.csv"), "--out", str(d / "p.csv")]) == 0
    with (d / "p.csv").open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0][-1] == "predicted_price" and len(rows) == 11

    assert main(["cv", str(d / "train.xlsx"), "--target", "price", "-k", "3"]) == 0
    assert main(["tune", str(d / "train.xlsx"), "--target", "price", "--alphas", "1",
                 "--degrees", "1", "2", "-k", "3", "--out", str(d / "g.xlsx")]) == 0


def test_train_function_reports_metrics(data_files):
    pipe, test_t = train(data_files / "train.xlsx", "price", test_size=0.2, seed=0)
    assert pipe.metrics["n_test"] == len(test_t) == 30
    assert pipe.metrics["test"]["r2"] > 0.99


def test_cli_errors(tmp_path, capsys):
    assert main(["train", str(tmp_path / "missing.csv"), "--target", "price"]) == 2
    (tmp_path / "t.csv").write_text("a,b\n1,2\n3,4\n")
    assert main(["train", str(tmp_path / "t.csv"), "--target", "price"]) == 2
    assert "error:" in capsys.readouterr().err
