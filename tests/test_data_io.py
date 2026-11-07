import pytest

from houseprice.data_io import Table, TableError, read_table, write_csv, write_excel


def test_csv_and_excel_give_the_same_table(tmp_path):
    columns = ["a", "b", "city"]
    rows = [[1.0, 2.5, "Paris"], [3.0, 4.0, "Lyon"]]
    write_csv(tmp_path / "t.csv", columns, rows)
    write_excel(tmp_path / "t.xlsx", {"data": (columns, rows)})
    for name in ("t.csv", "t.xlsx"):
        t = read_table(tmp_path / name)
        assert t.columns == columns
        assert t.rows == rows


def test_empty_cells_become_none(tmp_path):
    (tmp_path / "t.csv").write_text("a,b\n1,\n,2\n")
    assert read_table(tmp_path / "t.csv").rows == [[1.0, None], [None, 2.0]]


def test_blank_lines_are_skipped(tmp_path):
    (tmp_path / "t.csv").write_text("a,b\n1,2\n,\n3,4\n")
    assert len(read_table(tmp_path / "t.csv")) == 2


def test_select_sheet(tmp_path):
    write_excel(tmp_path / "t.xlsx", {"first": (["a"], [[1]]), "second": (["b"], [[2]])})
    assert read_table(tmp_path / "t.xlsx", sheet="second").columns == ["b"]
    with pytest.raises(TableError):
        read_table(tmp_path / "t.xlsx", sheet="nope")


@pytest.mark.parametrize("content", ["", "a,a\n1,2\n", ",a\n1,2\n", "a,b\n"])
def test_invalid_files(tmp_path, content):
    (tmp_path / "t.csv").write_text(content)
    with pytest.raises(TableError):
        read_table(tmp_path / "t.csv")


def test_unsupported_extension(tmp_path):
    (tmp_path / "t.txt").write_text("a\n1\n")
    with pytest.raises(TableError):
        read_table(tmp_path / "t.txt")


def test_table_column_and_take():
    t = Table(["a", "b"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert t.column("b") == [2.0, 4.0, 6.0]
    assert t.take([2, 0]).rows == [[5.0, 6.0], [1.0, 2.0]]
    with pytest.raises(TableError):
        t.column("c")
