"""Reading tables from Excel (.xlsx) or CSV files, and writing Excel reports.

A table has a header row (column names) followed by data rows. Cells are
returned as float (numbers), str (text) or None (empty cell).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook


class TableError(ValueError):
    """Raised when a file does not contain a usable table."""


Cell = float | str | None


@dataclass
class Table:
    columns: list[str]
    rows: list[list[Cell]]

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> list[Cell]:
        if name not in self.columns:
            raise TableError(f"column {name!r} not found (columns: {self.columns})")
        i = self.columns.index(name)
        return [row[i] for row in self.rows]

    def take(self, indices) -> "Table":
        return Table(list(self.columns), [self.rows[i] for i in indices])


def _convert(cell) -> Cell:
    if cell is None:
        return None
    if isinstance(cell, bool):
        return float(cell)
    if isinstance(cell, (int, float)):
        return float(cell)
    text = str(cell).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _build_table(path: Path, raw_rows: list) -> Table:
    if not raw_rows:
        raise TableError(f"{path}: file is empty")
    header = [str(c).strip() if c is not None else "" for c in raw_rows[0]]
    while header and header[-1] == "":
        header.pop()
    if not header or "" in header:
        raise TableError(f"{path}: every column needs a name in the first row")
    if len(set(header)) != len(header):
        raise TableError(f"{path}: duplicate column names")

    rows = []
    for raw in raw_rows[1:]:
        cells = [_convert(c) for c in list(raw)[: len(header)]]
        cells += [None] * (len(header) - len(cells))
        if all(c is None for c in cells):
            continue  # blank line
        rows.append(cells)
    if not rows:
        raise TableError(f"{path}: no data rows")
    return Table(header, rows)


def read_table(path: str | Path, sheet: str | None = None) -> Table:
    """Read a .xlsx or .csv file. `sheet` only applies to Excel files."""
    path = Path(path)
    if not path.exists():
        raise TableError(f"{path}: file not found")
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as fh:
            return _build_table(path, list(csv.reader(fh)))

    if suffix == ".xlsx":
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            if sheet is None:
                ws = wb.worksheets[0]
            elif sheet in wb.sheetnames:
                ws = wb[sheet]
            else:
                raise TableError(f"{path}: no sheet named {sheet!r} (found {wb.sheetnames})")
            raw_rows = list(ws.iter_rows(values_only=True))
        finally:
            wb.close()
        return _build_table(path, raw_rows)

    raise TableError(f"{path}: unsupported file type (use .xlsx or .csv)")


def write_excel(path: str | Path, sheets: dict[str, tuple[list[str], list[list]]]) -> None:
    """Write several sheets to one workbook: {sheet_name: (columns, rows)}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    for name, (columns, rows) in sheets.items():
        ws = wb.create_sheet(title=name[:31])
        ws.append(columns)
        for row in rows:
            ws.append(list(row))
        for i, col in enumerate(columns, start=1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = max(12, len(str(col)) + 2)
    wb.save(path)


def write_csv(path: str | Path, columns: list[str], rows: list[list]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
