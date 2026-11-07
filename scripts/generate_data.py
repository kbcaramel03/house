"""Generate the synthetic datasets in data/.

All values are fake. The "true" relationship used to generate prices is:

    price = 50_000 + 3_000 * surface_m2 + 8_000 * rooms
            - 600 * age_years - 2_500 * distance_center_km
            + city_bonus + noise

    city_bonus: Paris +80_000, Lyon +30_000, Marseille 0, Lille -10_000

A small fraction of numeric cells is left empty to mimic real-world missing data.

Writes:
    data/houses.xlsx      450 houses with prices (training data)
    data/houses.csv       the same table as CSV
    data/new_houses.csv   50 houses without prices (to try `houseprice predict`)

Usage: python scripts/generate_data.py
"""

import csv
from pathlib import Path

import numpy as np
from openpyxl import Workbook

COLUMNS = ["surface_m2", "rooms", "age_years", "distance_center_km", "city", "price"]
CITY_BONUS = {"Paris": 80_000, "Lyon": 30_000, "Marseille": 0, "Lille": -10_000}


def generate(n: int = 500, seed: int = 42, missing_rate: float = 0.03):
    rng = np.random.default_rng(seed)
    surface = rng.uniform(20, 200, n).round(1)
    rooms = np.clip((surface / 30 + rng.normal(0, 0.8, n)).round(), 1, 8).astype(int)
    age = rng.uniform(0, 80, n).round()
    distance = rng.uniform(0.5, 30, n).round(1)
    cities = rng.choice(list(CITY_BONUS), size=n, p=[0.3, 0.3, 0.25, 0.15])
    noise = rng.normal(0, 20_000, n)
    price = (50_000 + 3_000 * surface + 8_000 * rooms - 600 * age - 2_500 * distance
             + np.array([CITY_BONUS[c] for c in cities]) + noise)

    rows = []
    for i in range(n):
        row = [float(surface[i]), int(rooms[i]), int(age[i]), float(distance[i]),
               str(cities[i]), round(float(price[i]))]
        for col in (0, 2):  # surface_m2 and age_years are sometimes missing
            if rng.random() < missing_rate:
                row[col] = None
        rows.append(row)
    return rows


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows([["" if v is None else v for v in r] for r in rows])


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    rows = generate()
    train_rows, new_rows = rows[:450], rows[450:]

    wb = Workbook()
    ws = wb.active
    ws.title = "houses"
    ws.append(COLUMNS)
    for r in train_rows:
        ws.append(r)
    wb.save(data_dir / "houses.xlsx")

    write_csv(data_dir / "houses.csv", COLUMNS, train_rows)
    write_csv(data_dir / "new_houses.csv", COLUMNS[:-1], [r[:-1] for r in new_rows])
    print(f"wrote houses.xlsx, houses.csv and new_houses.csv to {data_dir}")


if __name__ == "__main__":
    main()
