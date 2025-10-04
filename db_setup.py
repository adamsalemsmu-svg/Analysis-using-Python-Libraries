from __future__ import annotations
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def build_iris_db(db_path: str = "data/iris.db", table: str = "iris") -> None:
    """
    Create a SQLite database containing the Iris dataset.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. If parent directories do not exist, they
        will be created. Defaults to ``data/iris.db``.
    table : str
        Name of the table to store the dataset. Defaults to ``iris``.
    """
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # normalize column names: remove spaces and units
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


if __name__ == "__main__":
    build_iris_db()
    print("✅ SQLite DB created at data/iris.db (table: iris)")
