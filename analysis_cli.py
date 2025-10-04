from __future__ import annotations
import argparse
import os
import sqlite3

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from dotenv import load_dotenv
from tabulate import tabulate

from bot_client import NeyBotClient
from sql_guard import is_safe_select, sanitize_limit


def train_model(random_state: int = 42):
    """
    Train a logistic regression model on the Iris dataset and print evaluation metrics.

    This function loads the Iris dataset, splits it into train and test sets with
    stratification, scales features, trains a logistic regression classifier, and
    prints the accuracy and classification report.

    Parameters
    ----------
    random_state : int, optional
        Random seed for the train/test split. Defaults to 42.

    Returns
    -------
    tuple
        (model, scaler, iris_dataset) where model is the trained classifier,
        scaler is the StandardScaler used, and iris_dataset is the sklearn
        dataset object (with target names).
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print("Model accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds, target_names=iris.target_names))
    return model, scaler, iris


def run_sql_query(db_path: str, sql: str) -> pd.DataFrame:
    """
    Execute a SQL query against a SQLite database and return the result as a DataFrame.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    sql : str
        The SQL query to execute.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the query results.
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def infer_schema_hint(db_path: str) -> str:
    """
    Construct a simple schema hint for the Ney bot based on the SQLite table.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.

    Returns
    -------
    str
        A string describing the table and its columns, e.g. ``"Table iris(species, sepal_length, …)"``.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("PRAGMA table_info('iris')", conn)
    cols = ", ".join(df["name"].tolist())
    return f"Table iris({cols})"


def ask_ney(question: str, *, db_path: str = "data/iris.db") -> pd.DataFrame:
    """
    Convert a natural language question to SQL using ney bot and execute it on the DB.

    Parameters
    ----------
    question : str
        The natural language question to answer.
    db_path : str, optional
        Path to the SQLite database file. Defaults to ``data/iris.db``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the query result.

    Raises
    ------
    ValueError
        If the generated SQL is deemed unsafe or empty.
    """
    load_dotenv()
    client = NeyBotClient()
    schema_hint = infer_schema_hint(db_path)
    sql = client.nl_to_sql(question, schema_hint=schema_hint)
    if not is_safe_select(sql):
        raise ValueError(f"Unsafe SQL blocked:\n{sql}")
    sql = sanitize_limit(sql)
    print("Generated SQL:\n", sql)
    return run_sql_query(db_path, sql)


def main():
    parser = argparse.ArgumentParser(description="Iris ML + NL-to-SQL demo")
    parser.add_argument(
        "--mode",
        choices=["train", "ask"],
        default="train",
        help="Mode to run: 'train' to train the model, 'ask' to ask a question",
    )
    parser.add_argument(
        "--q",
        help="Natural language question (required when --mode ask)",
    )
    parser.add_argument(
        "--db",
        default="data/iris.db",
        help="Path to the SQLite database (used in ask mode)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    else:
        if not args.q:
            raise SystemExit("Provide --q 'your question' for --mode ask")
        df = ask_ney(args.q, db_path=args.db)
        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
