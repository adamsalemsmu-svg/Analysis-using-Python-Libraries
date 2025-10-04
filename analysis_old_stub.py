import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# For SQL queries with DataFrames
try:
    from pandasql import sqldf
except ImportError:
    sqldf = None  # If pandasql isn't installed


def load_iris_df():
    """
    Load the Iris dataset into a pandas DataFrame with feature columns and a target label.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Clean up feature names: remove units and replace spaces with underscores
    feature_names = [name.replace(" (cm)", "").replace(" ", "_") for name in iris.feature_names]
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    return df, iris.target_names


def train_logistic_regression(df):
    """
    Train a logistic regression classifier on the Iris dataset and return metrics.
    Returns a tuple of (accuracy, classification_report, model).
    """
    X = df.drop('species', axis=1)
    y = df['species']

    # Split the data with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report, model


def run_sql_query(df, query):
    """
    Execute a SQL query on the given DataFrame using pandasql (if available).
    Returns a DataFrame with the query results.
    """
    if sqldf is None:
        raise ImportError(
            "pandasql is not installed. Install it with `pip install pandasql` to run SQL queries."
        )
    # pandasql uses local variables; provide df as a local variable to the query
    return sqldf(query, {"df": df})


def natural_language_to_sql(nl_query: str) -> str:
    """
    A minimal demonstration of mapping natural language questions to SQL.
    In a production system, this would call a language model (CSQLbot) to generate SQL.
    """
    q = nl_query.lower().strip()
    if "average sepal length by species" in q:
        return "SELECT species, AVG(sepal_length) AS avg_sepal_length FROM df GROUP BY species"
    elif "count of samples per species" in q:
        return "SELECT species, COUNT(*) AS count FROM df GROUP BY species"
    else:
        raise ValueError(f"No SQL mapping defined for query: '{nl_query}'")


def run_natural_language_query(df: pd.DataFrame, nl_query: str) -> pd.DataFrame:
    """
    Convert a natural-language query to SQL (using a stub function) and execute it on the DataFrame.
    """
    sql = natural_language_to_sql(nl_query)
    return run_sql_query(df, sql)


def main():
    # Load dataset
    df, target_names = load_iris_df()
    # Train the classifier and display metrics
    accuracy, report, model = train_logistic_regression(df)
    print(f"Model accuracy: {accuracy:.2f}")
    print("Classification report:")
    print(report)

    # Demonstrate a natural language query translated to SQL
    try:
        print("\nExample SQL query based on natural language request:")
        nl = "average sepal length by species"
        result = run_natural_language_query(df, nl)
        print(f"Natural language: {nl}")
        print("Query result:")
        print(result)
    except Exception as e:
        print(f"Natural language query failed: {e}")


if __name__ == "__main__":
    main()
