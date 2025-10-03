import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def main():
    """Load Iris dataset, train a classifier, and evaluate it."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], y, test_size=0.2, random_state=42
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
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))


if __name__ == "__main__":
    main()
