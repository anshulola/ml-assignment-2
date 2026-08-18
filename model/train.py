# trains 5 classification models on the breast cancer wisconsin dataset
# saves the trained models, the test split and the metrics table to disk
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

STUDENT_NAME = "Anshul Ola"
ROLL_NUMBER = "2025ac05001"

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
}


def main():
    print(f"ML Assignment 2, {STUDENT_NAME}, {ROLL_NUMBER}")
    data = load_breast_cancer(as_frame=True)
    # 569 rows and 30 features plus the target column
    df = data.frame
    feature_cols = list(data.feature_names)
    target_col = "target"

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_col], test_size=0.2, stratify=df[target_col], random_state=RANDOM_STATE
    )

    test_df = X_test.copy()
    test_df[target_col] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, estimator in MODELS.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "mcc": matthews_corrcoef(y_test, y_pred),
        }
        for key in ("accuracy", "auc", "precision", "recall", "f1"):
            assert 0.0 <= metrics[key] <= 1.0, f"{name} {key} out of range: {metrics[key]}"
        assert -1.0 <= metrics["mcc"] <= 1.0, f"{name} mcc out of range: {metrics['mcc']}"

        rows.append(metrics)
        joblib.dump(pipeline, ROOT / "model" / f"{name}.pkl")
        print(f"{name}: {metrics}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(ROOT / "model" / "metrics.csv", index=False)

    with open(ROOT / "model" / "feature_names.json", "w") as f:
        json.dump({"feature_names": feature_cols, "target_col": target_col, "target_names": list(data.target_names)}, f, indent=2)

    print("\nSaved test_data.csv, model/*.pkl, model/metrics.csv, model/feature_names.json")


if __name__ == "__main__":
    main()
