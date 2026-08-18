import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

STUDENT_NAME = "Anshul Ola"
ROLL_NUMBER = "2025ac05001"

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes (Gaussian)",
    "random_forest": "Random Forest (Ensemble)",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f"{name}.pkl") for name in MODEL_LABELS}


@st.cache_data
def load_feature_info():
    with open(MODEL_DIR / "feature_names.json") as f:
        return json.load(f)


@st.cache_data
def load_default_test_data():
    return pd.read_csv(ROOT / "test_data.csv")


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("Breast Cancer Classification, Model Comparison")
st.caption(f"ML Assignment 2, {STUDENT_NAME}, {ROLL_NUMBER}")
st.caption(
    "Dataset: UCI Breast Cancer Wisconsin (Diagnostic). Target: 0 is malignant, 1 is benign. "
    "Upload your own test CSV (same schema as test_data.csv) or use the bundled sample."
)

models = load_models()
feature_info = load_feature_info()
feature_names = feature_info["feature_names"]
target_col = feature_info["target_col"]

with st.sidebar:
    st.header("Controls")
    model_key = st.selectbox(
        "Choose a model", options=list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k]
    )
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled sample test_data.csv", value=uploaded is None)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = load_default_test_data()
else:
    st.info("Upload a CSV or check 'Use bundled sample' to see results.")
    st.stop()

missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing {len(missing_cols)} expected feature column(s), e.g. {missing_cols[:5]}")
    st.stop()

has_target = target_col in df.columns
X = df[feature_names]

pipeline = models[model_key]
y_pred = pipeline.predict(X)
y_proba = pipeline.predict_proba(X)[:, 1]

st.subheader(f"Results for {MODEL_LABELS[model_key]}")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Predictions (first 20 rows)**")
    preview = df.copy()
    preview["prediction"] = y_pred
    st.dataframe(preview.head(20), use_container_width=True)

if has_target:
    y_true = df[target_col]
    metrics = compute_metrics(y_true, y_pred, y_proba)

    with col2:
        st.write("**Evaluation metrics**")
        metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "value"})
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    st.write("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=feature_info["target_names"]).plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    st.write("**Classification Report**")
    report = classification_report(y_true, y_pred, target_names=feature_info["target_names"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

    st.divider()
    st.subheader("All models on this test data")
    all_rows = []
    for key, mdl in models.items():
        p = mdl.predict(X)
        pr = mdl.predict_proba(X)[:, 1]
        row = {"Model": MODEL_LABELS[key]}
        row.update(compute_metrics(y_true, p, pr))
        all_rows.append(row)
    comparison_df = pd.DataFrame(all_rows).set_index("Model")
    st.dataframe(comparison_df.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"), use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    comparison_df[["Accuracy", "F1 Score", "MCC"]].plot(kind="bar", ax=ax2)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
else:
    st.warning(f"No '{target_col}' column found, showing predictions only. Metrics need ground truth labels.")
