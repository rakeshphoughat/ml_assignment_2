import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# -----------------------------
# App Identity (Unique UI)
# -----------------------------
APP_TITLE = "CancerCheck ML Dashboard"
APP_SUBTITLE = "Breast Cancer (UCI WDBC) • 6 Models • 6 Metrics"

st.set_page_config(page_title=APP_TITLE, page_icon="🧬", layout="wide")
st.title(f"🧬 {APP_TITLE}")
st.caption(APP_SUBTITLE)

# -----------------------------
# Paths (IMPORTANT: use model/ per assignment PDF)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -----------------------------
# Model Registry (Friendly -> filename)
# -----------------------------
MODEL_REGISTRY = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN (k=5)": "KNN.pkl",
    "Naive Bayes (Gaussian)": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl",
}

# -----------------------------
# Helper Functions
# -----------------------------
def load_classifier(model_label: str):
    file_name = MODEL_REGISTRY[model_label]
    model_path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def compute_scores(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.50):
    """
    Computes required metrics. If model provides predict_proba, uses threshold for y_pred.
    """
    # Base prediction
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        y_prob = y_pred  # fallback for AUC (not ideal, but safe)

    scores = {
        "Accuracy": float(accuracy_score(y, y_pred)),
        "AUC": float(roc_auc_score(y, y_prob)),
        "Precision": float(precision_score(y, y_pred)),
        "Recall": float(recall_score(y, y_pred)),
        "F1": float(f1_score(y, y_pred)),
        "MCC": float(matthews_corrcoef(y, y_pred)),
        "cm": confusion_matrix(y, y_pred),
        "report": classification_report(y, y_pred, digits=4),
    }
    return scores

def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    return fig

# -----------------------------
# Tabs (Unique Layout)
# -----------------------------
tab_eval, tab_board, tab_about = st.tabs(["✅ Evaluate", "📊 Leaderboard", "ℹ️ About"])

with tab_eval:
    left, right = st.columns([1.1, 1.0])

    with left:
        st.subheader("1) Upload CSV")
        csv_in = st.file_uploader(
            "Upload a CSV file with features + a 'target' column (0/1).",
            type=["csv"]
        )
            # Download sample CSV button
    sample_path = os.path.join(BASE_DIR, "breast_cancer_test.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                label="⬇ Download Sample CSV",
                data=f,
                file_name="breast_cancer_sample.csv",
                mime="text/csv"
            )

        show_cm = st.checkbox("Show confusion matrix", value=True)
        show_report = st.checkbox("Show classification report", value=False)

    with right:
        st.subheader("2) Choose Model")
        chosen_model = st.selectbox("Model", list(MODEL_REGISTRY.keys()))
        threshold = st.slider(
            "Decision threshold (only for probability models)",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05
        )

    if csv_in is None:
        st.info("Upload a CSV to evaluate. Tip: include all 30 feature columns + a 'target' column.")
    else:
        df = pd.read_csv(csv_in)

        if "target" not in df.columns:
            st.error("Missing required column: 'target'. Please upload a labeled CSV.")
        else:
            y = df["target"].astype(int)
            X = df.drop(columns=["target"])

            try:
                clf = load_classifier(chosen_model)
                out = compute_scores(clf, X, y, threshold=threshold)

                st.markdown("### 📌 Metrics (on uploaded file)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{out['Accuracy']:.4f}")
                m2.metric("AUC", f"{out['AUC']:.4f}")
                m3.metric("MCC", f"{out['MCC']:.4f}")

                m4, m5, m6 = st.columns(3)
                m4.metric("Precision", f"{out['Precision']:.4f}")
                m5.metric("Recall", f"{out['Recall']:.4f}")
                m6.metric("F1", f"{out['F1']:.4f}")

                if show_cm:
                    st.pyplot(plot_confusion_matrix(out["cm"]))

                if show_report:
                    st.text(out["report"])

            except Exception as e:
                st.error(f"Error while evaluating: {e}")

#with tab_board:
#   st.subheader("Leaderboard (Offline Results)")
#    st.write("Paste your offline evaluation table here for reference (from your notebook).")

#    st.info(
#        "Optional: Export your results_df to results.csv and load it here to display a leaderboard."
#    )

with tab_about:
    st.subheader("About This App")
    st.markdown(
        """
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI WDBC  
**Task:** Binary classification (0 = malignant, 1 = benign)  
**Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost  
**Metrics:** Accuracy, AUC, Precision, Recall, F1 Score, MCC  

**Unique Feature:** Threshold-based classification for probability models (lets you explore precision/recall tradeoff).
"""
    )
