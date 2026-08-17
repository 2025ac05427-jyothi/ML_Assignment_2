
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 ML Classification Model Dashboard")

st.write(
    """
    This application compares multiple machine learning
    classification models on the selected test dataset.
    """
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = joblib.load(
    "model/metadata.pkl"
)

feature_names = metadata["feature_names"]
target_names = metadata["target_names"]


# ============================================================
# MODEL FILES
# ============================================================

model_files = {

    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "kNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "Random Forest":
        "model/random_forest.pkl"
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(model_files.keys())
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# ============================================================
# STOP IF NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload the test CSV file to continue."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

df = pd.read_csv(
    uploaded_file
)


st.subheader("Uploaded Dataset")

st.write(
    f"Dataset contains {df.shape[0]} rows "
    f"and {df.shape[1]} columns."
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if "target" not in df.columns:

    st.error(
        "Error: The uploaded CSV must contain "
        "a 'target' column."
    )

    st.stop()


# ============================================================
# CHECK FEATURE COLUMNS
# ============================================================

missing_features = [
    feature
    for feature in feature_names
    if feature not in df.columns
]

if len(missing_features) > 0:

    st.error(
        "The following required features are missing:"
    )

    st.write(
        missing_features
    )

    st.stop()


# ============================================================
# PREPARE INPUT DATA
# ============================================================

X_test = df[
    feature_names
]

y_test = df[
    "target"
]


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model = joblib.load(
    model_files[selected_model]
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# PROBABILITY / DECISION SCORE
# ============================================================

if hasattr(model, "predict_proba"):

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

else:

    y_probability = model.decision_function(
        X_test
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# SELECTED MODEL
# ============================================================

st.header("2. Selected Model")

st.success(
    f"Selected Model: {selected_model}"
)


# ============================================================
# METRICS
# ============================================================

st.header("3. Evaluation Metrics")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    st.metric(
        "AUC",
        f"{auc:.4f}"
    )


with col2:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )


with col3:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("4. Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=target_names,
    columns=target_names
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("5. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)


# ============================================================
# PREDICTIONS
# ============================================================

st.header("6. Predictions")

prediction_df = X_test.copy()

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = y_pred

st.dataframe(
    prediction_df,
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "Machine Learning Assignment - Classification Models"
)
