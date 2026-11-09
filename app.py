
import streamlit as st
import pandas as pd
import joblib
import os

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

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Prediction")
st.write("Machine Learning Model Comparison and Evaluation")

# Load saved preprocessing object
@st.cache_resource
def load_preprocessor():
    return joblib.load("models/preprocessor.joblib")

# Load saved models
@st.cache_resource
def load_models():
    model_files = {
        "Logistic Regression": "models/logistic_regression.joblib",
        "Decision Tree": "models/decision_tree.joblib",
        "KNN": "models/knn.joblib",
        "Naive Bayes": "models/naive_bayes.joblib",
        "SVM": "models/svm.joblib"
    }

    return {
        name: joblib.load(path)
        for name, path in model_files.items()
    }

preprocessor = load_preprocessor()
models = load_models()

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the target column y",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview")
    st.dataframe(data.head())

    if "y" not in data.columns:
        st.error("The uploaded CSV must contain the target column 'y'.")

    else:
        X = data.drop("y", axis=1)
        y_true = data["y"]

        if len(X.columns) != 16:
            st.error("The dataset must contain all 16 input features.")

        else:
            X_processed = preprocessor.transform(X)

            st.subheader("Select Machine Learning Model")

            selected_model = st.selectbox(
                "Choose a model",
                list(models.keys())
            )

            model = models[selected_model]

            y_pred = model.predict(X_processed)

            if selected_model == "SVM":
                y_score = model.decision_function(X_processed)
            else:
                y_score = model.predict_proba(X_processed)[:, 1]

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(
                (y_true == "yes").astype(int), y_score
            )
            precision = precision_score(
                y_true, y_pred, pos_label="yes", zero_division=0
            )
            recall = recall_score(
                y_true, y_pred, pos_label="yes", zero_division=0
            )
            f1 = f1_score(
                y_true, y_pred, pos_label="yes", zero_division=0
            )
            mcc = matthews_corrcoef(y_true, y_pred)

            st.subheader("Model Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_true, y_pred, labels=["no", "yes"]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"]
            )

            st.dataframe(cm_df)

            st.subheader("Classification Report")

            report = classification_report(
                y_true,
                y_pred,
                labels=["no", "yes"],
                target_names=["No", "Yes"],
                output_dict=True,
                zero_division=0
            )

            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("Model Comparison")

            if os.path.exists("model_metrics.csv"):
                comparison = pd.read_csv("model_metrics.csv")
                st.dataframe(comparison)

            st.success("Model evaluation completed successfully!")
