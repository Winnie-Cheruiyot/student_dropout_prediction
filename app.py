import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Load saved model and preprocessors ---
model = joblib.load("student_dropout_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# --- Streamlit App ---
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")
st.title("🎓 Student Dropout Prediction App")

st.markdown("""
This app uses a **Random Forest** model trained on academic and demographic data to predict if a student is likely to:
- Dropout
- Remain Enrolled
- Graduate
""")

# --- Sidebar for file upload ---
st.sidebar.header("Upload Student CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    st.subheader("Raw Uploaded Data")
    st.write(df.head())

    if 'Target' in df.columns:
        df['Target_encoded'] = le.transform(df['Target'])
        y = df['Target_encoded']
        X = df.drop(columns=['Target', 'Target_encoded'])
    else:
        st.warning("The uploaded dataset is missing the 'Target' column. Predictions will be shown without evaluation.")
        X = df
        y = None

    X_scaled = scaler.transform(X)

    # --- Predictions ---
    preds = model.predict(X_scaled)
    pred_labels = le.inverse_transform(preds)

    st.subheader("🔮 Predictions")
    df['Prediction'] = pred_labels
    st.write(df[['Prediction']].value_counts().rename("Count"))

    st.dataframe(df[['Prediction'] + list(X.columns)].head())

    # --- Visualizations ---
    st.subheader("📊 Visualizations")

    # Class distribution
    if 'Target' in df.columns:
        st.markdown("**Target Distribution:**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Target', palette='Set2', ax=ax1)
        ax1.set_title("Class Distribution")
        st.pyplot(fig1)

    # Correlation matrix
    st.markdown("**Correlation Matrix:**")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # Boxplot comparisons
    st.markdown("**Feature Distributions by Target:**")
    selected_features = st.multiselect("Choose up to 6 features", X.columns.tolist(), default=X.columns[:3].tolist())
    if selected_features and 'Target' in df.columns:
        fig3, axes = plt.subplots(nrows=1, ncols=len(selected_features), figsize=(5 * len(selected_features), 4))
        if len(selected_features) == 1:
            axes = [axes]
        for i, col in enumerate(selected_features):
            sns.boxplot(data=df, x='Target', y=col, ax=axes[i])
            axes[i].set_title(col)
        st.pyplot(fig3)

    # Model evaluation
    if y is not None:
        st.subheader("📈 Model Evaluation")
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        st.write(f"**Accuracy:** {acc:.2f}")

        st.markdown("**Classification Report:**")
        st.text(classification_report(y, y_pred, target_names=le.classes_))

        st.markdown("**Confusion Matrix:**")
        fig4, ax4 = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_scaled, y, display_labels=le.classes_, cmap='Blues', ax=ax4)
        st.pyplot(fig4)

else:
    st.info("Please upload a student dataset CSV file using the sidebar to begin.")
