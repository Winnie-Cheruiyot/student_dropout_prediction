# %%
# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import joblib
import streamlit as st


# %% [markdown]
#Selecting an ML Approach
# 
# ### ✅ Problem Type
# The goal of this project is to **predict early student dropouts** using existing data that includes academic records, attendance, socio-economic status, and other factors. This is a **binary classification** task where the output is either:
# 
# - `0` – Student is likely to continue  
# - `1` – Student is at risk of dropping out
# 
# ### 🧠 Chosen ML Category: Supervised Learning
# 
# **Supervised Learning** is the most appropriate approach because:
# - We have **labeled data**: past records where we know whether students dropped out.
# - The model needs to **learn from historical patterns** and make predictions on new, unseen students.
# - It allows us to evaluate performance using established classification metrics like accuracy, precision, recall, and F1-score.
# 
# ---
# 
# ### 🤖 Suitable Algorithms for Dropout Prediction
# 
# | Algorithm | Pros | Cons |
# |----------|------|------|
# | **Logistic Regression** | Simple, interpretable, good baseline | May underperform with complex patterns |
# | **Decision Tree** | Easy to visualize, handles non-linear relationships | Can overfit if not pruned |
# | **Random Forest** | Robust, handles missing data and imbalanced features well | Slower with very large datasets |
# | **Gradient Boosting (XGBoost)** | High accuracy, feature importance ranking | Requires hyperparameter tuning |
# | **Support Vector Machine (SVM)** | Good for high-dimensional data | Not ideal for large datasets |
# 
# ---
# 
# ### ⭐ Recommended Approach
# 
# We recommend starting with **Random Forest**, as it provides:
# - High accuracy and robustness to overfitting
# - Automatic feature importance evaluation
# - Compatibility with both categorical and numerical inputs
# 
# As a more advanced option, **XGBoost** can be used to further improve performance due to its regularization techniques and gradient boosting framework.
# 
# ---
# 
# **Conclusion:**  
# This problem is best solved using **Supervised Learning**, particularly with **Random Forest** as a strong baseline model. The flexibility and predictive power of tree-based models make them ideal for early dropout detection.
# 

# %%
# --- Step 2: Load Dataset ---

df = pd.read_csv("students_dropout.csv", sep=';')  # Adjust separator if needed
print(df.shape)


# %%
# 2. Encode the Target

le = LabelEncoder()
df['Target_encoded'] = le.fit_transform(df['Target'])  # Dropout=0, Enrolled=1, Graduate=2

# %%
# Save label encoder
joblib.dump(le, "label_encoder.pkl")



# %%
# 3. Select Features (drop Target and any leakage or identifiers)
X = df.drop(columns=['Target', 'Target_encoded'])
y = df['Target_encoded']

# %%
# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# %%
# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# %%
# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# %%
# 6. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
# 7. Evaluate Model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %%
# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# %%
# 8. Save Model
joblib.dump(model, "student_dropout_model.pkl")

# %%
# 9. Plot Boxplots for Feature Distributions
plt.figure(figsize=(15, 10))
for i, column in enumerate(X.columns[:8]):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df['Target'], y=df[column])
    plt.title(f"{column} by Target")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# 10. Class Distribution Bar Chart
sns.countplot(data=df, x='Target', palette='Set2')
plt.title("Target Class Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# %%
# 11. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# %%
# --- Step 9: Ethical Reflection ---

print("""
Ethical Reflection:
- Data bias risks: Imbalanced representation of demographics may cause unfair predictions.
- Model fairness: Ensure interventions based on predictions do not discriminate.
- Sustainability: Early prediction helps improve education retention and student support.
""")


