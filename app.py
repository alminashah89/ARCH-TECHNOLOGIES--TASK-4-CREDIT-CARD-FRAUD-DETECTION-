import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

# -----------------------------
# Load Dataset
# -----------------------------

data = pd.read_csv("creditcard.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# Class Distribution
print("\nClass Distribution:")
print(data["Class"].value_counts())

# -----------------------------
# Data Preprocessing
# -----------------------------

# Check Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Scale Time and Amount Features
scaler = StandardScaler()

data["scaled_amount"] = scaler.fit_transform(
    data["Amount"].values.reshape(-1, 1)
)

data["scaled_time"] = scaler.fit_transform(
    data["Time"].values.reshape(-1, 1)
)

# Drop original columns
data.drop(["Time", "Amount"], axis=1, inplace=True)

# -----------------------------
# Feature Selection
# -----------------------------

X = data.drop("Class", axis=1)
y = data["Class"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Handle Class Imbalance using SMOTE
# -----------------------------

print("\nBefore SMOTE:")
print(np.bincount(y_train))

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(
    X_train, y_train
)

print("\nAfter SMOTE:")
print(np.bincount(y_train_res))

# -----------------------------
# Model Training
# -----------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation Metrics
# -----------------------------

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_score = roc_auc_score(
    y_test,
    model.predict_proba(X_test)[:, 1]
)

print("\nROC-AUC Score:", roc_score)

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Fraud Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()