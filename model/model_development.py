import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# 1. Load the Breast Cancer Wisconsin dataset
print("Loading dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target # 0 = malignant, 1 = benign (sklearn default)
# Note: sklearn target: 0 is malignant (cancerous), 1 is benign (non-cancerous). 
# However, usually we want 1 to be the positive class (Malignant). Let's check or just stick to labels.
# In sklearn breast cancer dataset:
# class 0: Malignant
# class 1: Benign
# Let's inverse this for typical binary classification where 1 is the "event" (Cancer/Malignant).
# Or just keep track of labels. Let's remap for clarity: 1 = Malignant, 0 = Benign.
# Original: 0 -> Malignant, 1 -> Benign
# New: 1 -> Malignant, 0 -> Benign
df['diagnosis'] = df['diagnosis'].apply(lambda x: 0 if x == 1 else 1) 

print(f"Dataset shape: {df.shape}")

# 2. Data Preprocessing
# Selected features based on requirements (Radius, Texture, Perimeter, Area, Smoothness)
# Note: Feature names in sklearn might technically be 'mean radius' etc. Let's map them.
# The user asked for: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

print(f"Selected features: {selected_features}")
X = df[selected_features]
y = df['diagnosis']

# Handling missing values (Dataset usually has none, but good practice)
if X.isnull().sum().sum() > 0:
    print("Handling missing values...")
    X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Implement Machine Learning Algorithm (Logistic Regression)
print("Training Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# 5. Save the trained model and scaler
# We will save a dictionary containing both or just the model and scaler separately.
# To make it easy to load, let's save a pipeline or just the objects.
# Project requires 'breast_cancer_model.pkl'.
print("Saving model and scaler...")
model_data = {
    "model": model,
    "scaler": scaler,
    "features": selected_features
}

model_path = "breast_cancer_model.pkl"
joblib.dump(model_data, model_path)
print(f"Model saved to {os.path.abspath(model_path)}")

# 6. Demonstrate reloading
print("Verifying reload...")
loaded_data = joblib.load(model_path)
loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']

# Test prediction with a sample
sample = X_test.iloc[0].values.reshape(1, -1)
sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)[0]
prediction_label = "Malignant" if prediction == 1 else "Benign"
actual_label = "Malignant" if y_test.iloc[0] == 1 else "Benign"

print(f"Test sample prediction: {prediction_label} (Actual: {actual_label})")
print("Verification complete.")
