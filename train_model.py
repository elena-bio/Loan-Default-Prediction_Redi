# train_model.py
"""
Script to train the Loan Default Prediction model.
This script ensures a fixed set of features is used, avoiding mismatch errors in the demo.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==========================================
# 1. Load and Prepare Data
# ==========================================
print("Loading dataset...")
# Make sure the filename matches your actual train file
df = pd.read_csv("train_LZV4RXX.csv")

# Clean column names (remove extra spaces)
df.columns = [c.strip() for c in df.columns]

# Define the Target and Features
TARGET = 'loan_default'

# We explicitly define which columns we want to use for the demo.
# This prevents the model from expecting columns we don't have in the app (like 'loan_id').
NUMERIC_FEATURES = [
    'loan_amount', 
    'asset_cost', 
    'age', 
    'no_of_loans', 
    'no_of_curr_loans'
]

CATEGORICAL_FEATURES = [
    'education', 
    'proof_submitted'
]

# Feature Engineering: Create 'loan_to_asset' ratio
# We must do this before training so the model learns this relationship.
df['loan_to_asset'] = df['loan_amount'] / df['asset_cost']

# Add the new feature to the numeric list
FINAL_NUMERIC_FEATURES = NUMERIC_FEATURES + ['loan_to_asset']

# Select X (features) and y (target)
X = df[FINAL_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

print(f"Training with {len(X.columns)} features: {list(X.columns)}")

# ==========================================
# 2. Build the Pipeline
# ==========================================

# Pipeline for numeric columns: Fill missing values with Median -> Scale data
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical columns: Fill missing with Mode -> OneHot Encode
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ('num', num_pipe, FINAL_NUMERIC_FEATURES),
    ('cat', cat_pipe, CATEGORICAL_FEATURES)
])

# Create the full model pipeline with Random Forest
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# ==========================================
# 3. Train and Save
# ==========================================
print("Training the model...")
model.fit(X, y)
print("Model trained successfully.")

# Save the model to a file
output_file = 'best_loan_model.joblib'
joblib.dump(model, output_file)
print(f"Model saved to: {output_file}")