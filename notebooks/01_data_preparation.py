# Data Preparation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Load data
df = pd.read_csv('Telco_Customer_Churn_dataset.csv')
# Clean Data
# Step 1

df_clean = df.replace(r'^\s*$', None, regex=True)
blank_columns = df_clean.columns[df_clean.isna().all()]
columns_with_some_blanks = df_clean.columns[df_clean.isna().any()]
blank_rows = df_clean.index[df_clean.isna().all(axis=1)]
rows_with_some_blanks = df_clean.index[df_clean.isna().any(axis=1)]

print("Blank columns:", blank_columns.tolist())
print("Columns with some blanks:", columns_with_some_blanks.tolist())
print("Blank rows:", blank_rows.tolist())
print("Rows with some blanks:", rows_with_some_blanks.tolist())
# Step 2

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric (it has some blanks)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values
print("\nMissing values:\n", df.isna().sum())

# Drop rows with missing TotalCharges (small number, acceptable assumption)
df = df.dropna(subset=["TotalCharges"])

# Basic feature decisions

# Drop customerID (identifier, not predictive)
df = df.drop(columns=["customerID"])

# Target columns for later:
# - Revenue proxy: MonthlyCharges or TotalCharges (we'll use MonthlyCharges for regression)
# - Churn: 'Churn' (Yes/No)

# Encode target for churn (for later use)
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

# Separate numeric and categorical features
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Example train/test split for later tasks
# For churn classification

X_churn = df.drop(columns=["Churn", "ChurnFlag"])
y_churn = df["ChurnFlag"]

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

print("\nTrain/Test sizes (churn):", X_train_churn.shape, X_test_churn.shape)

# For revenue regression (using MonthlyCharges as target)
X_rev = df.drop(columns=["MonthlyCharges"])
y_rev = df["MonthlyCharges"]

X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(
    X_rev, y_rev, test_size=0.2, random_state=42
)

print("Train/Test sizes (revenue):", X_train_rev.shape, X_test_rev.shape)


# Save prepared data (optional)

df.to_csv('Telco_cleaned.csv', index=False)
print("\nCleaned data saved to Telco_cleaned.csv")





















