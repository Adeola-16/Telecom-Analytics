# Predicting Revenue (Regression)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load cleaned data
df = pd.read_csv('Telco_cleaned.csv')

# 2. Define target and features
target = "MonthlyCharges"
X = df.drop(columns=[target, "Churn", "ChurnFlag"])
y = df[target]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# 5. Model
model = LinearRegression()

from sklearn.pipeline import Pipeline

reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# 6. Fit
reg_pipeline.fit(X_train, y_train)

# 7. Evaluate
y_pred = reg_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

# 8. Print

print("Regression performance (MonthlyCharges):")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

# Business interpretation example
print("\nBusiness interpretation:")
print(f"On average, predictions are off by about £{mae:.2f} per month.")
print("Finance can use this to estimate expected monthly revenue per customer,")
print("with typical error around this amount.")
