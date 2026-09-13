# Predicting Churn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

# 1. Load cleaned data
df = pd.read_csv("Telco_cleaned.csv")

# 2. Define target and features
target = "ChurnFlag"
X = df.drop(columns=["Churn", "ChurnFlag"])
y = df[target]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# 3. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

# 5. Model (Logistic Regression with class_weight to care about churners)
clf = LogisticRegression(max_iter=1000, class_weight="balanced")

clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", clf)
])

# 6. Fit
clf_pipeline.fit(X_train, y_train)

# 7. Evaluate
y_pred = clf_pipeline.predict(X_test)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("Churn classification performance:")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC-AUC:   {roc:.3f}")

print("\nDetailed classification report:")
print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

# Business interpretation
print("\nBusiness interpretation:")
print("Recall tells us how many of the customers who actually churned we correctly flagged.")
print("Higher recall is valuable for the Retention team, even if precision is slightly lower,")
print("because missing churners is more costly than contacting a few extra customers.")
