# Segmenting Cutomers: K-Means

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Load cleaned data
df = pd.read_csv("Telco_cleaned.csv")

# 2. Choose features for segmentation
# Focus on account + service features (exclude Churn labels and IDs)
features_to_use = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

X = df[features_to_use]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# 3. Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

X_processed = preprocessor.fit_transform(X)

# 4. Elbow method to choose k
inertias = []
k_values = range(2, 9)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_processed)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow method for K-Means")
plt.tight_layout()
plt.show()

# Assume k=4 after inspecting elbow curve (you can adjust)
k_opt = 4
kmeans = KMeans(n_clusters=k_opt, random_state=42)
clusters = kmeans.fit_predict(X_processed)

df["Segment"] = clusters

print("\nSegment counts:")
print(df["Segment"].value_counts())

# 5. Simple segment profiling
segment_profile = df.groupby("Segment")[["tenure", "MonthlyCharges", "TotalCharges"]].mean()
print("\nSegment profile (numeric averages):")
print(segment_profile)

print("\nExample business profiling:")
for seg in sorted(df["Segment"].unique()):
    seg_data = df[df["Segment"] == seg]
    print(f"\nSegment {seg}:")
    print(f"- Avg tenure: {seg_data['tenure'].mean():.1f} months")
    print(f"- Avg MonthlyCharges: £{seg_data['MonthlyCharges'].mean():.2f}")
    print(f"- Avg TotalCharges: £{seg_data['TotalCharges'].mean():.2f}")
    print("- Contract mix:\n", seg_data["Contract"].value_counts(normalize=True).round(2))
    print("- InternetService mix:\n", seg_data["InternetService"].value_counts(normalize=True).round(2))
    print("Suggested campaign ideas based on price sensitivity, tenure, and contract type.")
