# FreshWave Telecom – Customer Analytics (Telco Customer Churn)

## Project Overview

**Project type:** Independent capstone project (GitHub portfolio)  
**Skills covered:** Supervised learning (regression & classification), Unsupervised learning (K-Means clustering)

This project uses the Telco Customer Churn dataset to answer three business questions for FreshWave Telecom:
1. Estimate customer revenue.
2. Predict customer churn.
3. Segment customers for targeted marketing.

## Dataset

Source: Kaggle – Telco Customer Churn (blastchar)  
Rows: 7,043 customers  
Columns: 21 (demographics, account info, subscribed services, churn label)

## Tasks

### 1. Data Preparation

- Cleaned `TotalCharges` and handled missing values.
- Dropped non-predictive identifiers (`customerID`).
- Encoded categorical variables and scaled numeric features.
- Used stratified train/test split for churn to reflect class imbalance.

### 2. Revenue Prediction (Regression)

- Target: `MonthlyCharges` as a proxy for monthly revenue.
- Model: Linear Regression with preprocessing pipeline.
- Metrics: MAE, RMSE, R².
- Business framing: typical monthly error in pounds and whether this is acceptable for Finance.

### 3. Churn Prediction (Classification)

- Target: `ChurnFlag` (Yes/No → 1/0).
- Model: Logistic Regression with `class_weight="balanced"`.
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Business framing: focus on Recall to catch likely churners for proactive retention.

### 4. Customer Segmentation (K-Means)

- Features: tenure, charges, contract type, payment method, internet and support services.
- Method: K-Means with elbow method to choose number of clusters.
- Output: 4 segments with profiles (tenure, charges, contract mix).
- Business framing: segment-specific campaign ideas (e.g. loyalty offers, upsell, retention).

## Repository Structure

- `notebooks/01_data_preparation.py`
- `notebooks/02_regression_revenue.py`
- `notebooks/03_classification_churn.py`
- `notebooks/04_clustering_segments.py`
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `requirements.txt`

## Key Findings / Business Recommendations

- **Finance:** Use the revenue model to estimate monthly revenue per customer with typical error around the MAE value.
- **Retention:** Use the churn model to generate a ranked list of high-risk customers for targeted outreach.
- **Marketing:** Use segments to design differentiated campaigns (e.g. high-tenure, high-value vs. low-tenure, price-sensitive customers).

---

You can now copy each file into your environment, install from `requirements.txt`, and run the scripts step by step.
