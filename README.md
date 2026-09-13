# 📘 FreshWave Telecom – Customer Analytics Project  
Machine Learning · Regression · Classification · Clustering  

---

## ⭐ Project Overview

This portfolio project analyses customer behaviour for **FreshWave Telecom** using the popular **Telco Customer Churn** dataset. It demonstrates end‑to‑end data‑science capability across:

- **Supervised Learning (Regression):** Predicting customer revenue  
- **Supervised Learning (Classification):** Predicting customer churn  
- **Unsupervised Learning (Clustering):** Segmenting customers for targeted marketing  

The project is structured to reflect real business workflows, with clear reasoning, modelling decisions, and business‑focused insights.

---

## 📂 Repository Structure

| Folder / File | Purpose |
|---------------|---------|
| **data/** | Raw and cleaned datasets |
| **notebooks/** | Python scripts for each modelling task |
| **visuals/** | Plots and charts generated during analysis |
| **requirements.txt** | Python dependencies for reproducibility |
| **README.md** | Project documentation |

---

## 📊 Dataset Summary

Source: Kaggle — *Telco Customer Churn (blastchar)*  
Rows: **7,043 customers**  
Columns: **21 features**, including:

- Demographics (gender, senior citizen, dependents)  
- Account information (tenure, contract type, payment method)  
- Service subscriptions (internet, streaming, tech support)  
- Charges (monthly and total)  
- Churn label (Yes/No)

---

## 🔧 1. Data Preparation

The dataset required several cleaning and preprocessing steps:

- Converted `TotalCharges` to numeric and handled blank values  
- Removed non‑predictive identifiers (`customerID`)  
- Created `ChurnFlag` (Yes/No → 1/0)  
- Split data into training and testing sets  
- Applied **One‑Hot Encoding** for categorical variables  
- Applied **Standard Scaling** for numeric variables  

These steps ensure fair model evaluation and reproducibility.

---

## 📈 2. Revenue Prediction (Regression)

**Business Question:**  
*How much monthly revenue is a customer likely to generate?*

**Target Variable:** `MonthlyCharges`  
**Model Used:** Linear Regression  
**Pipeline:** Scaling + One‑Hot Encoding + Regression

**Key Metrics:**

- **MAE** — average monthly error  
- **RMSE** — typical deviation  
- **R²** — proportion of variance explained  

**Business Interpretation:**  
The model provides a reasonable estimate of expected monthly revenue per customer. Finance can use this to forecast revenue and identify high‑value customers.

---

## 🔍 3. Churn Prediction (Classification)

**Business Question:**  
*Which customers are likely to cancel their service?*

**Target Variable:** `ChurnFlag`  
**Model Used:** Logistic Regression (balanced class weights)

**Key Metrics:**

- Accuracy  
- Precision  
- **Recall** (critical for churn)  
- F1‑Score  
- ROC‑AUC  

**Business Interpretation:**  
Recall is prioritised because missing a churner is more costly than contacting a customer who stays. The model helps the Retention team proactively reach out to high‑risk customers.

---

## 🎯 4. Customer Segmentation (K‑Means Clustering)

**Business Question:**  
*How can we group customers into meaningful segments for targeted marketing?*

**Features Used:** Tenure, charges, contract type, internet service, support services, streaming services.

**Steps:**

- Preprocessing (scaling + encoding)  
- Elbow method to determine optimal clusters  
- K‑Means clustering  
- Segment profiling  

**Example Segment Insights:**

- **Segment 0:** Long‑tenure, stable customers — ideal for loyalty rewards  
- **Segment 1:** High‑spend customers — suitable for premium upgrades  
- **Segment 2:** Low‑tenure, high churn risk — targeted retention offers  
- **Segment 3:** Price‑sensitive customers — promotional bundles  

---

## 🧠 Key Findings & Business Recommendations

### **Finance Team**
- Use the revenue model to estimate monthly revenue per customer  
- Identify high‑value customers for premium service offers  

### **Retention Team**
- Focus on customers flagged with high churn probability  
- Prioritise outreach to segments with low tenure and high monthly charges  

### **Marketing Team**
- Use customer segments to tailor campaigns  
- Avoid generic promotions; target segments based on behaviour and service usage  

---

## ⚙️ Reproducibility

Install dependencies:


Run scripts in order:

1. `01_data_preparation.py`  
2. `02_regression_revenue.py`  
3. `03_classification_churn.py`  
4. `04_clustering_segments.py`

---

## 🎓 About This Project

This project was completed as an independent capstone for a data‑science portfolio.  
It demonstrates practical machine‑learning skills, clear reasoning, and the ability to translate technical results into business value.

---
