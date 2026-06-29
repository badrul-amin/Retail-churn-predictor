# 🛒 Retail Member Churn Predictor

> An end-to-end machine learning project predicting customer churn for retail businesses — from raw transaction data to a live deployed web app.

🔗 **[Live Demo → retail-churn-predictor-analysis.streamlit.app](https://retail-churn-predictor-analysis.streamlit.app/)**

---

## 📌 Project Overview

Customer churn is one of the costliest problems in retail. This project builds a complete churn prediction pipeline using real transaction data — identifying which customers are at risk of lapsing so retention campaigns can be targeted effectively.

The model achieves a **ROC-AUC of 0.74** using only 3 interpretable RFM features, making it both accurate and explainable to business stakeholders.

---

## 🎯 Business Problem

> *"Which of our members are likely to stop purchasing in the next 90 days?"*

Instead of manually reviewing thousands of customer records, this model scores every customer with a churn probability and risk tier — enabling the marketing team to act proactively.

---

## 🗂️ Dataset

- **Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (via `ucimlrepo`)
- **Size:** 541,909 transactions → 397,884 after cleaning
- **Period:** December 2010 – December 2011
- **Customers:** 4,338 unique members

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 🔬 Methodology

### 1. Data Cleaning
Removed cancellations (negative quantity), zero-price entries, and rows with missing CustomerID — reducing noise from 541K to 397K rows.

### 2. Feature Engineering — RFM
Built 3 features per customer from the **observation window (Dec 2010 – Aug 2011)**:

| Feature | Description |
|---|---|
| **Recency** | Days since last purchase |
| **Frequency** | Number of unique purchase dates |
| **Monetary** | Total revenue generated |

### 3. Churn Labelling — Temporal Split
Defined churn using a **forward-looking prediction window (Sep – Dec 2011)**:
- Customers with **no purchase in the prediction window = churned (1)**
- This avoids data leakage — features and labels come from different time periods

> ⚠️ **Note on leakage:** An initial random split produced AUC = 1.0 (a red flag). Recency was computed from the same snapshot date used to define churn, so the model was reading the label directly from the feature. The temporal split corrects this.

### 4. Modelling
Trained and compared two classifiers:

| Model | ROC-AUC | Accuracy |
|---|---|---|
| **XGBoost** ✅ | **0.74** | **69%** |
| Random Forest | 0.70 | 65% |

XGBoost was selected as the final model. Recency was the dominant feature — customers inactive for 90+ days showed significantly higher churn probability.

### 5. Feature Enrichment Experiment
Added 5 additional features (`avg_order_value`, `total_items`, `unique_products`, `avg_quantity`, `revenue_per_visit`). AUC improved marginally to 0.742. Decision: retained 3-feature RFM model for simplicity, interpretability, and cleaner Streamlit UX.

---

## 🖥️ Streamlit App Features

- **Single prediction** — input RFM values manually, get instant churn probability + risk tier
- **Batch prediction** — upload a CSV of customers, download scored results
- **Risk tiers** — 🟢 Low / 🟡 Medium / 🔴 High based on churn probability
- **Sample CSV** — downloadable template for testing batch mode

---

## 📁 Project Structure

```
retail-churn-predictor/
├── app/
│   └── streamlit_app.py       # Streamlit web application
├── model/
│   ├── churn_model.pkl        # Trained XGBoost model
│   ├── scaler.pkl             # StandardScaler
│   └── feature_cols.pkl       # Feature column names
├── notebooks/
│   └── churn_analysis.ipynb   # Full EDA, feature engineering & modelling
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/badrul-amin/retail-churn-predictor.git
cd retail-churn-predictor
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 📊 Key Findings

- **41.2% churn rate** in the dataset — substantial enough to justify a retention programme
- **Recency is the strongest predictor** — customers inactive for 90+ days have significantly elevated churn risk
- **Frequency and monetary value** provide secondary signal, particularly for identifying high-value at-risk customers
- A simple 3-feature RFM model achieves 0.74 AUC — competitive with more complex feature sets

---

## 👤 Author

**Badrul Amin** — Data Analyst | Aspiring Data Scientist

📎 [Portfolio](https://badrul-amin.github.io/portfolio/) · [LinkedIn](https://www.linkedin.com/in/badrulamins/) · [GitHub](https://github.com/badrul-amin)
