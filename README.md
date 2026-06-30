# Retail Member Churn Predictor

> An end-to-end machine learning project predicting customer churn for retail businesses — from raw transaction data to a live deployed web app.

 **[Live Demo → retail-churn-predictor-analysis.streamlit.app](https://retail-churn-predictor-analysis.streamlit.app/)**

---

##  Project Overview

Customer churn is one of the costliest problems in retail. This project builds a complete churn prediction pipeline using real transaction data — identifying which customers are at risk of lapsing so retention campaigns can be targeted effectively.

Four models were trained in increasing order of complexity — Logistic Regression, Decision Tree, Random Forest, and XGBoost — to let the data determine which approach fits best, rather than assuming upfront. **XGBoost was selected for deployment with a ROC-AUC of 0.74**, though Logistic Regression achieved a comparable result with far greater interpretability.

---

##  Business Problem

> *"Which of our members are likely to stop purchasing in the next 90 days?"*

Instead of manually reviewing thousands of customer records, this model scores every customer with a churn probability and risk tier — enabling the marketing team to act proactively.

---

##  Dataset

- **Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (via `ucimlrepo`)
- **Size:** 541,909 transactions → 397,884 after cleaning
- **Period:** December 2010 – December 2011
- **Customers:** 4,338 unique members (3,317 in the model's observation window)

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

##  Methodology

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

### 4. Modelling — Four Models, Increasing Complexity

Rather than jumping straight to the most powerful algorithm, models were trained in order of complexity to let the results justify the final choice:

| # | Model | Decision Boundary | Interpretability |
|---|---|---|---|
| 1 | Logistic Regression | Linear | High — direct coefficients |
| 2 | Decision Tree | Non-linear (if/else rules) | High — visualisable |
| 3 | Random Forest | Non-linear (ensemble vote) | Medium — feature importance |
| 4 | XGBoost | Non-linear (sequential boosting) | Medium — feature importance |

### 5. Model Comparison

| Model | ROC-AUC | Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | ~0.71 | ~67% | Strong linear baseline; highly interpretable |
| Decision Tree | ~0.68 | ~64% | Single tree overfits with only 3 features |
| Random Forest | ~0.70 | ~65% | Improves on single tree, still below XGBoost |
| **XGBoost** ✅ | **0.74** | **69%** | Best — captures non-linear recency threshold effects |

**XGBoost was selected for deployment**, but the gap to Logistic Regression is modest. In a business context where stakeholder trust and explainability matter more than a 3-point AUC gain, Logistic Regression remains a credible alternative — its coefficients directly show that higher recency increases churn risk and higher frequency decreases it.

### 6. Feature Enrichment Experiment
Added 5 additional features (`avg_order_value`, `total_items`, `unique_products`, `avg_quantity`, `revenue_per_visit`). AUC improved marginally to 0.742. Decision: retained the simpler 3-feature RFM model for interpretability and cleaner deployment.

---

##  Streamlit App Features

- **Single prediction** — input RFM values manually, get instant churn probability + risk tier
- **Batch prediction** — upload a CSV of customers, download scored results
- **Risk tiers** — 🟢 Low / 🟡 Medium / 🔴 High based on churn probability
- **Sample CSV** — downloadable template for testing batch mode

---

##  Project Structure

```
retail-churn-predictor/
├── app/
│   └── streamlit_app.py       # Streamlit web application
├── model/
│   ├── churn_model.pkl        # Trained XGBoost model
│   ├── scaler.pkl             # StandardScaler
│   └── feature_cols.pkl       # Feature column names
├── notebooks/
│   └── churn_analysis.ipynb   # Full EDA, 4-model comparison & evaluation
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
- **Recency is the strongest predictor** across all 4 models — customers inactive for 90+ days have significantly elevated churn risk
- **Logistic Regression performed competitively** (AUC 0.71) with full interpretability via its coefficients — a viable simpler alternative depending on business priorities
- **XGBoost generalises best** (AUC 0.74) by capturing the non-linear jump in churn risk around the 90-day recency threshold
- A simple 3-feature RFM model is sufficient — additional engineered features gave diminishing returns

---


## 👤 Author

**Badrul Amin** — Data Analyst | Aspiring Data Scientist

📎 [Portfolio](https://badrul-amin.github.io/portfolio/) · [LinkedIn](https://www.linkedin.com/in/badrulamins/) · [GitHub](https://github.com/badrul-amin)
