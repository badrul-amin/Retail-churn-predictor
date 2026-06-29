import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config
st.set_page_config(
    page_title="Retail Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

# ── Load model artifacts
@st.cache_resource
def load_model():
    base = os.path.join(os.path.dirname(__file__), '..', 'model')
    with open(os.path.join(base, 'churn_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, 'feature_cols.pkl'), 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_cols = load_model()

# ── Header
st.title("🛒 Retail Member Churn Predictor")
st.markdown("Predict which customers are at risk of churning using their **RFM behaviour**.")
st.divider()

# ── Two modes: Single prediction vs Batch CSV
tab1, tab2 = st.tabs(["🔍 Single Customer", "📂 Batch Prediction (CSV)"])

# ══════════════════════════════════════════
# TAB 1 — Single Customer
# ══════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer RFM Values")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency   = st.number_input("Recency (days since last purchase)",
                                     min_value=0, max_value=500, value=45,
                                     help="Lower = more recent = less likely to churn")
    with col2:
        frequency = st.number_input("Frequency (number of orders)",
                                     min_value=1, max_value=300, value=5,
                                     help="Higher = more engaged customer")
    with col3:
        monetary  = st.number_input("Monetary (total spend £)",
                                     min_value=0.0, max_value=300000.0, value=500.0,
                                     help="Higher = more valuable customer")

    if st.button("🔮 Predict Churn Risk", type="primary"):
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=feature_cols)
        input_sc = scaler.transform(input_df)
        pred     = model.predict(input_sc)[0]
        prob     = model.predict_proba(input_sc)[0][1]

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            if pred == 1:
                st.error(f"⚠️ **HIGH CHURN RISK** — {prob*100:.1f}% probability")
                st.markdown("This customer is likely to lapse. Consider a re-engagement campaign.")
            else:
                st.success(f"✅ **LOW CHURN RISK** — {prob*100:.1f}% churn probability")
                st.markdown("This customer appears active. Focus on retention rewards.")

        with col_b:
            # Risk gauge using progress bar
            st.markdown("**Churn Probability**")
            st.progress(float(prob))
            st.caption(f"{prob*100:.1f}% risk score")

            risk_tier = "🔴 High" if prob > 0.7 else "🟡 Medium" if prob > 0.4 else "🟢 Low"
            st.metric("Risk Tier", risk_tier)

# ══════════════════════════════════════════
# TAB 2 — Batch CSV Upload
# ══════════════════════════════════════════
with tab2:
    st.subheader("Upload Customer RFM CSV")
    st.markdown("CSV must have columns: `recency`, `frequency`, `monetary`")

    # Sample CSV download
    sample = pd.DataFrame({
        'CustomerID': [12346, 12347, 12348],
        'recency':    [326, 2, 75],
        'frequency':  [1, 7, 4],
        'monetary':   [77183.60, 4310.00, 1797.24]
    })
    st.download_button("⬇️ Download Sample CSV", 
                        sample.to_csv(index=False), 
                        "sample_customers.csv", 
                        "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=['csv'])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_upload)} customers loaded**")
        st.dataframe(df_upload.head(), use_container_width=True)

        if all(c in df_upload.columns for c in feature_cols):
            X_up     = df_upload[feature_cols]
            X_up_sc  = scaler.transform(X_up)
            probs    = model.predict_proba(X_up_sc)[:, 1]
            preds    = model.predict(X_up_sc)

            df_upload['churn_probability'] = (probs * 100).round(1)
            df_upload['churn_prediction']  = preds
            df_upload['risk_tier'] = pd.cut(probs,
                                             bins=[0, 0.4, 0.7, 1.0],
                                             labels=['🟢 Low', '🟡 Medium', '🔴 High'])

            st.divider()
            st.subheader("📊 Results")

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Customers", len(df_upload))
            m2.metric("At-Risk (Churned)", int(preds.sum()))
            m3.metric("Avg Churn Probability", f"{probs.mean()*100:.1f}%")

            # Risk distribution
            st.markdown("**Risk Tier Breakdown**")
            tier_counts = df_upload['risk_tier'].value_counts()
            st.bar_chart(tier_counts)

            # Full results table
            st.markdown("**Full Results**")
            st.dataframe(df_upload.sort_values('churn_probability', ascending=False),
                         use_container_width=True)

            # Download results
            st.download_button("⬇️ Download Results CSV",
                                df_upload.to_csv(index=False),
                                "churn_predictions.csv",
                                "text/csv")
        else:
            st.error(f"❌ CSV must contain columns: {feature_cols}")

# ── Footer
st.divider()
st.caption("Built with XGBoost · Trained on UCI Online Retail Dataset · ROC-AUC: 0.74")
