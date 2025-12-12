import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="SecureVision",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return joblib.load("incident_model.pkl")

model = load_model()

# ---------------- AI Explanation ----------------
def explain_incident(row):
    reasons = []
    if row["severity"] in ["High", "Critical"]:
        reasons.append(f"Severity {row['severity']}")
    if row["ioc_reputation_score"] >= 70:
        reasons.append(f"IOC {int(row['ioc_reputation_score'])}")
    if row["num_vulns_asset"] >= 10:
        reasons.append(f"{int(row['num_vulns_asset'])} vulnerabilities")
    if row["asset_criticality"] >= 4:
        reasons.append("critical asset")
    if row["past_alerts_user_30d"] >= 7:
        reasons.append("repeated alerts")
    return " + ".join(reasons) if reasons else "No high-risk indicators detected"

# ---------------- Header ----------------
st.markdown("## 🔐 SecureVision")
st.caption(f"Incident Risk Intelligence Dashboard — {date.today()}")

st.info(
    "This platform analyzes security alerts and predicts real incidents "
    "using Machine Learning. Click the button below to run a demo analysis."
)

# ---------------- Demo Button ----------------
if st.button("🚀 Run Demo Analysis"):

    # -------- Generate Demo Data --------
    df = pd.DataFrame({
        "alert_type": np.random.choice(
            ["phishing", "malware", "brute_force", "port_scan"], 40),
        "severity": np.random.choice(
            ["Low", "Medium", "High", "Critical"], 40),
        "ioc_reputation_score": np.random.randint(10, 100, 40),
        "num_vulns_asset": np.random.randint(0, 25, 40),
        "asset_criticality": np.random.randint(1, 5, 40),
        "past_alerts_user_30d": np.random.randint(0, 15, 40),
    })

    # -------- Encode --------
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    preds = model.predict(df_encoded)
    probs = model.predict_proba(df_encoded)[:, 1]

    df["Prediction"] = preds
    df["Risk Score"] = probs
    df["AI Explanation"] = df.apply(explain_incident, axis=1)

    # -------- KPIs --------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Alerts", len(df))
    c2.metric("Predicted Incidents", int((preds == 1).sum()))
    c3.metric("Average Risk Score", round(probs.mean(), 2))

    st.divider()

    # -------- Gauge --------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probs.mean(),
        title={"text": "Incident Prediction"},
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.3], "color": "#a5d6a7"},
                {"range": [0.3, 0.6], "color": "#fff59d"},
                {"range": [0.6, 0.8], "color": "#ffcc80"},
                {"range": [0.8, 1], "color": "#ef9a9a"},
            ]
        }
    ))

    # -------- Risk Distribution --------
    df["Risk Level"] = pd.cut(
        df["Risk Score"],
        bins=[0, 0.3, 0.6, 0.8, 1],
        labels=["Low", "Medium", "High", "Critical"]
    )

    pie = px.pie(df, names="Risk Level", title="Risk Distribution")

    bar = px.bar(
        df["alert_type"].value_counts().reset_index(),
        x="index", y="alert_type",
        title="Alert Types"
    )

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(gauge, use_container_width=True)
    col2.plotly_chart(pie, use_container_width=True)
    col3.plotly_chart(bar, use_container_width=True)

    st.subheader("🧠 AI Explanation")
    st.write(df.sort_values("Risk Score", ascending=False)["AI Explanation"].iloc[0])

    st.subheader("🚨 Alerts")
    st.dataframe(
        df.sort_values("Risk Score", ascending=False),
        use_container_width=True
    )
