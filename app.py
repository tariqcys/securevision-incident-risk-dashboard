import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="SecureVision", layout="wide")

# ---------------- Custom Colors ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}
.kpi-box {
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
}
.kpi-blue { background: linear-gradient(135deg, #1e3c72, #2a5298); }
.kpi-red { background: linear-gradient(135deg, #c31432, #240b36); }
.kpi-orange { background: linear-gradient(135deg, #f7971e, #ffd200); color:black; }
.kpi-green { background: linear-gradient(135deg, #11998e, #38ef7d); }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("incident_model.pkl")

model = load_model()

# --------------------------------------------------
# AI Explanation
# --------------------------------------------------
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

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("## 🔐 SecureVision")
st.caption(f"Incident Risk Intelligence Platform — {date.today()}")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Alert Configuration")

alert_source = st.sidebar.selectbox(
    "Alert Source",
    ["SIEM", "AV", "email", "firewall", "third_party"]
)
alert_type = st.sidebar.selectbox(
    "Alert Type",
    ["phishing", "malware", "brute_force", "port_scan", "suspicious_login"]
)
severity = st.sidebar.selectbox(
    "Severity",
    ["Low", "Medium", "High", "Critical"]
)
asset_criticality = st.sidebar.slider("Asset Criticality", 1, 5, 3)
ioc_score = st.sidebar.slider("IOC Reputation Score", 0, 100, 60)
num_vulns = st.sidebar.slider("Number of Vulnerabilities", 0, 30, 5)
past_alerts = st.sidebar.slider("Past Alerts (30 days)", 0, 20, 3)

run = st.sidebar.button("🚀 Run Analysis")

# --------------------------------------------------
# Run Analysis
# --------------------------------------------------
if run:
    base = {
        "alert_source": alert_source,
        "alert_type": alert_type,
        "severity": severity,
        "has_ioc_from_haseen": 1 if ioc_score >= 70 else 0,
        "ioc_reputation_score": ioc_score,
        "num_vulns_asset": num_vulns,
        "cvss_max_score": round(np.random.uniform(0, 10), 1),
        "past_alerts_user_30d": past_alerts,
        "user_role": "employee",
        "asset_criticality": asset_criticality,
        "av_status": np.random.choice(["updated", "outdated", "not_installed"]),
        "detection_vector": np.random.choice(["network", "endpoint", "email_gateway"]),
        "email_indicator_score": np.random.randint(0, 100),
        "false_positive_rate_source": round(np.random.uniform(0, 1), 2),
    }

    df = pd.concat([pd.DataFrame([base])] * 25, ignore_index=True)
    df["time_index"] = range(len(df))

    df_enc = pd.get_dummies(df, drop_first=True)
    df_enc = df_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    preds = model.predict(df_enc)
    probs = model.predict_proba(df_enc)[:, 1]

    df["Prediction"] = preds
    df["Risk Score"] = probs
    df["AI Explanation"] = df.apply(explain_incident, axis=1)

    df["Risk Level"] = pd.cut(
        df["Risk Score"], [0, .3, .6, .8, 1],
        labels=["Low", "Medium", "High", "Critical"]
    )

    # ---------------- KPIs ----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-box kpi-blue'><h3>Total Alerts</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-box kpi-red'><h3>Incidents</h3><h2>{int((df['Prediction']==1).sum())}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-box kpi-orange'><h3>High / Critical</h3><h2>{int(df['Risk Level'].isin(['High','Critical']).sum())}</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-box kpi-green'><h3>Avg Risk</h3><h2>{round(df['Risk Score'].mean(),2)}</h2></div>", unsafe_allow_html=True)

    st.divider()

    # ---------------- Charts ----------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["Risk Score"].mean(),
        title={"text": "Incident Prediction"},
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.3], "color": "#4caf50"},
                {"range": [0.3, 0.6], "color": "#ffeb3b"},
                {"range": [0.6, 0.8], "color": "#ff9800"},
                {"range": [0.8, 1], "color": "#f44336"},
            ]
        }
    ))

    pie = px.pie(
        df,
        names="Risk Level",
        title="Risk Distribution",
        color="Risk Level",
        color_discrete_map={
            "Low": "#4caf50",
            "Medium": "#ffeb3b",
            "High": "#ff9800",
            "Critical": "#f44336"
        }
    )

    alert_counts = df["alert_type"].value_counts().reset_index()
    alert_counts.columns = ["Alert Type", "Count"]
    bar = px.bar(
        alert_counts,
        x="Alert Type",
        y="Count",
        title="Alert Types",
        color="Alert Type"
    )

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(gauge, use_container_width=True)
    col2.plotly_chart(pie, use_container_width=True)
    col3.plotly_chart(bar, use_container_width=True)

    # ---------------- Explanation ----------------
    st.subheader("🧠 AI Explanation")
    st.success(df.sort_values("Risk Score", ascending=False)["AI Explanation"].iloc[0])

    # ---------------- Table ----------------
    st.subheader("🚨 Alerts")
    st.dataframe(df.sort_values("Risk Score", ascending=False), use_container_width=True)

else:
    st.info("⬅️ Configure alerts and click **Run Analysis**")
