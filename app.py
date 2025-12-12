import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# ---------------- Page Config ----------------
st.set_page_config(page_title="SecureVision", layout="wide")

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
    "Select alert parameters on the left, then click **Run Analysis** "
    "to simulate SOC incident detection."
)

# ---------------- Sidebar Filters (LIKE IMAGE) ----------------
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

user_role = st.sidebar.selectbox(
    "User Role",
    ["employee", "admin", "manager"]
)

asset_criticality = st.sidebar.slider(
    "Asset Criticality", 1, 5, 3
)

ioc_score = st.sidebar.slider(
    "IOC Reputation Score", 0, 100, 60
)

vulns = st.sidebar.slider(
    "Number of Vulnerabilities", 0, 30, 5
)

past_alerts = st.sidebar.slider(
    "Past Alerts (30 days)", 0, 20, 3
)

run = st.sidebar.button("🚀 Run Analysis")

# ---------------- Run Analysis ----------------
if run:

    # -------- Create DataFrame from selections --------
    df = pd.DataFrame([{
        "alert_source": alert_source,
        "alert_type": alert_type,
        "severity": severity,
        "has_ioc_from_haseen": 1 if ioc_score >= 70 else 0,
        "ioc_reputation_score": ioc_score,
        "num_vulns_asset": vulns,
        "cvss_max_score": round(np.random.uniform(0, 10), 1),
        "past_alerts_user_30d": past_alerts,
        "user_role": user_role,
        "asset_criticality": asset_criticality,
        "av_status": np.random.choice(["updated", "outdated", "not_installed"]),
        "detection_vector": np.random.choice(["network", "endpoint", "email_gateway"]),
        "email_indicator_score": np.random.randint(0, 100),
        "false_positive_rate_source": round(np.random.uniform(0, 1), 2),
    }])

    # Duplicate row to simulate multiple alerts
    df = pd.concat([df]*25, ignore_index=True)

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

    # ---------------- KPIs ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Alerts", len(df))
    c2.metric("Predicted Incidents", int((preds == 1).sum()))
    c3.metric("Risk Score", round(probs.mean(), 2))

    st.divider()

    # ---------------- Gauge ----------------
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

    # ---------------- Risk Distribution ----------------
    df["Risk Level"] = pd.cut(
        df["Risk Score"],
        bins=[0, 0.3, 0.6, 0.8, 1],
        labels=["Low", "Medium", "High", "Critical"]
    )

    pie = px.pie(df, names="Risk Level", title="Risk Distribution")

    bar = px.bar(
        df["alert_type"].value_counts().reset_index(),
        x="index",
        y="alert_type",
        title="Alert Types",
        labels={"index": "Alert Type", "alert_type": "Count"}
    )

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(gauge, use_container_width=True)
    col2.plotly_chart(pie, use_container_width=True)
    col3.plotly_chart(bar, use_container_width=True)

    # ---------------- Explanation ----------------
    st.subheader("🧠 AI Explanation")
    st.write(df.sort_values("Risk Score", ascending=False)["AI Explanation"].iloc[0])

    # ---------------- Table ----------------
    st.subheader("🚨 Alerts")
    st.dataframe(
        df.sort_values("Risk Score", ascending=False),
        use_container_width=True
    )

else:
    st.warning("⬅️ Select alert options from the left and click **Run Analysis**")
