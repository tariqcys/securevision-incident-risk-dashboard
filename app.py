import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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
explainer = shap.TreeExplainer(model)

# --------------------------------------------------
# SHAP Text Explanation
# --------------------------------------------------
def generate_incident_reason(shap_row, feature_names, top_n=3):
    impact = list(zip(feature_names, shap_row))
    impact = sorted(impact, key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, val in impact[:top_n]:
        direction = "⬆️ زاد الخطورة" if val > 0 else "⬇️ خفّض الخطورة"
        clean_feat = feat.replace("_", " ").title()
        reasons.append(f"{clean_feat} {direction}")

    return reasons

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

    # simulate batch of alerts
    df = pd.concat([pd.DataFrame([base])] * 25, ignore_index=True)

    # Encode features
    X = pd.get_dummies(df, drop_first=True)
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predictions
    df["Prediction"] = model.predict(X)
    df["Risk Score"] = model.predict_proba(X)[:, 1]

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
        gauge={"axis": {"range": [0, 1]}}
    ))

    pie = px.pie(df, names="Risk Level", title="Risk Distribution")

    # ✅ FIXED BAR CHART
    alert_counts = df["alert_type"].value_counts().reset_index()
    alert_counts.columns = ["Alert Type", "Count"]

    bar = px.bar(
        alert_counts,
        x="Alert Type",
        y="Count",
        title="Alert Types"
    )

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(gauge, use_container_width=True)
    col2.plotly_chart(pie, use_container_width=True)
    col3.plotly_chart(bar, use_container_width=True)

    # ---------------- SHAP Visual Explanation ----------------
    st.subheader("🧠 Why this alert is risky")

    top_idx = df["Risk Score"].idxmax()
    shap_values = explainer.shap_values(X.loc[[top_idx]])

    reasons = generate_incident_reason(
        shap_values[1][0],
        X.columns
    )

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ffffff, #f1f5f9);
        border-left: 6px solid #6366f1;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-size: 16px;
    ">
    <b>Top contributing factors:</b><br><br>
    • {"<br>• ".join(reasons)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔥 Overall Risk Level")
    st.progress(float(df.loc[top_idx, "Risk Score"]))
    st.caption(f"Incident Risk Score: **{round(df.loc[top_idx, 'Risk Score'], 2)}**")

    with st.expander("📊 Feature Impact Breakdown"):
        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "Impact": shap_values[1][0]
        }).sort_values("Impact", key=abs, ascending=False).head(8)

        fig, ax = plt.subplots()
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df["Impact"]]
        ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
        ax.set_title("SHAP Feature Impact")
        ax.invert_yaxis()
        st.pyplot(fig)

    # ---------------- Table ----------------
    st.subheader("🚨 Alerts")
    st.dataframe(df.sort_values("Risk Score", ascending=False), use_container_width=True)

else:
    st.info("⬅️ Configure alerts and click **Run Analysis**")
