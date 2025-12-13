import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import date

# ==================================================
# Page Config
# ==================================================
st.set_page_config(page_title="SecureVision", layout="wide")

# ==================================================
# Style
# ==================================================
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,.08);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Load Model
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load("incident_model.pkl")

model = load_model()
explainer = shap.TreeExplainer(model)

# ==================================================
# Header
# ==================================================
st.title("🔐 SecureVision")
st.caption(f"Incident Risk Intelligence Platform — {date.today()}")

# ==================================================
# Sidebar Inputs
# ==================================================
st.sidebar.header("Alert Configuration")

alert_source = st.sidebar.selectbox(
    "Alert Source", ["SIEM", "AV", "email", "firewall", "third_party"]
)
alert_type = st.sidebar.selectbox(
    "Alert Type", ["phishing", "malware", "brute_force", "port_scan", "suspicious_login"]
)
severity = st.sidebar.selectbox(
    "Severity", ["Low", "Medium", "High", "Critical"]
)
asset_criticality = st.sidebar.slider("Asset Criticality", 1, 5, 3)
ioc_score = st.sidebar.slider("IOC Reputation Score", 0, 100, 60)
num_vulns = st.sidebar.slider("Number of Vulnerabilities", 0, 30, 5)
past_alerts = st.sidebar.slider("Past Alerts (30 days)", 0, 20, 3)

run = st.sidebar.button("🚀 Run Analysis")

# ==================================================
# Helper: Get SHAP row robustly (works across SHAP versions)
# ==================================================
def extract_shap_row(shap_values):
    """
    Returns a 1D SHAP vector for the positive class (if available),
    otherwise returns the single output vector.
    Handles list outputs and different array shapes safely.
    """
    # Case 1: list => usually multiclass or binary returning [class0, class1]
    if isinstance(shap_values, list):
        # Prefer class 1 if exists
        if len(shap_values) > 1:
            arr = shap_values[1]
        else:
            arr = shap_values[0]
    else:
        arr = shap_values

    arr = np.array(arr)

    # arr could be: (1, n_features) or (n_features,) or (n_samples, n_features)
    if arr.ndim == 3:
        # sometimes: (classes, samples, features) - take first sample
        arr = arr[0]

    if arr.ndim == 2:
        # take first sample
        arr = arr[0]

    # now flatten to 1D
    return arr.flatten()

def build_reason(shap_row_1d, columns, top_n=3):
    """
    Build top-N reason strings + dataframe for plotting.
    Ensures shap_row is 1D.
    """
    shap_row_1d = np.array(shap_row_1d).flatten()

    df_imp = pd.DataFrame({
        "feature": list(columns),
        "impact": shap_row_1d
    })

    df_imp = df_imp.sort_values("impact", key=abs, ascending=False).head(top_n)

    reasons = []
    for _, r in df_imp.iterrows():
        arrow = "⬆️" if r["impact"] > 0 else "⬇️"
        reasons.append(f"{r['feature'].replace('_',' ').title()} {arrow}")

    return reasons, df_imp

# ==================================================
# Run Analysis
# ==================================================
if run:
    # ---- Create simulated alert batch
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

    # ---- Encode
    X = pd.get_dummies(df, drop_first=True)
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    # ---- Predict
    df["Prediction"] = model.predict(X)
    df["Risk Score"] = model.predict_proba(X)[:, 1]

    df["Risk Level"] = pd.cut(
        df["Risk Score"],
        [0, .3, .6, .8, 1],
        labels=["Low", "Medium", "High", "Critical"]
    )

    # ==================================================
    # KPIs
    # ==================================================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", len(df))
    c2.metric("Incidents", int((df["Prediction"] == 1).sum()))
    c3.metric("High / Critical", int(df["Risk Level"].isin(["High", "Critical"]).sum()))
    c4.metric("Avg Risk", round(df["Risk Score"].mean(), 2))

    st.divider()

    # ==================================================
    # Charts
    # ==================================================
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(df["Risk Score"].mean()),
            gauge={"axis": {"range": [0, 1]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.plotly_chart(px.pie(df, names="Risk Level", title="Risk Distribution"),
                        use_container_width=True)

    with col3:
        ac = df["alert_type"].value_counts().reset_index()
        ac.columns = ["Alert Type", "Count"]
        st.plotly_chart(px.bar(ac, x="Alert Type", y="Count", title="Alert Types"),
                        use_container_width=True)

    # ==================================================
    # SHAP Explanation
    # ==================================================
    st.subheader("🧠 AI Decision Explanation")

    top_idx = df["Risk Score"].idxmax()
    x_one = X.loc[[top_idx]]

    shap_values = explainer.shap_values(x_one)
    shap_row = extract_shap_row(shap_values)

    reasons, top_df = build_reason(shap_row, X.columns, top_n=3)

    st.markdown(f"""
    <div class="card">
    <b>Top contributing factors:</b><br><br>
    • {"<br>• ".join(reasons)}
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(df.loc[top_idx, "Risk Score"]))
    st.caption(f"Incident Risk Score: **{round(float(df.loc[top_idx, 'Risk Score']), 2)}**")

    # Clear SHAP bar chart
    with st.expander("📊 Feature Impact Breakdown (Top 8)"):
        full_imp = pd.DataFrame({
            "feature": list(X.columns),
            "impact": shap_row
        }).sort_values("impact", key=abs, ascending=False).head(8)

        fig2, ax = plt.subplots()
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in full_imp["impact"]]
        ax.barh(full_imp["feature"], full_imp["impact"], color=colors)
        ax.set_title("SHAP Feature Impact")
        ax.invert_yaxis()
        st.pyplot(fig2)

    # ==================================================
    # Table
    # ==================================================
    st.subheader("🚨 Alerts")
    st.dataframe(df.sort_values("Risk Score", ascending=False), use_container_width=True)

else:
    st.info("⬅️ Configure inputs and click **Run Analysis**")
 
