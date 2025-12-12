import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# -------------------------------------------------
# Page config (Light / White background)
# -------------------------------------------------
st.set_page_config(
    page_title="SecureVision",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme (CSS)
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: #000000;
    }
    .stApp {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Load model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("incident_model.pkl")

model = load_model()

# -------------------------------------------------
# AI Explanation function
# -------------------------------------------------
def explain_incident(row):
    reasons = []

    if "severity" in row and row["severity"] in ["High", "Critical"]:
        reasons.append(f"Severity {row['severity']}")

    if "ioc_reputation_score" in row and row["ioc_reputation_score"] >= 70:
        reasons.append(f"IOC {int(row['ioc_reputation_score'])}")

    if "num_vulns_asset" in row and row["num_vulns_asset"] >= 10:
        reasons.append(f"{int(row['num_vulns_asset'])} vulnerabilities")

    if "asset_criticality" in row and row["asset_criticality"] >= 4:
        reasons.append("critical asset")

    if "past_alerts_user_30d" in row and row["past_alerts_user_30d"] >= 7:
        reasons.append("repeated alerts")

    return " + ".join(reasons) if reasons else "No high-risk indicators detected"

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("## 🔐 SecureVision")
st.caption(f"Incident Risk Intelligence Dashboard — {date.today()}")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Upload Alerts Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload incidents CSV file",
    type=["csv"]
)

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Encode input to match model features
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predictions
    preds = model.predict(df_encoded)
    probs = model.predict_proba(df_encoded)[:, 1]

    df["Prediction"] = preds
    df["Risk Score"] = probs

    # AI Explanation
    df["AI Explanation"] = df.apply(explain_incident, axis=1)

    # -------------------------------------------------
    # KPIs
    # -------------------------------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Alerts", len(df))
    k2.metric("Incidents Predicted", int((preds == 1).sum()))
    k3.metric("Average Risk Score", round(probs.mean(), 2))

    st.divider()

    # -------------------------------------------------
    # Gauge (Incident Prediction)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Risk Distribution
    # -------------------------------------------------
    df["Risk Level"] = pd.cut(
        df["Risk Score"],
        bins=[0, 0.3, 0.6, 0.8, 1],
        labels=["Low", "Medium", "High", "Critical"]
    )

    pie = px.pie(
        df,
        names="Risk Level",
        title="Risk Distribution"
    )

    # -------------------------------------------------
    # Alert Types
    # -------------------------------------------------
    bar = None
    if "alert_type" in df.columns:
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
    if bar:
        col3.plotly_chart(bar, use_container_width=True)

    # -------------------------------------------------
    # AI Explanation (Top Incident)
    # -------------------------------------------------
    st.subheader("🧠 AI Explanation")
    st.write(
        df.sort_values("Risk Score", ascending=False)["AI Explanation"].iloc[0]
    )

    st.divider()

    # -------------------------------------------------
    # Alerts Table
    # -------------------------------------------------
    show_cols = [
        c for c in
        ["alert_type", "severity", "Prediction", "Risk Score", "AI Explanation"]
        if c in df.columns
    ]

    st.subheader("🚨 Alerts")
    st.dataframe(
        df[show_cols].sort_values("Risk Score", ascending=False),
        use_container_width=True
    )

else:
    st.info("⬅️ Upload a CSV file to start the analysis")
