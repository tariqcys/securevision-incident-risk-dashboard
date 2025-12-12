import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="SecureVision", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("incident_model.pkl")

model = load_model()

st.markdown("## 🔐 SecureVision")
st.caption(f"Incident Risk Intelligence Dashboard — {date.today()}")

uploaded = st.sidebar.file_uploader("Upload incidents CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    df_enc = pd.get_dummies(df, drop_first=True)
    df_enc = df_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    preds = model.predict(df_enc)
    probs = model.predict_proba(df_enc)[:, 1]

    df["Prediction"] = preds
    df["Risk Score"] = probs

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Alerts", len(df))
    c2.metric("Predicted Incidents", int((preds == 1).sum()))
    c3.metric("Avg Risk Score", round(probs.mean(), 2))

    st.divider()

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probs.mean(),
        title={"text": "Incident Prediction"},
        gauge={
            "axis": {"range":[0,1]},
            "steps":[
                {"range":[0,0.3],"color":"#8fd19e"},
                {"range":[0.3,0.6],"color":"#ffe082"},
                {"range":[0.6,0.8],"color":"#ffb74d"},
                {"range":[0.8,1],"color":"#e57373"},
            ]
        }
    ))

    df["Risk Level"] = pd.cut(
        df["Risk Score"],
        bins=[0,0.3,0.6,0.8,1],
        labels=["Low","Medium","High","Critical"]
    )

    pie = px.pie(df, names="Risk Level", title="Risk Distribution")

    if "alert_type" in df.columns:
        bar = px.bar(
            df["alert_type"].value_counts().reset_index(),
            x="index", y="alert_type", title="Alert Types"
        )
    else:
        bar = None

    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(gauge, use_container_width=True)
    col2.plotly_chart(pie, use_container_width=True)
    if bar:
        col3.plotly_chart(bar, use_container_width=True)

    st.subheader("🚨 Alerts Table")
    show_cols = [c for c in ["alert_type","severity","Prediction","Risk Score"] if c in df.columns]
    st.dataframe(df[show_cols].sort_values("Risk Score", ascending=False), use_container_width=True)
else:
    st.info("⬅️ Upload your incidents CSV to start")
