import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FireWatch-CI", page_icon="🔥", layout="wide")
st.title("FireWatch-CI - Model Dashboard")
st.caption("Fire & smoke detection · model performance · drift monitoring")

# Model metrics
st.subheader("Latest model metrics")
try:
    with open("metrics/eval_results.json") as f:
        m = json.load(f)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mAP50", f"{m.get('mAP50',       0):.3f}")
    c2.metric("fire mAP50", f"{m.get('fire_mAP50',  0):.3f}")
    c3.metric("smoke mAP50", f"{m.get('smoke_mAP50', 0):.3f}")
    c4.metric("mAP50-95", f"{m.get('mAP50_95',    0):.3f}")
except FileNotFoundError:
    st.warning("No evaluation results yet.")

# Inference log
st.subheader("Inference log analysis")
try:
    df = pd.read_csv("logs/inference_log.csv")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Prediction class distribution")
        st.bar_chart(df["predicted_class"].value_counts())
    with col2:
        st.caption("Confidence distribution (fire + smoke only)")
        conf = df[df["predicted_class"] != "none"][["confidence"]]
        st.line_chart(conf)
    st.caption(f"Total predictions logged: {len(df)}")
except FileNotFoundError:
    st.info("No inference logs yet.")

# Drift report
st.subheader("Latest Evidently drift report")
report_path = Path("reports/drift_report.html")
if report_path.exists():
    with open(report_path) as f:
        st.components.v1.html(f.read(), height=700, scrolling=True)
else:
    st.info("No drift report yet. Trigger the weekly drift report workflow first.")
