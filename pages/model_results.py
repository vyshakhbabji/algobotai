#!/usr/bin/env python3
"""Streamlit page: Model Results & Analytics."""
import streamlit as st
import pandas as pd
from pathlib import Path
from algobot.analysis.model_scan import latest_scan, scan_universe

st.set_page_config(page_title="Model Results", page_icon="📊", layout="wide")

st.title("📊 Model Results & Analytics")
st.markdown("Review latest ensemble predictions, model test metrics, and distribution analytics.")

col_run, col_status = st.columns([1,3])
with col_run:
    if st.button("Run Fresh Scan", type="primary"):
        with st.spinner("Running model scan across universe..."):
            df_new = scan_universe()
            st.success(f"Scan complete. {len(df_new)} rows.")

df = latest_scan()
if df is None or df.empty:
    st.info("No scan found. Click 'Run Fresh Scan' to generate results.")
    st.stop()

st.subheader("Universe Summary")
meta_cols = ["symbol", "status", "signal", "expected_return_pct", "confidence", "quality", "best_r2", "best_direction", "avg_direction"]
try:
    st.dataframe(df[meta_cols].sort_values("symbol"), use_container_width=True, height=480)
except KeyError:
    st.dataframe(df, use_container_width=True)

valid = df[df['status'] == 'ok'] if 'status' in df.columns else df
if not valid.empty:
    st.subheader("Distributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predictions Available", f"{len(valid)}")
        if 'signal' in valid.columns:
            st.bar_chart(valid['signal'].value_counts())
    with c2:
        if 'expected_return_pct' in valid.columns:
            st.caption("Expected Return % (Histogram)")
            st.bar_chart(valid['expected_return_pct'])
    with c3:
        if 'confidence' in valid.columns and 'expected_return_pct' in valid.columns:
            st.caption("Confidence vs Expected Return")
            scatter_df = valid[['confidence', 'expected_return_pct']].dropna()
            if not scatter_df.empty:
                st.scatter_chart(scatter_df, x='confidence', y='expected_return_pct')

    if {'best_r2','best_direction','avg_direction'}.issubset(valid.columns):
        st.subheader("Model Quality Metrics")
        qual_cols = [c for c in ["symbol", "best_r2", "best_direction", "avg_direction", "models_used", "model_count"] if c in valid.columns]
        st.dataframe(valid[qual_cols].sort_values("best_direction", ascending=False), use_container_width=True)

    st.subheader("Filter & Explore")
    min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.01) if 'confidence' in valid.columns else 0.0
    min_dir = st.slider("Min Best Directional Accuracy", 0.0, 1.0, 0.5, 0.01) if 'best_direction' in valid.columns else 0.0
    filt = valid
    if 'confidence' in valid.columns:
        filt = filt[filt['confidence'] >= min_conf]
    if 'best_direction' in valid.columns:
        filt = filt[filt['best_direction'] >= min_dir]
    st.write(f"Filtered Symbols ({len(filt)}):")
    display_cols = [c for c in meta_cols if c in filt.columns]
    st.dataframe(filt[display_cols], use_container_width=True)
else:
    st.warning("No valid predictions in latest scan.")
