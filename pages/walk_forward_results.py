import streamlit as st
import pandas as pd
from algobot.metrics.store import load_walk_forward

st.title("Walk-Forward Evaluation Results")

symbol = st.text_input("Filter symbol (optional)")

df = load_walk_forward(symbol or None)
if df.empty:
    st.warning("No walk-forward metrics recorded yet.")
else:
    st.dataframe(df.tail(500))
    agg = df.groupby('symbol')['directional_accuracy'].mean().sort_values(ascending=False)
    st.subheader("Avg Directional Accuracy")
    st.bar_chart(agg)
