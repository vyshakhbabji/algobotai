import streamlit as st
import pandas as pd
from algobot.metrics.store import load_backtests

st.title("Backtest Comparisons")

bt = load_backtests()
if bt.empty:
    st.info("No backtests recorded yet.")
else:
    st.dataframe(bt)
    st.subheader("Summary")
    st.write(bt[['final_equity','total_return','max_dd','sharpe']].describe())
