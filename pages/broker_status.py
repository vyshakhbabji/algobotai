import streamlit as st
from algobot.broker.alpaca import AlpacaBroker, _ALPACA_OK
import os

st.title("Broker Status (Alpaca)")

if not _ALPACA_OK:
    st.error("alpaca-py not installed.")
else:
    api_key = os.getenv('ALPACA_API_KEY') or st.text_input('API Key', type='password')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or st.text_input('Secret Key', type='password')
    paper = st.checkbox('Paper Trading', value=True)
    if st.button('Connect'):
        try:
            broker = AlpacaBroker(api_key, secret_key, paper=paper)
            acct = broker.get_account()
            st.success("Connected")
            st.json(acct)
            st.subheader("Positions")
            st.json(broker.get_positions())
        except Exception as e:
            st.error(str(e))
