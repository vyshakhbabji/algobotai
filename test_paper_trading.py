#!/usr/bin/env python3
"""
Simple Paper Trading Test - Verify Dashboard is Working
"""

import streamlit as st
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🎯 Paper Trading Test",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Paper Trading System Test")
st.markdown("**Verifying Dashboard Functionality**")

# Check if our data files exist
data_files = {
    "Account Data": "paper_trading_account.json",
    "Positions": "paper_trading_positions.json", 
    "Trade History": "paper_trading_trades.json"
}

st.subheader("📁 Data Files Status")
for name, filename in data_files.items():
    if os.path.exists(filename):
        st.success(f"✅ {name}: {filename} - Found")
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if filename == "paper_trading_account.json":
                    st.write(f"   - Initial Capital: ${data.get('initial_capital', 0):,.2f}")
                    st.write(f"   - Current Cash: ${data.get('cash', 0):,.2f}")
                    st.write(f"   - Total Return: ${data.get('total_return', 0):,.2f}")
                elif filename == "paper_trading_positions.json":
                    st.write(f"   - Active Positions: {len(data)}")
                elif filename == "paper_trading_trades.json":
                    st.write(f"   - Total Trades: {len(data)}")
        except Exception as e:
            st.error(f"   - Error reading file: {str(e)}")
    else:
        st.warning(f"⚠️ {name}: {filename} - Not Found")

# Test the paper trading engine directly
st.subheader("🔧 Engine Test")

try:
    # Import and test the engine
    from live_paper_trading import PaperTradingEngine
    
    engine = PaperTradingEngine()
    st.success("✅ Paper Trading Engine loaded successfully!")
    
    # Display current account status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Initial Capital",
            f"${engine.account['initial_capital']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Current Cash", 
            f"${engine.account['cash']:,.2f}"
        )
    
    with col3:
        current_value = engine.update_portfolio_value()
        st.metric(
            "Portfolio Value",
            f"${current_value:,.2f}",
            f"${engine.account['total_return']:,.2f}"
        )
    
    # Test price fetching
    st.subheader("📈 Price Test")
    test_symbols = ["AAPL", "GOOGL", "MSFT"]
    
    price_data = []
    for symbol in test_symbols:
        price = engine.get_current_price(symbol)
        price_data.append({
            "Symbol": symbol,
            "Price": f"${price:.2f}" if price else "Failed",
            "Status": "✅ Success" if price else "❌ Failed"
        })
    
    import pandas as pd
    price_df = pd.DataFrame(price_data)
    st.dataframe(price_df, use_container_width=True)
    
    # Current positions
    if engine.positions:
        st.subheader("📊 Current Positions")
        positions_df = engine.get_position_details()
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("💡 No current positions - Starting fresh with $10,000!")
    
    # Recent trades
    if engine.trades:
        st.subheader("📋 Recent Trades")
        recent_trades = pd.DataFrame(engine.trades[-10:])  # Last 10 trades
        st.dataframe(recent_trades, use_container_width=True)
    else:
        st.info("💡 No trades yet - Ready to start trading!")

except Exception as e:
    st.error(f"❌ Engine Error: {str(e)}")
    st.code(f"Error details: {str(e)}")

# System info
st.subheader("💻 System Info")
st.write(f"**Current Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write(f"**Working Directory**: {os.getcwd()}")
st.write(f"**Python Path**: {os.sys.executable}")

# Manual refresh button
if st.button("🔄 Refresh Test", type="primary"):
    st.rerun()

st.markdown("---")
st.markdown("**This is a test dashboard to verify the paper trading system is working correctly.**")
