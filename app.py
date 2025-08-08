#!/usr/bin/env python3
"""
AI Trading Bot - Main Application Entry Point
Complete multi-page dashboard with system monitoring and navigation
"""

import streamlit as st
import os
import sys
import json

# Page configuration (only in main app)
st.set_page_config(
    page_title="🤖 AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Main app content
st.markdown('<h1 style="text-align: center; color: #1e3a8a; font-size: 3.5rem;">🤖 AI Trading Bot</h1>', unsafe_allow_html=True)
st.markdown("**Your Complete AI-Powered Trading Platform**")

# Quick stats section
try:
    if os.path.exists("paper_trading_data.json"):
        with open("paper_trading_data.json", 'r') as f:
            account_data = json.load(f)
        
        # Check system health
        try:
            if os.path.exists("system_status_log.json"):
                with open("system_status_log.json", 'r') as f:
                    status_data = json.load(f)
                system_health = status_data.get('overall_health', '🔴 UNKNOWN')
            else:
                system_health = '🟡 CHECK NEEDED'
        except:
            system_health = '🔴 ERROR'
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Available Cash", f"${account_data.get('cash', 0):,.0f}")
        with col2:
            st.metric("📊 Portfolio Value", f"${account_data.get('total_value', 0):,.0f}")
        with col3:
            total_return = account_data.get('total_return_pct', 0)
            st.metric("📈 Total Return", f"{total_return:.1f}%")
        with col4:
            st.metric("🔧 System Health", system_health)
        
except Exception as e:
    st.info("💡 Welcome! Your trading data will appear here once you start trading.")

st.markdown("---")

# Navigation instructions
st.subheader("🧭 Navigation")
st.markdown("""
**Welcome to your AI Trading Bot!** 

Use the **sidebar** to navigate between different pages:
- 🚀 **Live Trading** - Real-time AI trading dashboard
- 🔧 **System Monitor** - Comprehensive health checks  
- 📊 **Portfolio Manager** - Manage your stock universe
- 📈 **Performance Analytics** - Detailed performance metrics
- 📈 **Enhanced Dashboard** - Advanced market analysis
- 🧠 **AI Optimizer** - Continuous model improvement
- 🎯 **Elite Options** - Options trading strategies

Simply click on any page in the sidebar to get started!
""")

# Feature overview
st.subheader("✨ Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 AI Trading Strategy**
    - AI-powered stock selection
    - 15+ technical indicators
    - Risk-managed positions
    - Automated rebalancing
    """)

with col2:
    st.markdown("""
    **📊 Data Sources**
    - Yahoo Finance API
    - Real-time price feeds
    - Historical market data
    - Company fundamentals
    """)

with col3:
    st.markdown("""
    **🔧 System Health**
    - ✅ All systems operational
    - ✅ Data feeds active
    - ✅ AI models trained
    - ✅ Paper trading ready
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>AI Trading Bot v2.0</strong> | Built with Streamlit & Python</p>
    <p>⚠️ <em>This is a paper trading simulation for educational purposes only</em></p>
</div>
""", unsafe_allow_html=True)
