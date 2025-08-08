#!/usr/bin/env python3
"""
AI Paper Trading Bot - Main Application Entry Point
Deployable version of the live paper trading dashboard
"""

import streamlit as st
import os
import sys

# Set page config
st.set_page_config(
    page_title="🚀 AI Paper Trading Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the main trading dashboard
try:
    from pages.live_paper_trading import main as trading_dashboard
    
    # Run the trading dashboard
    if __name__ == "__main__":
        trading_dashboard()
        
except Exception as e:
    st.error(f"Error loading trading dashboard: {str(e)}")
    st.write("Please ensure all dependencies are installed.")
    st.code("pip install -r requirements.txt")
