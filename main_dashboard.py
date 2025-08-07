#!/usr/bin/env python3
"""
AI Trading Bot - Main Navigation Hub
Multi-page Streamlit application for comprehensive trading management
"""

import streamlit as st
import subprocess
import sys
import os

# Page configuration
st.set_page_config(
    page_title="🤖 AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .nav-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stats-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main navigation interface"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Trading Bot</h1>', unsafe_allow_html=True)
    st.markdown("**Your Complete AI-Powered Trading Platform**")
    
    # Quick stats
    try:
        # Try to load account data for quick stats
        import json
        if os.path.exists("paper_trading_data.json"):
            with open("paper_trading_data.json", 'r') as f:
                account_data = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="stats-card">
                    <h3>${account_data.get('cash', 0):,.0f}</h3>
                    <p>Available Cash</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="stats-card">
                    <h3>${account_data.get('total_value', 0):,.0f}</h3>
                    <p>Portfolio Value</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                total_return = account_data.get('total_return_pct', 0)
                st.markdown(f'''
                <div class="stats-card">
                    <h3>{total_return:.1f}%</h3>
                    <p>Total Return</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                # Count portfolio stocks
                portfolio_count = 0
                if os.path.exists("portfolio_universe.json"):
                    with open("portfolio_universe.json", 'r') as f:
                        portfolio_data = json.load(f)
                        portfolio_count = len(portfolio_data.get('stocks', []))
                
                st.markdown(f'''
                <div class="stats-card">
                    <h3>{portfolio_count}</h3>
                    <p>Tracked Stocks</p>
                </div>
                ''', unsafe_allow_html=True)
        
    except:
        pass  # If files don't exist, skip stats
    
    st.markdown("---")
    
    # Navigation menu
    st.subheader("🧭 Choose Your Dashboard")
    
    # Main navigation cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="nav-card">
            <h2>🚀 Live Trading Dashboard</h2>
            <p>Real-time AI trading with $10,000 capital</p>
            <p>✅ Live market data every 5 minutes</p>
            <p>✅ AI signals and automatic trades</p>
            <p>✅ Portfolio tracking and performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Live Trading", type="primary", use_container_width=True):
            st.switch_page("live_paper_trading.py")
    
    with col2:
        st.markdown("""
        <div class="nav-card">
            <h2>📈 Enhanced Dashboard</h2>
            <p>Advanced market analysis and scanning</p>
            <p>✅ Market scanner and top movers</p>
            <p>✅ Technical analysis charts</p>
            <p>✅ Advanced AI signals</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Enhanced Dashboard", type="primary", use_container_width=True):
            st.switch_page("enhanced_paper_trading_dashboard.py")
    
    # Portfolio management row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="nav-card">
            <h2>📊 Portfolio Manager</h2>
            <p>Manage your stock universe (up to 50 stocks)</p>
            <p>✅ Add/remove stocks dynamically</p>
            <p>✅ Sector and market cap analysis</p>
            <p>✅ Portfolio optimization tools</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Manage Portfolio", type="secondary", use_container_width=True):
            st.switch_page("portfolio_manager.py")
    
    with col2:
        st.markdown("""
        <div class="nav-card">
            <h2>📈 Performance Analytics</h2>
            <p>Individual stock analysis and AI backtesting</p>
            <p>✅ Forward backtesting (3-month tests)</p>
            <p>✅ AI model performance validation</p>
            <p>✅ Detailed stock performance metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("View Performance", type="secondary", use_container_width=True):
            st.switch_page("performance_dashboard.py")
    
    # Test dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="nav-card">
            <h2>🎯 System Test</h2>
            <p>Verify system functionality and data integrity</p>
            <p>✅ Engine diagnostics</p>
            <p>✅ Data file validation</p>
            <p>✅ Price feed testing</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run System Test", type="secondary", use_container_width=True):
            st.switch_page("test_paper_trading.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🆕 What's New?</h3>
            <p>• Portfolio Manager: Add/remove up to 50 stocks</p>
            <p>• Performance Dashboard: Forward backtesting</p>
            <p>• Enhanced Analytics: Individual stock analysis</p>
            <p>• Real-time Updates: 5-minute market refresh</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Trading Strategy**
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
        **🔧 System Status**
        - ✅ All systems operational
        - ✅ Data feeds active
        - ✅ AI models trained
        - ✅ Paper trading ready
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>AI Trading Bot v2.0</strong> | Built with Streamlit & Python</p>
        <p>⚠️ <em>This is a paper trading simulation for educational purposes only</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
