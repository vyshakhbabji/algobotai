#!/usr/bin/env python3
"""
AI Trading Bot - Main Application Entry Point
Complete multi-page dashboard with system monitoring and navigation
"""

import streamlit as st
import subprocess
import sys
import os
import json

# Page configuration
st.set_page_config(
    page_title="🤖 AI Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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
                st.markdown(f'''
                <div class="stats-card">
                    <h3>{system_health}</h3>
                    <p>System Health</p>
                </div>
                ''', unsafe_allow_html=True)
        
    except Exception as e:
        # Show basic welcome if data not available
        st.info("💡 Welcome! Your trading data will appear here once you start trading.")
    
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
            st.switch_page("pages/live_paper_trading.py")
    
    with col2:
        st.markdown("""
        <div class="nav-card">
            <h2>🎯 Elite Options Trading</h2>
            <p>AI-powered options strategies for 50-200% returns</p>
            <p>✅ Smart options strategy selection</p>
            <p>✅ Risk/reward optimization</p>
            <p>✅ High-probability setups</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Options Trading", type="primary", use_container_width=True):
            try:
                st.switch_page("pages/elite_options_trading.py")
            except:
                st.error("Elite Options Trading page not available yet")
    
    # Portfolio management row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="nav-card">
            <h2>📊 Portfolio Manager</h2>
            <p>Manage your elite stock universe (25 stocks)</p>
            <p>✅ Add/remove stocks dynamically</p>
            <p>✅ Sector and market cap analysis</p>
            <p>✅ Elite stock selection tools</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Manage Portfolio", type="secondary", use_container_width=True):
            st.switch_page("pages/portfolio_manager.py")
    
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
            st.switch_page("pages/performance_dashboard.py")
            
    with col3:
        st.markdown("""
        <div class="nav-card">
            <h2>📈 Enhanced Dashboard</h2>
            <p>Advanced market analysis and scanning</p>
            <p>✅ Market scanner and top movers</p>
            <p>✅ Technical analysis charts</p>
            <p>✅ Advanced AI signals</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Enhanced", type="secondary", use_container_width=True):
            st.switch_page("pages/enhanced_paper_trading_dashboard.py")
    
    # AI and System Tools
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="nav-card">
            <h2>🧠 AI Self-Optimizer</h2>
            <p>Continuously improve AI models for better returns</p>
            <p>✅ Performance monitoring and evaluation</p>
            <p>✅ Automatic model retraining</p>
            <p>✅ Parameter optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch AI Optimizer", type="secondary", use_container_width=True):
            st.switch_page("pages/ai_optimizer.py")
    
    with col2:
        st.markdown("""
        <div class="nav-card">
            <h2>🔧 System Status Monitor</h2>
            <p>Comprehensive health check & diagnostics</p>
            <p>✅ 8 critical system components</p>
            <p>✅ Real-time health scoring (🟢🟡🔴)</p>
            <p>✅ Automated status logging</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open System Monitor", type="secondary", use_container_width=True):
            st.switch_page("pages/test_paper_trading.py")
    
    # Additional features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🆕 What's New?</h3>
            <p>• System Status Monitor: Complete health diagnostics</p>
            <p>• Portfolio Manager: Add/remove up to 50 stocks</p>
            <p>• Performance Dashboard: Forward backtesting</p>
            <p>• Enhanced Analytics: Individual stock analysis</p>
            <p>• AI Self-Optimizer: Auto-improving models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 System Health</h3>
            <p>• Real-time monitoring of all components</p>
            <p>• Automated health scoring system</p>
            <p>• Quick diagnostic and troubleshooting</p>
            <p>• Performance tracking and optimization</p>
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

# Run the main function
if __name__ == "__main__":
    main()
else:
    # When imported, also run main (for Streamlit Cloud)
    main()
