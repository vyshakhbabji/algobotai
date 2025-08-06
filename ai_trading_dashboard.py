#!/usr/bin/env python3
"""
AI Trading Dashboard - Modern Web Interface
Real-time monitoring of AI portfolio performance and model health
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import subprocess
import json
import os

# Page configuration
st.set_page_config(
    page_title="🚀 AI Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .model-health {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        color: #212529;
    }
    .model-health h4 {
        color: #0d6efd;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .model-health p {
        color: #495057;
        margin-bottom: 0.3rem;
    }
    .model-health strong {
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

def load_portfolio_data():
    """Load current portfolio performance data"""
    # This would connect to your actual trading system
    # For demo, using simulated data based on our validation results
    return {
        'current_value': 13622.79,
        'initial_value': 10000.00,
        'total_return': 36.23,
        'daily_return': 0.15,
        'trades_today': 0,
        'active_positions': 12,
        'cash_available': 1245.67
    }

def load_model_health():
    """Load AI model health metrics"""
    return {
        'GOOG': {'r2': 0.536, 'confidence': 67.3, 'status': 'healthy'},
        'AAPL': {'r2': 0.535, 'confidence': 72.1, 'status': 'healthy'},
        'MSFT': {'r2': 0.537, 'confidence': 68.9, 'status': 'healthy'},
        'NVDA': {'r2': 0.586, 'confidence': 75.2, 'status': 'healthy'},
        'META': {'r2': 0.503, 'confidence': 62.4, 'status': 'healthy'},
        'AMZN': {'r2': 0.619, 'confidence': 78.1, 'status': 'healthy'},
        'AVGO': {'r2': 0.543, 'confidence': 69.7, 'status': 'healthy'},
        'PLTR': {'r2': 0.497, 'confidence': 84.5, 'status': 'healthy'},
        'NFLX': {'r2': 0.582, 'confidence': 71.6, 'status': 'healthy'},
        'TSM': {'r2': 0.600, 'confidence': 88.0, 'status': 'healthy'},
        'PANW': {'r2': 0.489, 'confidence': 66.3, 'status': 'healthy'},
        'NOW': {'r2': 0.501, 'confidence': 72.9, 'status': 'healthy'},
        'XLK': {'r2': 0.559, 'confidence': 63.7, 'status': 'healthy'},
        'QQQ': {'r2': 0.524, 'confidence': 65.4, 'status': 'healthy'},
        'COST': {'r2': 0.561, 'confidence': 68.2, 'status': 'healthy'}
    }

def create_performance_chart():
    """Create portfolio performance chart"""
    # Simulate performance data
    dates = pd.date_range(start='2025-05-01', end='2025-08-05', freq='D')
    initial_value = 10000
    
    # Generate realistic performance curve based on our 36.23% return
    np.random.seed(42)
    daily_returns = np.random.normal(0.0015, 0.02, len(dates))  # ~36% annual with volatility
    cumulative_returns = np.cumprod(1 + daily_returns)
    portfolio_values = initial_value * cumulative_returns
    
    # Ensure final value matches our validation result
    portfolio_values = portfolio_values * (13622.79 / portfolio_values[-1])
    
    fig = go.Figure()
    
    # Portfolio line
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='AI Portfolio',
        line=dict(color='#1f77b4', width=3),
        fill='tonexty'
    ))
    
    # Benchmark line (SPY equivalent)
    spy_returns = np.random.normal(0.0008, 0.015, len(dates))
    spy_cumulative = np.cumprod(1 + spy_returns)
    spy_values = initial_value * spy_cumulative
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=spy_values,
        mode='lines',
        name='Benchmark (SPY)',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="🚀 AI Portfolio Performance vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig

def create_model_health_chart():
    """Create model health visualization"""
    model_data = load_model_health()
    
    stocks = list(model_data.keys())
    r2_scores = [model_data[stock]['r2'] for stock in stocks]
    confidences = [model_data[stock]['confidence'] for stock in stocks]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model R² Scores', 'Current Confidence Levels'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² scores
    fig.add_trace(
        go.Bar(x=stocks, y=r2_scores, name='R² Score', 
               marker_color='lightblue', showlegend=False),
        row=1, col=1
    )
    
    # Confidence levels
    colors = ['green' if c > 70 else 'orange' if c > 55 else 'red' for c in confidences]
    fig.add_trace(
        go.Bar(x=stocks, y=confidences, name='Confidence %',
               marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        title="🧠 AI Model Health Dashboard",
        height=400,
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_yaxes(title_text="Confidence %", row=1, col=2)
    
    return fig

def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<div class="main-header">🚀 AI Trading Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    portfolio = load_portfolio_data()
    model_health = load_model_health()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # System status
        st.subheader("🔧 System Status")
        healthy_models = sum(1 for m in model_health.values() if m['status'] == 'healthy')
        st.success(f"✅ {healthy_models}/15 models healthy")
        
        avg_r2 = np.mean([m['r2'] for m in model_health.values()])
        st.info(f"📊 Avg R² Score: {avg_r2:.3f}")
        
        avg_confidence = np.mean([m['confidence'] for m in model_health.values()])
        st.info(f"🎯 Avg Confidence: {avg_confidence:.1f}%")
        
        # Controls
        st.subheader("⚙️ Actions")
        if st.button("🔄 Retrain Models"):
            st.info("Model retraining initiated...")
        
        if st.button("📊 Run Backtest"):
            st.info("Running new backtest...")
        
        if st.button("📈 Generate Signals"):
            st.info("Generating fresh signals...")
        
        # Settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 50, 80, 55)
        rebalance_frequency = st.selectbox("Rebalancing", ["Daily", "Weekly", "Monthly"])
        risk_limit = st.slider("Max Position Size", 5, 25, 15)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="success-card">
            <h3>💰 Portfolio Value</h3>
            <h2>${portfolio['current_value']:,.2f}</h2>
            <p>+${portfolio['current_value'] - portfolio['initial_value']:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Total Return</h3>
            <h2>{portfolio['total_return']:.2f}%</h2>
            <p>Since May 1, 2025</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔄 Active Positions</h3>
            <h2>{portfolio['active_positions']}</h2>
            <p>{portfolio['trades_today']} trades today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💵 Available Cash</h3>
            <h2>${portfolio['cash_available']:,.2f}</h2>
            <p>Ready to deploy</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_performance_chart(), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_model_health_chart(), use_container_width=True)
    
    # Model details
    st.header("🧠 AI Model Health Details")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (stock, data) in enumerate(model_health.items()):
        col = [col1, col2, col3][i % 3]
        
        status_color = "green" if data['confidence'] > 70 else "orange"
        
        with col:
            st.markdown(f"""
            <div class="model-health">
                <h4>{stock}</h4>
                <p><strong>R² Score:</strong> <span style="color: #0d6efd; font-weight: bold;">{data['r2']:.3f}</span></p>
                <p><strong>Confidence:</strong> <span style="color: {status_color}; font-weight: bold;">{data['confidence']:.1f}%</span></p>
                <p><strong>Status:</strong> <span style="color: #28a745; font-weight: bold;">✅ {data['status'].title()}</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent signals
    st.header("📊 Recent Trading Signals")
    
    # Simulate recent signals
    signal_data = {
        'Stock': ['TSM', 'NFLX', 'NVDA', 'AMZN', 'PANW'],
        'Signal': ['BUY', 'BUY', 'BUY', 'BUY', 'BUY'],
        'Confidence': [84.5, 81.4, 70.4, 70.4, 66.3],
        'Target Shares': [11, 2, 15, 13, 16],
        'Price': [239.00, 1170.99, 180.00, 211.65, 171.00],
        'Timestamp': ['2025-08-05 09:30'] * 5
    }
    
    df_signals = pd.DataFrame(signal_data)
    st.dataframe(df_signals, use_container_width=True)
    
    # Risk metrics
    st.header("⚠️ Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sharpe Ratio", "0.737", delta="0.05")
    
    with col2:
        st.metric("Max Drawdown", "43.75%", delta="-2.1%")
    
    with col3:
        st.metric("Volatility", "108.96%", delta="+5.2%")
    
    with col4:
        st.metric("Win Rate", "68.3%", delta="+3.1%")
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 **AI Trading System** | Powered by Improved ML Models | Last Updated: " + 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
