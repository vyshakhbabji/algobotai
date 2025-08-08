#!/usr/bin/env python3
"""
Elite Options Trading Dashboard - Streamlit Page
Created: August 7, 2025
Purpose: Streamlit interface for AI-powered options trading recommendations
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from elite_options_trader import EliteOptionsTrader

def main():
    # Header
    st.title("🚀 Elite Options Trading System")
    st.markdown("**AI-Powered Options Strategy Recommendations for Maximum Returns**")
    
    # Initialize options trader
    if 'options_trader' not in st.session_state:
        with st.spinner("Initializing Elite Options Trading System..."):
            st.session_state.options_trader = EliteOptionsTrader()
    
    trader = st.session_state.options_trader
    
    # Sidebar controls
    st.sidebar.header("🎯 Options Analysis Controls")
    
    # Mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode:",
        ["Daily Opportunities", "Single Stock Analysis", "Portfolio Scan"]
    )
    
    # Risk tolerance
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance:",
        ["Conservative", "Moderate", "Aggressive"]
    )
    
    # Main content area
    if analysis_mode == "Daily Opportunities":
        show_daily_opportunities(trader, risk_tolerance)
    elif analysis_mode == "Single Stock Analysis":
        show_single_stock_analysis(trader, risk_tolerance)
    elif analysis_mode == "Portfolio Scan":
        show_portfolio_scan(trader, risk_tolerance)

def show_daily_opportunities(trader, risk_tolerance):
    """Show daily options trading opportunities"""
    st.header("📊 Daily Options Trading Report")
    
    # Generate report button
    if st.button("🔄 Generate Fresh Daily Report", type="primary"):
        with st.spinner("Analyzing market conditions and generating opportunities..."):
            report = trader.generate_daily_options_report()
            st.session_state.daily_report = report
    
    # Show cached report if available
    if 'daily_report' in st.session_state:
        report = st.session_state.daily_report
        
        # Market Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Market Condition",
                report['market_condition'].title(),
                delta=None
            )
        
        with col2:
            st.metric(
                "Market Volatility",
                report['market_volatility'],
                delta=None
            )
        
        with col3:
            st.metric(
                "Total Opportunities",
                report['total_opportunities'],
                delta=None
            )
        
        with col4:
            st.metric(
                "High Confidence Trades",
                report['high_confidence_trades'],
                delta=None
            )
        
        # Top Opportunities
        st.subheader("🎯 Top Options Opportunities")
        
        if report['top_opportunities']:
            for i, opportunity in enumerate(report['top_opportunities'][:5], 1):
                with st.expander(f"#{i} {opportunity['symbol']} - {opportunity['strategy']} (Confidence: {opportunity['confidence_score']})"):
                    
                    # Create two columns for details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📋 Trade Details:**")
                        st.write(f"**Action:** {opportunity['action']}")
                        st.write(f"**Reasoning:** {opportunity['reasoning']}")
                        st.write(f"**Risk Level:** {opportunity['risk_level']}")
                        st.write(f"**Time Frame:** {opportunity['time_frame']}")
                    
                    with col2:
                        st.markdown("**💰 Profit/Risk Analysis:**")
                        st.write(f"**Expected Return:** {opportunity['expected_return']}")
                        st.write(f"**Win Probability:** {opportunity['win_probability']}")
                        st.write(f"**Max Profit:** {opportunity['max_profit']}")
                        st.write(f"**Max Loss:** {opportunity['max_loss']}")
                    
                    # Entry cost and position sizing
                    if 'entry_cost' in opportunity:
                        st.markdown("**💸 Entry Requirements:**")
                        st.write(f"**Entry Cost:** {opportunity['entry_cost']}")
                        if 'position_size' in opportunity:
                            st.write(f"**Position Size:** {opportunity['position_size']}")
                    
                    # Exit plan
                    if 'exit_plan' in opportunity:
                        st.markdown("**🎯 Exit Strategy:**")
                        st.write(opportunity['exit_plan'])
        
        # Strategy Distribution Chart
        if report['strategy_distribution']:
            st.subheader("📈 Strategy Distribution")
            
            strategies = list(report['strategy_distribution'].keys())
            counts = list(report['strategy_distribution'].values())
            
            fig = px.pie(
                values=counts,
                names=strategies,
                title="Recommended Options Strategies Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk Summary
        st.subheader("⚠️ Portfolio Risk Analysis")
        risk = report['risk_summary']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Recommended Allocation",
                f"{risk['recommended_allocation']}%",
                help="Maximum portfolio allocation for all options trades"
            )
        
        with col2:
            st.metric(
                "High Risk Trades",
                risk['high_risk_trades'],
                help="Number of high-risk options strategies"
            )
        
        with col3:
            st.metric(
                "Overall Risk Level",
                risk['risk_level'],
                help="Portfolio-wide risk assessment"
            )
        
        # Recommendations
        st.subheader("💡 AI Recommendations")
        recommendations = report['recommendations']
        
        if recommendations['immediate_action']:
            st.markdown("**🚨 Immediate Action Required:**")
            for action in recommendations['immediate_action']:
                st.success(f"✅ {action['symbol']} - {action['strategy']} (Confidence: {action['confidence_score']})")
        
        if recommendations['watch_list']:
            st.markdown("**👀 Watch List:**")
            for watch in recommendations['watch_list']:
                st.info(f"📊 {watch['symbol']} - {watch['strategy']} (Confidence: {watch['confidence_score']})")
        
        st.markdown(f"**📊 Market Outlook:** {recommendations['market_outlook'].title()}")
        st.markdown(f"**🌊 Volatility Environment:** {recommendations['volatility_environment']}")
    
    else:
        st.info("👆 Click 'Generate Fresh Daily Report' to analyze current market opportunities")

def show_single_stock_analysis(trader, risk_tolerance):
    """Show detailed analysis for a single stock"""
    st.header("🔍 Single Stock Options Analysis")
    
    # Stock symbol input
    symbol = st.text_input(
        "Enter Stock Symbol:",
        value="AAPL",
        help="Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)"
    ).upper().strip()
    
    if symbol and st.button(f"🔎 Analyze {symbol} Options", type="primary"):
        with st.spinner(f"Analyzing options opportunities for {symbol}..."):
            try:
                # Get recommendation
                recommendation = trader.recommend_strategy(symbol, risk_tolerance.lower())
                
                # Display results
                st.success(f"✅ Analysis Complete for {symbol}")
                
                # Main recommendation card
                with st.container():
                    st.subheader(f"🎯 Recommended Strategy: {recommendation['strategy']}")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Confidence Score", recommendation['confidence_score'])
                    
                    with col2:
                        st.metric("Expected Return", recommendation['expected_return'])
                    
                    with col3:
                        st.metric("Win Probability", recommendation['win_probability'])
                    
                    with col4:
                        st.metric("Risk Level", recommendation['risk_level'])
                
                # Detailed analysis
                st.subheader("📋 Detailed Trade Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎯 Trade Action:**")
                    st.code(recommendation['action'], language=None)
                    
                    st.markdown("**💭 AI Reasoning:**")
                    st.write(recommendation['reasoning'])
                    
                    st.markdown("**⏰ Time Frame:**")
                    st.write(recommendation['time_frame'])
                
                with col2:
                    st.markdown("**💰 Profit/Loss Analysis:**")
                    st.write(f"**Max Profit:** {recommendation['max_profit']}")
                    st.write(f"**Max Loss:** {recommendation['max_loss']}")
                    
                    if 'breakeven' in recommendation:
                        st.write(f"**Breakeven:** {recommendation['breakeven']}")
                    
                    if 'entry_cost' in recommendation:
                        st.markdown("**💸 Entry Requirements:**")
                        st.write(f"**Entry Cost:** {recommendation['entry_cost']}")
                
                # Additional details
                if 'exit_plan' in recommendation:
                    st.subheader("🎯 Exit Strategy")
                    st.info(recommendation['exit_plan'])
                
                # Risk factors
                risk_factors = []
                if 'iv_impact' in recommendation:
                    risk_factors.append(f"**Volatility Risk:** {recommendation['iv_impact']}")
                if 'theta_risk' in recommendation:
                    risk_factors.append(f"**Time Decay:** {recommendation['theta_risk']}")
                if 'event_risk' in recommendation:
                    risk_factors.append(f"**Event Risk:** {recommendation['event_risk']}")
                
                if risk_factors:
                    st.subheader("⚠️ Risk Factors")
                    for risk in risk_factors:
                        st.warning(risk)
                
            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {str(e)}")

def show_portfolio_scan(trader, risk_tolerance):
    """Show portfolio-wide options scan"""
    st.header("📊 Portfolio Options Scan")
    st.markdown("Scanning all portfolio stocks for the best options opportunities...")
    
    if st.button("🔍 Scan Portfolio for Options", type="primary"):
        with st.spinner("Scanning portfolio stocks for options opportunities..."):
            try:
                # Get opportunities
                opportunities = trader.scan_portfolio_opportunities()
                
                if opportunities:
                    st.success(f"✅ Found {len(opportunities)} high-confidence opportunities")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Calculate summary stats
                    avg_confidence = sum(float(op['confidence_score'].replace('%', '')) for op in opportunities) / len(opportunities)
                    high_confidence = len([op for op in opportunities if float(op['confidence_score'].replace('%', '')) > 75])
                    strategies = list(set(op['strategy'] for op in opportunities))
                    
                    with col1:
                        st.metric("Total Opportunities", len(opportunities))
                    
                    with col2:
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    
                    with col3:
                        st.metric("High Confidence (>75%)", high_confidence)
                    
                    with col4:
                        st.metric("Unique Strategies", len(strategies))
                    
                    # Opportunities table
                    st.subheader("🎯 All Opportunities")
                    
                    # Create DataFrame for display
                    df_data = []
                    for op in opportunities:
                        df_data.append({
                            'Symbol': op['symbol'],
                            'Strategy': op['strategy'],
                            'Confidence': op['confidence_score'],
                            'Expected Return': op['expected_return'],
                            'Win Probability': op['win_probability'],
                            'Risk Level': op['risk_level'],
                            'Time Frame': op['time_frame']
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Detailed view
                    st.subheader("📋 Detailed Opportunities")
                    
                    # Filter by confidence
                    min_confidence = st.slider(
                        "Minimum Confidence Level (%)",
                        min_value=50,
                        max_value=100,
                        value=60,
                        step=5
                    )
                    
                    filtered_ops = [
                        op for op in opportunities 
                        if float(op['confidence_score'].replace('%', '')) >= min_confidence
                    ]
                    
                    for i, opportunity in enumerate(filtered_ops, 1):
                        with st.expander(f"#{i} {opportunity['symbol']} - {opportunity['strategy']} (Confidence: {opportunity['confidence_score']})"):
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📋 Trade Details:**")
                                st.write(f"**Action:** {opportunity['action']}")
                                st.write(f"**Reasoning:** {opportunity['reasoning']}")
                                st.write(f"**Risk Level:** {opportunity['risk_level']}")
                                
                            with col2:
                                st.markdown("**💰 Returns:**")
                                st.write(f"**Expected Return:** {opportunity['expected_return']}")
                                st.write(f"**Win Probability:** {opportunity['win_probability']}")
                                st.write(f"**Max Profit:** {opportunity['max_profit']}")
                                st.write(f"**Max Loss:** {opportunity['max_loss']}")
                            
                            if 'exit_plan' in opportunity:
                                st.markdown("**🎯 Exit Strategy:**")
                                st.info(opportunity['exit_plan'])
                
                else:
                    st.warning("⚠️ No high-confidence options opportunities found in current market conditions")
                    st.info("💡 Try adjusting risk tolerance or wait for better market conditions")
                    
            except Exception as e:
                st.error(f"❌ Error scanning portfolio: {str(e)}")
    
    # Portfolio info
    st.subheader("📊 Portfolio Information")
    
    try:
        # Show current portfolio stocks
        portfolio_file = os.path.join(parent_dir, 'elite_portfolio_universe.json')
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                portfolio_data = json.load(f)
            
            st.info(f"📈 Elite Portfolio: {len(portfolio_data['stocks'])} stocks selected for AI trading")
            
            # Show stocks in a nice grid
            cols = st.columns(5)
            for i, stock in enumerate(portfolio_data['stocks'][:25]):
                with cols[i % 5]:
                    st.write(f"**{stock}**")
        else:
            st.warning("⚠️ Elite portfolio not found. Run elite stock selector first.")
            
    except Exception as e:
        st.error(f"❌ Error loading portfolio: {str(e)}")

if __name__ == "__main__":
    main()
