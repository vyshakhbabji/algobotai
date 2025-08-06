#!/usr/bin/env python3
"""
Enhanced Paper Trading Dashboard
Real-time market monitoring and trading interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import requests
from live_paper_trading import PaperTradingEngine

# Password Protection (same as main dashboard)
def check_password():
    """Returns True if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "102326":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.title("🔐 Enhanced Trading Dashboard - Secure Access")
        st.markdown("**Please enter the access password to continue:**")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.markdown("---")
        st.markdown("*Authorized users only. This system manages a $10,000 AI trading portfolio.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.title("🔐 Enhanced Trading Dashboard - Secure Access")
        st.markdown("**Please enter the access password to continue:**")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😞 Password incorrect. Please try again.")
        st.markdown("---")
        st.markdown("*Authorized users only. This system manages a $10,000 AI trading portfolio.*")
        return False
    else:
        # Password correct
        return True

# Page configuration
st.set_page_config(
    page_title="📈 Paper Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .profit-positive {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .profit-negative {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stock-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        color: #212529;
    }
    .buy-signal {
        border-left: 4px solid #28a745 !important;
        background: rgba(40, 167, 69, 0.1);
    }
    .sell-signal {
        border-left: 4px solid #dc3545 !important;
        background: rgba(220, 53, 69, 0.1);
    }
    .hold-signal {
        border-left: 4px solid #ffc107 !important;
        background: rgba(255, 193, 7, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def get_market_data():
    """Get real-time market data for our stock universe"""
    stocks = [
        "AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AVGO",
        "PLTR", "NFLX", "TSM", "PANW", "NOW", "XLK", "QQQ", "COST",
        # Additional testing stocks
        "SHOP", "ROKU", "SNOW", "CRWD", "ZM", "DOCU", "OKTA", "DDOG"
    ]
    
    market_data = []
    
    for symbol in stocks:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            info = ticker.info
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                volume = hist['Volume'].iloc[-1]
                avg_volume = hist['Volume'].mean()
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                
                market_data.append({
                    "Symbol": symbol,
                    "Price": current_price,
                    "Change": change,
                    "Change %": change_pct,
                    "Volume": volume,
                    "Vol Ratio": volume_ratio,
                    "Market Cap": info.get('marketCap', 'N/A'),
                    "Sector": info.get('sector', 'N/A')
                })
        except Exception as e:
            st.write(f"Error fetching data for {symbol}: {str(e)}")
    
    return pd.DataFrame(market_data)

def analyze_market_sentiment():
    """Analyze overall market sentiment"""
    try:
        # Get major indices
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "DOW": "^DJI",
            "VIX": "^VIX"
        }
        
        sentiment_data = []
        
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change_pct = ((current - prev) / prev) * 100
                
                sentiment_data.append({
                    "Index": name,
                    "Value": current,
                    "Change %": change_pct
                })
        
        return pd.DataFrame(sentiment_data)
    except Exception as e:
        st.error(f"Error analyzing market sentiment: {str(e)}")
        return pd.DataFrame()

def main():
    # Check password first
    if not check_password():
        return
        
    st.title("📈 Enhanced Paper Trading Dashboard")
    st.markdown("**Live Market Monitoring & AI Trading Simulation**")
    
    # Initialize trading engine
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = PaperTradingEngine()
    
    engine = st.session_state.trading_engine
    
    # Sidebar - Quick Stats
    st.sidebar.title("Portfolio Overview")
    
    # Update portfolio value
    current_value = engine.update_portfolio_value()
    
    st.sidebar.metric(
        "Portfolio Value",
        f"${current_value:.2f}",
        f"${engine.account['total_return']:.2f} ({engine.account['total_return_pct']:.2f}%)"
    )
    
    st.sidebar.metric(
        "Available Cash",
        f"${engine.account['cash']:.2f}"
    )
    
    st.sidebar.metric(
        "Positions",
        len(engine.positions)
    )
    
    # Sidebar - Market Indices
    st.sidebar.subheader("Market Indices")
    sentiment_df = analyze_market_sentiment()
    
    if not sentiment_df.empty:
        for _, row in sentiment_df.iterrows():
            color = "green" if row['Change %'] >= 0 else "red"
            st.sidebar.markdown(f"""
            **{row['Index']}**: {row['Value']:.2f} 
            <span style="color: {color}">({row['Change %']:.2f}%)</span>
            """, unsafe_allow_html=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Trading", "📊 Portfolio", "📈 Market Scanner", "📋 Trade History"])
    
    with tab1:
        st.header("🎯 Live Trading Interface")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Market Data & Signals")
            
            if st.button("🔄 Refresh Market Data", type="primary"):
                with st.spinner("Fetching live market data..."):
                    market_df = get_market_data()
                    
                    if not market_df.empty:
                        # Add signal column based on technical analysis
                        market_df['Signal'] = market_df.apply(lambda row: 
                            "🟢 BUY" if row['Change %'] > 2 and row['Vol Ratio'] > 1.5 
                            else "🔴 SELL" if row['Change %'] < -2 
                            else "🟡 HOLD", axis=1)
                        
                        # Color code the dataframe
                        def color_negative_red(val):
                            if isinstance(val, (int, float)):
                                color = 'red' if val < 0 else 'green' if val > 0 else 'black'
                                return f'color: {color}'
                            return ''
                        
                        styled_df = market_df.style.applymap(
                            color_negative_red, subset=['Change', 'Change %']
                        )
                        
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        # Quick trade buttons
                        st.subheader("⚡ Quick Trade")
                        
                        # Show top movers
                        top_gainers = market_df.nlargest(3, 'Change %')
                        top_losers = market_df.nsmallest(3, 'Change %')
                        
                        col_gain, col_lose = st.columns(2)
                        
                        with col_gain:
                            st.markdown("**🚀 Top Gainers**")
                            for _, stock in top_gainers.iterrows():
                                if st.button(f"BUY {stock['Symbol']} (${stock['Price']:.2f}, +{stock['Change %']:.1f}%)", 
                                           key=f"buy_{stock['Symbol']}"):
                                    shares = int(min(1000, engine.account['cash'] * 0.05) / stock['Price'])
                                    if shares > 0:
                                        success, message = engine.execute_trade(
                                            stock['Symbol'], "BUY", shares, stock['Price'], 
                                            75.0, "Quick Buy - Top Gainer"
                                        )
                                        if success:
                                            st.success(message)
                                            engine.save_data()
                                        else:
                                            st.error(message)
                        
                        with col_lose:
                            st.markdown("**📉 Top Losers**")
                            for _, stock in top_losers.iterrows():
                                # Only show sell if we own the stock
                                if stock['Symbol'] in engine.positions:
                                    current_shares = engine.positions[stock['Symbol']]['shares']
                                    if st.button(f"SELL {stock['Symbol']} ({current_shares} shares)", 
                                               key=f"sell_{stock['Symbol']}"):
                                        success, message = engine.execute_trade(
                                            stock['Symbol'], "SELL", current_shares, stock['Price'],
                                            75.0, "Quick Sell - Cut Losses"
                                        )
                                        if success:
                                            st.success(message)
                                            engine.save_data()
                                        else:
                                            st.error(message)
        
        with col2:
            st.subheader("Manual Trading")
            
            # Manual trade form
            with st.form("manual_trade"):
                trade_symbol = st.selectbox("Symbol", [
                    "AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AVGO",
                    "PLTR", "NFLX", "TSM", "PANW", "NOW", "XLK", "QQQ", "COST",
                    "SHOP", "ROKU", "SNOW", "CRWD", "ZM", "DOCU", "OKTA", "DDOG"
                ])
                trade_action = st.selectbox("Action", ["BUY", "SELL"])
                trade_amount = st.number_input("Amount ($)", min_value=100, value=1000, step=100)
                
                submitted = st.form_submit_button("Execute Trade")
                
                if submitted:
                    current_price = engine.get_current_price(trade_symbol)
                    if current_price:
                        shares = int(trade_amount / current_price)
                        if shares > 0:
                            success, message = engine.execute_trade(
                                trade_symbol, trade_action, shares, current_price,
                                90.0, "Manual Trade"
                            )
                            if success:
                                st.success(message)
                                engine.save_data()
                            else:
                                st.error(message)
                        else:
                            st.error("Amount too small for any shares")
                    else:
                        st.error(f"Could not get price for {trade_symbol}")
    
    with tab2:
        st.header("📊 Portfolio Analysis")
        
        if engine.positions:
            # Current positions
            positions_df = engine.get_position_details()
            st.subheader("Current Positions")
            st.dataframe(positions_df, use_container_width=True)
            
            # Portfolio charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Portfolio allocation pie chart
                if not positions_df.empty:
                    values = [float(val.replace('$', '').replace(',', '')) 
                             for val in positions_df['Current Value'].tolist()]
                    
                    fig = px.pie(
                        values=values, 
                        names=positions_df['Symbol'].tolist(),
                        title="Portfolio Allocation by Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # P&L chart
                if not positions_df.empty:
                    pl_values = [float(val.replace('$', '').replace(',', '')) 
                               for val in positions_df['Unrealized P&L'].tolist()]
                    
                    colors = ['green' if val >= 0 else 'red' for val in pl_values]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=positions_df['Symbol'].tolist(),
                            y=pl_values,
                            marker_color=colors
                        )
                    ])
                    fig.update_layout(title="Unrealized P&L by Position")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Portfolio metrics
            st.subheader("Portfolio Metrics")
            
            total_cost = sum([float(val.replace('$', '').replace(',', '')) 
                            for val in positions_df['Cost Basis'].tolist()])
            total_value = sum([float(val.replace('$', '').replace(',', '')) 
                             for val in positions_df['Current Value'].tolist()])
            total_pl = total_value - total_cost
            total_pl_pct = (total_pl / total_cost) * 100 if total_cost > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Invested", f"${total_cost:.2f}")
            with col2:
                st.metric("Current Value", f"${total_value:.2f}")
            with col3:
                st.metric("Unrealized P&L", f"${total_pl:.2f}", f"{total_pl_pct:.2f}%")
            with col4:
                st.metric("Cash Position", f"${engine.account['cash']:.2f}")
        
        else:
            st.info("No current positions. Start trading to build your portfolio!")
    
    with tab3:
        st.header("📈 Market Scanner")
        
        # Real-time market scanner
        if st.button("🔍 Scan Markets", type="primary"):
            with st.spinner("Scanning markets for opportunities..."):
                market_df = get_market_data()
                
                if not market_df.empty:
                    # Market overview
                    st.subheader("Market Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        gainers = len(market_df[market_df['Change %'] > 0])
                        st.metric("Gainers", gainers)
                    
                    with col2:
                        losers = len(market_df[market_df['Change %'] < 0])
                        st.metric("Losers", losers)
                    
                    with col3:
                        avg_change = market_df['Change %'].mean()
                        st.metric("Avg Change", f"{avg_change:.2f}%")
                    
                    with col4:
                        high_volume = len(market_df[market_df['Vol Ratio'] > 1.5])
                        st.metric("High Volume", high_volume)
                    
                    # Filters and sorting
                    st.subheader("Market Data")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        min_change = st.slider("Min Change %", -10.0, 10.0, -10.0)
                    with col2:
                        max_change = st.slider("Max Change %", -10.0, 10.0, 10.0)
                    with col3:
                        sort_by = st.selectbox("Sort by", ["Change %", "Volume", "Vol Ratio", "Price"])
                    
                    # Apply filters
                    filtered_df = market_df[
                        (market_df['Change %'] >= min_change) & 
                        (market_df['Change %'] <= max_change)
                    ].sort_values(sort_by, ascending=False)
                    
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    # Market heatmap
                    st.subheader("Performance Heatmap")
                    
                    fig = px.treemap(
                        market_df,
                        path=['Sector', 'Symbol'],
                        values='Volume',
                        color='Change %',
                        color_continuous_scale='RdYlGn',
                        title="Market Performance by Sector"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("📋 Trade History & Performance")
        
        if engine.trades:
            # Recent trades
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(engine.trades[-20:])  # Last 20 trades
            st.dataframe(trades_df, use_container_width=True)
            
            # Performance analysis
            st.subheader("Performance Analysis")
            
            # Calculate metrics
            buy_trades = [t for t in engine.trades if t['action'] == 'BUY']
            sell_trades = [t for t in engine.trades if t['action'] == 'SELL']
            
            total_trades = len(engine.trades)
            total_buy_value = sum([t['total_value'] for t in buy_trades])
            total_sell_value = sum([t['total_value'] for t in sell_trades])
            
            realized_pl = sum([t.get('profit_loss', 0) for t in sell_trades])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Buy Volume", f"${total_buy_value:.2f}")
            with col3:
                st.metric("Sell Volume", f"${total_sell_value:.2f}")
            with col4:
                st.metric("Realized P&L", f"${realized_pl:.2f}")
            
            # Portfolio value over time
            if len(engine.trades) > 1:
                st.subheader("Portfolio Performance Timeline")
                
                # Create timeline of portfolio value
                portfolio_timeline = []
                current_cash = engine.account['initial_capital']
                current_positions = {}
                
                for trade in engine.trades:
                    timestamp = pd.to_datetime(trade['timestamp'])
                    
                    if trade['action'] == 'BUY':
                        current_cash -= trade['total_value']
                        if trade['symbol'] in current_positions:
                            current_positions[trade['symbol']] += trade['shares']
                        else:
                            current_positions[trade['symbol']] = trade['shares']
                    else:
                        current_cash += trade['total_value']
                        if trade['symbol'] in current_positions:
                            current_positions[trade['symbol']] -= trade['shares']
                            if current_positions[trade['symbol']] <= 0:
                                del current_positions[trade['symbol']]
                    
                    # Calculate portfolio value
                    portfolio_value = current_cash
                    for symbol, shares in current_positions.items():
                        current_price = engine.get_current_price(symbol)
                        if current_price:
                            portfolio_value += shares * current_price
                    
                    portfolio_timeline.append({
                        'timestamp': timestamp,
                        'portfolio_value': portfolio_value,
                        'cash': current_cash,
                        'positions_value': portfolio_value - current_cash
                    })
                
                timeline_df = pd.DataFrame(portfolio_timeline)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=timeline_df['timestamp'],
                    y=timeline_df['portfolio_value'],
                    mode='lines+markers',
                    name='Total Portfolio Value',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=timeline_df['timestamp'],
                    y=timeline_df['cash'],
                    mode='lines',
                    name='Cash',
                    line=dict(color='green', dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=timeline_df['timestamp'],
                    y=timeline_df['positions_value'],
                    mode='lines',
                    name='Positions Value',
                    line=dict(color='orange', dash='dot')
                ))
                
                fig.add_hline(
                    y=engine.account['initial_capital'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Starting Capital"
                )
                
                fig.update_layout(
                    title="Portfolio Performance Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No trades yet. Start trading to see your performance history!")
    
    # Footer with auto-refresh
    st.markdown("---")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"*Last updated: {current_time} | Auto-refresh: Every 5 minutes*")
    
    # Auto-refresh every 5 minutes during market hours
    if st.button("🔄 Refresh Now"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
