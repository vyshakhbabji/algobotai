#!/usr/bin/env python3
"""
Live Paper Trading Engine
Real-time AI trading simulation with $10,000 starting capital
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time
import pickle
from improved_ai_portfolio_manager import ImprovedAIPortfolioManager

# Password Protection
# Password protection removed for easier access

# Page configuration
st.set_page_config(
    page_title="🚀 Live Paper Trading Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .profit-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .trade-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        color: #212529;
    }
    .buy-signal {
        background: rgba(40, 167, 69, 0.1);
        border-left: 4px solid #28a745;
    }
    .sell-signal {
        background: rgba(220, 53, 69, 0.1);
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class PaperTradingEngine:
    def __init__(self):
        self.data_file = "paper_trading_data.json"
        self.positions_file = "current_positions.json"
        self.trades_file = "trade_history.json"
        self.initialize_account()
        
    def initialize_account(self):
        """Initialize or load paper trading account"""
        if not os.path.exists(self.data_file):
            # Fresh start with $10,000
            self.account = {
                "cash": 10000.00,
                "initial_capital": 10000.00,
                "total_value": 10000.00,
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "start_date": datetime.now().strftime("%Y-%m-%d"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.positions = {}
            self.trades = []
            self.save_data()
        else:
            self.load_data()
    
    def save_data(self):
        """Save all trading data"""
        with open(self.data_file, 'w') as f:
            json.dump(self.account, f, indent=2)
        with open(self.positions_file, 'w') as f:
            json.dump(self.positions, f, indent=2)
        with open(self.trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)
    
    def load_data(self):
        """Load existing trading data"""
        with open(self.data_file, 'r') as f:
            self.account = json.load(f)
        with open(self.positions_file, 'r') as f:
            self.positions = json.load(f)
        if os.path.exists(self.trades_file):
            with open(self.trades_file, 'r') as f:
                self.trades = json.load(f)
        else:
            self.trades = []
    
    def get_current_price(self, symbol):
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
            else:
                # Fallback to daily data
                data = ticker.history(period="5d")
                return data['Close'].iloc[-1]
        except:
            return None
    
    def execute_trade(self, symbol, action, shares, price, confidence, reason="AI Signal"):
        """Execute a paper trade"""
        trade_value = shares * price
        
        if action.upper() == "BUY":
            if self.account["cash"] >= trade_value:
                # Execute buy
                self.account["cash"] -= trade_value
                if symbol in self.positions:
                    # Average down
                    total_shares = self.positions[symbol]["shares"] + shares
                    total_cost = self.positions[symbol]["total_cost"] + trade_value
                    self.positions[symbol] = {
                        "shares": total_shares,
                        "avg_price": total_cost / total_shares,
                        "total_cost": total_cost,
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    # New position
                    self.positions[symbol] = {
                        "shares": shares,
                        "avg_price": price,
                        "total_cost": trade_value,
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                
                # Record trade
                trade = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "action": "BUY",
                    "shares": shares,
                    "price": price,
                    "total_value": trade_value,
                    "confidence": confidence,
                    "reason": reason,
                    "cash_after": self.account["cash"]
                }
                self.trades.append(trade)
                return True, f"Bought {shares} shares of {symbol} at ${price:.2f}"
            else:
                return False, f"Insufficient cash. Need ${trade_value:.2f}, have ${self.account['cash']:.2f}"
        
        elif action.upper() == "SELL":
            if symbol in self.positions and self.positions[symbol]["shares"] >= shares:
                # Execute sell
                proceeds = shares * price
                self.account["cash"] += proceeds
                
                # Update position
                self.positions[symbol]["shares"] -= shares
                cost_basis = shares * self.positions[symbol]["avg_price"]
                self.positions[symbol]["total_cost"] -= cost_basis
                
                if self.positions[symbol]["shares"] == 0:
                    del self.positions[symbol]
                
                # Record trade
                trade = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "action": "SELL",
                    "shares": shares,
                    "price": price,
                    "total_value": proceeds,
                    "confidence": confidence,
                    "reason": reason,
                    "cash_after": self.account["cash"],
                    "profit_loss": proceeds - cost_basis
                }
                self.trades.append(trade)
                return True, f"Sold {shares} shares of {symbol} at ${price:.2f}"
            else:
                return False, f"Insufficient shares. Have {self.positions.get(symbol, {}).get('shares', 0)} shares"
    
    def update_portfolio_value(self):
        """Calculate current portfolio value"""
        portfolio_value = self.account["cash"]
        
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                portfolio_value += position["shares"] * current_price
        
        self.account["total_value"] = portfolio_value
        self.account["total_return"] = portfolio_value - self.account["initial_capital"]
        self.account["total_return_pct"] = (self.account["total_return"] / self.account["initial_capital"]) * 100
        self.account["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return portfolio_value
    
    def get_position_details(self):
        """Get detailed position information"""
        position_details = []
        
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                current_value = position["shares"] * current_price
                unrealized_pl = current_value - position["total_cost"]
                unrealized_pl_pct = (unrealized_pl / position["total_cost"]) * 100
                
                position_details.append({
                    "Symbol": symbol,
                    "Shares": position["shares"],
                    "Avg Price": f"${position['avg_price']:.2f}",
                    "Current Price": f"${current_price:.2f}",
                    "Cost Basis": f"${position['total_cost']:.2f}",
                    "Current Value": f"${current_value:.2f}",
                    "Unrealized P&L": f"${unrealized_pl:.2f}",
                    "Unrealized P&L %": f"{unrealized_pl_pct:.2f}%"
                })
        
        return pd.DataFrame(position_details)
    
    def reset_account(self):
        """Reset account to initial state"""
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.positions_file):
            os.remove(self.positions_file)
        if os.path.exists(self.trades_file):
            os.remove(self.trades_file)
        self.initialize_account()

def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">🚀 Live Paper Trading Engine</h1>', unsafe_allow_html=True)
    
    # Initialize trading engine
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = PaperTradingEngine()
    
    engine = st.session_state.trading_engine
    
    # Sidebar controls
    st.sidebar.title("Trading Controls")
    
    # Reset button
    if st.sidebar.button("🔄 Reset Account", type="secondary"):
        engine.reset_account()
        st.sidebar.success("Account reset to $10,000!")
        st.rerun()
    
    # Manual trade section
    st.sidebar.subheader("Manual Trade")
    manual_symbol = st.sidebar.selectbox("Stock Symbol", [
        "AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AVGO", 
        "PLTR", "NFLX", "TSM", "PANW", "NOW", "XLK", "QQQ", "COST"
    ])
    manual_action = st.sidebar.selectbox("Action", ["BUY", "SELL"])
    manual_shares = st.sidebar.number_input("Shares", min_value=1, value=10)
    
    if st.sidebar.button("Execute Manual Trade"):
        current_price = engine.get_current_price(manual_symbol)
        if current_price:
            success, message = engine.execute_trade(
                manual_symbol, manual_action, manual_shares, current_price, 100.0, "Manual Trade"
            )
            if success:
                st.sidebar.success(message)
                engine.save_data()
            else:
                st.sidebar.error(message)
        else:
            st.sidebar.error(f"Could not get price for {manual_symbol}")
    
    # Update portfolio value
    current_portfolio_value = engine.update_portfolio_value()
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="success-card">
            <h3>💰 Portfolio Value</h3>
            <h2>${current_portfolio_value:.2f}</h2>
            <p>Starting: ${engine.account['initial_capital']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        return_color = "success-card" if engine.account['total_return'] >= 0 else "profit-card"
        st.markdown(f"""
        <div class="{return_color}">
            <h3>📈 Total Return</h3>
            <h2>${engine.account['total_return']:.2f}</h2>
            <p>({engine.account['total_return_pct']:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💵 Available Cash</h3>
            <h2>${engine.account['cash']:.2f}</h2>
            <p>Ready to invest</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        days_trading = (datetime.now() - datetime.strptime(engine.account['start_date'], "%Y-%m-%d")).days
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Days Trading</h3>
            <h2>{days_trading}</h2>
            <p>Since {engine.account['start_date']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Trading Section
    st.header("🧠 AI Trading Signals")
    
    # Initialize AI manager
    if st.button("🚀 Generate AI Trading Signals", type="primary"):
        with st.spinner("Analyzing markets with AI..."):
            try:
                # Load AI models and generate signals
                ai_manager = ImprovedAIPortfolioManager()
                
                # Get signals for our stock universe
                stocks = ["AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMZN", "AVGO", 
                         "PLTR", "NFLX", "TSM", "PANW", "NOW", "XLK", "QQQ", "COST"]
                
                signals = []
                for stock in stocks:
                    try:
                        # Get current data for the stock
                        ticker = yf.Ticker(stock)
                        current_data = ticker.history(period="60d")  # Get 60 days for feature calculation
                        
                        if not current_data.empty:
                            confidence = ai_manager.get_prediction_strength(stock, current_data)
                            current_price = engine.get_current_price(stock)
                            
                            if current_price and confidence > 55:  # Only high confidence signals
                                action = "BUY" if confidence > 75 else "HOLD"
                                if action == "BUY":
                                    # Calculate position size based on confidence
                                    max_position_value = engine.account["cash"] * 0.1  # Max 10% per position
                                    confidence_factor = min(confidence / 100, 1.0)
                                    position_value = max_position_value * confidence_factor
                                    shares = int(position_value / current_price)
                                
                                if shares > 0:
                                    signals.append({
                                        "Symbol": stock,
                                        "Action": action,
                                        "Current Price": current_price,
                                        "Target Price": f"${current_price * 1.05:.2f}",  # 5% target
                                        "Confidence": confidence,
                                        "Suggested Shares": shares,
                                        "Position Value": shares * current_price
                                    })
                            else:
                                # Add HOLD signals for tracking
                                signals.append({
                                    "Symbol": stock,
                                    "Action": "HOLD",
                                    "Current Price": current_price,
                                    "Target Price": f"${current_price:.2f}",
                                    "Confidence": confidence,
                                    "Suggested Shares": 0,
                                    "Position Value": 0
                                })
                    except Exception as e:
                        st.write(f"Error processing {stock}: {str(e)}")
                
                if signals:
                    st.subheader("🎯 Current AI Signals")
                    signals_df = pd.DataFrame(signals)
                    st.dataframe(signals_df, use_container_width=True)
                    
                    # Auto-execute high confidence signals
                    st.subheader("⚡ Auto-Execute High Confidence Trades")
                    
                    for signal in signals:
                        if signal["Confidence"] > 75:  # Very high confidence
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.write(f"**{signal['Symbol']}** - {signal['Action']} {signal['Suggested Shares']} shares at ${signal['Current Price']:.2f}")
                                st.write(f"Confidence: {signal['Confidence']:.1f}% | Value: ${signal['Position Value']:.2f}")
                            
                            with col2:
                                if st.button(f"Execute {signal['Symbol']}", key=f"exec_{signal['Symbol']}"):
                                    success, message = engine.execute_trade(
                                        signal['Symbol'], 
                                        signal['Action'], 
                                        signal['Suggested Shares'], 
                                        signal['Current Price'],
                                        signal['Confidence'],
                                        f"AI Signal - {signal['Confidence']:.1f}% confidence"
                                    )
                                    if success:
                                        st.success(message)
                                        engine.save_data()
                                        st.rerun()
                                    else:
                                        st.error(message)
                            
                            with col3:
                                st.write(f"Target: ${signal['Target Price']:.2f}")
                else:
                    st.info("No high-confidence AI signals available at this time.")
                    
            except Exception as e:
                st.error(f"Error generating AI signals: {str(e)}")
    
    # Current Positions
    st.header("📊 Current Positions")
    if engine.positions:
        positions_df = engine.get_position_details()
        st.dataframe(positions_df, use_container_width=True)
        
        # Portfolio allocation chart
        if not positions_df.empty:
            # Extract numeric values for the pie chart
            symbols = positions_df['Symbol'].tolist()
            values = [float(val.replace('$', '').replace(',', '')) for val in positions_df['Current Value'].tolist()]
            
            fig = px.pie(values=values, names=symbols, title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No current positions. Start trading to see your portfolio here!")
    
    # Recent Trades
    st.header("📋 Recent Trades")
    if engine.trades:
        # Show last 10 trades
        recent_trades = engine.trades[-10:]
        trades_df = pd.DataFrame(recent_trades)
        
        # Style the dataframe
        def color_action(val):
            if val == 'BUY':
                return 'background-color: rgba(40, 167, 69, 0.1)'
            elif val == 'SELL':
                return 'background-color: rgba(220, 53, 69, 0.1)'
            return ''
        
        styled_df = trades_df.style.applymap(color_action, subset=['action'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Performance chart
        if len(engine.trades) > 1:
            # Calculate running portfolio value
            running_values = []
            current_cash = engine.account['initial_capital']
            current_positions = {}
            
            for trade in engine.trades:
                if trade['action'] == 'BUY':
                    current_cash -= trade['total_value']
                    if trade['symbol'] in current_positions:
                        current_positions[trade['symbol']] += trade['shares']
                    else:
                        current_positions[trade['symbol']] = trade['shares']
                elif trade['action'] == 'SELL':
                    current_cash += trade['total_value']
                    current_positions[trade['symbol']] -= trade['shares']
                    if current_positions[trade['symbol']] <= 0:
                        del current_positions[trade['symbol']]
                
                # Calculate portfolio value at this point
                portfolio_value = current_cash
                for symbol, shares in current_positions.items():
                    current_price = engine.get_current_price(symbol)
                    if current_price:
                        portfolio_value += shares * current_price
                
                running_values.append({
                    'timestamp': trade['timestamp'],
                    'portfolio_value': portfolio_value
                })
            
            if running_values:
                perf_df = pd.DataFrame(running_values)
                perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
                
                fig = px.line(perf_df, x='timestamp', y='portfolio_value', 
                             title='Portfolio Performance Over Time')
                fig.add_hline(y=engine.account['initial_capital'], 
                             line_dash="dash", line_color="red",
                             annotation_text="Starting Capital")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades yet. Execute your first trade to see history here!")
    
    # Save data
    engine.save_data()
    
    # Auto-refresh
    st.markdown("---")
    st.markdown(f"*Last updated: {engine.account['last_updated']} | Auto-refreshing every 5 minutes*")
    
    # Auto-refresh every 5 minutes
    time.sleep(300)
    st.rerun()

if __name__ == "__main__":
    main()
