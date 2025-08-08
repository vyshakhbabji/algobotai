#!/usr/bin/env python3
"""
Alpaca Paper Trading Engine
Real-time paper trading with Alpaca Markets API
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import plotly.graph_objects as go
import plotly.express as px

class AlpacaPaperTradingEngine:
    """Advanced Alpaca Paper Trading Engine with Real Market Data"""
    
    def __init__(self):
        self.setup_credentials()
        self.initialize_clients()
        self.load_configuration()
    
    def setup_credentials(self):
        """Setup Alpaca API credentials"""
        # Check for credentials in environment or streamlit secrets
        if hasattr(st, 'secrets') and 'alpaca' in st.secrets:
            self.api_key = st.secrets['alpaca']['api_key']
            self.secret_key = st.secrets['alpaca']['secret_key']
        else:
            # For local development - you'll need to add these
            self.api_key = os.getenv('ALPACA_API_KEY', '')
            self.secret_key = os.getenv('ALPACA_SECRET_KEY', '')
        
        self.base_url = 'https://paper-api.alpaca.markets'  # Paper trading URL
        
    def initialize_clients(self):
        """Initialize Alpaca trading and data clients"""
        try:
            if self.api_key and self.secret_key:
                # Trading client for orders and account info
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=True  # Enable paper trading
                )
                
                # Data client for market data
                self.data_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
                
                self.connected = True
                st.success("✅ Connected to Alpaca Paper Trading!")
            else:
                self.connected = False
                st.error("❌ Alpaca API credentials not found!")
                
        except Exception as e:
            self.connected = False
            st.error(f"❌ Failed to connect to Alpaca: {str(e)}")
    
    def load_configuration(self):
        """Load trading configuration"""
        self.config = {
            "max_position_size": 0.1,  # Max 10% per position
            "stop_loss_pct": 0.05,     # 5% stop loss
            "take_profit_pct": 0.15,   # 15% take profit
            "min_volume": 100000,      # Minimum daily volume
            "max_positions": 10        # Maximum number of positions
        }
    
    def get_account_info(self):
        """Get current account information"""
        if not self.connected:
            return None
            
        try:
            account = self.trading_client.get_account()
            return {
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'day_trade_count': account.daytrade_count,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            st.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        if not self.connected:
            return []
            
        try:
            positions = self.trading_client.get_all_positions()
            position_list = []
            
            for position in positions:
                position_list.append({
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price)
                })
            
            return position_list
            
        except Exception as e:
            st.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        if not self.connected:
            return None
            
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            latest_quote = self.data_client.get_stock_latest_quote(request)
            
            if symbol in latest_quote:
                quote = latest_quote[symbol]
                return (float(quote.bid_price) + float(quote.ask_price)) / 2
            else:
                return None
                
        except Exception as e:
            st.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def place_market_order(self, symbol, quantity, side):
        """Place a market order"""
        if not self.connected:
            return False, "Not connected to Alpaca"
            
        try:
            # Prepare the order request
            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit the order
            market_order = self.trading_client.submit_order(order_data=market_order_data)
            
            return True, f"Order placed successfully: {market_order.id}"
            
        except Exception as e:
            return False, f"Error placing order: {str(e)}"
    
    def place_limit_order(self, symbol, quantity, side, limit_price):
        """Place a limit order"""
        if not self.connected:
            return False, "Not connected to Alpaca"
            
        try:
            # Prepare the limit order request
            limit_order_data = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            # Submit the order
            limit_order = self.trading_client.submit_order(order_data=limit_order_data)
            
            return True, f"Limit order placed successfully: {limit_order.id}"
            
        except Exception as e:
            return False, f"Error placing limit order: {str(e)}"
    
    def get_orders(self, status="open"):
        """Get orders by status"""
        if not self.connected:
            return []
            
        try:
            orders = self.trading_client.get_orders()
            order_list = []
            
            for order in orders:
                if status == "all" or order.status.lower() == status.lower():
                    order_list.append({
                        'id': order.id,
                        'symbol': order.symbol,
                        'quantity': float(order.qty),
                        'side': order.side.value,
                        'order_type': order.order_type.value,
                        'status': order.status.value,
                        'limit_price': float(order.limit_price) if order.limit_price else None,
                        'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                        'submitted_at': order.submitted_at
                    })
            
            return order_list
            
        except Exception as e:
            st.error(f"Error getting orders: {str(e)}")
            return []
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        if not self.connected:
            return False, "Not connected to Alpaca"
            
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True, "Order cancelled successfully"
            
        except Exception as e:
            return False, f"Error cancelling order: {str(e)}"
    
    def calculate_position_size(self, symbol, price, account_value):
        """Calculate optimal position size based on risk management"""
        max_investment = account_value * self.config["max_position_size"]
        shares = int(max_investment / price)
        return max(shares, 1)  # At least 1 share
    
    def analyze_stock(self, symbol):
        """Perform basic stock analysis"""
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            
            if current_price:
                # Calculate position sizing recommendations
                account_info = self.get_account_info()
                if account_info:
                    recommended_shares = self.calculate_position_size(
                        symbol, current_price, account_info['portfolio_value']
                    )
                    
                    return {
                        'symbol': symbol,
                        'current_price': current_price,
                        'recommended_shares': recommended_shares,
                        'investment_amount': current_price * recommended_shares,
                        'stop_loss_price': current_price * (1 - self.config["stop_loss_pct"]),
                        'take_profit_price': current_price * (1 + self.config["take_profit_pct"])
                    }
            
            return None
            
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")
            return None

def main():
    """Main Alpaca Paper Trading Interface"""
    st.title("🏦 Alpaca Paper Trading Engine")
    st.markdown("**Real-time paper trading with Alpaca Markets**")
    
    # Initialize the trading engine
    if 'alpaca_engine' not in st.session_state:
        st.session_state.alpaca_engine = AlpacaPaperTradingEngine()
    
    engine = st.session_state.alpaca_engine
    
    # Setup instructions if not connected
    if not engine.connected:
        st.warning("⚠️ **Alpaca API Setup Required**")
        st.markdown("""
        ### 🔧 Setup Instructions:
        
        1. **Create Alpaca Account**: Go to [alpaca.markets](https://alpaca.markets) and sign up
        2. **Get Paper Trading Keys**: 
           - Login to your Alpaca account
           - Go to "Paper Trading" section
           - Generate API Key and Secret Key
        3. **Add Credentials**:
           - For Streamlit Cloud: Add to secrets.toml
           - For local: Set environment variables
        
        ```toml
        # secrets.toml (for Streamlit Cloud)
        [alpaca]
        api_key = "YOUR_API_KEY"
        secret_key = "YOUR_SECRET_KEY"
        ```
        
        ```bash
        # Environment variables (for local)
        export ALPACA_API_KEY="your_api_key"
        export ALPACA_SECRET_KEY="your_secret_key"
        ```
        """)
        return
    
    # Main dashboard if connected
    st.success("🟢 **Connected to Alpaca Paper Trading**")
    
    # Account overview
    account_info = engine.get_account_info()
    if account_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Value", f"${account_info['portfolio_value']:,.2f}")
        with col2:
            st.metric("Buying Power", f"${account_info['buying_power']:,.2f}")
        with col3:
            st.metric("Cash", f"${account_info['cash']:,.2f}")
        with col4:
            st.metric("Equity", f"${account_info['equity']:,.2f}")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Trading", "💼 Positions", "📋 Orders", "⚙️ Settings"])
    
    with tab1:
        st.subheader("🎯 Stock Trading")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stock symbol input
            symbol = st.text_input("Stock Symbol", value="AAPL", key="alpaca_symbol").upper()
            
            if symbol:
                # Analyze the stock
                analysis = engine.analyze_stock(symbol)
                
                if analysis:
                    st.success(f"✅ **{symbol}** - Current Price: ${analysis['current_price']:.2f}")
                    
                    # Trading form
                    with st.form(f"trade_form_{symbol}"):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            order_type = st.selectbox("Order Type", ["Market", "Limit"])
                        
                        with col_b:
                            side = st.selectbox("Side", ["BUY", "SELL"])
                        
                        with col_c:
                            quantity = st.number_input("Quantity", 
                                                     value=analysis['recommended_shares'], 
                                                     min_value=1)
                        
                        if order_type == "Limit":
                            limit_price = st.number_input("Limit Price", 
                                                        value=analysis['current_price'], 
                                                        min_value=0.01,
                                                        step=0.01)
                        
                        submitted = st.form_submit_button("Place Order", type="primary")
                        
                        if submitted:
                            if order_type == "Market":
                                success, message = engine.place_market_order(symbol, quantity, side)
                            else:
                                success, message = engine.place_limit_order(symbol, quantity, side, limit_price)
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
        
        with col2:
            st.subheader("📈 Analysis")
            if symbol and 'analysis' in locals() and analysis:
                st.metric("Recommended Shares", analysis['recommended_shares'])
                st.metric("Investment Amount", f"${analysis['investment_amount']:,.2f}")
                st.metric("Stop Loss", f"${analysis['stop_loss_price']:.2f}")
                st.metric("Take Profit", f"${analysis['take_profit_price']:.2f}")
    
    with tab2:
        st.subheader("💼 Current Positions")
        
        positions = engine.get_positions()
        
        if positions:
            df = pd.DataFrame(positions)
            
            # Format the dataframe for display
            df['market_value'] = df['market_value'].apply(lambda x: f"${x:,.2f}")
            df['avg_entry_price'] = df['avg_entry_price'].apply(lambda x: f"${x:.2f}")
            df['current_price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
            df['unrealized_pl'] = df['unrealized_pl'].apply(lambda x: f"${x:,.2f}")
            df['unrealized_plpc'] = df['unrealized_plpc'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(df, use_container_width=True)
            
            # Position summary
            total_positions = len(positions)
            total_value = sum([float(p['market_value'].replace('$', '').replace(',', '')) for p in positions])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Positions", total_positions)
            with col2:
                st.metric("Total Position Value", f"${total_value:,.2f}")
        else:
            st.info("📭 No current positions")
    
    with tab3:
        st.subheader("📋 Orders")
        
        # Order status filter
        status_filter = st.selectbox("Filter by Status", ["open", "filled", "cancelled", "all"])
        
        orders = engine.get_orders(status_filter)
        
        if orders:
            # Display orders
            for order in orders:
                with st.expander(f"{order['symbol']} - {order['side']} {order['quantity']} shares ({order['status']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Order ID**: {order['id']}")
                        st.write(f"**Type**: {order['order_type']}")
                        st.write(f"**Status**: {order['status']}")
                    
                    with col2:
                        if order['limit_price']:
                            st.write(f"**Limit Price**: ${order['limit_price']:.2f}")
                        st.write(f"**Filled**: {order['filled_qty']}/{order['quantity']}")
                        st.write(f"**Submitted**: {order['submitted_at']}")
                    
                    with col3:
                        if order['status'].lower() == 'open':
                            if st.button(f"Cancel Order", key=f"cancel_{order['id']}"):
                                success, message = engine.cancel_order(order['id'])
                                if success:
                                    st.success("✅ Order cancelled")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
        else:
            st.info("📭 No orders found")
    
    with tab4:
        st.subheader("⚙️ Trading Settings")
        
        st.markdown("### 🎛️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_position = st.slider("Max Position Size (%)", 
                                   min_value=1, 
                                   max_value=50, 
                                   value=int(engine.config["max_position_size"]*100))
            
            stop_loss = st.slider("Stop Loss (%)", 
                                 min_value=1, 
                                 max_value=20, 
                                 value=int(engine.config["stop_loss_pct"]*100))
        
        with col2:
            take_profit = st.slider("Take Profit (%)", 
                                   min_value=5, 
                                   max_value=50, 
                                   value=int(engine.config["take_profit_pct"]*100))
            
            max_positions = st.slider("Max Positions", 
                                     min_value=1, 
                                     max_value=20, 
                                     value=engine.config["max_positions"])
        
        if st.button("💾 Save Settings"):
            engine.config.update({
                "max_position_size": max_position/100,
                "stop_loss_pct": stop_loss/100,
                "take_profit_pct": take_profit/100,
                "max_positions": max_positions
            })
            st.success("✅ Settings saved!")

if __name__ == "__main__":
    main()
