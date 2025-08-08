#!/usr/bin/env python3
"""
Portfolio Manager - Add/Remove Stocks for AI Trading
Manage up to 50 stocks in the trading universe
"""

import streamlit as st
import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_ai_portfolio_manager import ImprovedAIPortfolioManager

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
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
    .stock-card {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .remove-stock {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class PortfolioManager:
    def __init__(self):
        # Point to parent directory for JSON files
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.portfolio_file = os.path.join(parent_dir, "portfolio_universe.json")
        self.max_stocks = 50
        self.load_portfolio()
    
    def load_portfolio(self):
        """Load current portfolio universe"""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r') as f:
                data = json.load(f)
                self.stocks = data.get('stocks', [])
                self.created_date = data.get('created_date', datetime.now().strftime("%Y-%m-%d"))
                self.last_updated = data.get('last_updated', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            # Default stocks
            self.stocks = [
                'GOOG', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'AVGO', 'PLTR', 
                'NFLX', 'TSM', 'PANW', 'NOW', 'XLK', 'QQQ', 'COST'
            ]
            self.created_date = datetime.now().strftime("%Y-%m-%d")
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_portfolio()
    
    def save_portfolio(self):
        """Save portfolio universe to file"""
        data = {
            'stocks': self.stocks,
            'created_date': self.created_date,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_stocks': len(self.stocks),
            'max_capacity': self.max_stocks
        }
        with open(self.portfolio_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.last_updated = data['last_updated']
    
    def add_stock(self, symbol):
        """Add a stock to the portfolio"""
        symbol = symbol.upper().strip()
        
        if len(self.stocks) >= self.max_stocks:
            return False, f"Portfolio full! Maximum {self.max_stocks} stocks allowed."
        
        if symbol in self.stocks:
            return False, f"{symbol} is already in your portfolio."
        
        # Validate stock exists
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info or 'symbol' not in info:
                return False, f"{symbol} is not a valid stock symbol."
        except:
            return False, f"Could not validate {symbol}. Please check the symbol."
        
        self.stocks.append(symbol)
        self.save_portfolio()
        return True, f"✅ {symbol} added successfully!"
    
    def remove_stock(self, symbol):
        """Remove a stock from the portfolio"""
        if symbol in self.stocks:
            self.stocks.remove(symbol)
            self.save_portfolio()
            return True, f"✅ {symbol} removed successfully!"
        return False, f"{symbol} not found in portfolio."
    
    def get_stock_info(self, symbol):
        """Get basic info about a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            return {
                'symbol': symbol,
                'price': current_price,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0)
            }
        except:
            return None

def main():
    """Main Portfolio Manager Interface"""
    st.markdown('<h1 class="main-header">📊 Portfolio Manager</h1>', unsafe_allow_html=True)
    st.markdown("**Manage your AI trading universe (Max 50 stocks)**")
    
    # Initialize portfolio manager
    pm = PortfolioManager()
    
    # Portfolio overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{len(pm.stocks)}</h3>
            <p>Active Stocks</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        remaining = pm.max_stocks - len(pm.stocks)
        st.markdown(f'''
        <div class="metric-card">
            <h3>{remaining}</h3>
            <p>Slots Available</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{pm.max_stocks}</h3>
            <p>Max Capacity</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        capacity_pct = (len(pm.stocks) / pm.max_stocks) * 100
        st.markdown(f'''
        <div class="metric-card">
            <h3>{capacity_pct:.1f}%</h3>
            <p>Portfolio Full</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Add new stock section
    st.subheader("➕ Add New Stock")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_symbol = st.text_input(
            "Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA)", 
            placeholder="AAPL",
            key="new_stock"
        ).upper()
    
    with col2:
        if st.button("🔍 Validate & Add", type="primary"):
            if new_symbol:
                success, message = pm.add_stock(new_symbol)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter a stock symbol.")
    
    # Quick add popular stocks
    st.subheader("⚡ Quick Add Popular Stocks")
    popular_stocks = [
        'TSLA', 'SHOP', 'ROKU', 'SQ', 'PYPL', 'ZOOM', 'DOCU', 'TWLO', 
        'OKTA', 'DDOG', 'SNOW', 'CRWD', 'NET', 'FSLY', 'MDB'
    ]
    
    # Filter out stocks already in portfolio
    available_popular = [s for s in popular_stocks if s not in pm.stocks]
    
    if available_popular:
        cols = st.columns(5)
        for i, symbol in enumerate(available_popular[:10]):  # Show first 10
            with cols[i % 5]:
                if st.button(f"+ {symbol}", key=f"add_{symbol}"):
                    success, message = pm.add_stock(symbol)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    st.markdown("---")
    
    # Current portfolio
    st.subheader("📈 Current Portfolio Universe")
    
    if pm.stocks:
        # Get stock info for display
        stock_data = []
        
        with st.spinner("Loading stock information..."):
            for symbol in pm.stocks:
                info = pm.get_stock_info(symbol)
                if info:
                    stock_data.append(info)
                else:
                    stock_data.append({
                        'symbol': symbol,
                        'price': 0,
                        'name': 'Data Unavailable',
                        'sector': 'Unknown',
                        'market_cap': 0
                    })
        
        # Display as cards
        cols_per_row = 3
        for i in range(0, len(stock_data), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < len(stock_data):
                    stock = stock_data[i + j]
                    
                    with cols[j]:
                        market_cap_text = f"${stock['market_cap']/1e9:.1f}B" if stock['market_cap'] > 0 else "N/A"
                        
                        st.markdown(f'''
                        <div class="stock-card">
                            <h4>{stock['symbol']}</h4>
                            <p><strong>${stock['price']:.2f}</strong></p>
                            <p>{stock['name'][:30]}...</p>
                            <p>Sector: {stock['sector']}</p>
                            <p>Market Cap: {market_cap_text}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        if st.button(f"🗑️ Remove {stock['symbol']}", key=f"remove_{stock['symbol']}"):
                            success, message = pm.remove_stock(stock['symbol'])
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
        
        # Portfolio statistics
        st.markdown("---")
        st.subheader("📊 Portfolio Statistics")
        
        if stock_data:
            df = pd.DataFrame(stock_data)
            
            # Sector distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏢 Sector Distribution")
                sector_counts = df['sector'].value_counts()
                st.bar_chart(sector_counts)
            
            with col2:
                st.subheader("💰 Market Cap Distribution")
                df['market_cap_range'] = pd.cut(df['market_cap'], 
                                               bins=[0, 1e9, 10e9, 100e9, float('inf')],
                                               labels=['<$1B', '$1B-$10B', '$10B-$100B', '>$100B'])
                cap_counts = df['market_cap_range'].value_counts()
                st.bar_chart(cap_counts)
    
    else:
        st.info("No stocks in portfolio. Add some stocks to get started!")
    
    # Export/Import section
    st.markdown("---")
    st.subheader("💾 Portfolio Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Portfolio", type="secondary"):
            portfolio_data = {
                'stocks': pm.stocks,
                'exported_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_stocks': len(pm.stocks)
            }
            st.download_button(
                label="⬇️ Download portfolio.json",
                data=json.dumps(portfolio_data, indent=2),
                file_name=f"portfolio_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        st.info("💡 **Pro Tip**: Choose a diverse mix of sectors and market caps for better AI performance!")
    
    # Footer info
    st.markdown("---")
    st.markdown(f"**Last Updated**: {pm.last_updated} | **Created**: {pm.created_date}")
    
    # Auto-refresh
    if st.button("🔄 Refresh Portfolio", type="secondary"):
        st.rerun()

if __name__ == "__main__":
    main()
