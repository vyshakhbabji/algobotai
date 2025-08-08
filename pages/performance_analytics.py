#!/usr/bin/env python3
"""
Performance Dashboard - Individual Stock Analysis & Backtesting
Track how each stock performs and test AI model performance
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_ai_portfolio_manager import ImprovedAIPortfolioManager
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📈 Performance Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .perf-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .positive {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
    }
    .negative {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    }
    .neutral {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

class PerformanceAnalyzer:
    def __init__(self):
        self.load_portfolio()
        self.ai_manager = ImprovedAIPortfolioManager()
    
    def load_portfolio(self):
        """Load current portfolio universe"""
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        portfolio_file = os.path.join(parent_dir, "portfolio_universe.json")
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                data = json.load(f)
                self.stocks = data.get('stocks', [])
        else:
            # Fallback to default stocks
            self.stocks = [
                'GOOG', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'AVGO', 'PLTR', 
                'NFLX', 'TSM', 'PANW', 'NOW', 'XLK', 'QQQ', 'COST'
            ]
    
    def get_stock_performance(self, symbol, period="1y"):
        """Get comprehensive stock performance data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist = ticker.history(period=period)
            if hist.empty:
                return None
            
            # Calculate performance metrics
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            total_return = (current_price - start_price) / start_price * 100
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Max drawdown
            rolling_max = hist['Close'].expanding().max()
            drawdown = (hist['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Sharpe ratio (simplified, assuming 0% risk-free rate)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            
            # Get company info
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'start_price': start_price,
                'total_return': total_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'historical_data': hist
            }
        except Exception as e:
            st.error(f"Error getting data for {symbol}: {str(e)}")
            return None
    
    def run_forward_backtest(self, symbol, train_months=12, test_months=3):
        """Run forward backtest: train on X months, test on next Y months"""
        try:
            # Calculate date ranges
            end_date = datetime.now() - timedelta(days=test_months * 30)
            start_date = end_date - timedelta(days=train_months * 30)
            test_end_date = datetime.now()
            
            ticker = yf.Ticker(symbol)
            
            # Get training data
            train_data = ticker.history(start=start_date, end=end_date)
            if len(train_data) < 50:
                return None
            
            # Get test data
            test_data = ticker.history(start=end_date, end=test_end_date)
            if len(test_data) < 20:
                return None
            
            # Train AI model on historical data
            self.ai_manager.stock_universe = [symbol]  # Focus on this stock only
            model, scaler, r2_score = self.ai_manager.train_improved_model(symbol, data=train_data)
            
            if model is None:
                return None
            
            # Test on forward data
            test_predictions = []
            test_returns = []
            
            for i in range(len(test_data) - 5):  # Need 5 days ahead for target
                # Get features for current day
                current_slice = test_data.iloc[:i+50] if i+50 < len(test_data) else test_data.iloc[:i+1]
                
                if len(current_slice) < 50:
                    continue
                
                # Calculate features
                df_features = current_slice.copy()
                df_features = self.ai_manager.calculate_improved_features(df_features)
                
                feature_cols = [
                    'price_vs_sma10', 'price_vs_sma30', 'price_vs_sma50',
                    'momentum_5', 'momentum_10', 'momentum_20',
                    'volatility_10', 'volatility_30',
                    'volume_ratio', 'volume_momentum',
                    'rsi_normalized', 'bb_position', 'bb_squeeze',
                    'macd', 'macd_histogram', 'price_position'
                ]
                
                latest_features = df_features[feature_cols].iloc[-1:].values
                
                if np.isnan(latest_features).any():
                    continue
                
                # Make prediction
                features_scaled = scaler.transform(latest_features)
                predicted_return = model.predict(features_scaled)[0]
                
                # Get actual return (5 days ahead)
                if i + 5 < len(test_data):
                    current_price = test_data['Close'].iloc[i]
                    future_price = test_data['Close'].iloc[i + 5]
                    actual_return = (future_price - current_price) / current_price
                    
                    test_predictions.append(predicted_return)
                    test_returns.append(actual_return)
            
            if not test_predictions:
                return None
            
            # Calculate backtest results with multiple strategies
            predictions_array = np.array(test_predictions)
            returns_array = np.array(test_returns)
            
            # Strategy 1: Conservative (1% threshold)
            buy_signals_conservative = predictions_array > 0.01
            strategy_returns_conservative = np.where(buy_signals_conservative, returns_array, 0)
            
            # Strategy 2: Moderate (0.5% threshold) 
            buy_signals_moderate = predictions_array > 0.005
            strategy_returns_moderate = np.where(buy_signals_moderate, returns_array, 0)
            
            # Strategy 3: Aggressive (0.1% threshold)
            buy_signals_aggressive = predictions_array > 0.001
            strategy_returns_aggressive = np.where(buy_signals_aggressive, returns_array, 0)
            
            # Use moderate strategy as default
            buy_signals = buy_signals_moderate
            strategy_returns = strategy_returns_moderate
            
            # Calculate cumulative performance for all strategies
            cumulative_strategy = (1 + strategy_returns).cumprod()
            cumulative_market = (1 + returns_array).cumprod()
            cumulative_conservative = (1 + strategy_returns_conservative).cumprod()
            cumulative_aggressive = (1 + strategy_returns_aggressive).cumprod()
            
            strategy_total_return = (cumulative_strategy[-1] - 1) * 100
            market_total_return = (cumulative_market[-1] - 1) * 100
            conservative_return = (cumulative_conservative[-1] - 1) * 100
            aggressive_return = (cumulative_aggressive[-1] - 1) * 100
            
            return {
                'symbol': symbol,
                'train_period': f"{train_months} months",
                'test_period': f"{test_months} months",
                'model_r2': r2_score,
                'total_predictions': len(test_predictions),
                'buy_signals': np.sum(buy_signals),
                'buy_signals_conservative': np.sum(buy_signals_conservative),
                'buy_signals_aggressive': np.sum(buy_signals_aggressive),
                'strategy_return': strategy_total_return,
                'market_return': market_total_return,
                'conservative_return': conservative_return,
                'aggressive_return': aggressive_return,
                'alpha': strategy_total_return - market_total_return,
                'predictions': test_predictions,
                'actual_returns': test_returns,
                'dates': test_data.index[-len(test_predictions):].tolist()
            }
            
        except Exception as e:
            st.error(f"Backtest error for {symbol}: {str(e)}")
            return None

def main():
    """Main Performance Dashboard"""
    st.markdown('<h1 class="main-header">📈 Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Individual Stock Analysis & AI Model Performance**")
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    
    if not analyzer.stocks:
        st.warning("No stocks in portfolio! Please add stocks in the Portfolio Manager first.")
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analysis Controls")
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock for Detailed Analysis",
        analyzer.stocks
    )
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1y", "2y", "3y", "5y"],
        index=1
    )
    
    # Backtest parameters
    st.sidebar.subheader("🧪 Backtest Settings")
    train_months = st.sidebar.slider("Training Period (months)", 6, 24, 12)
    test_months = st.sidebar.slider("Testing Period (months)", 1, 6, 3)
    
    # Portfolio Overview
    st.subheader("📊 Portfolio Performance Overview")
    
    with st.spinner("Loading portfolio performance..."):
        portfolio_perf = []
        
        for symbol in analyzer.stocks[:10]:  # Limit to first 10 for speed
            perf = analyzer.get_stock_performance(symbol, period=time_period)
            if perf:
                portfolio_perf.append(perf)
        
        if portfolio_perf:
            # Create performance summary
            df_perf = pd.DataFrame([{
                'Symbol': p['symbol'],
                'Price': f"${p['current_price']:.2f}",
                'Return (%)': f"{p['total_return']:.1f}%",
                'Volatility (%)': f"{p['volatility']:.1f}%",
                'Max Drawdown (%)': f"{p['max_drawdown']:.1f}%",
                'Sharpe Ratio': f"{p['sharpe_ratio']:.2f}",
                'Sector': p['sector']
            } for p in portfolio_perf])
            
            st.dataframe(df_perf, use_container_width=True)
            
            # Performance cards
            col1, col2, col3, col4 = st.columns(4)
            
            avg_return = np.mean([p['total_return'] for p in portfolio_perf])
            avg_volatility = np.mean([p['volatility'] for p in portfolio_perf])
            avg_sharpe = np.mean([p['sharpe_ratio'] for p in portfolio_perf if not np.isnan(p['sharpe_ratio'])])
            
            with col1:
                card_class = "positive" if avg_return > 0 else "negative"
                st.markdown(f'''
                <div class="perf-card {card_class}">
                    <h3>{avg_return:.1f}%</h3>
                    <p>Avg Return</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="perf-card neutral">
                    <h3>{avg_volatility:.1f}%</h3>
                    <p>Avg Volatility</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                card_class = "positive" if avg_sharpe > 1 else "neutral"
                st.markdown(f'''
                <div class="perf-card {card_class}">
                    <h3>{avg_sharpe:.2f}</h3>
                    <p>Avg Sharpe</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'''
                <div class="perf-card neutral">
                    <h3>{len(portfolio_perf)}</h3>
                    <p>Stocks Analyzed</p>
                </div>
                ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed Stock Analysis
    st.subheader(f"🔍 Detailed Analysis: {selected_stock}")
    
    # Get detailed performance for selected stock
    with st.spinner(f"Analyzing {selected_stock}..."):
        detailed_perf = analyzer.get_stock_performance(selected_stock, period=time_period)
        
        if detailed_perf:
            # Stock info cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${detailed_perf['current_price']:.2f}"
                )
            
            with col2:
                st.metric(
                    "Total Return",
                    f"{detailed_perf['total_return']:.1f}%",
                    delta=f"{detailed_perf['total_return']:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Volatility",
                    f"{detailed_perf['volatility']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "Max Drawdown",
                    f"{detailed_perf['max_drawdown']:.1f}%"
                )
            
            # Price chart
            st.subheader("📈 Price Chart")
            
            fig = go.Figure()
            
            hist_data = detailed_perf['historical_data']
            
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"{selected_stock} Price Performance",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_volume = go.Figure()
            
            fig_volume.add_trace(go.Bar(
                x=hist_data.index,
                y=hist_data['Volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.8)'
            ))
            
            fig_volume.update_layout(
                title=f"{selected_stock} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        else:
            st.error(f"Could not load performance data for {selected_stock}")
    
    st.markdown("---")
    
    # AI Model Backtest
    st.subheader("🧠 AI Model Backtest")
    st.markdown(f"**Forward Testing**: Train on {train_months} months, test on next {test_months} months")
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner(f"Running AI backtest for {selected_stock}..."):
            backtest_result = analyzer.run_forward_backtest(
                selected_stock, 
                train_months=train_months, 
                test_months=test_months
            )
            
            if backtest_result:
                # Backtest results - Multiple strategies
                st.subheader("🎯 Strategy Comparison")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    card_class = "positive" if backtest_result['strategy_return'] > 0 else "negative"
                    st.markdown(f'''
                    <div class="perf-card {card_class}">
                        <h3>{backtest_result['strategy_return']:.1f}%</h3>
                        <p>Moderate Strategy (0.5%)</p>
                        <small>{backtest_result['buy_signals']} signals</small>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    card_class = "positive" if backtest_result['conservative_return'] > 0 else "negative"
                    st.markdown(f'''
                    <div class="perf-card {card_class}">
                        <h3>{backtest_result['conservative_return']:.1f}%</h3>
                        <p>Conservative (1%)</p>
                        <small>{backtest_result['buy_signals_conservative']} signals</small>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    card_class = "positive" if backtest_result['aggressive_return'] > 0 else "negative"
                    st.markdown(f'''
                    <div class="perf-card {card_class}">
                        <h3>{backtest_result['aggressive_return']:.1f}%</h3>
                        <p>Aggressive (0.1%)</p>
                        <small>{backtest_result['buy_signals_aggressive']} signals</small>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col4:
                    card_class = "positive" if backtest_result['market_return'] > 0 else "negative"
                    st.markdown(f'''
                    <div class="perf-card {card_class}">
                        <h3>{backtest_result['market_return']:.1f}%</h3>
                        <p>Market (Buy & Hold)</p>
                        <small>Baseline</small>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f'''
                    <div class="perf-card neutral">
                        <h3>{backtest_result['buy_signals']}</h3>
                        <p>Buy Signals</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Additional metrics
                st.subheader("📊 Backtest Details")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model R² Score**: {backtest_result['model_r2']:.3f}")
                    st.write(f"**Total Predictions**: {backtest_result['total_predictions']}")
                    st.write(f"**Training Period**: {backtest_result['train_period']}")
                    st.write(f"**Testing Period**: {backtest_result['test_period']}")
                
                with col2:
                    signal_rate = (backtest_result['buy_signals'] / backtest_result['total_predictions']) * 100
                    st.write(f"**Buy Signal Rate**: {signal_rate:.1f}%")
                    st.write(f"**Strategy Outperformed**: {'✅ Yes' if backtest_result['alpha'] > 0 else '❌ No'}")
                    st.write(f"**Prediction Accuracy**: Model dependent")
                
                # Performance visualization
                if len(backtest_result['predictions']) > 0:
                    st.subheader("📈 Prediction vs Reality")
                    
                    df_predictions = pd.DataFrame({
                        'Date': backtest_result['dates'],
                        'Predicted_Return': backtest_result['predictions'],
                        'Actual_Return': backtest_result['actual_returns']
                    })
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_predictions['Date'],
                        y=df_predictions['Predicted_Return'],
                        mode='lines+markers',
                        name='AI Predictions',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_predictions['Date'],
                        y=df_predictions['Actual_Return'],
                        mode='lines+markers',
                        name='Actual Returns',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title=f"AI Predictions vs Actual Returns - {selected_stock}",
                        xaxis_title="Date",
                        yaxis_title="Return",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("Could not run backtest. Insufficient data or model training failed.")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Note**: Past performance does not guarantee future results. This is for educational purposes only.")

if __name__ == "__main__":
    main()
