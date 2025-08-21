import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import uuid
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to Python path to import our elite AI trader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from elite_ai_trader import EliteAITrader
except ImportError:
    st.error("Elite AI Trader module not found. Please ensure elite_ai_trader.py is in the root directory.")
    st.stop()

# Streamlit page configuration - only set if not already set
if "page_config_set" not in st.session_state:
    try:
        st.set_page_config(
            page_title="🚀 Elite AI Trader",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.session_state.page_config_set = True
    except:
        pass

# Initialize session state with unique keys
def init_session_state():
    session_id = str(uuid.uuid4())[:8]
    
    if f"elite_trader_{session_id}" not in st.session_state:
        st.session_state[f"elite_trader_{session_id}"] = None
    if f"predictions_{session_id}" not in st.session_state:
        st.session_state[f"predictions_{session_id}"] = None
    if f"model_performance_{session_id}" not in st.session_state:
        st.session_state[f"model_performance_{session_id}"] = None
    if f"training_status_{session_id}" not in st.session_state:
        st.session_state[f"training_status_{session_id}"] = "Not Started"
        
    return session_id

def main():
    session_id = init_session_state()
    
    # Elite AI Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🚀 ELITE AI TRADER 🤖
        </h1>
        <h3 style="color: #f0f0f0; text-align: center; margin: 0.5rem 0 0 0;">
            Next-Generation Ensemble ML Trading Engine
        </h3>
        <p style="color: #e0e0e0; text-align: center; margin: 0.5rem 0 0 0;">
            5 Advanced Models • 87+ Features • Real-time Predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Controls
    st.sidebar.markdown("## 🎯 Elite AI Controls")
    
    # Stock Selection
    st.sidebar.markdown("### 📈 Stock Universe")
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    
    selected_stocks = st.sidebar.multiselect(
        "Select stocks for Elite AI analysis:",
        options=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 
                'SPY', 'QQQ', 'IWM', 'ARKK', 'BTC-USD', 'ETH-USD'],
        default=default_stocks[:5],
        help="Choose 3-8 stocks for optimal ensemble performance"
    )
    
    # Training Configuration
    st.sidebar.markdown("### 🤖 AI Configuration")
    training_period = st.sidebar.selectbox(
        "Training Period",
        ["3y", "2y", "1y", "6mo"],
        index=0,
        help="More data = better AI predictions"
    )
    
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        ["1d", "3d", "1w", "2w"],
        index=0,
        help="How far ahead to predict"
    )
    
    # Elite AI Training Button
    if st.sidebar.button("🚀 TRAIN ELITE AI", type="primary", use_container_width=True):
        if len(selected_stocks) < 3:
            st.sidebar.error("Please select at least 3 stocks")
        elif len(selected_stocks) > 10:
            st.sidebar.error("Please select maximum 10 stocks")
        else:
            st.session_state[f"training_status_{session_id}"] = "Training..."
            st.rerun()
    
    # Main Content Area
    if st.session_state[f"training_status_{session_id}"] == "Training...":
        train_elite_ai(session_id, selected_stocks, training_period)
    elif st.session_state[f"predictions_{session_id}"] is not None:
        display_elite_predictions(session_id)
    else:
        display_elite_dashboard()

def train_elite_ai(session_id, symbols, period):
    """Train the Elite AI Ensemble"""
    
    st.markdown("## 🤖 Elite AI Training in Progress...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize Elite AI Trader
        status_text.text("🚀 Initializing Elite AI Trader...")
        progress_bar.progress(10)
        
        trader = EliteAITrader()
        st.session_state[f"elite_trader_{session_id}"] = trader
        
        # Train models
        status_text.text(f"🎯 Training ensemble models for {len(symbols)} stocks...")
        progress_bar.progress(30)
        
        # Train the ensemble
        results = trader.train_ensemble(symbols)
        progress_bar.progress(80)
        
        # Generate predictions
        status_text.text("🔮 Generating elite predictions...")
        predictions = trader.predict(symbols)
        
        # Store results
        st.session_state[f"predictions_{session_id}"] = predictions
        st.session_state[f"model_performance_{session_id}"] = results
        st.session_state[f"training_status_{session_id}"] = "Complete"
        
        progress_bar.progress(100)
        status_text.text("✅ Elite AI Training Complete!")
        
        st.success("🚀 Elite AI Ensemble successfully trained and ready for predictions!")
        st.balloons()
        
        # Auto-refresh to show results
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Elite AI Training Error: {str(e)}")
        st.session_state[f"training_status_{session_id}"] = "Error"

def display_elite_predictions(session_id):
    """Display Elite AI Predictions and Analysis"""
    
    predictions_df = st.session_state[f"predictions_{session_id}"]
    
    if predictions_df is None or predictions_df.empty:
        st.warning("No predictions available. Please train the Elite AI first.")
        return
    
    # Elite Predictions Summary
    st.markdown("## 🔮 Elite AI Predictions")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        buy_signals = len(predictions_df[predictions_df['signal'] == 'BUY'])
        st.metric("🟢 BUY Signals", buy_signals)
    
    with col2:
        sell_signals = len(predictions_df[predictions_df['signal'] == 'SELL'])
        st.metric("🔴 SELL Signals", sell_signals)
    
    with col3:
        hold_signals = len(predictions_df[predictions_df['signal'] == 'HOLD'])
        st.metric("🟡 HOLD Signals", hold_signals)
    
    with col4:
        avg_confidence = predictions_df['confidence'].mean()
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
    
    # Predictions Table
    st.markdown("### 📊 Elite Predictions Table")
    
    # Style the predictions dataframe
    def style_predictions(df):
        def color_predictions(val):
            if val > 1:
                return 'color: #00ff00; font-weight: bold'  # Green for positive
            elif val < -1:
                return 'color: #ff4444; font-weight: bold'  # Red for negative
            else:
                return 'color: #ffaa00; font-weight: bold'  # Orange for neutral
        
        def color_signals(val):
            if val == 'BUY':
                return 'background-color: #004400; color: white; font-weight: bold'
            elif val == 'SELL':
                return 'background-color: #440000; color: white; font-weight: bold'
            else:
                return 'background-color: #444400; color: white; font-weight: bold'
        
        styled = df.style.applymap(color_predictions, subset=['predicted_return'])
        styled = styled.applymap(color_signals, subset=['signal'])
        return styled
    
    # Display styled table
    display_df = predictions_df.copy()
    display_df['predicted_return'] = display_df['predicted_return'].apply(lambda x: f"{x:.2f}%")
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(
        style_predictions(display_df),
        use_container_width=True,
        height=300
    )
    
    # Visualization Section
    st.markdown("### 📈 Elite AI Visualizations")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction Returns Chart
        fig_returns = px.bar(
            predictions_df, 
            x='symbol', 
            y='predicted_return',
            color='predicted_return',
            color_continuous_scale='RdYlGn',
            title="🎯 Predicted Returns by Stock",
            labels={'predicted_return': 'Predicted Return (%)', 'symbol': 'Stock Symbol'}
        )
        fig_returns.update_layout(height=400)
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        # Confidence vs Returns Scatter
        fig_confidence = px.scatter(
            predictions_df,
            x='confidence',
            y='predicted_return',
            color='signal',
            size='confidence',
            hover_data=['symbol'],
            title="🎯 Confidence vs Predicted Returns",
            color_discrete_map={'BUY': '#00ff00', 'SELL': '#ff4444', 'HOLD': '#ffaa00'}
        )
        fig_confidence.update_layout(height=400)
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    # Signal Distribution Pie Chart
    signal_counts = predictions_df['signal'].value_counts()
    fig_pie = px.pie(
        values=signal_counts.values,
        names=signal_counts.index,
        title="📊 Trading Signal Distribution",
        color_discrete_map={'BUY': '#00ff00', 'SELL': '#ff4444', 'HOLD': '#ffaa00'}
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Model Performance (if available)
    if st.session_state[f"model_performance_{session_id}"] is not None:
        display_model_performance(session_id)

def display_model_performance(session_id):
    """Display Model Performance Metrics"""
    
    st.markdown("### 🏆 Elite AI Model Performance")
    
    performance_data = st.session_state[f"model_performance_{session_id}"]
    
    if isinstance(performance_data, dict):
        # Create performance DataFrame
        perf_rows = []
        for symbol, metrics in performance_data.items():
            if isinstance(metrics, dict):
                perf_rows.append({
                    'Symbol': symbol,
                    'R² Score': metrics.get('r2_score', 0),
                    'Direction Accuracy': metrics.get('direction_accuracy', 0),
                    'Status': 'Trained ✅'
                })
        
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)
            
            # Display performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_r2 = perf_df['R² Score'].mean()
                st.metric("📊 Avg R² Score", f"{avg_r2:.3f}")
            
            with col2:
                avg_direction = perf_df['Direction Accuracy'].mean()
                st.metric("🎯 Avg Direction Accuracy", f"{avg_direction:.1%}")
            
            with col3:
                models_trained = len(perf_df)
                st.metric("🤖 Models Trained", models_trained)
            
            # Performance table
            st.dataframe(perf_df, use_container_width=True, height=200)

def display_elite_dashboard():
    """Display Elite AI Dashboard when not trained"""
    
    st.markdown("## 🚀 Welcome to Elite AI Trader")
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 Advanced AI Models
        - **XGBoost**: Gradient boosting excellence
        - **LightGBM**: Lightning-fast predictions  
        - **CatBoost**: Categorical data mastery
        - **Random Forest**: Ensemble wisdom
        - **Gradient Boosting**: Traditional power
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Elite Features (87+)
        - **Technical Indicators**: RSI, MACD, Bollinger
        - **Price Action**: Support/Resistance levels
        - **Volume Analysis**: Money flow insights
        - **Momentum**: Trend strength metrics
        - **Volatility**: Risk assessment tools
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Smart Predictions
        - **Multi-timeframe**: 1D to 2W horizons
        - **Confidence Scores**: Risk-adjusted signals
        - **Signal Generation**: BUY/SELL/HOLD
        - **Performance Tracking**: Model validation
        - **Real-time Updates**: Live market data
        """)
    
    # Instructions
    st.markdown("""
    ---
    ## 🚀 Getting Started
    
    1. **Select Stocks**: Choose 3-8 stocks from the sidebar
    2. **Configure AI**: Set training period and prediction horizon  
    3. **Train Elite AI**: Click the training button to build your ensemble
    4. **Get Predictions**: View AI-powered trading signals and analysis
    
    **Pro Tip**: Start with large-cap stocks (AAPL, MSFT, GOOGL) for best results!
    """)
    
    # Sample predictions display
    st.markdown("### 📈 Sample Elite AI Output")
    
    sample_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        'Predicted Return': ['2.3%', '-0.8%', '1.5%', '-2.1%', '0.4%'],
        'Confidence': [0.85, 0.72, 0.78, 0.91, 0.65],
        'Signal': ['BUY', 'HOLD', 'BUY', 'SELL', 'HOLD']
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True, height=200)
    
    st.markdown("*This is sample data. Train the Elite AI to get real predictions!*")

if __name__ == "__main__":
    main()
