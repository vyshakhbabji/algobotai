#!/usr/bin/env python3
"""
Self-Improving AI Model System
Continuously evaluates and optimizes trading models for better performance
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
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
    .good-performance {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
    }
    .poor-performance {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    }
    .optimization-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SelfImprovingAI:
    def __init__(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.optimization_file = os.path.join(parent_dir, "model_optimization_log.json")
        self.performance_file = os.path.join(parent_dir, "model_performance_history.json")
        self.load_optimization_history()
        self.ai_manager = ImprovedAIPortfolioManager()
    
    def load_optimization_history(self):
        """Load historical optimization data"""
        if os.path.exists(self.optimization_file):
            with open(self.optimization_file, 'r') as f:
                self.optimization_history = json.load(f)
        else:
            self.optimization_history = {
                'optimizations': [],
                'best_parameters': {},
                'performance_trends': {},
                'last_optimization': None
            }
        
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                self.performance_history = json.load(f)
        else:
            self.performance_history = {
                'daily_performance': [],
                'model_scores': [],
                'prediction_accuracy': []
            }
    
    def save_optimization_data(self):
        """Save optimization data to file"""
        with open(self.optimization_file, 'w') as f:
            json.dump(self.optimization_history, f, indent=2, default=str)
        
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2, default=str)
    
    def evaluate_model_performance(self, symbol, days_back=30):
        """Evaluate how well the model has been performing recently"""
        try:
            # Get recent data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back + 100)  # Extra data for features
            
            df = ticker.history(start=start_date, end=end_date)
            if len(df) < 50:
                return None
            
            # Calculate features
            df = self.ai_manager.calculate_improved_features(df)
            
            # Get last 30 days for evaluation
            recent_df = df.tail(days_back)
            
            if len(recent_df) < 10:
                return None
            
            # Load current model if it exists
            if symbol not in self.ai_manager.models or self.ai_manager.models[symbol] is None:
                return None
            
            model, scaler, r2_score = self.ai_manager.models[symbol]
            
            feature_cols = [
                'price_vs_sma10', 'price_vs_sma30', 'price_vs_sma50',
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_30',
                'volume_ratio', 'volume_momentum',
                'rsi_normalized', 'bb_position', 'bb_squeeze',
                'macd', 'macd_histogram', 'price_position'
            ]
            
            predictions = []
            actual_returns = []
            
            for i in range(len(recent_df) - 5):
                # Get features for day i
                features = recent_df[feature_cols].iloc[i:i+1].values
                
                if np.isnan(features).any():
                    continue
                
                # Make prediction
                features_scaled = scaler.transform(features)
                predicted_return = model.predict(features_scaled)[0]
                
                # Get actual return (5 days ahead)
                if i + 5 < len(recent_df):
                    current_price = recent_df['Close'].iloc[i]
                    future_price = recent_df['Close'].iloc[i + 5]
                    actual_return = (future_price - current_price) / current_price
                    
                    predictions.append(predicted_return)
                    actual_returns.append(actual_return)
            
            if not predictions:
                return None
            
            # Calculate performance metrics
            predictions = np.array(predictions)
            actual_returns = np.array(actual_returns)
            
            # Accuracy metrics
            mae = mean_absolute_error(actual_returns, predictions)
            mse = mean_squared_error(actual_returns, predictions)
            correlation = np.corrcoef(predictions, actual_returns)[0, 1] if len(predictions) > 1 else 0
            
            # Direction accuracy (did we predict the right direction?)
            direction_correct = np.sum(np.sign(predictions) == np.sign(actual_returns)) / len(predictions)
            
            # Profitability if we traded on predictions
            buy_signals = predictions > 0.005  # 0.5% threshold
            if np.sum(buy_signals) > 0:
                profitable_trades = np.sum((actual_returns[buy_signals] > 0))
                win_rate = profitable_trades / np.sum(buy_signals)
                avg_return_when_bought = np.mean(actual_returns[buy_signals])
            else:
                win_rate = 0
                avg_return_when_bought = 0
            
            return {
                'symbol': symbol,
                'evaluation_period': days_back,
                'total_predictions': len(predictions),
                'mae': mae,
                'mse': mse,
                'correlation': correlation,
                'direction_accuracy': direction_correct,
                'buy_signals': np.sum(buy_signals),
                'win_rate': win_rate,
                'avg_return_when_bought': avg_return_when_bought,
                'model_r2': r2_score,
                'needs_improvement': direction_correct < 0.6 or win_rate < 0.5,
                'performance_score': (direction_correct + win_rate + max(0, correlation)) / 3
            }
            
        except Exception as e:
            st.error(f"Error evaluating {symbol}: {str(e)}")
            return None
    
    def optimize_model_parameters(self, symbol, performance_data):
        """Optimize model parameters based on poor performance"""
        try:
            st.info(f"🔧 Optimizing model parameters for {symbol}...")
            
            # Get training data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='2y')
            
            if len(df) < 200:
                return None
            
            # Calculate features
            df = self.ai_manager.calculate_improved_features(df)
            
            feature_cols = [
                'price_vs_sma10', 'price_vs_sma30', 'price_vs_sma50',
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_30',
                'volume_ratio', 'volume_momentum',
                'rsi_normalized', 'bb_position', 'bb_squeeze',
                'macd', 'macd_histogram', 'price_position'
            ]
            
            # Prepare data
            X = df[feature_cols].dropna()
            y = df['future_return_5d'].dropna()
            
            # Align X and y
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            
            if len(X) < 100:
                return None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Hyperparameter optimization
            if performance_data['performance_score'] < 0.4:
                # Poor performance - try different models
                models_to_try = {
                    'RandomForest': {
                        'model': RandomForestRegressor(random_state=42),
                        'params': {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [5, 10, 15, None],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                    },
                    'GradientBoosting': {
                        'model': GradientBoostingRegressor(random_state=42),
                        'params': {
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'max_depth': [3, 5, 7],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                    }
                }
            else:
                # Moderate performance - fine-tune current model
                models_to_try = {
                    'RandomForest': {
                        'model': RandomForestRegressor(random_state=42),
                        'params': {
                            'n_estimators': [100, 150, 200],
                            'max_depth': [10, 15, 20],
                            'min_samples_split': [2, 3, 5]
                        }
                    }
                }
            
            best_model = None
            best_score = -np.inf
            best_params = None
            best_model_name = None
            
            tscv = TimeSeriesSplit(n_splits=5)
            
            for model_name, model_config in models_to_try.items():
                st.info(f"🔍 Testing {model_name} with different parameters...")
                
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=tscv,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_scaled, y)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_model_name = model_name
            
            if best_model is not None:
                # Update the model in AI manager
                self.ai_manager.models[symbol] = (best_model, scaler, best_score)
                
                # Log optimization
                optimization_record = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'trigger': 'poor_performance',
                    'old_performance': performance_data,
                    'new_model': best_model_name,
                    'new_parameters': best_params,
                    'new_r2_score': best_score,
                    'improvement': best_score - performance_data['model_r2']
                }
                
                self.optimization_history['optimizations'].append(optimization_record)
                self.optimization_history['best_parameters'][symbol] = {
                    'model': best_model_name,
                    'params': best_params,
                    'score': best_score
                }
                self.optimization_history['last_optimization'] = datetime.now().isoformat()
                
                self.save_optimization_data()
                
                return optimization_record
            
            return None
            
        except Exception as e:
            st.error(f"Error optimizing {symbol}: {str(e)}")
            return None
    
    def run_comprehensive_evaluation(self, symbols=None):
        """Run comprehensive evaluation and optimization for all symbols"""
        if symbols is None:
            # Load portfolio stocks
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            portfolio_file = os.path.join(parent_dir, "portfolio_universe.json")
            
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    data = json.load(f)
                    symbols = data.get('stocks', [])
            else:
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']
        
        evaluation_results = []
        optimization_results = []
        
        # First, train models for all symbols
        st.info("🤖 Training/updating models for all symbols...")
        self.ai_manager.train_all_models()
        
        # Then evaluate each one
        for symbol in symbols[:10]:  # Limit to 10 for performance
            st.info(f"📊 Evaluating performance for {symbol}...")
            
            performance = self.evaluate_model_performance(symbol)
            if performance:
                evaluation_results.append(performance)
                
                # If performance is poor, optimize
                if performance['needs_improvement']:
                    st.warning(f"⚠️ {symbol} needs improvement (Score: {performance['performance_score']:.2f})")
                    optimization = self.optimize_model_parameters(symbol, performance)
                    if optimization:
                        optimization_results.append(optimization)
                        st.success(f"✅ {symbol} model optimized!")
                else:
                    st.success(f"✅ {symbol} performing well (Score: {performance['performance_score']:.2f})")
        
        return evaluation_results, optimization_results

def main():
    """Main Self-Improving AI Interface"""
    st.markdown('<h1 class="main-header">🧠 AI Self-Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("**Continuously improving AI models for better trading performance**")
    
    # Initialize self-improving AI
    ai_optimizer = SelfImprovingAI()
    
    # Display current optimization status
    st.subheader("📊 Current Optimization Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_optimizations = len(ai_optimizer.optimization_history.get('optimizations', []))
        st.markdown(f'''
        <div class="metric-card">
            <h3>{total_optimizations}</h3>
            <p>Total Optimizations</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        best_params = len(ai_optimizer.optimization_history.get('best_parameters', {}))
        st.markdown(f'''
        <div class="metric-card">
            <h3>{best_params}</h3>
            <p>Optimized Models</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        last_opt = ai_optimizer.optimization_history.get('last_optimization')
        if last_opt:
            days_ago = (datetime.now() - datetime.fromisoformat(last_opt)).days
            st.markdown(f'''
            <div class="metric-card">
                <h3>{days_ago}</h3>
                <p>Days Since Last Opt</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="metric-card">
                <h3>Never</h3>
                <p>Last Optimization</p>
            </div>
            ''', unsafe_allow_html=True)
    
    with col4:
        if st.button("🚀 Run Full Optimization", type="primary"):
            st.rerun()
    
    # Control panel
    st.markdown("---")
    st.subheader("🎛️ Optimization Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        evaluation_mode = st.selectbox(
            "Evaluation Mode",
            ["Quick (5 stocks)", "Standard (10 stocks)", "Full Portfolio", "Single Stock"]
        )
        
        if evaluation_mode == "Single Stock":
            # Load portfolio for stock selection
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            portfolio_file = os.path.join(parent_dir, "portfolio_universe.json")
            
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    data = json.load(f)
                    portfolio_stocks = data.get('stocks', [])
            else:
                portfolio_stocks = ['AAPL', 'GOOGL', 'MSFT']
                
            selected_stock = st.selectbox("Select Stock", portfolio_stocks)
    
    with col2:
        evaluation_days = st.slider("Evaluation Period (days)", 7, 60, 30)
        auto_optimize = st.checkbox("Auto-optimize poor performers", value=True)
    
    # Run evaluation
    if st.button("📊 Run Performance Evaluation", type="primary"):
        with st.spinner("🧠 Evaluating AI models..."):
            
            if evaluation_mode == "Single Stock":
                symbols = [selected_stock]
            elif evaluation_mode == "Quick (5 stocks)":
                symbols = None  # Will use first 5 from portfolio
            else:
                symbols = None  # Will use portfolio or default
            
            evaluation_results, optimization_results = ai_optimizer.run_comprehensive_evaluation(symbols)
            
            if evaluation_results:
                # Display results
                st.subheader("📈 Performance Evaluation Results")
                
                # Create performance summary
                df_results = pd.DataFrame(evaluation_results)
                
                # Performance cards
                avg_score = df_results['performance_score'].mean()
                needs_improvement = df_results['needs_improvement'].sum()
                avg_direction_accuracy = df_results['direction_accuracy'].mean()
                avg_win_rate = df_results['win_rate'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    card_class = "good-performance" if avg_score > 0.6 else "poor-performance"
                    st.markdown(f'''
                    <div class="metric-card {card_class}">
                        <h3>{avg_score:.2f}</h3>
                        <p>Avg Performance Score</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    card_class = "poor-performance" if needs_improvement > 0 else "good-performance"
                    st.markdown(f'''
                    <div class="metric-card {card_class}">
                        <h3>{needs_improvement}</h3>
                        <p>Models Need Improvement</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    card_class = "good-performance" if avg_direction_accuracy > 0.6 else "poor-performance"
                    st.markdown(f'''
                    <div class="metric-card {card_class}">
                        <h3>{avg_direction_accuracy:.1%}</h3>
                        <p>Direction Accuracy</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col4:
                    card_class = "good-performance" if avg_win_rate > 0.5 else "poor-performance"
                    st.markdown(f'''
                    <div class="metric-card {card_class}">
                        <h3>{avg_win_rate:.1%}</h3>
                        <p>Win Rate</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Detailed results table
                st.subheader("📋 Detailed Performance Metrics")
                
                display_df = df_results[[
                    'symbol', 'performance_score', 'direction_accuracy', 'win_rate', 
                    'correlation', 'buy_signals', 'avg_return_when_bought', 'needs_improvement'
                ]].copy()
                
                display_df['performance_score'] = display_df['performance_score'].round(3)
                display_df['direction_accuracy'] = (display_df['direction_accuracy'] * 100).round(1)
                display_df['win_rate'] = (display_df['win_rate'] * 100).round(1)
                display_df['correlation'] = display_df['correlation'].round(3)
                display_df['avg_return_when_bought'] = (display_df['avg_return_when_bought'] * 100).round(2)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Optimization results
                if optimization_results:
                    st.subheader("⚡ Optimization Results")
                    
                    for opt in optimization_results:
                        st.markdown(f'''
                        <div class="optimization-card">
                            <h3>🔧 {opt['symbol']} Optimized</h3>
                            <p><strong>New Model:</strong> {opt['new_model']}</p>
                            <p><strong>R² Score:</strong> {opt['old_performance']['model_r2']:.3f} → {opt['new_r2_score']:.3f}</p>
                            <p><strong>Improvement:</strong> +{opt['improvement']:.3f}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Performance visualization
                if len(evaluation_results) > 1:
                    st.subheader("📊 Performance Visualization")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_results['symbol'],
                        y=df_results['performance_score'],
                        mode='markers+lines',
                        name='Performance Score',
                        marker=dict(
                            size=10,
                            color=df_results['performance_score'],
                            colorscale='RdYlGn',
                            showscale=True
                        )
                    ))
                    
                    fig.add_hline(y=0.6, line_dash="dash", line_color="green", 
                                annotation_text="Good Performance Threshold")
                    
                    fig.update_layout(
                        title="Model Performance Scores by Stock",
                        xaxis_title="Stock Symbol",
                        yaxis_title="Performance Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("No evaluation results obtained. Check your portfolio or try different settings.")
    
    # Optimization history
    if ai_optimizer.optimization_history.get('optimizations'):
        st.markdown("---")
        st.subheader("📚 Optimization History")
        
        recent_optimizations = ai_optimizer.optimization_history['optimizations'][-5:]  # Last 5
        
        for opt in reversed(recent_optimizations):
            timestamp = datetime.fromisoformat(opt['timestamp']).strftime("%Y-%m-%d %H:%M")
            improvement = opt.get('improvement', 0)
            
            st.markdown(f'''
            <div class="optimization-card">
                <h4>{opt['symbol']} - {timestamp}</h4>
                <p><strong>Trigger:</strong> {opt['trigger']}</p>
                <p><strong>New Model:</strong> {opt['new_model']}</p>
                <p><strong>R² Improvement:</strong> {improvement:+.3f}</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Note**: The AI continuously learns and improves. Models are automatically optimized when performance drops below acceptable thresholds.")

if __name__ == "__main__":
    main()
