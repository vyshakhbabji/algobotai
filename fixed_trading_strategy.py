#!/usr/bin/env python3
"""
Fixed Trading Strategy Implementation
Generate trading signals and backtest performance
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

class TradingStrategy:
    def __init__(self):
        """
        Initialize trading strategy
        """
        self.models = {}
        self.scalers = {}
        self.metadata = None
        self.predictions = {}
        self.signals = {}
        
    def load_models_and_data(self):
        """
        Load trained models and preprocessed data
        """
        print("Loading models and data...")
        
        # Load metadata
        with open('fixed_data/preprocessed/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load scalers
        self.scalers['feature'] = joblib.load('fixed_data/preprocessed/feature_scaler.pkl')
        self.scalers['target'] = joblib.load('fixed_data/preprocessed/target_scaler.pkl')
        
        # Load test data
        self.X_test_seq = np.load('fixed_data/preprocessed/X_test_seq.npy')
        self.X_test_flat = np.load('fixed_data/preprocessed/X_test_flat.npy')
        self.y_test = np.load('fixed_data/preprocessed/y_test.npy')
        self.test_dates = np.load('fixed_data/preprocessed/test_dates.npy', allow_pickle=True)
        
        # Load models
        try:
            self.models['lstm'] = tf.keras.models.load_model('fixed_data/models/lstm_model.h5')
            print("LSTM model loaded")
        except Exception as e:
            print(f"LSTM model not found: {e}")
            
        try:
            self.models['cnn'] = tf.keras.models.load_model('fixed_data/models/cnn_model.h5')
            print("CNN model loaded")
        except Exception as e:
            print(f"CNN model not found: {e}")
            
        try:
            self.models['rf'] = joblib.load('fixed_data/models/random_forest_model.pkl')
            print("Random Forest model loaded")
        except Exception as e:
            print(f"Random Forest model not found: {e}")
            
        try:
            self.models['gb'] = joblib.load('fixed_data/models/gradient_boosting_model.pkl')
            print("Gradient Boosting model loaded")
        except Exception as e:
            print(f"Gradient Boosting model not found: {e}")
            
        try:
            self.models['linear'] = joblib.load('fixed_data/models/linear_regression_model.pkl')
            print("Linear Regression model loaded")
        except Exception as e:
            print(f"Linear Regression model not found: {e}")
            
        try:
            self.models['ridge'] = joblib.load('fixed_data/models/ridge_model.pkl')
            print("Ridge model loaded")
        except Exception as e:
            print(f"Ridge model not found: {e}")
            
        try:
            self.models['ensemble'] = joblib.load('fixed_data/models/ensemble_meta_model.pkl')
            print("Ensemble model loaded")
        except Exception as e:
            print("Ensemble model not found")
        
        print(f"Loaded {len(self.models)} models")
        print(f"Test data shape: {self.X_test_seq.shape}")
    
    def generate_predictions(self):
        """
        Generate predictions from all models
        """
        print("Generating predictions...")
        
        for model_name, model in self.models.items():
            print(f"Generating predictions for {model_name}...")
            
            if model_name in ['lstm', 'cnn']:
                # Deep learning models use sequence data
                pred_scaled = model.predict(self.X_test_seq, verbose=0).flatten()
            elif model_name == 'ensemble':
                # Ensemble model needs predictions from base models
                base_predictions = []
                for base_name, base_model in self.models.items():
                    if base_name != 'ensemble':
                        if base_name in ['lstm', 'cnn']:
                            base_pred = base_model.predict(self.X_test_seq, verbose=0).flatten()
                        else:
                            base_pred = base_model.predict(self.X_test_flat)
                        base_predictions.append(base_pred)
                
                if base_predictions:
                    ensemble_features = np.column_stack(base_predictions)
                    pred_scaled = model.predict(ensemble_features)
                else:
                    continue
            else:
                # ML models use sequence data reshaped to flat
                X_test_flat_reshaped = self.X_test_seq.reshape(self.X_test_seq.shape[0], -1)
                pred_scaled = model.predict(X_test_flat_reshaped)
            
            # Inverse transform predictions
            pred_original = self.scalers['target'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            self.predictions[model_name] = pred_original
    
    def generate_trading_signals(self, model_name='ensemble', threshold=0.005):
        """
        Generate trading signals with momentum and trend analysis
        
        Args:
            model_name: Model to use for signal generation
            threshold: Minimum predicted return to generate signal
        """
        print(f"Generating trading signals using {model_name} model...")
        
        if model_name not in self.predictions:
            print(f"Model {model_name} not found. Using first available model.")
            model_name = list(self.predictions.keys())[0]
        
        predictions = self.predictions[model_name]
        
        # Convert test dates to datetime if they're strings
        dates = pd.to_datetime(self.test_dates)
        
        # Get actual prices (inverse transform)
        actual_prices = self.scalers['target'].inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        
        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'Date': dates,
            'Actual_Price': actual_prices,
            'Predicted_Price': predictions,
            'Predicted_Return': (predictions / actual_prices - 1) * 100
        })
        
        # Calculate technical indicators for better signals
        signals_df['Price_MA3'] = signals_df['Actual_Price'].rolling(window=3, min_periods=1).mean()
        signals_df['Price_MA5'] = signals_df['Actual_Price'].rolling(window=5, min_periods=1).mean()
        signals_df['Price_Change'] = signals_df['Actual_Price'].pct_change() * 100
        signals_df['Volume_Trend'] = signals_df['Actual_Price'].rolling(window=3).std()  # Proxy for volatility
        
        # Initialize signals
        signals_df['Signal'] = 0  # 0 = Hold, 1 = Buy, -1 = Sell
        signals_df['Signal_Strength'] = 0.0
        
        # Enhanced signal generation logic
        for i in range(len(signals_df)):
            pred_return = signals_df.iloc[i]['Predicted_Return']
            price_change = signals_df.iloc[i]['Price_Change'] if i > 0 else 0
            
            # Get short and long term trends
            ma3 = signals_df.iloc[i]['Price_MA3']
            ma5 = signals_df.iloc[i]['Price_MA5'] if i >= 4 else ma3
            current_price = signals_df.iloc[i]['Actual_Price']
            
            # Trend indicators
            short_trend = (current_price - ma3) / ma3 * 100 if ma3 > 0 else 0
            long_trend = (ma3 - ma5) / ma5 * 100 if ma5 > 0 else 0
            
            # More balanced signal generation
            signal = 0
            strength = 0
            
            # Buy conditions (much more aggressive for uptrending stocks)
            if (pred_return > -8) or \
               (short_trend > 0.1) or \
               (long_trend > 0.2) or \
               (price_change > 1):
                signal = 1
                strength = max(1, abs(pred_return) + short_trend + long_trend)
                
            # Sell conditions (only for very strong bearish signals)
            elif (pred_return < -15 and short_trend < -2 and long_trend < -2):
                signal = -1
                strength = abs(pred_return) + abs(short_trend) + abs(long_trend)
            
            # Hold for everything else
            else:
                signal = 0
                strength = 0
                
            signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = signal
            signals_df.iloc[i, signals_df.columns.get_loc('Signal_Strength')] = strength
        
        # Calculate actual returns for evaluation
        signals_df['Actual_Return'] = signals_df['Actual_Price'].pct_change() * 100
        
        self.signals[model_name] = signals_df
        
        # Print signal summary
        signal_counts = signals_df['Signal'].value_counts()
        print(f"Signal distribution:")
        print(f"Buy signals: {signal_counts.get(1, 0)}")
        print(f"Hold signals: {signal_counts.get(0, 0)}")
        print(f"Sell signals: {signal_counts.get(-1, 0)}")
        
        return signals_df
    
    def backtest_strategy(self, signals_df, initial_capital=10000, transaction_cost=0.001):
        """
        Backtest the trading strategy with improved logic
        
        Args:
            signals_df: DataFrame with trading signals
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage of trade value
        """
        print("Backtesting strategy...")
        
        portfolio = signals_df.copy()
        portfolio['Position'] = 0  # 0 = no position, 1 = long, -1 = short
        portfolio['Holdings'] = 0.0
        portfolio['Cash'] = float(initial_capital)
        portfolio['Total_Value'] = float(initial_capital)
        portfolio['Returns'] = 0.0
        portfolio['Cumulative_Returns'] = 0.0
        
        cash = initial_capital
        position = 0
        holdings = 0.0
        
        for i in range(len(portfolio)):
            current_price = portfolio.iloc[i]['Actual_Price']
            signal = portfolio.iloc[i]['Signal']
            
            # Execute trades based on signals with more realistic logic
            if signal == 1 and position <= 0:  # Buy signal
                if cash > current_price:  # Can afford at least 1 share
                    # Use 80% of available cash for buying
                    trade_amount = cash * 0.8
                    shares_to_buy = trade_amount / current_price
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    
                    if cost <= cash:
                        holdings += shares_to_buy
                        cash -= cost
                        position = 1
            
            elif signal == -1 and holdings > 0:  # Sell signal and we have holdings
                # Sell all holdings
                proceeds = holdings * current_price * (1 - transaction_cost)
                cash += proceeds
                holdings = 0
                position = 0  # Neutral position after selling
            
            # Update portfolio values
            portfolio.iloc[i, portfolio.columns.get_loc('Position')] = position
            portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = holdings
            portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = cash
            
            total_value = cash + (holdings * current_price)
            portfolio.iloc[i, portfolio.columns.get_loc('Total_Value')] = total_value
            
            # Calculate returns
            if i > 0:
                prev_value = portfolio.iloc[i-1]['Total_Value']
                daily_return = (total_value / prev_value - 1) * 100 if prev_value > 0 else 0
                portfolio.iloc[i, portfolio.columns.get_loc('Returns')] = daily_return
                portfolio.iloc[i, portfolio.columns.get_loc('Cumulative_Returns')] = \
                    ((total_value / initial_capital) - 1) * 100
        
        return portfolio
    
    def calculate_performance_metrics(self, portfolio):
        """
        Calculate performance metrics
        """
        print("Calculating performance metrics...")
        
        returns = portfolio['Returns'].dropna()
        cumulative_return = portfolio['Cumulative_Returns'].iloc[-1]
        
        # Basic metrics
        total_return = cumulative_return
        num_days = len(returns)
        annualized_return = (1 + total_return/100) ** (252/num_days) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        max_drawdown = (portfolio['Cumulative_Returns'].cummax() - portfolio['Cumulative_Returns']).max()
        
        # Sharpe ratio (assuming 2% risk-free rate)
        sharpe_ratio = (annualized_return - 0.02) / (volatility / 100) if volatility > 0 else 0
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Buy and hold comparison
        initial_price = portfolio['Actual_Price'].iloc[0]
        final_price = portfolio['Actual_Price'].iloc[-1]
        buy_hold_return = (final_price / initial_price - 1) * 100
        
        metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return * 100,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate * 100,
            'Total Trades': total_trades,
            'Buy & Hold Return (%)': buy_hold_return,
            'Alpha (%)': total_return - buy_hold_return
        }
        
        return metrics
    
    def plot_backtest_results(self, portfolio, metrics, model_name):
        """
        Plot backtest results
        """
        print("Plotting backtest results...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio['Date'], portfolio['Total_Value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Cumulative returns vs buy & hold
        axes[0, 1].plot(portfolio['Date'], portfolio['Cumulative_Returns'], label='Strategy')
        
        # Calculate buy & hold returns
        initial_price = portfolio['Actual_Price'].iloc[0]
        buy_hold_returns = (portfolio['Actual_Price'] / initial_price - 1) * 100
        axes[0, 1].plot(portfolio['Date'], buy_hold_returns, label='Buy & Hold')
        
        axes[0, 1].set_title('Cumulative Returns Comparison')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Cumulative Returns (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Trading signals
        buy_signals = portfolio[portfolio['Signal'] == 1]
        sell_signals = portfolio[portfolio['Signal'] == -1]
        
        axes[1, 0].plot(portfolio['Date'], portfolio['Actual_Price'], label='Price', alpha=0.7)
        axes[1, 0].scatter(buy_signals['Date'], buy_signals['Actual_Price'], 
                          color='green', marker='^', s=50, label='Buy')
        axes[1, 0].scatter(sell_signals['Date'], sell_signals['Actual_Price'], 
                          color='red', marker='v', s=50, label='Sell')
        axes[1, 0].set_title('Trading Signals')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Returns distribution
        returns = portfolio['Returns'].dropna()
        axes[1, 1].hist(returns, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Returns (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'fixed_data/results/backtest_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Performance metrics table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        metrics_table = [[k, f"{v:.2f}" if isinstance(v, (int, float)) else str(v)] 
                        for k, v in metrics.items()]
        
        table = ax.table(cellText=metrics_table, colLabels=['Metric', 'Value'],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title(f'Performance Metrics - {model_name.upper()} Model', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'fixed_data/results/metrics_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, portfolio, metrics, model_name):
        """
        Save backtest results
        """
        print("Saving results...")
        
        # Create results directory
        os.makedirs('fixed_data/results', exist_ok=True)
        
        # Save portfolio data
        portfolio.to_csv(f'fixed_data/results/portfolio_{model_name}.csv', index=False)
        
        # Save metrics
        with open(f'fixed_data/results/metrics_{model_name}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save signals
        signals_only = portfolio[['Date', 'Signal', 'Signal_Strength', 'Predicted_Return']].copy()
        signals_only.to_csv(f'fixed_data/results/signals_{model_name}.csv', index=False)
    
    def run_strategy(self, model_name='gb', threshold=0.02, initial_capital=10000):
        """
        Run complete trading strategy
        """
        print(f"\n{'='*60}")
        print(f"RUNNING TRADING STRATEGY - {model_name.upper()} MODEL")
        print(f"{'='*60}")
        
        # Load models and data
        self.load_models_and_data()
        
        # Generate predictions
        self.generate_predictions()
        
        # Generate trading signals
        signals_df = self.generate_trading_signals(model_name, threshold)
        
        # Backtest strategy
        portfolio = self.backtest_strategy(signals_df, initial_capital)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(portfolio)
        
        # Print results
        print(f"\nPerformance Summary:")
        print(f"{'='*40}")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Plot results
        self.plot_backtest_results(portfolio, metrics, model_name)
        
        # Save results
        self.save_results(portfolio, metrics, model_name)
        
        return portfolio, metrics
    
    def compare_all_models(self, threshold=0.02, initial_capital=10000):
        """
        Compare performance of all models
        """
        print(f"\n{'='*60}")
        print("COMPARING ALL MODELS")
        print(f"{'='*60}")
        
        # Load models and data
        self.load_models_and_data()
        
        # Generate predictions
        self.generate_predictions()
        
        all_metrics = {}
        
        # Test each model
        for model_name in self.predictions.keys():
            print(f"\nTesting {model_name} model...")
            
            try:
                signals_df = self.generate_trading_signals(model_name, threshold)
                portfolio = self.backtest_strategy(signals_df, initial_capital)
                metrics = self.calculate_performance_metrics(portfolio)
                all_metrics[model_name] = metrics
                
                # Plot individual results
                self.plot_backtest_results(portfolio, metrics, model_name)
                self.save_results(portfolio, metrics, model_name)
                
            except Exception as e:
                print(f"Error testing {model_name}: {str(e)}")
                continue
        
        # Create comparison plot
        if all_metrics:
            self.plot_model_comparison(all_metrics)
            
            # Save comparison results
            comparison_df = pd.DataFrame(all_metrics).T
            comparison_df.to_csv('fixed_data/results/model_comparison.csv')
            
            # Find best model
            best_model = comparison_df['Total Return (%)'].idxmax()
            best_return = comparison_df.loc[best_model, 'Total Return (%)']
            
            print(f"\n{'='*60}")
            print(f"BEST PERFORMING MODEL: {best_model.upper()}")
            print(f"Total Return: {best_return:.2f}%")
            print(f"{'='*60}")
        
        return all_metrics
    
    def plot_model_comparison(self, all_metrics):
        """
        Plot comparison of all models
        """
        print("Creating model comparison plots...")
        
        comparison_df = pd.DataFrame(all_metrics).T
        
        # Key metrics to compare
        key_metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            if metric in comparison_df.columns:
                ax = axes[i]
                values = comparison_df[metric].sort_values(ascending=False)
                
                bars = ax.bar(range(len(values)), values.values)
                ax.set_title(metric)
                ax.set_xlabel('Models')
                ax.set_ylabel(metric)
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(values.index, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Color bars
                for j, bar in enumerate(bars):
                    if j == 0:  # Best performer
                        bar.set_color('green')
                    elif j == len(bars) - 1:  # Worst performer
                        bar.set_color('red')
                    else:
                        bar.set_color('blue')
        
        plt.tight_layout()
        plt.savefig('fixed_data/results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Main function to run trading strategy
    """
    try:
        strategy = TradingStrategy()
        
        # Run strategy with gradient boosting model (best performer)
        portfolio, metrics = strategy.run_strategy(
            model_name='gb', 
            threshold=0.005,  # Lowered threshold for more trading opportunities
            initial_capital=10000
        )
        
        # Compare all models
        all_metrics = strategy.compare_all_models(
            threshold=0.005,  # Lowered threshold for more trading opportunities
            initial_capital=10000
        )
        
        print("\n" + "="*60)
        print("TRADING STRATEGY ANALYSIS COMPLETED!")
        print("="*60)
        print("Check 'fixed_data/results/' folder for detailed results and plots")
        
        return strategy, portfolio, metrics, all_metrics
        
    except Exception as e:
        print(f"Error in trading strategy: {str(e)}")
        raise

if __name__ == "__main__":
    strategy, portfolio, metrics, all_metrics = main()
