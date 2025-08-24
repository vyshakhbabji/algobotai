"""
Performance Analysis and Plotting for ML Trading Sy            # 6. Generate Summary Report
            self.generate_summary_report(analysis_dir)
            
            # 7. Analyze Bad Trades (NEW)
            self.analyze_bad_trades(analysis_dir)
            
            # 8. Signal Quality Analysis (NEW)
            self.analyze_signal_quality(analysis_dir)em
Generates comprehensive performance metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformancePlotter:
    """Generate comprehensive performance analysis plots and metrics"""
    
    def __init__(self, strategy, analyzers: Dict, output_dir: Path):
        self.strategy = strategy
        self.analyzers = analyzers
        self.output_dir = output_dir
        self.trading_start_date = datetime(2024, 8, 22)
        
    def generate_full_analysis(self):
        """Generate all performance analysis plots and metrics"""
        print("📊 Generating comprehensive performance analysis...")
        
        # Create analysis directory
        analysis_dir = self.output_dir / "performance_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Portfolio Performance
            self.plot_portfolio_performance(analysis_dir)
            
            # 2. ML Model Performance
            self.plot_ml_performance(analysis_dir)
            
            # 3. Trading Performance
            self.plot_trading_metrics(analysis_dir)
            
            # 4. Risk Analysis
            self.plot_risk_analysis(analysis_dir)
            
            # 5. Sector/Stock Analysis
            self.plot_stock_performance(analysis_dir)
            
            # 6. Overfitting Analysis
            self.analyze_overfitting(analysis_dir)
            
            # 7. Generate Summary Report
            self.generate_summary_report(analysis_dir)
            
            print(f"✅ Complete analysis saved to: {analysis_dir}")
            
        except Exception as e:
            print(f"⚠️ Error generating analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_portfolio_performance(self, output_dir: Path):
        """Plot portfolio performance over time"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
            
            # Get equity curve data
            if hasattr(self.strategy, 'daily_values') and self.strategy.daily_values:
                df = pd.DataFrame(self.strategy.daily_values)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Separate training and trading periods
                training_df = df[df.index < self.trading_start_date]
                trading_df = df[df.index >= self.trading_start_date]
                
                # 1. Equity Curve
                axes[0,0].plot(training_df.index, training_df['value'], 
                             label='Training Period', color='blue', alpha=0.7)
                axes[0,0].plot(trading_df.index, trading_df['value'], 
                             label='Trading Period', color='red', linewidth=2)
                axes[0,0].axvline(x=self.trading_start_date, color='green', 
                                linestyle='--', label='Trading Start')
                axes[0,0].set_title('Portfolio Value Over Time')
                axes[0,0].set_ylabel('Portfolio Value ($)')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
                
                # 2. Returns
                df['returns'] = df['value'].pct_change()
                trading_returns = trading_df['returns'].dropna()
                
                axes[0,1].plot(trading_returns.index, trading_returns.cumsum() * 100)
                axes[0,1].set_title('Cumulative Returns (Trading Period)')
                axes[0,1].set_ylabel('Cumulative Return (%)')
                axes[0,1].grid(True, alpha=0.3)
                
                # 3. Drawdown
                running_max = trading_df['value'].expanding().max()
                drawdown = (trading_df['value'] - running_max) / running_max * 100
                
                axes[1,0].fill_between(drawdown.index, drawdown, 0, 
                                     alpha=0.3, color='red')
                axes[1,0].plot(drawdown.index, drawdown, color='red')
                axes[1,0].set_title('Drawdown (Trading Period)')
                axes[1,0].set_ylabel('Drawdown (%)')
                axes[1,0].grid(True, alpha=0.3)
                
                # 4. Rolling Sharpe Ratio
                if len(trading_returns) > 30:
                    rolling_sharpe = trading_returns.rolling(30).mean() / trading_returns.rolling(30).std() * np.sqrt(252)
                    axes[1,1].plot(rolling_sharpe.index, rolling_sharpe)
                    axes[1,1].set_title('30-Day Rolling Sharpe Ratio')
                    axes[1,1].set_ylabel('Sharpe Ratio')
                    axes[1,1].grid(True, alpha=0.3)
                else:
                    axes[1,1].text(0.5, 0.5, 'Insufficient data for rolling Sharpe', 
                                 ha='center', va='center', transform=axes[1,1].transAxes)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'portfolio_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting portfolio performance: {e}")
    
    def plot_ml_performance(self, output_dir: Path):
        """Plot ML model performance metrics"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('ML Model Performance Analysis', fontsize=16, fontweight='bold')
            
            if hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics:
                ml_data = []
                for date, metrics in self.strategy.ml_metrics.items():
                    for symbol, symbol_metrics in metrics.items():
                        ml_data.append({
                            'date': pd.to_datetime(date),
                            'symbol': symbol,
                            'regime_accuracy': symbol_metrics.get('regime_accuracy', 0),
                            'strength_r2': symbol_metrics.get('strength_r2', 0),
                            'feature_importance': symbol_metrics.get('feature_importance', 0)
                        })
                
                if ml_data:
                    df = pd.DataFrame(ml_data)
                    trading_df = df[df['date'] >= self.trading_start_date]
                    
                    # 1. Regime Accuracy Over Time
                    avg_accuracy = trading_df.groupby('date')['regime_accuracy'].mean()
                    axes[0,0].plot(avg_accuracy.index, avg_accuracy)
                    axes[0,0].set_title('Average Regime Prediction Accuracy')
                    axes[0,0].set_ylabel('Accuracy')
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # 2. Strength R² Over Time
                    avg_r2 = trading_df.groupby('date')['strength_r2'].mean()
                    axes[0,1].plot(avg_r2.index, avg_r2)
                    axes[0,1].set_title('Average Signal Strength R²')
                    axes[0,1].set_ylabel('R² Score')
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # 3. Model Performance by Symbol
                    symbol_performance = trading_df.groupby('symbol').agg({
                        'regime_accuracy': 'mean',
                        'strength_r2': 'mean'
                    }).sort_values('regime_accuracy', ascending=False)
                    
                    axes[0,2].bar(range(len(symbol_performance)), symbol_performance['regime_accuracy'])
                    axes[0,2].set_title('Regime Accuracy by Symbol')
                    axes[0,2].set_ylabel('Average Accuracy')
                    axes[0,2].set_xticks(range(len(symbol_performance)))
                    axes[0,2].set_xticklabels(symbol_performance.index, rotation=45)
                    
                    # 4. Accuracy Distribution
                    axes[1,0].hist(trading_df['regime_accuracy'], bins=20, alpha=0.7, edgecolor='black')
                    axes[1,0].set_title('Regime Accuracy Distribution')
                    axes[1,0].set_xlabel('Accuracy')
                    axes[1,0].set_ylabel('Frequency')
                    
                    # 5. R² Distribution
                    axes[1,1].hist(trading_df['strength_r2'], bins=20, alpha=0.7, edgecolor='black')
                    axes[1,1].set_title('Signal Strength R² Distribution')
                    axes[1,1].set_xlabel('R² Score')
                    axes[1,1].set_ylabel('Frequency')
                    
                    # 6. Correlation between Accuracy and R²
                    axes[1,2].scatter(trading_df['regime_accuracy'], trading_df['strength_r2'], alpha=0.6)
                    axes[1,2].set_title('Regime Accuracy vs Signal Strength R²')
                    axes[1,2].set_xlabel('Regime Accuracy')
                    axes[1,2].set_ylabel('Signal Strength R²')
                    
                    # Add correlation coefficient
                    corr = trading_df['regime_accuracy'].corr(trading_df['strength_r2'])
                    axes[1,2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                                 transform=axes[1,2].transAxes, fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ml_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting ML performance: {e}")
    
    def plot_trading_metrics(self, output_dir: Path):
        """Plot trading performance metrics"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Trading Performance Analysis', fontsize=16, fontweight='bold')
            
            if hasattr(self.strategy, 'trade_log') and self.strategy.trade_log:
                trades_df = pd.DataFrame(self.strategy.trade_log)
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                
                # Filter to trading period
                trading_trades = trades_df[trades_df['date'] >= self.trading_start_date]
                
                if len(trading_trades) > 0:
                    # 1. P&L Distribution
                    axes[0,0].hist(trading_trades['pnl'], bins=20, alpha=0.7, edgecolor='black')
                    axes[0,0].set_title('Trade P&L Distribution')
                    axes[0,0].set_xlabel('P&L ($)')
                    axes[0,0].set_ylabel('Frequency')
                    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    
                    # 2. Cumulative P&L
                    trading_trades = trading_trades.sort_values('date')
                    trading_trades['cumulative_pnl'] = trading_trades['pnl'].cumsum()
                    axes[0,1].plot(trading_trades['date'], trading_trades['cumulative_pnl'])
                    axes[0,1].set_title('Cumulative P&L Over Time')
                    axes[0,1].set_ylabel('Cumulative P&L ($)')
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # 3. Win/Loss by Symbol
                    symbol_performance = trading_trades.groupby('symbol').agg({
                        'pnl': ['sum', 'count', lambda x: (x > 0).sum()]
                    }).round(2)
                    symbol_performance.columns = ['Total_PnL', 'Total_Trades', 'Winning_Trades']
                    symbol_performance['Win_Rate'] = symbol_performance['Winning_Trades'] / symbol_performance['Total_Trades']
                    
                    top_performers = symbol_performance.nlargest(10, 'Total_PnL')
                    axes[0,2].bar(range(len(top_performers)), top_performers['Total_PnL'])
                    axes[0,2].set_title('Top 10 Performers by Total P&L')
                    axes[0,2].set_ylabel('Total P&L ($)')
                    axes[0,2].set_xticks(range(len(top_performers)))
                    axes[0,2].set_xticklabels(top_performers.index, rotation=45)
                    
                    # 4. Position Size vs P&L
                    axes[1,0].scatter(trading_trades['size'], trading_trades['pnl'], alpha=0.6)
                    axes[1,0].set_title('Position Size vs P&L')
                    axes[1,0].set_xlabel('Position Size (shares)')
                    axes[1,0].set_ylabel('P&L ($)')
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # 5. Trade Duration Analysis (if available)
                    if 'exit_date' in trading_trades.columns:
                        trading_trades['duration'] = (pd.to_datetime(trading_trades['exit_date']) - 
                                                    trading_trades['date']).dt.days
                        axes[1,1].hist(trading_trades['duration'], bins=20, alpha=0.7, edgecolor='black')
                        axes[1,1].set_title('Trade Duration Distribution')
                        axes[1,1].set_xlabel('Duration (days)')
                        axes[1,1].set_ylabel('Frequency')
                    else:
                        axes[1,1].text(0.5, 0.5, 'Trade duration data not available', 
                                     ha='center', va='center', transform=axes[1,1].transAxes)
                    
                    # 6. Monthly P&L
                    trading_trades['month'] = trading_trades['date'].dt.to_period('M')
                    monthly_pnl = trading_trades.groupby('month')['pnl'].sum()
                    axes[1,2].bar(range(len(monthly_pnl)), monthly_pnl.values)
                    axes[1,2].set_title('Monthly P&L')
                    axes[1,2].set_ylabel('Monthly P&L ($)')
                    axes[1,2].set_xticks(range(len(monthly_pnl)))
                    axes[1,2].set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45)
                    axes[1,2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'trading_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting trading metrics: {e}")
    
    def plot_risk_analysis(self, output_dir: Path):
        """Plot risk analysis metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
            
            if hasattr(self.strategy, 'daily_values') and self.strategy.daily_values:
                df = pd.DataFrame(self.strategy.daily_values)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Filter to trading period
                trading_df = df[df.index >= self.trading_start_date]
                trading_df['returns'] = trading_df['value'].pct_change().dropna()
                
                if len(trading_df) > 1:
                    # 1. Return Distribution
                    axes[0,0].hist(trading_df['returns'] * 100, bins=30, alpha=0.7, edgecolor='black')
                    axes[0,0].set_title('Daily Returns Distribution')
                    axes[0,0].set_xlabel('Daily Return (%)')
                    axes[0,0].set_ylabel('Frequency')
                    axes[0,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    
                    # Add normal distribution overlay
                    mean_ret = trading_df['returns'].mean() * 100
                    std_ret = trading_df['returns'].std() * 100
                    x = np.linspace(trading_df['returns'].min() * 100, 
                                  trading_df['returns'].max() * 100, 100)
                    normal_dist = ((1/(std_ret * np.sqrt(2 * np.pi))) * 
                                 np.exp(-0.5 * ((x - mean_ret) / std_ret) ** 2))
                    axes[0,0].plot(x, normal_dist * len(trading_df) * 
                                 (trading_df['returns'].max() - trading_df['returns'].min()) * 100 / 30, 
                                 'r-', alpha=0.7, label='Normal Distribution')
                    axes[0,0].legend()
                    
                    # 2. Volatility Over Time
                    rolling_vol = trading_df['returns'].rolling(30).std() * np.sqrt(252) * 100
                    axes[0,1].plot(rolling_vol.index, rolling_vol)
                    axes[0,1].set_title('30-Day Rolling Volatility')
                    axes[0,1].set_ylabel('Annualized Volatility (%)')
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # 3. VaR Analysis
                    var_95 = np.percentile(trading_df['returns'], 5) * 100
                    var_99 = np.percentile(trading_df['returns'], 1) * 100
                    
                    axes[1,0].hist(trading_df['returns'] * 100, bins=30, alpha=0.7, edgecolor='black')
                    axes[1,0].axvline(x=var_95, color='orange', linestyle='--', 
                                    label=f'95% VaR: {var_95:.2f}%')
                    axes[1,0].axvline(x=var_99, color='red', linestyle='--', 
                                    label=f'99% VaR: {var_99:.2f}%')
                    axes[1,0].set_title('Value at Risk (VaR)')
                    axes[1,0].set_xlabel('Daily Return (%)')
                    axes[1,0].set_ylabel('Frequency')
                    axes[1,0].legend()
                    
                    # 4. Risk-Return Scatter (if we have benchmark data)
                    # For now, show risk metrics as text
                    risk_metrics = {
                        'Annualized Return': f"{trading_df['returns'].mean() * 252 * 100:.2f}%",
                        'Annualized Volatility': f"{trading_df['returns'].std() * np.sqrt(252) * 100:.2f}%",
                        'Sharpe Ratio': f"{trading_df['returns'].mean() / trading_df['returns'].std() * np.sqrt(252):.2f}",
                        'Max Daily Loss': f"{trading_df['returns'].min() * 100:.2f}%",
                        'Max Daily Gain': f"{trading_df['returns'].max() * 100:.2f}%",
                        '95% VaR': f"{var_95:.2f}%",
                        '99% VaR': f"{var_99:.2f}%"
                    }
                    
                    axes[1,1].axis('off')
                    text_str = '\n'.join([f'{k}: {v}' for k, v in risk_metrics.items()])
                    axes[1,1].text(0.1, 0.5, text_str, transform=axes[1,1].transAxes, 
                                 fontsize=12, verticalalignment='center',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
                    axes[1,1].set_title('Risk Metrics Summary')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'risk_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting risk analysis: {e}")
    
    def plot_stock_performance(self, output_dir: Path):
        """Plot individual stock performance"""
        try:
            if hasattr(self.strategy, 'trade_log') and self.strategy.trade_log:
                trades_df = pd.DataFrame(self.strategy.trade_log)
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trading_trades = trades_df[trades_df['date'] >= self.trading_start_date]
                
                if len(trading_trades) > 0:
                    # Stock performance summary
                    stock_summary = trading_trades.groupby('symbol').agg({
                        'pnl': ['sum', 'mean', 'count'],
                        'size': 'mean'
                    }).round(2)
                    
                    stock_summary.columns = ['Total_PnL', 'Avg_PnL', 'Trade_Count', 'Avg_Size']
                    stock_summary['Win_Rate'] = trading_trades.groupby('symbol')['pnl'].apply(lambda x: (x > 0).mean())
                    
                    # Create subplots
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    fig.suptitle('Individual Stock Performance', fontsize=16, fontweight='bold')
                    
                    # 1. Total P&L by Stock
                    top_stocks = stock_summary.nlargest(15, 'Total_PnL')
                    axes[0,0].barh(range(len(top_stocks)), top_stocks['Total_PnL'])
                    axes[0,0].set_title('Total P&L by Stock (Top 15)')
                    axes[0,0].set_xlabel('Total P&L ($)')
                    axes[0,0].set_yticks(range(len(top_stocks)))
                    axes[0,0].set_yticklabels(top_stocks.index)
                    
                    # 2. Win Rate by Stock
                    win_rate_sorted = stock_summary.sort_values('Win_Rate', ascending=False).head(15)
                    axes[0,1].barh(range(len(win_rate_sorted)), win_rate_sorted['Win_Rate'] * 100)
                    axes[0,1].set_title('Win Rate by Stock (Top 15)')
                    axes[0,1].set_xlabel('Win Rate (%)')
                    axes[0,1].set_yticks(range(len(win_rate_sorted)))
                    axes[0,1].set_yticklabels(win_rate_sorted.index)
                    
                    # 3. Trade Count vs Average P&L
                    axes[1,0].scatter(stock_summary['Trade_Count'], stock_summary['Avg_PnL'], 
                                    s=stock_summary['Total_PnL']*2, alpha=0.6)
                    axes[1,0].set_title('Trade Frequency vs Average P&L')
                    axes[1,0].set_xlabel('Number of Trades')
                    axes[1,0].set_ylabel('Average P&L ($)')
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # 4. Risk-Return by Stock
                    stock_volatility = trading_trades.groupby('symbol')['pnl'].std()
                    axes[1,1].scatter(stock_volatility, stock_summary['Avg_PnL'], alpha=0.6)
                    axes[1,1].set_title('Risk-Return Profile by Stock')
                    axes[1,1].set_xlabel('P&L Volatility ($)')
                    axes[1,1].set_ylabel('Average P&L ($)')
                    axes[1,1].grid(True, alpha=0.3)
                    
                    # Add stock labels for outliers
                    for symbol in stock_summary.index:
                        if (stock_summary.loc[symbol, 'Total_PnL'] > stock_summary['Total_PnL'].quantile(0.9) or
                            stock_summary.loc[symbol, 'Total_PnL'] < stock_summary['Total_PnL'].quantile(0.1)):
                            x = stock_volatility[symbol] if symbol in stock_volatility else 0
                            y = stock_summary.loc[symbol, 'Avg_PnL']
                            axes[1,1].annotate(symbol, (x, y), xytext=(5, 5), 
                                             textcoords='offset points', fontsize=8)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / 'stock_performance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Save detailed stock summary to CSV
                    stock_summary.to_csv(output_dir / 'stock_performance_summary.csv')
            
        except Exception as e:
            print(f"Error plotting stock performance: {e}")
    
    def analyze_overfitting(self, output_dir: Path):
        """Analyze potential overfitting issues"""
        try:
            analysis = {
                'training_vs_trading_performance': {},
                'model_stability': {},
                'signal_consistency': {}
            }
            
            # Compare training vs trading performance
            if hasattr(self.strategy, 'daily_values') and self.strategy.daily_values:
                df = pd.DataFrame(self.strategy.daily_values)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df['returns'] = df['value'].pct_change()
                
                training_df = df[df.index < self.trading_start_date]
                trading_df = df[df.index >= self.trading_start_date]
                
                if len(training_df) > 0 and len(trading_df) > 0:
                    training_returns = training_df['returns'].dropna()
                    trading_returns = trading_df['returns'].dropna()
                    
                    analysis['training_vs_trading_performance'] = {
                        'training_sharpe': training_returns.mean() / training_returns.std() * np.sqrt(252) if len(training_returns) > 1 else 0,
                        'trading_sharpe': trading_returns.mean() / trading_returns.std() * np.sqrt(252) if len(trading_returns) > 1 else 0,
                        'training_volatility': training_returns.std() * np.sqrt(252) if len(training_returns) > 1 else 0,
                        'trading_volatility': trading_returns.std() * np.sqrt(252) if len(trading_returns) > 1 else 0,
                        'training_return': training_returns.mean() * 252 if len(training_returns) > 0 else 0,
                        'trading_return': trading_returns.mean() * 252 if len(trading_returns) > 0 else 0
                    }
            
            # Analyze ML model stability
            if hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics:
                accuracy_over_time = []
                r2_over_time = []
                
                for date, metrics in self.strategy.ml_metrics.items():
                    date_obj = pd.to_datetime(date)
                    if date_obj >= self.trading_start_date:
                        avg_accuracy = np.mean([m.get('regime_accuracy', 0) for m in metrics.values()])
                        avg_r2 = np.mean([m.get('strength_r2', 0) for m in metrics.values()])
                        accuracy_over_time.append(avg_accuracy)
                        r2_over_time.append(avg_r2)
                
                if accuracy_over_time and r2_over_time:
                    analysis['model_stability'] = {
                        'accuracy_trend': np.polyfit(range(len(accuracy_over_time)), accuracy_over_time, 1)[0],
                        'r2_trend': np.polyfit(range(len(r2_over_time)), r2_over_time, 1)[0],
                        'accuracy_volatility': np.std(accuracy_over_time),
                        'r2_volatility': np.std(r2_over_time)
                    }
            
            # Create overfitting analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Overfitting Analysis', fontsize=16, fontweight='bold')
            
            # 1. Training vs Trading Performance Comparison
            if analysis['training_vs_trading_performance']:
                perf = analysis['training_vs_trading_performance']
                metrics = ['Sharpe Ratio', 'Volatility', 'Annual Return']
                training_vals = [perf['training_sharpe'], perf['training_volatility'], perf['training_return']]
                trading_vals = [perf['trading_sharpe'], perf['trading_volatility'], perf['trading_return']]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                axes[0,0].bar(x - width/2, training_vals, width, label='Training', alpha=0.7)
                axes[0,0].bar(x + width/2, trading_vals, width, label='Trading', alpha=0.7)
                axes[0,0].set_title('Training vs Trading Performance')
                axes[0,0].set_xticks(x)
                axes[0,0].set_xticklabels(metrics)
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
            
            # 2. Model Accuracy Trend
            if hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics:
                trading_dates = []
                trading_accuracies = []
                
                for date, metrics in self.strategy.ml_metrics.items():
                    date_obj = pd.to_datetime(date)
                    if date_obj >= self.trading_start_date:
                        trading_dates.append(date_obj)
                        avg_accuracy = np.mean([m.get('regime_accuracy', 0) for m in metrics.values()])
                        trading_accuracies.append(avg_accuracy)
                
                if trading_dates and trading_accuracies:
                    axes[0,1].plot(trading_dates, trading_accuracies, 'o-', alpha=0.7)
                    # Add trend line
                    if len(trading_accuracies) > 1:
                        z = np.polyfit(range(len(trading_accuracies)), trading_accuracies, 1)
                        p = np.poly1d(z)
                        axes[0,1].plot(trading_dates, p(range(len(trading_accuracies))), 
                                     "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}')
                        axes[0,1].legend()
                    
                    axes[0,1].set_title('Model Accuracy Over Trading Period')
                    axes[0,1].set_ylabel('Average Accuracy')
                    axes[0,1].grid(True, alpha=0.3)
            
            # 3. & 4. Overfitting Indicators (text summary)
            overfitting_indicators = []
            
            if analysis['training_vs_trading_performance']:
                perf = analysis['training_vs_trading_performance']
                sharpe_diff = perf['training_sharpe'] - perf['trading_sharpe']
                if sharpe_diff > 0.5:
                    overfitting_indicators.append(f"⚠️ Sharpe ratio dropped by {sharpe_diff:.2f}")
                
                vol_diff = abs(perf['training_volatility'] - perf['trading_volatility'])
                if vol_diff > 0.1:
                    overfitting_indicators.append(f"⚠️ Volatility changed by {vol_diff:.2f}")
            
            if analysis['model_stability']:
                if analysis['model_stability']['accuracy_trend'] < -0.001:
                    overfitting_indicators.append("⚠️ Model accuracy declining over time")
                
                if analysis['model_stability']['accuracy_volatility'] > 0.1:
                    overfitting_indicators.append("⚠️ High model accuracy volatility")
            
            if not overfitting_indicators:
                overfitting_indicators = ["✅ No major overfitting indicators detected"]
            
            axes[1,0].axis('off')
            axes[1,0].text(0.1, 0.5, '\n'.join(overfitting_indicators), 
                         transform=axes[1,0].transAxes, fontsize=12, 
                         verticalalignment='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
            axes[1,0].set_title('Overfitting Indicators')
            
            # Model recommendations
            recommendations = [
                "📊 Monitor model performance consistency",
                "🔄 Consider periodic model retraining",
                "📈 Track prediction accuracy trends",
                "⚖️ Balance model complexity vs stability",
                "🎯 Use out-of-sample validation"
            ]
            
            axes[1,1].axis('off')
            axes[1,1].text(0.1, 0.5, '\n'.join(recommendations), 
                         transform=axes[1,1].transAxes, fontsize=12, 
                         verticalalignment='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
            axes[1,1].set_title('Recommendations')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save analysis to JSON
            with open(output_dir / 'overfitting_analysis.json', 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
        except Exception as e:
            print(f"Error in overfitting analysis: {e}")
    
    def generate_summary_report(self, output_dir: Path):
        """Generate a comprehensive summary report"""
        try:
            report = {
                'backtest_summary': {},
                'performance_metrics': {},
                'ml_model_summary': {},
                'risk_metrics': {},
                'trading_summary': {},
                'recommendations': []
            }
            
            # Backtest Summary
            report['backtest_summary'] = {
                'training_period': f"2022-08-22 to 2024-08-22",
                'trading_period': f"2024-08-22 to 2025-08-22",
                'initial_capital': 100000,
                'symbols_traded': len(self.strategy.symbols) if hasattr(self.strategy, 'symbols') else 0
            }
            
            # Performance Metrics
            if hasattr(self.strategy, 'daily_values') and self.strategy.daily_values:
                df = pd.DataFrame(self.strategy.daily_values)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                trading_df = df[df.index >= self.trading_start_date]
                if len(trading_df) > 0:
                    initial_value = trading_df['value'].iloc[0]
                    final_value = trading_df['value'].iloc[-1]
                    total_return = (final_value - initial_value) / initial_value
                    
                    trading_df['returns'] = trading_df['value'].pct_change().dropna()
                    if len(trading_df['returns']) > 1:
                        sharpe = trading_df['returns'].mean() / trading_df['returns'].std() * np.sqrt(252)
                        volatility = trading_df['returns'].std() * np.sqrt(252)
                        max_dd = ((trading_df['value'] - trading_df['value'].expanding().max()) / trading_df['value'].expanding().max()).min()
                        
                        report['performance_metrics'] = {
                            'total_return_pct': round(total_return * 100, 2),
                            'annualized_return_pct': round(trading_df['returns'].mean() * 252 * 100, 2),
                            'sharpe_ratio': round(sharpe, 2),
                            'volatility_pct': round(volatility * 100, 2),
                            'max_drawdown_pct': round(max_dd * 100, 2),
                            'final_portfolio_value': round(final_value, 2)
                        }
            
            # Trading Summary
            if hasattr(self.strategy, 'trade_log') and self.strategy.trade_log:
                trades_df = pd.DataFrame(self.strategy.trade_log)
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trading_trades = trades_df[trades_df['date'] >= self.trading_start_date]
                
                if len(trading_trades) > 0:
                    winning_trades = len(trading_trades[trading_trades['pnl'] > 0])
                    total_trades = len(trading_trades)
                    
                    report['trading_summary'] = {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'win_rate_pct': round(winning_trades / total_trades * 100, 2),
                        'average_trade_pnl': round(trading_trades['pnl'].mean(), 2),
                        'total_trading_pnl': round(trading_trades['pnl'].sum(), 2),
                        'best_trade': round(trading_trades['pnl'].max(), 2),
                        'worst_trade': round(trading_trades['pnl'].min(), 2)
                    }
            
            # ML Model Summary
            if hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics:
                all_accuracies = []
                all_r2_scores = []
                
                for date, metrics in self.strategy.ml_metrics.items():
                    date_obj = pd.to_datetime(date)
                    if date_obj >= self.trading_start_date:
                        for symbol_metrics in metrics.values():
                            all_accuracies.append(symbol_metrics.get('regime_accuracy', 0))
                            all_r2_scores.append(symbol_metrics.get('strength_r2', 0))
                
                if all_accuracies and all_r2_scores:
                    report['ml_model_summary'] = {
                        'average_regime_accuracy': round(np.mean(all_accuracies), 3),
                        'average_strength_r2': round(np.mean(all_r2_scores), 3),
                        'model_updates': len(self.strategy.ml_metrics),
                        'accuracy_std': round(np.std(all_accuracies), 3),
                        'r2_std': round(np.std(all_r2_scores), 3)
                    }
            
            # Generate recommendations
            recommendations = []
            
            if 'performance_metrics' in report:
                perf = report['performance_metrics']
                if perf.get('sharpe_ratio', 0) > 1.5:
                    recommendations.append("✅ Strong risk-adjusted returns achieved")
                elif perf.get('sharpe_ratio', 0) < 0.5:
                    recommendations.append("⚠️ Consider improving risk-adjusted returns")
                
                if perf.get('max_drawdown_pct', 0) < -20:
                    recommendations.append("⚠️ High drawdown - consider risk management improvements")
                
                if perf.get('volatility_pct', 0) > 30:
                    recommendations.append("⚠️ High volatility - consider position sizing adjustments")
            
            if 'trading_summary' in report:
                trade = report['trading_summary']
                if trade.get('win_rate_pct', 0) < 40:
                    recommendations.append("⚠️ Low win rate - review signal generation")
                elif trade.get('win_rate_pct', 0) > 60:
                    recommendations.append("✅ Good win rate achieved")
            
            if 'ml_model_summary' in report:
                ml = report['ml_model_summary']
                if ml.get('average_regime_accuracy', 0) < 0.6:
                    recommendations.append("⚠️ Low ML accuracy - consider feature engineering")
                elif ml.get('average_regime_accuracy', 0) > 0.75:
                    recommendations.append("✅ Strong ML model performance")
            
            if not recommendations:
                recommendations.append("📊 Review detailed analysis for optimization opportunities")
            
            report['recommendations'] = recommendations
            
            # Save comprehensive report
            with open(output_dir / 'comprehensive_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate markdown report
            self._generate_markdown_report(report, output_dir)
            
            print("📋 Summary report generated")
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
    
    def _generate_markdown_report(self, report: Dict, output_dir: Path):
        """Generate a markdown summary report"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            markdown = f"""# ML Trading System Performance Report
            
Generated: {timestamp}

## Executive Summary

### Backtest Overview
- **Training Period**: {report.get('backtest_summary', {}).get('training_period', 'N/A')}
- **Trading Period**: {report.get('backtest_summary', {}).get('trading_period', 'N/A')}
- **Initial Capital**: ${report.get('backtest_summary', {}).get('initial_capital', 0):,}
- **Symbols Traded**: {report.get('backtest_summary', {}).get('symbols_traded', 0)}

### Key Performance Metrics
"""
            
            if 'performance_metrics' in report:
                perf = report['performance_metrics']
                markdown += f"""
- **Total Return**: {perf.get('total_return_pct', 0):.2f}%
- **Annualized Return**: {perf.get('annualized_return_pct', 0):.2f}%
- **Sharpe Ratio**: {perf.get('sharpe_ratio', 0):.2f}
- **Volatility**: {perf.get('volatility_pct', 0):.2f}%
- **Max Drawdown**: {perf.get('max_drawdown_pct', 0):.2f}%
- **Final Portfolio Value**: ${perf.get('final_portfolio_value', 0):,.2f}
"""
            
            if 'trading_summary' in report:
                trade = report['trading_summary']
                markdown += f"""
### Trading Statistics
- **Total Trades**: {trade.get('total_trades', 0)}
- **Win Rate**: {trade.get('win_rate_pct', 0):.2f}%
- **Average Trade P&L**: ${trade.get('average_trade_pnl', 0):.2f}
- **Total Trading P&L**: ${trade.get('total_trading_pnl', 0):.2f}
- **Best Trade**: ${trade.get('best_trade', 0):.2f}
- **Worst Trade**: ${trade.get('worst_trade', 0):.2f}
"""
            
            if 'ml_model_summary' in report:
                ml = report['ml_model_summary']
                markdown += f"""
### ML Model Performance
- **Average Regime Accuracy**: {ml.get('average_regime_accuracy', 0):.3f}
- **Average Signal Strength R²**: {ml.get('average_strength_r2', 0):.3f}
- **Model Updates**: {ml.get('model_updates', 0)}
- **Accuracy Std Dev**: {ml.get('accuracy_std', 0):.3f}
- **R² Std Dev**: {ml.get('r2_std', 0):.3f}
"""
            
            markdown += f"""
### Recommendations
"""
            for rec in report.get('recommendations', []):
                markdown += f"- {rec}\n"
            
            markdown += f"""
## Detailed Analysis

Refer to the generated plots for detailed analysis:
- `portfolio_performance.png` - Portfolio value and returns over time
- `ml_performance.png` - ML model accuracy and performance metrics
- `trading_metrics.png` - Trading performance and P&L analysis
- `risk_analysis.png` - Risk metrics and volatility analysis
- `stock_performance.png` - Individual stock performance breakdown
- `overfitting_analysis.png` - Overfitting detection and model stability

## Data Files
- `comprehensive_report.json` - Complete numerical results
- `stock_performance_summary.csv` - Detailed stock-by-stock performance
- `overfitting_analysis.json` - Overfitting analysis results
"""
            
            with open(output_dir / 'performance_report.md', 'w') as f:
                f.write(markdown)
                
        except Exception as e:
            print(f"Error generating markdown report: {e}")
    
    def analyze_bad_trades(self, output_dir: Path):
        """Analyze bad trades in detail - what went wrong and why"""
        try:
            print("🔍 Analyzing bad trades...")
            
            # Get trade statistics from strategy
            if hasattr(self.strategy, 'get_trade_stats'):
                trade_stats = self.strategy.get_trade_stats()
                trade_log = trade_stats.get('trade_log', [])
            else:
                trade_log = getattr(self.strategy, 'trade_log', [])
            
            if not trade_log:
                print("⚠️ No trade data available for bad trade analysis")
                return
            
            trades_df = pd.DataFrame(trade_log)
            
            # Only analyze completed trades (with P&L)
            completed_trades = trades_df[trades_df['pnl'] != 0].copy() if 'pnl' in trades_df.columns else trades_df.copy()
            
            if len(completed_trades) == 0:
                print("⚠️ No completed trades available for analysis")
                return
            
            # Convert date column if it exists
            if 'date' in completed_trades.columns:
                completed_trades['date'] = pd.to_datetime(completed_trades['date'])
            
            # Filter to trading period if we have dates
            trading_trades = completed_trades.copy()
            if 'date' in completed_trades.columns and hasattr(self, 'trading_start_date'):
                trading_trades = completed_trades[completed_trades['date'] >= self.trading_start_date].copy()
            
            if len(trading_trades) == 0:
                print("⚠️ No trades in trading period for analysis")
                return
            
            # Define bad trades (losing trades or bottom 25% of P&L)
            if 'pnl' in trading_trades.columns:
                losing_trades = trading_trades[trading_trades['pnl'] < 0].copy()
                pnl_threshold = trading_trades['pnl'].quantile(0.25)
                bad_trades = trading_trades[trading_trades['pnl'] <= pnl_threshold].copy()
                good_trades = trading_trades[trading_trades['pnl'] > trading_trades['pnl'].quantile(0.75)].copy()
            else:
                print("⚠️ No P&L data available for bad trade analysis")
                return
            
            print(f"📊 Analyzing {len(bad_trades)} bad trades vs {len(good_trades)} good trades")
            
            # Create detailed analysis
            bad_trade_analysis = {
                'summary': {
                    'total_bad_trades': len(bad_trades),
                    'bad_trade_threshold': pnl_threshold,
                    'total_loss_from_bad_trades': bad_trades['pnl'].sum(),
                    'average_bad_trade_loss': bad_trades['pnl'].mean(),
                    'worst_trade_loss': bad_trades['pnl'].min(),
                    'bad_trades_percentage': len(bad_trades) / len(trading_trades) * 100
                },
                'by_symbol': {},
                'by_signal_strength': {},
                'by_time_period': {},
                'patterns': []
            }
            
            # Analysis by symbol
            bad_by_symbol = bad_trades.groupby('symbol').agg({
                'pnl': ['count', 'sum', 'mean'],
                'size': 'mean'
            }).round(2)
            bad_by_symbol.columns = ['bad_trade_count', 'total_loss', 'avg_loss', 'avg_size']
            
            good_by_symbol = good_trades.groupby('symbol').agg({
                'pnl': ['count', 'sum', 'mean'],
                'size': 'mean'
            }).round(2)
            good_by_symbol.columns = ['good_trade_count', 'total_profit', 'avg_profit', 'avg_size']
            
            # Analysis by signal strength (if available)
            if 'signal_strength' in bad_trades.columns:
                bad_trade_analysis['by_signal_strength'] = {
                    'bad_trades_avg_strength': bad_trades['signal_strength'].mean(),
                    'good_trades_avg_strength': good_trades['signal_strength'].mean(),
                    'strength_difference': good_trades['signal_strength'].mean() - bad_trades['signal_strength'].mean()
                }
            
            # Time-based analysis
            bad_trades['hour'] = bad_trades['date'].dt.hour
            bad_trades['day_of_week'] = bad_trades['date'].dt.day_name()
            bad_trades['month'] = bad_trades['date'].dt.month
            
            bad_trade_analysis['by_time_period'] = {
                'worst_hours': bad_trades.groupby('hour')['pnl'].mean().nsmallest(3).to_dict(),
                'worst_days': bad_trades.groupby('day_of_week')['pnl'].mean().nsmallest(3).to_dict(),
                'worst_months': bad_trades.groupby('month')['pnl'].mean().nsmallest(3).to_dict()
            }
            
            # Pattern identification
            patterns = []
            
            # Pattern 1: Large position size correlates with bad trades
            if bad_trades['size'].mean() > good_trades['size'].mean():
                patterns.append(f"⚠️ Bad trades tend to have larger position sizes (avg: {bad_trades['size'].mean():.0f} vs {good_trades['size'].mean():.0f})")
            
            # Pattern 2: Certain symbols consistently underperform
            bad_symbols = bad_by_symbol.nlargest(5, 'bad_trade_count').index.tolist()
            if bad_symbols:
                patterns.append(f"⚠️ Worst performing symbols: {', '.join(bad_symbols)}")
            
            # Pattern 3: Signal strength analysis (if available)
            if 'signal_strength' in bad_trades.columns:
                if bad_trades['signal_strength'].mean() < good_trades['signal_strength'].mean():
                    patterns.append(f"⚠️ Bad trades have weaker signals (avg: {bad_trades['signal_strength'].mean():.3f} vs {good_trades['signal_strength'].mean():.3f})")
            
            # Pattern 4: Time-based patterns
            worst_hour = min(bad_trade_analysis['by_time_period']['worst_hours'], 
                           key=bad_trade_analysis['by_time_period']['worst_hours'].get)
            patterns.append(f"⚠️ Worst trading hour: {worst_hour}:00 (avg loss: ${bad_trade_analysis['by_time_period']['worst_hours'][worst_hour]:.2f})")
            
            bad_trade_analysis['patterns'] = patterns
            
            # Create visualizations
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            fig.suptitle('Bad Trade Analysis - Root Cause Investigation', fontsize=16, fontweight='bold')
            
            # 1. P&L Distribution Comparison
            axes[0,0].hist(bad_trades['pnl'], bins=20, alpha=0.7, label='Bad Trades', color='red', edgecolor='black')
            axes[0,0].hist(good_trades['pnl'], bins=20, alpha=0.7, label='Good Trades', color='green', edgecolor='black')
            axes[0,0].set_title('P&L Distribution: Bad vs Good Trades')
            axes[0,0].set_xlabel('P&L ($)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].legend()
            axes[0,0].axvline(x=0, color='black', linestyle='--', alpha=0.7)
            
            # 2. Position Size Comparison
            size_comparison = pd.DataFrame({
                'Bad Trades': [bad_trades['size'].mean(), bad_trades['size'].std()],
                'Good Trades': [good_trades['size'].mean(), good_trades['size'].std()]
            }, index=['Mean', 'Std Dev'])
            
            size_comparison.plot(kind='bar', ax=axes[0,1], color=['red', 'green'], alpha=0.7)
            axes[0,1].set_title('Position Size: Bad vs Good Trades')
            axes[0,1].set_ylabel('Position Size (shares)')
            axes[0,1].tick_params(axis='x', rotation=0)
            
            # 3. Bad Trades by Symbol
            worst_symbols = bad_by_symbol.nlargest(10, 'total_loss')
            axes[1,0].barh(range(len(worst_symbols)), worst_symbols['total_loss'], color='red', alpha=0.7)
            axes[1,0].set_title('Total Losses by Symbol (Top 10 Worst)')
            axes[1,0].set_xlabel('Total Loss ($)')
            axes[1,0].set_yticks(range(len(worst_symbols)))
            axes[1,0].set_yticklabels(worst_symbols.index)
            
            # 4. Bad Trades Over Time
            bad_trades_daily = bad_trades.groupby(bad_trades['date'].dt.date)['pnl'].sum()
            axes[1,1].plot(bad_trades_daily.index, bad_trades_daily.values, 'ro-', alpha=0.7, markersize=4)
            axes[1,1].set_title('Daily Bad Trade Losses Over Time')
            axes[1,1].set_ylabel('Daily Loss ($)')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
            
            # 5. Signal Strength Comparison (if available)
            if 'signal_strength' in bad_trades.columns and 'signal_strength' in good_trades.columns:
                axes[2,0].boxplot([bad_trades['signal_strength'], good_trades['signal_strength']], 
                                labels=['Bad Trades', 'Good Trades'])
                axes[2,0].set_title('Signal Strength Distribution')
                axes[2,0].set_ylabel('Signal Strength')
                axes[2,0].grid(True, alpha=0.3)
            else:
                axes[2,0].text(0.5, 0.5, 'Signal strength data not available', 
                             ha='center', va='center', transform=axes[2,0].transAxes)
                axes[2,0].set_title('Signal Strength Analysis')
            
            # 6. Key Insights Summary
            axes[2,1].axis('off')
            insights_text = f"""Bad Trade Analysis Summary:
            
📊 {len(bad_trades)} bad trades analyzed
💰 Total loss: ${bad_trades['pnl'].sum():,.2f}
📉 Avg loss per bad trade: ${bad_trades['pnl'].mean():.2f}
📈 Bad trades: {len(bad_trades)/len(trading_trades)*100:.1f}% of all trades

Key Patterns Found:
"""
            for pattern in patterns:
                insights_text += f"\n{pattern}"
            
            axes[2,1].text(0.05, 0.95, insights_text, transform=axes[2,1].transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_dir / 'bad_trade_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save detailed bad trade data
            bad_trades_detailed = bad_trades.copy()
            bad_trades_detailed['trade_rank'] = 'Bad'
            good_trades_detailed = good_trades.copy()
            good_trades_detailed['trade_rank'] = 'Good'
            
            combined_analysis = pd.concat([bad_trades_detailed, good_trades_detailed])
            combined_analysis.to_csv(output_dir / 'detailed_trade_analysis.csv', index=False)
            
            # Save analysis results
            with open(output_dir / 'bad_trade_analysis.json', 'w') as f:
                json.dump(bad_trade_analysis, f, indent=2, default=str)
            
            print(f"📋 Bad trade analysis saved to {output_dir}")
            
        except Exception as e:
            print(f"❌ Error in bad trade analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_signal_quality(self, output_dir: Path):
        """Analyze signal quality and prediction accuracy"""
        try:
            print("🎯 Analyzing signal quality...")
            
            signal_analysis = {
                'signal_accuracy': {},
                'signal_strength_vs_outcome': {},
                'false_signals': {},
                'recommendations': []
            }
            
            # Get ML metrics if available
            if hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics:
                all_predictions = []
                
                for date, metrics in self.strategy.ml_metrics.items():
                    date_obj = pd.to_datetime(date)
                    if date_obj >= self.trading_start_date:
                        for symbol, symbol_metrics in metrics.items():
                            all_predictions.append({
                                'date': date_obj,
                                'symbol': symbol,
                                'regime_accuracy': symbol_metrics.get('regime_accuracy', 0),
                                'strength_r2': symbol_metrics.get('strength_r2', 0),
                                'feature_importance': symbol_metrics.get('feature_importance', {})
                            })
                
                if all_predictions:
                    pred_df = pd.DataFrame(all_predictions)
                    
                    # Signal accuracy over time
                    accuracy_over_time = pred_df.groupby('date')['regime_accuracy'].mean()
                    r2_over_time = pred_df.groupby('date')['strength_r2'].mean()
                    
                    # Signal quality by symbol
                    symbol_quality = pred_df.groupby('symbol').agg({
                        'regime_accuracy': 'mean',
                        'strength_r2': 'mean'
                    }).round(3)
                    
                    signal_analysis['signal_accuracy'] = {
                        'overall_avg_accuracy': pred_df['regime_accuracy'].mean(),
                        'overall_avg_r2': pred_df['strength_r2'].mean(),
                        'accuracy_trend': np.polyfit(range(len(accuracy_over_time)), accuracy_over_time.values, 1)[0] if len(accuracy_over_time) > 1 else 0,
                        'best_performing_symbols': symbol_quality.nlargest(5, 'regime_accuracy').to_dict(),
                        'worst_performing_symbols': symbol_quality.nsmallest(5, 'regime_accuracy').to_dict()
                    }
            
            # Analyze signal strength vs trade outcomes (if trade data available)
            if hasattr(self.strategy, 'trade_log') and self.strategy.trade_log:
                trades_df = pd.DataFrame(self.strategy.trade_log)
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trading_trades = trades_df[trades_df['date'] >= self.trading_start_date]
                
                if 'signal_strength' in trading_trades.columns and len(trading_trades) > 0:
                    # Correlate signal strength with outcomes
                    strength_vs_pnl = trading_trades[['signal_strength', 'pnl']].corr().iloc[0,1]
                    
                    # Analyze different strength buckets
                    trading_trades['strength_bucket'] = pd.cut(trading_trades['signal_strength'], 
                                                             bins=5, labels=['Very Weak', 'Weak', 'Medium', 'Strong', 'Very Strong'])
                    
                    bucket_performance = trading_trades.groupby('strength_bucket').agg({
                        'pnl': ['mean', 'count', lambda x: (x > 0).mean()]
                    }).round(3)
                    bucket_performance.columns = ['avg_pnl', 'trade_count', 'win_rate']
                    
                    signal_analysis['signal_strength_vs_outcome'] = {
                        'correlation_strength_pnl': strength_vs_pnl,
                        'bucket_performance': bucket_performance.to_dict(),
                        'optimal_strength_threshold': trading_trades[trading_trades['pnl'] > 0]['signal_strength'].quantile(0.25)
                    }
            
            # Create signal quality visualizations
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Signal Quality & Prediction Analysis', fontsize=16, fontweight='bold')
            
            # 1. Signal Accuracy Over Time
            if hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics:
                trading_dates = []
                trading_accuracies = []
                trading_r2 = []
                
                for date, metrics in self.strategy.ml_metrics.items():
                    date_obj = pd.to_datetime(date)
                    if date_obj >= self.trading_start_date:
                        trading_dates.append(date_obj)
                        avg_accuracy = np.mean([m.get('regime_accuracy', 0) for m in metrics.values()])
                        avg_r2 = np.mean([m.get('strength_r2', 0) for m in metrics.values()])
                        trading_accuracies.append(avg_accuracy)
                        trading_r2.append(avg_r2)
                
                if trading_dates and trading_accuracies:
                    axes[0,0].plot(trading_dates, trading_accuracies, 'b-', label='Regime Accuracy', alpha=0.7)
                    axes[0,0].plot(trading_dates, trading_r2, 'r-', label='Signal Strength R²', alpha=0.7)
                    axes[0,0].set_title('ML Model Accuracy Over Time')
                    axes[0,0].set_ylabel('Accuracy / R²')
                    axes[0,0].legend()
                    axes[0,0].grid(True, alpha=0.3)
                else:
                    axes[0,0].text(0.5, 0.5, 'No ML metrics available', ha='center', va='center', transform=axes[0,0].transAxes)
            
            # 2. Signal Strength vs P&L Correlation
            if hasattr(self.strategy, 'trade_log') and self.strategy.trade_log:
                trades_df = pd.DataFrame(self.strategy.trade_log)
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trading_trades = trades_df[trades_df['date'] >= self.trading_start_date]
                
                if 'signal_strength' in trading_trades.columns and len(trading_trades) > 0:
                    axes[0,1].scatter(trading_trades['signal_strength'], trading_trades['pnl'], alpha=0.6)
                    axes[0,1].set_title('Signal Strength vs Trade Outcome')
                    axes[0,1].set_xlabel('Signal Strength')
                    axes[0,1].set_ylabel('P&L ($)')
                    
                    # Add correlation text
                    corr = trading_trades['signal_strength'].corr(trading_trades['pnl'])
                    axes[0,1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                                 transform=axes[0,1].transAxes, fontsize=10)
                    axes[0,1].grid(True, alpha=0.3)
                else:
                    axes[0,1].text(0.5, 0.5, 'Signal strength data not available', 
                                 ha='center', va='center', transform=axes[0,1].transAxes)
            
            # 3. Performance by Signal Strength Bucket
            if (hasattr(self.strategy, 'trade_log') and self.strategy.trade_log and 
                'signal_strength_vs_outcome' in signal_analysis and 
                'bucket_performance' in signal_analysis['signal_strength_vs_outcome']):
                
                bucket_data = signal_analysis['signal_strength_vs_outcome']['bucket_performance']
                if bucket_data and 'avg_pnl' in bucket_data:
                    buckets = list(bucket_data['avg_pnl'].keys())
                    avg_pnls = list(bucket_data['avg_pnl'].values())
                    
                    axes[0,2].bar(buckets, avg_pnls, alpha=0.7, 
                                color=['red' if x < 0 else 'green' for x in avg_pnls])
                    axes[0,2].set_title('Average P&L by Signal Strength')
                    axes[0,2].set_ylabel('Average P&L ($)')
                    axes[0,2].tick_params(axis='x', rotation=45)
                    axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.7)
                else:
                    axes[0,2].text(0.5, 0.5, 'Signal strength buckets not available', 
                                 ha='center', va='center', transform=axes[0,2].transAxes)
            else:
                axes[0,2].text(0.5, 0.5, 'Signal strength analysis not available', 
                             ha='center', va='center', transform=axes[0,2].transAxes)
            
            # 4. Model Accuracy Distribution
            if hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics:
                all_accuracies = []
                for date, metrics in self.strategy.ml_metrics.items():
                    date_obj = pd.to_datetime(date)
                    if date_obj >= self.trading_start_date:
                        for symbol_metrics in metrics.values():
                            all_accuracies.append(symbol_metrics.get('regime_accuracy', 0))
                
                if all_accuracies:
                    axes[1,0].hist(all_accuracies, bins=20, alpha=0.7, edgecolor='black')
                    axes[1,0].set_title('Model Accuracy Distribution')
                    axes[1,0].set_xlabel('Accuracy')
                    axes[1,0].set_ylabel('Frequency')
                    axes[1,0].axvline(x=np.mean(all_accuracies), color='red', linestyle='--', 
                                    label=f'Mean: {np.mean(all_accuracies):.3f}')
                    axes[1,0].legend()
                else:
                    axes[1,0].text(0.5, 0.5, 'No accuracy data available', 
                                 ha='center', va='center', transform=axes[1,0].transAxes)
            
            # 5. Signal Quality by Symbol
            if (hasattr(self.strategy, 'ml_metrics') and self.strategy.ml_metrics and 
                'signal_accuracy' in signal_analysis and 
                'best_performing_symbols' in signal_analysis['signal_accuracy']):
                
                best_symbols = signal_analysis['signal_accuracy']['best_performing_symbols'].get('regime_accuracy', {})
                if best_symbols:
                    symbols = list(best_symbols.keys())[:10]  # Top 10
                    accuracies = list(best_symbols.values())[:10]
                    
                    axes[1,1].barh(range(len(symbols)), accuracies, alpha=0.7)
                    axes[1,1].set_title('Top 10 Symbols by Model Accuracy')
                    axes[1,1].set_xlabel('Average Accuracy')
                    axes[1,1].set_yticks(range(len(symbols)))
                    axes[1,1].set_yticklabels(symbols)
                else:
                    axes[1,1].text(0.5, 0.5, 'Symbol accuracy data not available', 
                                 ha='center', va='center', transform=axes[1,1].transAxes)
            
            # 6. Signal Quality Summary
            axes[1,2].axis('off')
            
            # Generate recommendations
            recommendations = []
            
            if 'signal_accuracy' in signal_analysis:
                avg_acc = signal_analysis['signal_accuracy'].get('overall_avg_accuracy', 0)
                if avg_acc < 0.6:
                    recommendations.append("⚠️ Low model accuracy - consider feature engineering")
                elif avg_acc > 0.75:
                    recommendations.append("✅ Strong model accuracy achieved")
                
                trend = signal_analysis['signal_accuracy'].get('accuracy_trend', 0)
                if trend < -0.001:
                    recommendations.append("⚠️ Model accuracy declining - retrain more frequently")
                elif trend > 0.001:
                    recommendations.append("✅ Model accuracy improving over time")
            
            if 'signal_strength_vs_outcome' in signal_analysis:
                corr = signal_analysis['signal_strength_vs_outcome'].get('correlation_strength_pnl', 0)
                if corr < 0.1:
                    recommendations.append("⚠️ Weak signal-outcome correlation - review signal generation")
                elif corr > 0.3:
                    recommendations.append("✅ Good signal-outcome correlation")
            
            if not recommendations:
                recommendations = ["📊 Review detailed metrics for optimization opportunities"]
            
            signal_analysis['recommendations'] = recommendations
            
            summary_text = "Signal Quality Analysis:\n\n"
            if 'signal_accuracy' in signal_analysis:
                summary_text += f"📊 Avg Model Accuracy: {signal_analysis['signal_accuracy'].get('overall_avg_accuracy', 0):.3f}\n"
                summary_text += f"📈 Avg Signal R²: {signal_analysis['signal_accuracy'].get('overall_avg_r2', 0):.3f}\n\n"
            
            summary_text += "Recommendations:\n"
            for rec in recommendations:
                summary_text += f"{rec}\n"
            
            axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_dir / 'signal_quality_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save signal analysis
            with open(output_dir / 'signal_quality_analysis.json', 'w') as f:
                json.dump(signal_analysis, f, indent=2, default=str)
            
            print(f"🎯 Signal quality analysis saved to {output_dir}")
            
        except Exception as e:
            print(f"❌ Error in signal quality analysis: {e}")
            import traceback
            traceback.print_exc()
