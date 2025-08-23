"""
Custom Analyzers for Backtrader ML Trading System
"""

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Dict, Any
import math


class SharpeAnalyzer(bt.Analyzer):
    """
    Calculate Sharpe Ratio with √252 annualization
    """
    
    params = (
        ('risk_free_rate', 0.02),  # 2% risk-free rate
        ('annualize', True),
    )
    
    def __init__(self):
        self.returns = []
        self.start_value = None
        
    def next(self):
        if self.start_value is None:
            self.start_value = self.strategy.broker.getvalue()
        
        current_value = self.strategy.broker.getvalue()
        if len(self.returns) == 0:
            daily_return = 0.0
        else:
            prev_value = self.start_value if len(self.returns) == 0 else self.prev_value
            daily_return = (current_value / prev_value) - 1
        
        self.returns.append(daily_return)
        self.prev_value = current_value
    
    def get_analysis(self):
        if len(self.returns) < 2:
            return {'sharpe_ratio': 0.0}
        
        returns_array = np.array(self.returns[1:])  # Skip first zero return
        
        if self.params.annualize:
            # Annualized Sharpe with √252
            excess_returns = returns_array - (self.params.risk_free_rate / 252)
            sharpe = np.mean(excess_returns) * math.sqrt(252) / (np.std(returns_array) + 1e-8)
        else:
            # Simple Sharpe
            excess_returns = returns_array - (self.params.risk_free_rate / 252)
            sharpe = np.mean(excess_returns) / (np.std(returns_array) + 1e-8)
        
        return {
            'sharpe_ratio': sharpe,
            'mean_daily_return': np.mean(returns_array),
            'daily_volatility': np.std(returns_array),
            'annualized_return': np.mean(returns_array) * 252,
            'annualized_volatility': np.std(returns_array) * math.sqrt(252)
        }


class CustomMetricsAnalyzer(bt.Analyzer):
    """
    Calculate additional custom metrics like Calmar, Sortino, etc.
    """
    
    def __init__(self):
        self.values = []
        self.returns = []
        self.peak = 0
        self.max_drawdown = 0
        self.drawdowns = []
        
    def next(self):
        value = self.strategy.broker.getvalue()
        self.values.append(value)
        
        # Calculate returns
        if len(self.values) > 1:
            daily_return = (value / self.values[-2]) - 1
            self.returns.append(daily_return)
        
        # Track drawdowns
        if value > self.peak:
            self.peak = value
        
        current_drawdown = (self.peak - value) / self.peak if self.peak > 0 else 0
        self.drawdowns.append(current_drawdown)
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def get_analysis(self):
        if len(self.values) < 2:
            return {}
        
        returns_array = np.array(self.returns)
        final_value = self.values[-1]
        initial_value = self.values[0]
        
        # Total return
        total_return = (final_value / initial_value) - 1
        
        # Annualized return (CAGR)
        days = len(self.values)
        cagr = (final_value / initial_value) ** (252 / days) - 1
        
        # Calmar ratio
        calmar_ratio = cagr / self.max_drawdown if self.max_drawdown > 0 else 0
        
        # Sortino ratio
        negative_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(negative_returns) * math.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (np.mean(returns_array) * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # Average drawdown
        avg_drawdown = np.mean(self.drawdowns) if self.drawdowns else 0
        
        # Win rate (for completed trades - approximation)
        positive_days = len(returns_array[returns_array > 0])
        total_trading_days = len(returns_array)
        win_rate = positive_days / total_trading_days if total_trading_days > 0 else 0
        
        # Volatility
        volatility = np.std(returns_array) * math.sqrt(252)
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cagr': cagr,
            'cagr_pct': cagr * 100,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_pct': avg_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'trading_days': total_trading_days,
            'final_value': final_value,
            'initial_value': initial_value
        }


class TurnoverAnalyzer(bt.Analyzer):
    """
    Calculate portfolio turnover rate
    """
    
    def __init__(self):
        self.trades = 0
        self.total_volume = 0
        self.values = []
        
    def notify_order(self, order):
        if order.status == order.Completed:
            self.trades += 1
            self.total_volume += abs(order.executed.value)
    
    def next(self):
        self.values.append(self.strategy.broker.getvalue())
    
    def get_analysis(self):
        if len(self.values) < 2:
            return {'turnover': 0, 'trades': 0}
        
        avg_portfolio_value = np.mean(self.values)
        turnover = self.total_volume / (2 * avg_portfolio_value) if avg_portfolio_value > 0 else 0
        
        return {
            'turnover': turnover,
            'total_trades': self.trades,
            'total_volume': self.total_volume,
            'avg_portfolio_value': avg_portfolio_value
        }


class ExposureAnalyzer(bt.Analyzer):
    """
    Calculate portfolio exposure percentage
    """
    
    def __init__(self):
        self.daily_exposure = []
        
    def next(self):
        total_value = self.strategy.broker.getvalue()
        cash = self.strategy.broker.getcash()
        invested_value = total_value - cash
        
        exposure_pct = (invested_value / total_value * 100) if total_value > 0 else 0
        self.daily_exposure.append(exposure_pct)
    
    def get_analysis(self):
        if not self.daily_exposure:
            return {'avg_exposure_pct': 0}
        
        return {
            'avg_exposure_pct': np.mean(self.daily_exposure),
            'min_exposure_pct': np.min(self.daily_exposure),
            'max_exposure_pct': np.max(self.daily_exposure),
            'final_exposure_pct': self.daily_exposure[-1] if self.daily_exposure else 0
        }
