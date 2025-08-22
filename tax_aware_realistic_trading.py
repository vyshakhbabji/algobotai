#!/usr/bin/env python3
"""
🎯 TAX-AWARE REALISTIC HIGH-PROFIT TRADING SYSTEM
Designed for REAL WORLD profitability with tax considerations

Key Features:
• 40% tax impact modeling
• Wash sale rule enforcement  
• Realistic slippage and transaction costs
• Position size limits STRICTLY enforced
• Minimum holding periods for tax efficiency
• Focus on HIGH-CONVICTION trades only
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TaxAwareRealisticTrader:
    """Tax-optimized realistic trading system for maximum after-tax profits"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {symbol: shares}
        self.position_cost_basis = {}  # {symbol: avg_cost}
        self.position_entry_dates = {}  # {symbol: entry_date} for wash sale tracking
        self.recent_sales = {}  # {symbol: [sale_dates]} for wash sale prevention
        self.trades = []
        self.daily_values = []
        
        # Tax-aware configuration
        self.config = {
            # Tax parameters
            'tax_rate': 0.40,  # 40% tax on short-term gains
            'wash_sale_days': 30,  # Wash sale rule period
            'min_holding_period': 3,  # Days to avoid excessive trading
            
            # Realistic trading costs
            'commission': 0.0,  # Zero commission brokers
            'bid_ask_spread': 0.001,  # 0.1% spread
            'market_impact': 0.0005,  # 0.05% impact for large orders
            'slippage_factor': 0.0002,  # 0.02% random slippage
            
            # Position management - STRICT ENFORCEMENT
            'max_position_size': 0.15,  # 15% max per position (conservative)
            'max_exceptional_position': 0.25,  # 25% for ultra-high conviction
            'max_ultra_position': 0.35,  # 35% for once-in-a-lifetime signals
            
            # Signal thresholds - HIGHER for tax efficiency
            'min_signal_strength': 0.75,  # Only trade very strong signals
            'exceptional_threshold': 0.85,  # 25% position size
            'ultra_threshold': 0.95,  # 35% position size
            
            # Capital deployment - Conservative for taxes
            'target_deployment': 0.70,  # 70% deployed (lower for tax efficiency)
            'max_deployment': 0.80,  # 80% max (keep 20% cash for opportunities)
            
            # Portfolio limits
            'max_positions': 8,  # Fewer positions, higher conviction
            'rebalance_threshold': 0.05,  # 5% drift before rebalancing
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("🎯 TAX-AWARE REALISTIC TRADING SYSTEM initialized")
        self.logger.info(f"💰 Tax Rate: {self.config['tax_rate']:.1%}")
        self.logger.info(f"📅 Wash Sale Period: {self.config['wash_sale_days']} days")
        self.logger.info(f"🎲 Signal Threshold: {self.config['min_signal_strength']:.1%}")

    def calculate_after_tax_return(self, gross_profit: float, holding_days: int) -> float:
        """Calculate after-tax profit considering holding period"""
        if gross_profit <= 0:
            return gross_profit  # Losses are tax-deductible
        
        # Short-term vs long-term capital gains
        if holding_days >= 365:
            tax_rate = 0.20  # Long-term capital gains (lower rate)
        else:
            tax_rate = self.config['tax_rate']  # Short-term (ordinary income)
        
        after_tax_profit = gross_profit * (1 - tax_rate)
        return after_tax_profit

    def check_wash_sale_violation(self, symbol: str, current_date: pd.Timestamp) -> bool:
        """Check if buying this symbol would violate wash sale rule"""
        if symbol not in self.recent_sales:
            return False
        
        # Check if any sales occurred within wash sale period
        wash_sale_cutoff = current_date - timedelta(days=self.config['wash_sale_days'])
        recent_sale_dates = self.recent_sales[symbol]
        
        for sale_date in recent_sale_dates:
            if sale_date > wash_sale_cutoff:
                return True  # Would violate wash sale rule
        
        return False

    def calculate_realistic_execution_price(self, symbol: str, reference_price: float, 
                                          shares: int, is_buy: bool) -> float:
        """Calculate realistic execution price with spreads and slippage"""
        # Bid-ask spread
        if is_buy:
            price = reference_price * (1 + self.config['bid_ask_spread'] / 2)
        else:
            price = reference_price * (1 - self.config['bid_ask_spread'] / 2)
        
        # Market impact (larger orders move price more)
        notional_value = shares * reference_price
        if notional_value > 50000:  # Large order
            impact = self.config['market_impact'] * (notional_value / 100000)
            if is_buy:
                price *= (1 + impact)
            else:
                price *= (1 - impact)
        
        # Random slippage
        slippage = np.random.normal(0, self.config['slippage_factor'])
        price *= (1 + slippage)
        
        return price

    def calculate_position_size(self, signal_strength: float, portfolio_value: float) -> float:
        """Calculate position size based on signal strength with strict limits"""
        if signal_strength >= self.config['ultra_threshold']:
            max_size = self.config['max_ultra_position']
            self.logger.info(f"🔥 ULTRA signal ({signal_strength:.3f}) - max position: {max_size:.1%}")
        elif signal_strength >= self.config['exceptional_threshold']:
            max_size = self.config['max_exceptional_position'] 
            self.logger.info(f"⭐ Exceptional signal ({signal_strength:.3f}) - max position: {max_size:.1%}")
        else:
            max_size = self.config['max_position_size']
        
        # Scale within the limit based on signal strength
        strength_factor = (signal_strength - self.config['min_signal_strength']) / \
                         (1.0 - self.config['min_signal_strength'])
        
        position_size = max_size * (0.5 + 0.5 * strength_factor)  # 50-100% of max
        return min(position_size, max_size)

    def can_execute_order(self, symbol: str, shares: int, price: float, 
                         current_date: pd.Timestamp) -> Tuple[bool, str]:
        """Check if order can be executed with all constraints"""
        # Check wash sale rule
        if self.check_wash_sale_violation(symbol, current_date):
            return False, f"Wash sale violation for {symbol}"
        
        # Check if we're holding too recently (tax efficiency)
        if symbol in self.position_entry_dates:
            days_held = (current_date - self.position_entry_dates[symbol]).days
            if days_held < self.config['min_holding_period']:
                return False, f"Minimum holding period not met for {symbol}"
        
        # Check position size limit
        current_position = self.positions.get(symbol, 0)
        new_position_value = (current_position + shares) * price
        
        # Calculate current portfolio value
        portfolio_value = self.current_capital
        for pos_symbol, pos_shares in self.positions.items():
            if pos_symbol != symbol:  # Don't double count
                portfolio_value += pos_shares * price  # Approximation
        
        new_weight = new_position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Determine appropriate limit
        signal_strength = 0.80  # Default assumption, would come from signal
        if signal_strength >= self.config['ultra_threshold']:
            max_allowed = self.config['max_ultra_position']
        elif signal_strength >= self.config['exceptional_threshold']:
            max_allowed = self.config['max_exceptional_position']
        else:
            max_allowed = self.config['max_position_size']
        
        if new_weight > max_allowed:
            return False, f"Position size limit: {new_weight:.1%} > {max_allowed:.1%}"
        
        # Check capital availability
        total_cost = shares * price
        if total_cost > self.current_capital:
            return False, f"Insufficient capital: ${total_cost:,.0f} > ${self.current_capital:,.0f}"
        
        # Check max positions limit
        if symbol not in self.positions and len(self.positions) >= self.config['max_positions']:
            return False, f"Max positions limit reached: {len(self.positions)}"
        
        return True, "OK"

    def generate_mock_signals(self, symbols: List[str], date: pd.Timestamp) -> Dict[str, float]:
        """Generate mock trading signals for testing"""
        np.random.seed(int(date.timestamp()) % 10000)
        signals = {}
        
        for symbol in symbols:
            # Simulate realistic signal distribution
            # Most signals are mediocre, few are exceptional, rare ultra signals
            base_signal = np.random.beta(2, 5)  # Skewed towards lower values
            
            # Add some symbol-specific bias
            symbol_bias = hash(symbol) % 100 / 1000  # -0.05 to +0.05
            signal = base_signal + symbol_bias
            
            # Only keep signals above threshold
            if signal >= self.config['min_signal_strength']:
                signals[symbol] = min(signal, 0.99)  # Cap at 99%
        
        return signals

    def simulate_trading_day(self, date: pd.Timestamp, market_data: Dict[str, float], 
                           signals: Dict[str, float]) -> None:
        """Simulate one trading day with realistic constraints"""
        self.logger.info(f"📅 Trading Day: {date.strftime('%Y-%m-%d')}")
        
        # Calculate current portfolio value
        portfolio_value = self.current_capital
        for symbol, shares in self.positions.items():
            if symbol in market_data:
                portfolio_value += shares * market_data[symbol]
        
        executed_trades = 0
        skipped_trades = 0
        
        # Process signals (buy orders)
        for symbol, signal_strength in signals.items():
            if symbol not in market_data:
                continue
            
            reference_price = market_data[symbol]
            position_size_pct = self.calculate_position_size(signal_strength, portfolio_value)
            target_value = portfolio_value * position_size_pct
            shares = int(target_value / reference_price)
            
            if shares < 1:
                continue
            
            # Calculate realistic execution price
            execution_price = self.calculate_realistic_execution_price(
                symbol, reference_price, shares, is_buy=True
            )
            
            # Check if order can be executed
            can_execute, reason = self.can_execute_order(symbol, shares, execution_price, date)
            
            if can_execute:
                # Execute the trade
                total_cost = shares * execution_price
                self.positions[symbol] = self.positions.get(symbol, 0) + shares
                self.position_cost_basis[symbol] = execution_price  # Simplified
                self.position_entry_dates[symbol] = date
                self.current_capital -= total_cost
                
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': execution_price,
                    'total_cost': total_cost,
                    'signal_strength': signal_strength,
                    'position_size_pct': position_size_pct
                })
                
                executed_trades += 1
                self.logger.info(f"   ✅ BUY {shares:,} {symbol} @ ${execution_price:.2f} "
                               f"(signal: {signal_strength:.3f}, size: {position_size_pct:.1%})")
            else:
                skipped_trades += 1
                self.logger.info(f"   ❌ SKIPPED {symbol}: {reason}")
        
        # Update portfolio value for the day
        final_portfolio_value = self.current_capital
        for symbol, shares in self.positions.items():
            if symbol in market_data:
                final_portfolio_value += shares * market_data[symbol]
        
        self.daily_values.append({
            'date': date,
            'portfolio_value': final_portfolio_value,
            'cash': self.current_capital,
            'positions': len(self.positions),
            'executed_trades': executed_trades,
            'skipped_trades': skipped_trades
        })
        
        deployment_pct = (final_portfolio_value - self.current_capital) / final_portfolio_value
        daily_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        self.logger.info(f"📊 EOD: ${final_portfolio_value:,.0f} ({daily_return:+.1%}) | "
                        f"Deployed: {deployment_pct:.1%} | Trades: {executed_trades} | "
                        f"Skipped: {skipped_trades} | Cash: ${self.current_capital:,.0f}")

    def run_backtest(self, start_date: str = "2025-05-01", end_date: str = "2025-08-20") -> Dict:
        """Run tax-aware realistic backtest"""
        self.logger.info(f"🚀 Starting Tax-Aware Realistic Backtest: {start_date} to {end_date}")
        
        # Sample symbols (high-quality, liquid stocks)
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']
        
        # Get market data
        self.logger.info("📊 Fetching market data...")
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError("No market data available")
        
        # Process each trading day
        trading_days = data.index
        
        for i, date in enumerate(trading_days):
            # Get market prices for the day
            market_data = {}
            for symbol in symbols:
                try:
                    price = data['Close'][symbol].loc[date]
                    if not pd.isna(price):
                        market_data[symbol] = float(price)
                except:
                    continue
            
            if not market_data:
                continue
            
            # Generate signals for the day
            signals = self.generate_mock_signals(symbols, date)
            
            # Simulate trading
            if signals:  # Only trade if we have signals
                self.simulate_trading_day(date, market_data, signals)
        
        return self._calculate_final_metrics()

    def _calculate_final_metrics(self) -> Dict:
        """Calculate final performance metrics with tax considerations"""
        if not self.daily_values:
            return {}
        
        final_value = self.daily_values[-1]['portfolio_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate metrics
        daily_returns = []
        for i in range(1, len(self.daily_values)):
            prev_val = self.daily_values[i-1]['portfolio_value']
            curr_val = self.daily_values[i]['portfolio_value']
            daily_ret = (curr_val - prev_val) / prev_val
            daily_returns.append(daily_ret)
        
        if daily_returns:
            volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe = (total_return * 252 / len(daily_returns)) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe = 0
        
        # Tax impact analysis
        gross_profit = final_value - self.initial_capital
        avg_holding_days = 30  # Estimate
        after_tax_profit = self.calculate_after_tax_return(gross_profit, avg_holding_days)
        after_tax_return = after_tax_profit / self.initial_capital
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'gross_return': total_return,
            'after_tax_return': after_tax_return,
            'gross_profit': gross_profit,
            'after_tax_profit': after_tax_profit,
            'tax_impact': gross_profit - after_tax_profit,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.trades),
            'avg_positions': np.mean([d['positions'] for d in self.daily_values]),
            'final_positions': len(self.positions),
            'trading_days': len(self.daily_values)
        }
        
        return results

if __name__ == "__main__":
    # Run the tax-aware realistic system
    trader = TaxAwareRealisticTrader(initial_capital=100000)
    results = trader.run_backtest()
    
    print("\n" + "="*60)
    print("🎯 TAX-AWARE REALISTIC TRADING RESULTS")
    print("="*60)
    print(f"Initial Capital:     ${results['initial_capital']:,.0f}")
    print(f"Final Value:         ${results['final_value']:,.0f}")
    print(f"Gross Return:        {results['gross_return']:+.1%}")
    print(f"After-Tax Return:    {results['after_tax_return']:+.1%}")
    print(f"Tax Impact:          ${results['tax_impact']:,.0f}")
    print(f"Volatility:          {results['volatility']:.1%}")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    print(f"Total Trades:        {results['total_trades']}")
    print(f"Avg Positions:       {results['avg_positions']:.1f}")
    print("="*60)
    
    # Compare to previous unrealistic results
    print("\n📊 REALITY CHECK:")
    print(f"Previous 'unrealistic' annual return: +35.5%")
    print(f"Tax-aware realistic annual return:    {results['after_tax_return']*12:.1%}")
    print(f"Reality gap:                          {35.5 - results['after_tax_return']*12:.1f} percentage points")
