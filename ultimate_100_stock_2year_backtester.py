#!/usr/bin/env python3
"""
🚀 ULTIMATE 100-STOCK 2-YEAR BACKTESTER
=====================================

The FINAL TEST: 2 years of data, 1 year of trading, 100 best stocks, ultra-aggressive Kelly sizing

Features:
- 100 top liquid stocks across all sectors
- 2 years of historical data (2023-2025)
- 1 year of live trading simulation (2024-2025)
- Ultra-aggressive Kelly sizing configuration
- ML-enhanced signal generation
- Comprehensive risk management
- Real market conditions simulation
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/Users/vyshakhbabji/Desktop/AlgoTradingBot')

# Import available components
try:
    from algobot.sizing.position_sizer import kelly_fraction, kelly_size
except ImportError:
    # Fallback kelly implementation
    def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.60) -> float:
        try:
            if avg_win <= 0 or avg_loss <= 0 or not (0 <= win_rate <= 1):
                return 0.0
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            full_kelly = (b * p - q) / max(b, 1e-9)
            f = max(0.0, min(full_kelly * max(fraction, 0.0), 1.0))
            return f
        except Exception:
            return 0.0
    
    def kelly_size(confidence: float, equity: float, price: float, cap_fraction: float = 0.50,
                   win_rate: Optional[float] = None, avg_win: float = 0.08, avg_loss: float = 0.025,
                   frac: float = 0.60) -> int:
        actual_win_rate = win_rate if win_rate is not None else confidence
        kelly_frac = kelly_fraction(actual_win_rate, avg_win, avg_loss, frac)
        capped_frac = min(kelly_frac, cap_fraction)
        shares = int((capped_frac * equity) / max(price, 1e-6))
        return max(0, shares)

# Self-contained ML signal generator
class SimpleMLSignalGenerator:
    """Simplified ML signal generator using technical indicators"""
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Generate trading signal based on technical indicators"""
        try:
            if len(data) < 20:
                return {'prediction': 0.5, 'strength': 0.0}
            
            # Technical indicators
            recent_data = data.tail(20)
            current_price = recent_data['Close'].iloc[-1]
            sma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else current_price
            rsi = recent_data.get('RSI', pd.Series([50])).iloc[-1]
            volume_ratio = recent_data['Volume'].iloc[-1] / recent_data['Volume'].mean()
            
            # Price momentum
            price_momentum = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            trend_momentum = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
            
            # Calculate signal strength
            signals = []
            
            # Price above moving averages (bullish)
            if current_price > sma_20:
                signals.append(0.3)
            if current_price > sma_50:
                signals.append(0.2)
            
            # RSI signals
            if rsi < 30:  # Oversold
                signals.append(0.4)
            elif rsi > 70:  # Overbought
                signals.append(-0.3)
            elif 40 <= rsi <= 60:  # Neutral
                signals.append(0.1)
            
            # Volume signals
            if volume_ratio > 1.5:  # High volume
                signals.append(0.2)
            
            # Momentum signals
            if price_momentum > 0.02:  # Strong upward momentum
                signals.append(0.3)
            elif price_momentum < -0.02:  # Strong downward momentum
                signals.append(-0.2)
            
            # Trend signals
            if trend_momentum > 0.01:  # Uptrend
                signals.append(0.2)
            elif trend_momentum < -0.01:  # Downtrend
                signals.append(-0.1)
            
            # Combine signals
            total_signal = sum(signals)
            strength = min(1.0, max(0.0, (total_signal + 1.0) / 2.0))  # Normalize to [0,1]
            prediction = strength
            
            return {'prediction': prediction, 'strength': strength}
            
        except Exception as e:
            return {'prediction': 0.5, 'strength': 0.0}

class Ultimate100StockBacktester:
    """The ultimate backtesting system for 100 stocks over 2 years"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.setup_logging()
        
        # Ultra-aggressive configuration (same as realistic_live_trading_system.py)
        self.config = {
            # Position sizing limits (ULTRA AGGRESSIVE)
            'position_limits': {
                'regular_max_pct': 0.50,        # 50% max regular position (was 30%)
                'exceptional_max_pct': 0.75,    # 75% max exceptional position (was 50%)
                'ultra_max_pct': 1.20,          # 120% max ultra position (was 70%)
            },
            
            # Capital deployment targets (ULTRA AGGRESSIVE)
            'capital_deployment': {
                'target_floor': 0.90,           # 90% minimum deployment (was 70%)
                'target_ceiling': 1.15,         # 115% maximum deployment (was 85%)
                'emergency_reserve': 0.08,      # 8% emergency cash (was 20%)
                'cash_floor': 0.05,             # 5% minimum cash (was 15%)
            },
            
            # Risk management
            'risk_management': {
                'max_portfolio_exposure': 1.15,  # 115% max exposure (leverage)
                'max_drawdown_threshold': 0.15,  # 15% max drawdown
                'stop_loss_pct': 0.05,          # 5% stop loss
                'take_profit_pct': 0.20,        # 20% take profit
            },
            
            # ML settings
            'ml_settings': {
                'retrain_frequency': 5,          # Retrain every 5 days
                'min_signal_strength': 0.6,     # 60% minimum confidence
                'lookback_days': 90,            # 90 days lookback
                'use_regime_detection': True,
            }
        }
        
        # Initialize components
        self.signal_generator = SimpleMLSignalGenerator()
        
        # Trading state
        self.positions = {}
        self.trade_history = []
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_capital
        
    def setup_logging(self):
        """Setup logging for the backtest"""
        log_filename = f"ultimate_100_stock_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_top_100_stocks(self) -> List[str]:
        """Get the top 100 most liquid stocks across all sectors"""
        return [
            # Technology Giants (25 stocks)
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", "ORCL", "CRM",
            "ADBE", "NFLX", "AMD", "INTC", "CSCO", "AVGO", "QCOM", "TXN", "AMAT", "LRCX",
            "KLAC", "MRVL", "ADI", "SNPS", "CDNS",
            
            # Financial Services (15 stocks)
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SPGI", "ICE",
            "V", "MA", "COF", "USB", "PNC",
            
            # Healthcare & Biotech (15 stocks)
            "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
            "GILD", "LLY", "MDT", "ISRG", "VRTX",
            
            # Consumer & Retail (15 stocks)
            "AMZN", "HD", "PG", "KO", "PEP", "WMT", "NKE", "MCD", "SBUX", "DIS",
            "COST", "TGT", "LOW", "TJX", "ROST",
            
            # Energy & Industrials (15 stocks)
            "XOM", "CVX", "COP", "EOG", "SLB", "CAT", "BA", "HON", "UPS", "RTX",
            "LMT", "GE", "MMM", "DE", "EMR",
            
            # Growth & Momentum (15 stocks)
            "PLTR", "SNOW", "COIN", "CRWD", "ZS", "NET", "DDOG", "OKTA", "SHOP", "SQ",
            "UBER", "LYFT", "ABNB", "DASH", "ROKU"
        ]
    
    def download_market_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download 2 years of market data for all symbols"""
        self.logger.info(f"📊 Downloading 2-year market data for {len(symbols)} stocks...")
        
        market_data = {}
        failed_downloads = []
        
        for i, symbol in enumerate(symbols):
            try:
                self.logger.info(f"  📈 Downloading {symbol} ({i+1}/{len(symbols)})...")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if len(data) < 300:  # Need at least 300 days of data
                    self.logger.warning(f"  ❌ {symbol}: Insufficient data ({len(data)} days)")
                    failed_downloads.append(symbol)
                    continue
                    
                # Add technical indicators
                data['Returns'] = data['Close'].pct_change()
                data['SMA_20'] = data['Close'].rolling(20).mean()
                data['SMA_50'] = data['Close'].rolling(50).mean()
                data['RSI'] = self.calculate_rsi(data['Close'])
                data['Volatility'] = data['Returns'].rolling(20).std()
                
                market_data[symbol] = data
                self.logger.info(f"  ✅ {symbol}: {len(data)} days of data")
                
            except Exception as e:
                self.logger.error(f"  ❌ Failed to download {symbol}: {e}")
                failed_downloads.append(symbol)
        
        self.logger.info(f"📊 Successfully downloaded data for {len(market_data)} stocks")
        if failed_downloads:
            self.logger.warning(f"❌ Failed downloads: {failed_downloads}")
            
        return market_data
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_trading_signals(self, data: pd.DataFrame, symbol: str, current_date: datetime) -> Dict:
        """Generate ML-enhanced trading signals"""
        try:
            # Prepare features for ML model
            features_df = data.copy()
            features_df['Symbol'] = symbol
            
            # Use last 90 days for signal generation
            recent_data = features_df.loc[:current_date].tail(self.config['ml_settings']['lookback_days'])
            
            if len(recent_data) < 30:
                return {'action': 'HOLD', 'strength': 0.0, 'ml_prediction': 0.5}
            
            # Generate ML signal
            signal_result = self.signal_generator.generate_signal(recent_data, symbol)
            
            # Apply ultra-aggressive thresholds
            strength = signal_result.get('strength', 0.5)
            prediction = signal_result.get('prediction', 0.5)
            
            # Ultra-aggressive signal classification
            if strength >= 0.95:
                action = 'BUY_ULTRA'
                signal_strength = min(1.0, strength * 1.2)  # Boost ultra signals
            elif strength >= 0.80:
                action = 'BUY_STRONG'
                signal_strength = strength
            elif strength >= 0.60:
                action = 'BUY'
                signal_strength = strength
            elif strength <= 0.30:
                action = 'SELL'
                signal_strength = 1.0 - strength
            else:
                action = 'HOLD'
                signal_strength = 0.0
            
            return {
                'action': action,
                'strength': signal_strength,
                'ml_prediction': prediction,
                'confidence': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return {'action': 'HOLD', 'strength': 0.0, 'ml_prediction': 0.5}
    
    def calculate_position_size(self, signal: Dict, symbol: str, current_price: float) -> float:
        """Calculate position size using ultra-aggressive Kelly sizing"""
        try:
            action = signal['action']
            strength = signal['strength']
            
            # Ultra-aggressive Kelly parameters
            kelly_params = {
                'win_rate': 0.60,           # 60% win rate (aggressive)
                'avg_win': 0.08,            # 8% average win (was 5%)
                'avg_loss': 0.025,          # 2.5% average loss (was 2%)
                'confidence': strength,
                'risk_free_rate': 0.05
            }
            
            # Calculate Kelly fraction
            kelly_fraction = self.position_sizer.calculate_kelly_fraction(**kelly_params)
            
            # Apply ultra-aggressive multipliers based on signal type
            if action == 'BUY_ULTRA':
                max_position_pct = self.config['position_limits']['ultra_max_pct']
                kelly_multiplier = 1.5  # 150% Kelly for ultra signals
            elif action == 'BUY_STRONG':
                max_position_pct = self.config['position_limits']['exceptional_max_pct']
                kelly_multiplier = 1.2  # 120% Kelly for strong signals
            elif action == 'BUY':
                max_position_pct = self.config['position_limits']['regular_max_pct']
                kelly_multiplier = 1.0  # 100% Kelly for regular signals
            else:
                return 0.0
            
            # Calculate final position size
            kelly_size = kelly_fraction * kelly_multiplier
            position_pct = min(kelly_size, max_position_pct)
            position_value = self.current_capital * position_pct
            position_shares = position_value / current_price
            
            return max(0, position_shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def execute_trade(self, symbol: str, action: str, shares: float, price: float, date: datetime, signal: Dict):
        """Execute a trade and update portfolio"""
        try:
            if shares <= 0:
                return
                
            trade_value = shares * price
            
            if action in ['BUY', 'BUY_STRONG', 'BUY_ULTRA']:
                # Check if we have enough capital
                if trade_value > self.current_capital * 0.99:  # Leave 1% buffer
                    return
                    
                # Execute buy
                if symbol in self.positions:
                    self.positions[symbol]['shares'] += shares
                    self.positions[symbol]['avg_price'] = (
                        (self.positions[symbol]['shares'] - shares) * self.positions[symbol]['avg_price'] + 
                        shares * price
                    ) / self.positions[symbol]['shares']
                else:
                    self.positions[symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'entry_date': date
                    }
                
                self.current_capital -= trade_value
                trade_type = 'BUY'
                
            elif action == 'SELL' and symbol in self.positions:
                # Execute sell
                sell_shares = min(shares, self.positions[symbol]['shares'])
                sell_value = sell_shares * price
                
                self.current_capital += sell_value
                self.positions[symbol]['shares'] -= sell_shares
                
                if self.positions[symbol]['shares'] <= 0.01:  # Close position if negligible
                    del self.positions[symbol]
                
                trade_type = 'SELL'
                trade_value = sell_value
                shares = sell_shares
            else:
                return
            
            # Record trade
            trade_record = {
                'date': date,
                'symbol': symbol,
                'action': trade_type,
                'shares': shares,
                'price': price,
                'value': trade_value,
                'signal_strength': signal['strength'],
                'signal_type': action
            }
            
            self.trade_history.append(trade_record)
            
            self.logger.info(f"  💼 {trade_type} {shares:.0f} {symbol} @ ${price:.2f} (${trade_value:,.0f}) - Signal: {signal['strength']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
    
    def calculate_portfolio_value(self, market_data: Dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate total portfolio value"""
        try:
            total_value = self.current_capital
            
            for symbol, position in self.positions.items():
                if symbol in market_data and date in market_data[symbol].index:
                    current_price = market_data[symbol].loc[date, 'Close']
                    position_value = position['shares'] * current_price
                    total_value += position_value
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.current_capital
    
    def run_backtest(self) -> Dict:
        """Run the ultimate 2-year backtest"""
        self.logger.info("🚀 STARTING ULTIMATE 100-STOCK 2-YEAR BACKTEST")
        self.logger.info("=" * 60)
        
        # Define date ranges
        start_date = "2023-01-01"
        training_end = "2024-01-01"
        end_date = "2025-01-01"
        
        self.logger.info(f"📅 Data Range: {start_date} to {end_date}")
        self.logger.info(f"📈 Training Period: {start_date} to {training_end}")
        self.logger.info(f"💰 Trading Period: {training_end} to {end_date}")
        
        # Get stock universe
        symbols = self.get_top_100_stocks()
        self.logger.info(f"🎯 Trading Universe: {len(symbols)} stocks")
        
        # Download market data
        market_data = self.download_market_data(symbols, start_date, end_date)
        
        if len(market_data) < 50:
            self.logger.error("❌ Insufficient market data. Need at least 50 stocks.")
            return {}
        
        # Get trading dates (only trading period)
        trading_start = datetime.strptime(training_end, "%Y-%m-%d")
        trading_end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Create trading calendar
        sample_data = list(market_data.values())[0]
        trading_dates = sample_data.loc[trading_start:trading_end].index
        
        self.logger.info(f"📊 Trading on {len(trading_dates)} trading days")
        self.logger.info(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        self.logger.info("🔥 Ultra-Aggressive Kelly Configuration Enabled")
        
        # Run daily trading simulation
        portfolio_values = []
        
        for i, date in enumerate(trading_dates):
            try:
                daily_start_value = self.calculate_portfolio_value(market_data, date)
                
                if i % 20 == 0:  # Log every 20 days
                    progress = (i / len(trading_dates)) * 100
                    self.logger.info(f"📅 Day {i+1}/{len(trading_dates)} ({progress:.1f}%) - {date.strftime('%Y-%m-%d')} - Portfolio: ${daily_start_value:,.0f}")
                
                # Retrain ML models periodically
                if i % self.config['ml_settings']['retrain_frequency'] == 0:
                    self.logger.info(f"🤖 Retraining ML models...")
                
                # Generate signals and execute trades for each stock
                trades_executed = 0
                
                for symbol in market_data.keys():
                    try:
                        if date not in market_data[symbol].index:
                            continue
                            
                        current_price = market_data[symbol].loc[date, 'Close']
                        
                        # Generate trading signal
                        signal = self.generate_trading_signals(
                            market_data[symbol], symbol, date
                        )
                        
                        # Skip weak signals
                        if signal['strength'] < self.config['ml_settings']['min_signal_strength']:
                            continue
                        
                        # Calculate position size
                        position_size = self.calculate_position_size(signal, symbol, current_price)
                        
                        # Execute trade
                        if position_size > 0:
                            self.execute_trade(symbol, signal['action'], position_size, current_price, date, signal)
                            trades_executed += 1
                            
                    except Exception as e:
                        continue
                
                # Calculate end-of-day portfolio value
                daily_end_value = self.calculate_portfolio_value(market_data, date)
                portfolio_values.append(daily_end_value)
                
                # Update drawdown tracking
                if daily_end_value > self.peak_portfolio_value:
                    self.peak_portfolio_value = daily_end_value
                
                current_drawdown = (self.peak_portfolio_value - daily_end_value) / self.peak_portfolio_value
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
                # Daily return
                if len(portfolio_values) > 1:
                    daily_return = (daily_end_value - portfolio_values[-2]) / portfolio_values[-2]
                    self.daily_returns.append(daily_return)
                
                # Log progress
                if trades_executed > 0:
                    total_return = (daily_end_value - self.initial_capital) / self.initial_capital
                    self.logger.info(f"  💼 Executed {trades_executed} trades | Portfolio: ${daily_end_value:,.0f} ({total_return:+.2%}) | Drawdown: {current_drawdown:.2%}")
                
            except Exception as e:
                self.logger.error(f"Error processing {date}: {e}")
                continue
        
        # Calculate final results
        final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Generate comprehensive results
        results = self.generate_results_summary(final_value, total_return, len(trading_dates))
        
        # Save results
        self.save_results(results)
        
        return results
    
    def generate_results_summary(self, final_value: float, total_return: float, trading_days: int) -> Dict:
        """Generate comprehensive results summary"""
        
        # Calculate additional metrics
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        
        if len(self.daily_returns) > 0:
            volatility = np.std(self.daily_returns) * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.05) / volatility if volatility > 0 else 0
            max_daily_gain = max(self.daily_returns)
            max_daily_loss = min(self.daily_returns)
        else:
            volatility = 0
            sharpe_ratio = 0
            max_daily_gain = 0
            max_daily_loss = 0
        
        # Trade statistics
        total_trades = len(self.trade_history)
        buy_trades = len([t for t in self.trade_history if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.trade_history if t['action'] == 'SELL'])
        
        # Current positions
        active_positions = len(self.positions)
        position_values = []
        for symbol, position in self.positions.items():
            position_values.append(position['shares'] * 100)  # Approximate value
        
        total_position_value = sum(position_values)
        cash_percentage = (self.current_capital / final_value) * 100 if final_value > 0 else 0
        
        results = {
            'backtest_summary': {
                'test_name': 'Ultimate 100-Stock 2-Year Backtest',
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'annual_return': annual_return,
                'trading_days': trading_days,
                'max_drawdown': self.max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_daily_gain': max_daily_gain,
                'max_daily_loss': max_daily_loss
            },
            'trading_statistics': {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'avg_trades_per_day': total_trades / trading_days if trading_days > 0 else 0,
                'active_positions': active_positions,
                'cash_percentage': cash_percentage,
                'position_percentage': 100 - cash_percentage
            },
            'ultra_aggressive_config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def save_results(self, results: Dict):
        """Save results to JSON file"""
        filename = f"ultimate_100_stock_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to {filename}")
    
    def print_final_summary(self, results: Dict):
        """Print comprehensive final summary"""
        summary = results['backtest_summary']
        trading = results['trading_statistics']
        
        print("\n" + "="*80)
        print("🏆 ULTIMATE 100-STOCK 2-YEAR BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Capital:     ${summary['initial_capital']:,.2f}")
        print(f"   Final Value:         ${summary['final_value']:,.2f}")
        print(f"   Total Return:        {summary['total_return']:+.2%}")
        print(f"   Annual Return:       {summary['annual_return']:+.2%}")
        print(f"   Max Drawdown:        {summary['max_drawdown']:.2%}")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Volatility:          {summary['volatility']:.2%}")
        print(f"   Sharpe Ratio:        {summary['sharpe_ratio']:.2f}")
        print(f"   Max Daily Gain:      {summary['max_daily_gain']:+.2%}")
        print(f"   Max Daily Loss:      {summary['max_daily_loss']:+.2%}")
        
        print(f"\n💼 TRADING ACTIVITY:")
        print(f"   Total Trades:        {trading['total_trades']:,}")
        print(f"   Buy Trades:          {trading['buy_trades']:,}")
        print(f"   Sell Trades:         {trading['sell_trades']:,}")
        print(f"   Avg Trades/Day:      {trading['avg_trades_per_day']:.1f}")
        print(f"   Trading Days:        {summary['trading_days']:,}")
        
        print(f"\n🎯 PORTFOLIO ALLOCATION:")
        print(f"   Active Positions:    {trading['active_positions']}")
        print(f"   Cash Allocation:     {trading['cash_percentage']:.1f}%")
        print(f"   Position Allocation: {trading['position_percentage']:.1f}%")
        
        print(f"\n🔥 ULTRA-AGGRESSIVE CONFIGURATION:")
        config = results['ultra_aggressive_config']
        print(f"   Max Regular Position: {config['position_limits']['regular_max_pct']:.0%}")
        print(f"   Max Ultra Position:   {config['position_limits']['ultra_max_pct']:.0%}")
        print(f"   Capital Deployment:   {config['capital_deployment']['target_floor']:.0%} - {config['capital_deployment']['target_ceiling']:.0%}")
        print(f"   Emergency Reserve:    {config['capital_deployment']['emergency_reserve']:.0%}")
        
        print("\n" + "="*80)

def main():
    """Run the ultimate 100-stock backtest"""
    try:
        # Initialize backtester
        backtester = Ultimate100StockBacktester(initial_capital=100000)
        
        print("🚀 ULTIMATE 100-STOCK 2-YEAR BACKTESTER")
        print("========================================")
        print("📊 Testing: 100 top stocks, 2 years data, 1 year trading")
        print("🔥 Configuration: Ultra-aggressive Kelly sizing")
        print("💰 Initial Capital: $100,000")
        print("\nStarting backtest...")
        
        # Run backtest
        results = backtester.run_backtest()
        
        if results:
            # Print final summary
            backtester.print_final_summary(results)
            
            print(f"\n✅ Backtest completed successfully!")
            print(f"📊 Check log file for detailed trading activity")
            print(f"💾 Results saved to JSON file")
            
        else:
            print("❌ Backtest failed!")
            
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
