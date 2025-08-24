"""
Backtrader ML Trading Strategy

Implements the exact same ML trading logic as realistic_live_trading_system.py
with proper Kelly sizing and position management.
"""

import backtrader as bt
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime
import yfinance as yf

from .signal_generator import MLSignalGenerator


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.60) -> float:
    """Compute fractional Kelly bet fraction of equity - same as realistic_live_trading_system"""
    try:
        if avg_win <= 0 or avg_loss <= 0 or not (0 <= win_rate <= 1):
            return 0.0
        # Full Kelly for win/loss formulation: f* = (bp - q)/b, where b = avg_win/avg_loss
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        full_kelly = (b * p - q) / max(b, 1e-9)
        # Apply fractional Kelly and clamp to [0, 1]
        f = max(0.0, min(full_kelly * max(fraction, 0.0), 1.0))
        return f
    except Exception:
        return 0.0


def kelly_size(confidence: float, equity: float, price: float, cap_fraction: float = 0.15,
               win_rate: float = None, avg_win: float = 0.08, avg_loss: float = 0.025,
               frac: float = 0.60) -> int:
    """Return integer shares using Kelly sizing - same as realistic_live_trading_system"""
    wr = confidence if win_rate is None else win_rate
    f = kelly_fraction(wr, avg_win, avg_loss, fraction=frac)
    # Cap the Kelly sizing by cap_fraction to avoid oversized bets
    f_capped = min(max(f, 0.0), max(cap_fraction, 0.0))
    value = equity * f_capped
    return int(value / price) if price > 0 else 0


class MLTradingStrategy(bt.Strategy):
    """
    ML Trading Strategy for Backtrader - Exact replica of realistic_live_trading_system.py
    """
    
    params = (
        ('signal_threshold', 0.25),
        ('max_positions', 15),
        ('max_position_size', 0.40),
        ('max_symbol_exposure', 0.15),
        ('max_net_exposure', 0.95),
        ('target_invested_floor', 0.75),
        ('target_invested_ceiling', 0.95),
        ('kelly_multiplier', 0.5),
        ('max_position_size_exceptional', 0.60),
        ('avg_win', 0.08),
        ('avg_loss', 0.025),
        ('min_training_days', 60),
        ('config', None),
    )

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML signal generator
        self.signal_generator = MLSignalGenerator(self.params.config or {})
        
        # Track performance
        self.trade_log = []
        self.daily_values = []
        self.ml_metrics = {}  # Store ML performance metrics
        self.current_signals = {}  # Store current signals for trade logging
        
        # Data storage for ML training
        self.symbol_data = {}  # {symbol: DataFrame}
        self.symbols = []
        
        # Get symbols from data feeds
        for data in self.datas:
            symbol = data._name
            self.symbols.append(symbol)
            self.symbol_data[symbol] = []
        
        self.logger.info(f"Strategy initialized with {len(self.symbols)} symbols")
        
        # Track rebalancing
        self.last_rebalance_date = None
        self.rebalance_frequency = 5  # Rebalance every 5 days
        
        # Performance tracking
        self.start_value = self.broker.getvalue()
        
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        self.logger.info(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Track order execution with enhanced logging"""
        if order.status in [order.Completed]:
            symbol = order.data._name
            current_date = self.datas[0].datetime.datetime(0)
            
            # Get signal information for this trade
            signal_info = self.current_signals.get(symbol, {})
            signal_strength = signal_info.get('strength', 0)
            ml_enhanced = signal_info.get('ml_enhanced', False)
            signal_features = signal_info.get('features', {})
            
            if order.isbuy():
                self.log(f'BUY EXECUTED: {symbol}, Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Cost: {order.executed.value:.2f}')
                
                # Log detailed trade info for analysis
                self.trade_log.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'value': order.executed.value,
                    'signal_strength': signal_strength,
                    'ml_enhanced': ml_enhanced,
                    'signal_features': signal_features,
                    'pnl': 0,  # Will be updated on exit
                    'fees': order.executed.comm or 0
                })
                
            elif order.issell():
                self.log(f'SELL EXECUTED: {symbol}, Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Value: {order.executed.value:.2f}')
                
                # Calculate P&L for completed trades
                position = self.getposition(order.data)
                if len(self.trade_log) > 0:
                    # Find corresponding buy order(s) and update P&L
                    for i in range(len(self.trade_log) - 1, -1, -1):
                        if (self.trade_log[i]['symbol'] == symbol and 
                            self.trade_log[i]['action'] == 'BUY' and 
                            self.trade_log[i]['pnl'] == 0):
                            
                            # Calculate P&L for this trade
                            buy_price = self.trade_log[i]['price']
                            sell_price = order.executed.price
                            trade_size = min(abs(order.executed.size), self.trade_log[i]['size'])
                            
                            pnl = (sell_price - buy_price) * trade_size
                            pnl -= (order.executed.comm or 0) + self.trade_log[i]['fees']
                            
                            self.trade_log[i]['pnl'] = pnl
                            self.trade_log[i]['exit_date'] = current_date
                            self.trade_log[i]['exit_price'] = sell_price
                            break
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Failed: {order.data._name}, Status: {order.getstatusname()}')

    def train_models(self, current_date):
        """Train ML models for all symbols"""
        try:
            trained_count = 0
            for symbol in self.symbols:
                data_df = self.get_data_for_symbol(symbol, current_date)
                if len(data_df) >= self.params.min_training_days:
                    if self.signal_generator.train_daily_models(symbol, data_df, current_date):
                        trained_count += 1
            
            if trained_count > 0:
                self.log(f'Trained ML models for {trained_count} symbols')
                
        except Exception as e:
            self.logger.error(f"Error training models: {e}")

    def get_data_for_symbol(self, symbol: str, current_date: datetime) -> pd.DataFrame:
        """Get historical data for a symbol up to current date"""
        try:
            # Find the data feed for this symbol
            data_feed = None
            for data in self.datas:
                if data._name == symbol:
                    data_feed = data
                    break
            
            if data_feed is None:
                return pd.DataFrame()
            
            # Extract data up to current date
            data_list = []
            for i in range(len(data_feed)):
                if i >= len(data_feed):
                    break
                    
                # Get OHLCV data
                try:
                    dt = data_feed.datetime.datetime(-i)
                    if dt > current_date:
                        continue
                        
                    data_point = {
                        'Date': dt,
                        'Open': data_feed.open[-i],
                        'High': data_feed.high[-i],
                        'Low': data_feed.low[-i],
                        'Close': data_feed.close[-i],
                        'Volume': getattr(data_feed, 'volume', [1])[-i] if hasattr(data_feed, 'volume') else 1
                    }
                    data_list.append(data_point)
                except IndexError:
                    break
            
            if not data_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, signal: Dict[str, Any], symbol: str) -> int:
        """
        Calculate position size using Kelly criterion - EXACT replica of realistic_live_trading_system
        """
        try:
            signal_strength = signal['strength']
            price = signal['price']
            
            # Check if signal meets threshold
            if signal_strength < self.params.signal_threshold:
                return 0
            
            # Get current portfolio value (trading capital)
            current_value = self.broker.getvalue()
            
            # Get actual Backtrader position for this symbol (CRITICAL FIX)
            data_feed = self.getdatabyname(symbol)
            if data_feed is None:
                return 0
                
            actual_position = self.getposition(data_feed).size
            current_position_value = abs(actual_position) * price
            position_weight = current_position_value / current_value if current_value > 0 else 0
            
            # Position limits based on signal strength (like realistic_live_trading_system)
            if signal_strength >= 0.85:  # Exceptional signals
                max_allowed = min(self.params.max_position_size_exceptional, self.params.max_symbol_exposure)
            else:
                max_allowed = min(self.params.max_position_size, self.params.max_symbol_exposure)
            
            # If we already have a large position, don't add more
            if position_weight >= max_allowed:
                return 0
            
            # CRITICAL: Don't allow new positions if we have any short positions
            if actual_position < 0:
                return 0
            
            # Kelly sizing using the exact same logic as realistic_live_trading_system
            shares = kelly_size(
                confidence=signal_strength,
                equity=current_value,
                price=price,
                cap_fraction=max_allowed,
                avg_win=self.params.avg_win,
                avg_loss=self.params.avg_loss,
                frac=self.params.kelly_multiplier
            )
            
            # Count actual positions with Backtrader
            active_positions = sum(1 for data in self.datas if abs(self.getposition(data).size) > 0)
            
            # If this is a new position, check max positions limit
            if actual_position == 0 and active_positions >= self.params.max_positions:
                return 0
            
            # If we already have a position, only allow adding up to the limit
            if actual_position > 0:
                # Calculate how much more we can add
                max_total_value = current_value * max_allowed
                current_value_in_symbol = current_position_value
                available_value = max_total_value - current_value_in_symbol
                
                if available_value <= 0:
                    return 0
                    
                max_additional_shares = int(available_value / price)
                shares = min(shares, max_additional_shares)
            
            return max(0, shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0

    def should_exit_position(self, symbol: str, signal: Dict[str, Any]) -> tuple:
        """
        Determine if we should exit a position - EXACT replica of realistic_live_trading_system
        Returns (should_exit, exit_size)
        """
        # Get actual Backtrader position for this symbol (CRITICAL FIX)
        data_feed = self.getdatabyname(symbol)
        if data_feed is None:
            return False, 0
            
        actual_position = self.getposition(data_feed).size
        
        if actual_position <= 0:
            return False, 0
        
        current_price = signal['price']
        entry_price = self.getposition(data_feed).price
        
        # Exit on strong sell signal
        if signal['signal'] == 'SELL':
            signal_strength = signal['strength']
            
            # Calculate position return
            position_return = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Adjust thresholds based on profit/loss
            sell_threshold_adjustment = 0.0
            if position_return > 0.15:  # If we have >15% profit
                sell_threshold_adjustment = -0.10  # Make it easier to sell
            else:
                sell_threshold_adjustment = 0.05   # Make it harder to sell
            
            # Different exit strategies based on signal strength
            if signal_strength > (0.80 + sell_threshold_adjustment):
                # Full exit on very strong sell signal
                return True, actual_position
            elif signal_strength > (0.70 + sell_threshold_adjustment):
                # Partial exit on strong sell signal
                partial_size = max(1, int(actual_position * 0.5))
                return True, partial_size
        
        # Risk-based exits (like realistic_live_trading_system)
        if entry_price > 0:
            gain_pct = (current_price / entry_price) - 1
            
            # Take profits on strong gains (15%+)
            if gain_pct > 0.15:
                # Take partial profits
                profit_taking_size = max(1, int(actual_position * 0.3))
                return True, profit_taking_size
            
            # Stop loss on significant losses (8%+)
            if gain_pct < -0.08:
                # Exit full position on stop loss
                return True, actual_position
        
        return False, 0

    def next(self):
        """Main strategy logic - called on each bar - EXACT replica of realistic_live_trading_system"""
        try:
            current_date = self.datas[0].datetime.datetime(0)
            
            # Skip if not enough data
            if len(self.datas[0]) < 60:
                return
            
            # ONLY START TRADING ON AUG 22, 2024 (after 2 years of training)
            trading_start_date = datetime(2024, 8, 22)
            if current_date < trading_start_date:
                # During training period - train models weekly to speed up backtest
                if len(self.datas[0]) % 25 == 0:  # Train every 25 days (~monthly)
                    self.train_models(current_date)
                return
            
            # During trading period - train ML models more frequently (every 5 days)
            if len(self.datas[0]) % 5 == 0:
                self.train_models(current_date)
            
            # Generate signals for all symbols
            signals = {}
            for symbol in self.symbols:
                data_df = self.get_data_for_symbol(symbol, current_date)
                if len(data_df) >= 50:  # Minimum data required
                    signal = self.signal_generator.generate_signal(symbol, data_df, current_date)
                    signals[symbol] = signal
                    # Store signals for trade logging
                    self.current_signals[symbol] = signal
            
            # 1. Process exit signals first (CRITICAL - prevents position accumulation)
            for symbol, signal in signals.items():
                should_exit, exit_size = self.should_exit_position(symbol, signal)
                if should_exit and exit_size > 0:
                    data_feed = self.getdatabyname(symbol)
                    if data_feed:
                        self.sell(data=data_feed, size=exit_size)
                        self.log(f'EXIT SIGNAL: {symbol}, Strength: {signal["strength"]:.3f}')
            
            # 2. Process buy signals (enter new positions or add to existing)
            for symbol, signal in signals.items():
                if signal['signal'] == 'BUY':
                    position_size = self.calculate_position_size(signal, symbol)
                    
                    if position_size > 0:
                        data_feed = self.getdatabyname(symbol)
                        if data_feed:
                            self.buy(data=data_feed, size=position_size)
                            self.log(f'BUY SIGNAL: {symbol}, Strength: {signal["strength"]:.3f}, '
                                   f'Size: {position_size}, ML Enhanced: {signal.get("ml_enhanced", False)}')
            
            # 3. Portfolio monitoring (like realistic_live_trading_system)
            current_value = self.broker.getvalue()
            
            # Count active positions using Backtrader's position tracking
            active_positions = sum(1 for data in self.datas if abs(self.getposition(data).size) > 0)
            
            self.daily_values.append({
                'date': current_date,
                'portfolio_value': current_value,
                'positions': active_positions,
                'return_pct': ((current_value - self.start_value) / self.start_value) * 100
            })
            
            # Log portfolio status every 5 days
            if len(self.datas[0]) % 5 == 0:
                total_return = ((current_value - self.start_value) / self.start_value) * 100
                self.log(f'Portfolio Value: ${current_value:,.0f}, Return: {total_return:+.1f}%, Positions: {active_positions}')
            
        except Exception as e:
            self.logger.error(f"Error in next() method: {e}")
            
    def stop(self):
        """Called when backtest ends"""
        try:
            final_value = self.broker.getvalue()
            total_return = ((final_value - self.start_value) / self.start_value) * 100
            total_trades = len(self.trade_log)
            
            self.log('Strategy Completed!')
            self.log(f'Starting Value: ${self.start_value:,.0f}')
            self.log(f'Final Value: ${final_value:,.0f}')
            self.log(f'Total Return: {total_return:+.2f}%')
            self.log(f'Total Trades: {total_trades}')
            
        except Exception as e:
            self.logger.error(f"Error in stop(): {e}")

    def get_trade_stats(self):
        """Return comprehensive trade statistics for analysis"""
        return {
            'trade_log': self.trade_log,
            'signal_history': getattr(self, 'signal_history', []),
            'ml_performance': self.ml_performance,
            'total_trades': len([t for t in self.trade_log if t.get('pnl', 0) != 0]),
            'winning_trades': len([t for t in self.trade_log if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in self.trade_log if t.get('pnl', 0) < 0]),
            'total_pnl': sum([t.get('pnl', 0) for t in self.trade_log]),
            'avg_win': np.mean([t['pnl'] for t in self.trade_log if t.get('pnl', 0) > 0]) if any(t.get('pnl', 0) > 0 for t in self.trade_log) else 0,
            'avg_loss': np.mean([t['pnl'] for t in self.trade_log if t.get('pnl', 0) < 0]) if any(t.get('pnl', 0) < 0 for t in self.trade_log) else 0
        }
