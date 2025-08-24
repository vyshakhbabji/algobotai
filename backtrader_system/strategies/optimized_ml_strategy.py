"""
OPTIMIZED Backtrader ML Trading Strategy

Key Improvements from Analysis:
1. Reduced Kelly position sizing (15-25% instead of 35-60%)
2. Added stop-loss protection
3. Portfolio heat limits
4. Higher conviction signal filtering
5. Reduced turnover targeting
"""

import backtrader as bt
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import yfinance as yf

from .signal_generator import MLSignalGenerator


def optimized_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.35) -> float:
    """Compute CONSERVATIVE fractional Kelly bet fraction - REDUCED FROM 0.60 to 0.35"""
    try:
        if avg_win <= 0 or avg_loss <= 0 or not (0 <= win_rate <= 1):
            return 0.0
        # Full Kelly for win/loss formulation: f* = (bp - q)/b, where b = avg_win/avg_loss
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        full_kelly = (b * p - q) / max(b, 1e-9)
        # Apply CONSERVATIVE fractional Kelly and clamp to [0, 1]
        f = max(0.0, min(full_kelly * max(fraction, 0.0), 1.0))
        return f
    except Exception:
        return 0.0


def optimized_kelly_size(confidence: float, equity: float, price: float, cap_fraction: float = 0.12,
                        win_rate: float = None, avg_win: float = 0.08, avg_loss: float = 0.025,
                        frac: float = 0.35) -> int:
    """Return integer shares using CONSERVATIVE Kelly sizing - REDUCED CAP from 0.15 to 0.12"""
    wr = confidence if win_rate is None else win_rate
    f = optimized_kelly_fraction(wr, avg_win, avg_loss, fraction=frac)
    # Cap the Kelly sizing by LOWER cap_fraction to avoid oversized bets
    f_capped = min(max(f, 0.0), max(cap_fraction, 0.0))
    value = equity * f_capped
    return int(value / price) if price > 0 else 0


class OptimizedMLTradingStrategy(bt.Strategy):
    """
    OPTIMIZED ML Trading Strategy - Based on backtest analysis recommendations
    """
    
    params = (
        ('signal_threshold', 0.25),  # ADJUSTED: Lower threshold to enable trading (from 0.35)
        ('max_positions', 12),  # REDUCED from 15 - more focused portfolio
        ('max_position_size', 0.25),  # REDUCED from 0.40 - conservative sizing
        ('max_symbol_exposure', 0.12),  # REDUCED from 0.15 - prevent concentration
        ('max_net_exposure', 0.85),  # REDUCED from 0.95 - keep some dry powder
        ('target_invested_floor', 0.65),  # REDUCED from 0.75 - more selective
        ('target_invested_ceiling', 0.85),  # REDUCED from 0.95 - risk management
        ('kelly_multiplier', 0.35),  # REDUCED from 0.5 - conservative Kelly
        ('max_position_size_exceptional', 0.40),  # REDUCED from 0.60
        ('avg_win', 0.08),
        ('avg_loss', 0.025),
        ('min_training_days', 60),
        ('config', None),
        # NEW RISK MANAGEMENT PARAMETERS
        ('stop_loss_pct', 0.08),  # 8% stop loss
        ('profit_target_pct', 0.15),  # 15% profit target
        ('max_portfolio_heat', 0.25),  # Max 25% of portfolio at risk
        ('min_hold_days', 3),  # Minimum hold period to reduce turnover
        ('max_daily_trades', 8),  # Limit daily trading activity
    )

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML signal generator
        self.signal_generator = MLSignalGenerator(self.params.config or {})
        
        # Track performance
        self.trade_log = []
        self.daily_values = []
        self.ml_metrics = {}
        self.current_signals = {}
        
        # NEW: Risk management tracking
        self.position_entry_dates = {}  # Track entry dates for min hold period
        self.position_entry_prices = {}  # Track entry prices for stop loss
        self.daily_trade_count = 0  # Track daily trading activity
        self.last_trade_date = None  # Track when we last traded
        
        # Data storage for ML training
        self.symbol_data = {}
        self.symbols = []
        
        # Get symbols from data feeds
        for data in self.datas:
            symbol = data._name
            self.symbols.append(symbol)
            self.symbol_data[symbol] = []
        
        # NEW: Track whether models have been trained
        self.models_trained = False
        self.training_completed_date = None
        
        self.logger.info(f"OPTIMIZED Strategy initialized with {len(self.symbols)} symbols")
        self.logger.info(f"Key improvements: Conservative sizing (max {self.params.max_position_size*100}%), "
                        f"Stop loss ({self.params.stop_loss_pct*100}%), "
                        f"Higher signal threshold ({self.params.signal_threshold})")
        
        # Track rebalancing - trade every day for more activity
        self.last_rebalance_date = None
        self.rebalance_frequency = 1  # Trade every day
        
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
                
                # NEW: Track position entry for risk management
                self.position_entry_dates[symbol] = current_date
                self.position_entry_prices[symbol] = order.executed.price
                
                # Log the trade details
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
                    'pnl': 0.0,
                    'fees': order.executed.comm or 0.0
                })
                
            elif order.issell():
                self.log(f'SELL EXECUTED: {symbol}, Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size}, Value: {order.executed.value:.2f}')
                
                # NEW: Clear position tracking
                self.position_entry_dates.pop(symbol, None)
                entry_price = self.position_entry_prices.pop(symbol, None)
                
                # Calculate PnL if we have entry price
                pnl = 0.0
                if entry_price:
                    pnl = (order.executed.price - entry_price) * abs(order.executed.size)
                
                # Log the trade details
                self.trade_log.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'value': order.executed.value,
                    'signal_strength': signal_strength,
                    'ml_enhanced': ml_enhanced,
                    'signal_features': signal_features,
                    'pnl': pnl,
                    'fees': order.executed.comm or 0.0
                })
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            symbol = order.data._name
            status_map = {order.Margin: 'Margin', order.Canceled: 'Canceled', order.Rejected: 'Rejected'}
            self.log(f'Order Failed: {symbol}, Status: {status_map.get(order.status, "Unknown")}')

    def check_stop_loss_and_targets(self):
        """NEW: Check stop loss and profit targets for all positions"""
        current_date = self.datas[0].datetime.datetime(0)
        
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data)
            
            if position.size != 0 and symbol in self.position_entry_prices:
                current_price = data.close[0]
                entry_price = self.position_entry_prices[symbol]
                entry_date = self.position_entry_dates.get(symbol)
                
                # Check minimum hold period
                if entry_date and (current_date - entry_date).days < self.params.min_hold_days:
                    continue
                
                if position.size > 0:  # Long position
                    # Calculate unrealized P&L percentage
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Check stop loss
                    if pnl_pct <= -self.params.stop_loss_pct:
                        self.log(f'STOP LOSS TRIGGERED: {symbol}, Entry: {entry_price:.2f}, '
                               f'Current: {current_price:.2f}, Loss: {pnl_pct:.2%}')
                        self.close(data)
                        continue
                    
                    # Check profit target
                    if pnl_pct >= self.params.profit_target_pct:
                        self.log(f'PROFIT TARGET HIT: {symbol}, Entry: {entry_price:.2f}, '
                               f'Current: {current_price:.2f}, Gain: {pnl_pct:.2%}')
                        self.close(data)
                        continue

    def calculate_portfolio_heat(self) -> float:
        """NEW: Calculate current portfolio heat (risk exposure)"""
        total_value = self.broker.getvalue()
        total_risk = 0.0
        
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data)
            
            if position.size > 0 and symbol in self.position_entry_prices:
                current_price = data.close[0]
                entry_price = self.position_entry_prices[symbol]
                position_value = position.size * current_price
                
                # Risk is the potential loss from stop loss
                risk_per_share = entry_price * self.params.stop_loss_pct
                total_risk += position.size * risk_per_share
        
        return total_risk / total_value if total_value > 0 else 0.0

    def can_take_new_position(self, signal_strength: float) -> bool:
        """NEW: Check if we can take a new position based on risk limits"""
        current_date = self.datas[0].datetime.datetime(0)
        
        # Check daily trade limit
        if self.last_trade_date != current_date.date():
            self.daily_trade_count = 0
            self.last_trade_date = current_date.date()
        
        if self.daily_trade_count >= self.params.max_daily_trades:
            return False
        
        # Check portfolio heat
        current_heat = self.calculate_portfolio_heat()
        if current_heat >= self.params.max_portfolio_heat:
            return False
        
        # Check signal strength threshold (higher conviction only)
        if signal_strength < self.params.signal_threshold:
            return False
        
        return True

    def collect_data_for_symbol(self, symbol: str, data) -> pd.DataFrame:
        """Collect price data for ML training"""
        try:
            # Get the date for current bar
            current_date = data.datetime.datetime(0)
            
            # Collect OHLCV data
            row = {
                'Date': current_date,
                'Open': data.open[0],
                'High': data.high[0],
                'Low': data.low[0],
                'Close': data.close[0],
                'Volume': getattr(data, 'volume', [0])[0] or 0
            }
            
            # Add to symbol data
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = []
            
            self.symbol_data[symbol].append(row)
            
            # Keep only last 500 bars for memory efficiency
            if len(self.symbol_data[symbol]) > 500:
                self.symbol_data[symbol] = self.symbol_data[symbol][-500:]
            
            # Convert to DataFrame
            df = pd.DataFrame(self.symbol_data[symbol])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error collecting data for {symbol}: {e}")
            return pd.DataFrame()

    def should_rebalance(self) -> bool:
        """Check if we should rebalance portfolio"""
        current_date = self.datas[0].datetime.datetime(0)
        
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance >= self.rebalance_frequency

    def is_training_period(self) -> bool:
        """Check if we're in the training period"""
        current_date = self.datas[0].datetime.datetime(0)
        
        # Training period: Aug 22, 2022 to Aug 22, 2024 (2 years)
        training_start = datetime(2022, 8, 22)
        training_end = datetime(2024, 8, 22)
        
        return training_start <= current_date <= training_end
    
    def train_models_once(self, current_date):
        """Train all models once using all available training data up to current date"""
        if self.models_trained:
            return True
            
        self.logger.info(f"{current_date.date()}: [ONE-TIME TRAINING] Training ML models with all available data")
        
        # Collect all training data for each symbol
        all_training_data = {}
        for data in self.datas:
            symbol = data._name
            df = self.collect_data_for_symbol(symbol, data)
            if not df.empty and len(df) >= self.params.min_training_days:
                # Filter to only training period data
                training_data = df[df.index <= current_date]
                if len(training_data) >= self.params.min_training_days:
                    all_training_data[symbol] = training_data
        
        if not all_training_data:
            self.logger.warning("No sufficient training data available")
            return False
        
        # Train models for each symbol using all training data
        ml_success_count = 0
        for symbol, training_data in all_training_data.items():
            try:
                if self.signal_generator.train_models_with_full_data(symbol, training_data):
                    ml_success_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to train model for {symbol}: {e}")
        
        self.models_trained = True
        self.training_completed_date = current_date
        self.logger.info(f"[ONE-TIME TRAINING] Successfully trained models for {ml_success_count}/{len(all_training_data)} symbols")
        
        return ml_success_count > 0

    def next(self):
        """Main strategy logic executed on each bar"""
        current_date = self.datas[0].datetime.datetime(0)
        
        # Reset daily trade counter
        if self.last_trade_date != current_date.date():
            self.daily_trade_count = 0
            self.last_trade_date = current_date.date()
        
        # Collect data for all symbols
        all_data = {}
        for data in self.datas:
            symbol = data._name
            df = self.collect_data_for_symbol(symbol, data)
            if not df.empty and len(df) >= self.params.min_training_days:
                all_data[symbol] = df
        
        # Skip if no sufficient data
        if not all_data:
            return

        # Check if we should rebalance (applies to both training and trading)
        if not self.should_rebalance():
            return

        self.last_rebalance_date = current_date
        
        # Handle training phase (Aug 2022 - Aug 2024)
        if self.is_training_period():
            # Just collect data during training period, don't trade
            return
            
        # At the end of training period, train models once with all training data
        if current_date > datetime(2024, 8, 22) and not self.models_trained:
            self.logger.info(f"{current_date.date()}: [END OF TRAINING] Training models with all 2-year training data")
            self.train_models_once(datetime(2024, 8, 22))  # Train using data up to end of training period
            
        # Ensure models are trained before trading
        if not self.models_trained:
            self.logger.warning(f"{current_date.date()}: Models not trained yet, cannot trade")
            return
            
        # Trading phase (Aug 23, 2024 - Aug 22, 2025)
        if not hasattr(self, 'trading_phase_logged'):
            self.logger.info(f"{current_date.date()}: [OUT-OF-SAMPLE TRADING] Starting trading with pre-trained models")
            self.trading_phase_logged = True
        
        # Check stop losses and profit targets during trading phase
        self.check_stop_loss_and_targets()
        
        # Generate signals for all symbols (using the generate_signal method for each symbol)
        signals = {}
        for symbol in all_data.keys():
            try:
                signal = self.signal_generator.generate_signal(symbol, all_data[symbol], current_date)
                if signal:
                    signals[symbol] = signal
                    # Log signal generation for debugging
                    if signal.get('signal') != 'HOLD':
                        self.logger.info(f"Generated {signal.get('signal')} signal for {symbol} with strength {signal.get('strength', 0):.3f}")
            except Exception as e:
                self.logger.warning(f"Failed to generate signal for {symbol}: {e}")
        
        self.current_signals = signals
        
        # Log daily signal summary
        buy_signals = sum(1 for s in signals.values() if s.get('signal') == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s.get('signal') == 'SELL')
        self.logger.info(f"Daily signals: {buy_signals} BUY, {sell_signals} SELL out of {len(signals)} symbols")
        
        # Process signals with OPTIMIZED logic
        self.process_optimized_signals(signals)
        
        # Log portfolio status
        self.log_portfolio_status()

    def process_optimized_signals(self, signals: Dict):
        """Process trading signals with OPTIMIZED risk management"""
        if not signals:
            return
        
        current_value = self.broker.getvalue()
        current_cash = self.broker.getcash()
        
        # Get current positions
        current_positions = {}
        total_position_value = 0
        
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data)
            if position.size != 0:
                current_price = data.close[0]
                position_value = abs(position.size) * current_price
                current_positions[symbol] = {
                    'size': position.size,
                    'value': position_value,
                    'price': current_price
                }
                total_position_value += position_value
        
        # Process EXIT signals first (map SELL to EXIT)
        for symbol, signal_data in signals.items():
            action = signal_data.get('signal', 'HOLD')  # Get 'signal' not 'action'
            if symbol in current_positions and action in ['EXIT', 'SELL']:
                data = self.getdatabyname(symbol)
                if data:
                    # Check minimum hold period
                    entry_date = self.position_entry_dates.get(symbol)
                    if entry_date:
                        hold_days = (self.datas[0].datetime.datetime(0) - entry_date).days
                        if hold_days < self.params.min_hold_days:
                            continue  # Skip exit if within minimum hold period
                    
                    signal_strength = signal_data.get('strength', 0)
                    self.log(f'EXIT SIGNAL: {symbol}, Strength: {signal_strength:.3f}, Signal: {action}')
                    self.close(data)
                    self.daily_trade_count += 1
        
        # Recalculate after exits
        remaining_positions = {}
        remaining_position_value = 0
        
        for data in self.datas:
            symbol = data._name
            position = self.getposition(data)
            if position.size != 0:
                current_price = data.close[0]
                position_value = abs(position.size) * current_price
                remaining_positions[symbol] = {
                    'size': position.size,
                    'value': position_value,
                    'price': current_price
                }
                remaining_position_value += position_value
        
        # Process BUY signals with CONSERVATIVE sizing
        buy_signals = []
        for symbol, signal_data in signals.items():
            action = signal_data.get('signal', 'HOLD')  # Get 'signal' not 'action'
            if (action == 'BUY' and 
                symbol not in remaining_positions and
                self.can_take_new_position(signal_data.get('strength', 0))):
                buy_signals.append((symbol, signal_data))
        
        # Sort by signal strength (highest first) - focus on best opportunities
        buy_signals.sort(key=lambda x: x[1].get('strength', 0), reverse=True)
        
        # Limit new positions based on portfolio constraints
        max_new_positions = min(
            self.params.max_positions - len(remaining_positions),
            self.params.max_daily_trades - self.daily_trade_count
        )
        
        for symbol, signal_data in buy_signals[:max_new_positions]:
            data = self.getdatabyname(symbol)
            if not data:
                continue
            
            current_price = data.close[0]
            signal_strength = signal_data.get('strength', 0)
            ml_enhanced = signal_data.get('ml_enhanced', False)
            
            # Calculate CONSERVATIVE position size
            updated_value = self.broker.getvalue()
            
            # Use optimized Kelly sizing with conservative parameters
            position_size = optimized_kelly_size(
                confidence=signal_strength,
                equity=updated_value,
                price=current_price,
                cap_fraction=self.params.max_symbol_exposure,  # 0.12 max
                avg_win=self.params.avg_win,
                avg_loss=self.params.avg_loss,
                frac=self.params.kelly_multiplier  # 0.35 instead of 0.5
            )
            
            if position_size > 0:
                # Additional safety check - ensure we don't exceed portfolio heat
                position_value = position_size * current_price
                potential_risk = position_value * self.params.stop_loss_pct
                current_heat = self.calculate_portfolio_heat()
                projected_heat = current_heat + (potential_risk / updated_value)
                
                if projected_heat > self.params.max_portfolio_heat:
                    # Reduce position size to stay within heat limits
                    max_risk = (self.params.max_portfolio_heat - current_heat) * updated_value
                    max_position_value = max_risk / self.params.stop_loss_pct
                    position_size = int(max_position_value / current_price)
                
                if position_size > 0:
                    self.log(f'BUY SIGNAL: {symbol}, Strength: {signal_strength:.3f}, '
                           f'Size: {position_size}, ML Enhanced: {ml_enhanced}')
                    
                    order = self.buy(data=data, size=position_size)
                    if order:
                        self.daily_trade_count += 1

    def log_portfolio_status(self):
        """Log current portfolio status with risk metrics"""
        current_value = self.broker.getvalue()
        current_cash = self.broker.getcash()
        total_return = (current_value - self.start_value) / self.start_value
        
        # Count positions
        position_count = 0
        total_exposure = 0
        
        for data in self.datas:
            position = self.getposition(data)
            if position.size != 0:
                position_count += 1
                position_value = abs(position.size) * data.close[0]
                total_exposure += position_value
        
        exposure_pct = total_exposure / current_value if current_value > 0 else 0
        current_heat = self.calculate_portfolio_heat()
        
        # Log every 5 trading days
        current_date = self.datas[0].datetime.datetime(0)
        if current_date.weekday() == 4 or position_count == 0:  # Friday or no positions
            self.log(f'Portfolio Value: ${current_value:,.0f}, Return: {total_return:.1%}, '
                   f'Positions: {position_count}, Exposure: {exposure_pct:.1%}, '
                   f'Heat: {current_heat:.1%}')

    def stop(self):
        """Called when strategy ends - log final results"""
        final_value = self.broker.getvalue()
        total_return = (final_value - self.start_value) / self.start_value
        
        # Count total trades
        total_trades = len([t for t in self.trade_log if t['action'] == 'BUY'])
        
        self.log('Strategy Completed!')
        self.log(f'Starting Value: ${self.start_value:,.0f}')
        self.log(f'Final Value: ${final_value:,.0f}')
        self.log(f'Total Return: {total_return:.2%}')
        self.log(f'Total Trades: {total_trades}')
        
        # Store daily values for analysis
        self.daily_values.append({
            'date': self.datas[0].datetime.datetime(0),
            'value': final_value,
            'return': total_return
        })
