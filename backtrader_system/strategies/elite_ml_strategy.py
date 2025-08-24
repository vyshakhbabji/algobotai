"""
ELITE Backtrader ML Trading Strategy - Maximum Accuracy & Profit

Combines:
1. Enhanced ML Signal Generator (XGBoost + LightGBM + Neural Networks)
2. Optimized risk management from optimized_ml_strategy
3. Advanced position sizing
4. Multi-timeframe analysis
5. Regime-aware trading
"""

import backtrader as bt
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import yfinance as yf

from .enhanced_ml_signal_generator import EnhancedMLSignalGenerator


def elite_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, confidence: float, fraction: float = 0.25) -> float:
    """Elite Kelly sizing with confidence adjustment"""
    try:
        if avg_win <= 0 or avg_loss <= 0 or not (0 <= win_rate <= 1):
            return 0.0
        
        # Full Kelly calculation
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        full_kelly = (b * p - q) / max(b, 1e-9)
        
        # Adjust by confidence and apply conservative fractional Kelly
        confidence_adjusted_kelly = full_kelly * confidence
        conservative_kelly = confidence_adjusted_kelly * fraction
        
        # Cap at maximum position size
        return max(0.0, min(conservative_kelly, 0.15))  # Max 15% position
        
    except Exception:
        return 0.0


def elite_position_size(signal_data: Dict[str, Any], equity: float, price: float, 
                       max_position_pct: float = 0.15) -> int:
    """Calculate position size using elite Kelly with ML confidence"""
    
    confidence = signal_data.get('confidence', 0.5)
    predicted_strength = signal_data.get('predicted_strength', 0.5)
    volatility = signal_data.get('volatility', 0.02)
    
    # Estimated win rate based on confidence and strength
    estimated_win_rate = 0.5 + (confidence - 0.5) * 0.6  # Scale to 0.2-0.8 range
    
    # Estimated win/loss based on volatility and strength
    avg_win = max(0.02, predicted_strength * 0.15)  # 2-15% potential win
    avg_loss = max(0.01, volatility * 2)  # 2x volatility as potential loss
    
    # Kelly fraction
    kelly_fraction = elite_kelly_fraction(estimated_win_rate, avg_win, avg_loss, confidence)
    
    # Volatility adjustment - reduce size in high volatility
    vol_adjustment = max(0.5, 1 - (volatility - 0.02) * 5)  # Reduce if vol > 2%
    
    # Final position size
    position_value = equity * kelly_fraction * vol_adjustment
    position_value = min(position_value, equity * max_position_pct)
    
    shares = int(position_value / price) if price > 0 else 0
    
    return max(0, shares)


class EliteMLTradingStrategy(bt.Strategy):
    """
    Elite ML Trading Strategy for Maximum Accuracy and Profit
    """
    
    params = (
        ('signal_threshold', 0.15),  # Minimum confidence for signal
        ('max_positions', 12),       # Maximum concurrent positions  
        ('max_position_size', 0.15), # Maximum position size (15%)
        ('max_symbol_exposure', 0.12), # Maximum exposure per symbol
        ('min_hold_days', 2),        # Minimum holding period
        ('stop_loss_pct', 0.08),     # Stop loss percentage
        ('profit_target_pct', 0.25), # Profit target percentage
        ('trailing_stop_pct', 0.12), # Trailing stop percentage
        ('max_portfolio_heat', 0.30), # Maximum portfolio risk
        ('config', {}),              # ML configuration
        ('rebalance_frequency', 5),  # Rebalance every N days
        ('volatility_filter', 0.05), # Max volatility for entry
        ('volume_filter', 0.8),      # Min volume ratio for entry
    )
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML Signal Generator
        self.ml_generator = EnhancedMLSignalGenerator(self.params.config)
        
        # Strategy state
        self.positions_data = {}  # Track position details
        self.daily_values = []    # Track portfolio value
        self.trade_log = []       # Trade history
        self.signals_history = [] # Signal history for analysis
        
        # Performance tracking
        self.trades_won = 0
        self.trades_lost = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # Market data storage for ML
        self.market_data = {}
        self.data_feeds = {}
        
        # Training flags
        self.models_trained = False
        self.training_data_length = 500  # Need 2+ years for training
        
        # Last rebalance date
        self.last_rebalance = None
        
        # Store data feed references
        for d in self.datas:
            symbol = d._name
            self.data_feeds[symbol] = d
            self.market_data[symbol] = []
        
        self.logger.info("🚀 Elite ML Trading Strategy initialized")
        self.logger.info(f"📊 Monitoring {len(self.datas)} symbols")
        self.logger.info(f"🎯 Signal threshold: {self.params.signal_threshold}")
        self.logger.info(f"💰 Max position size: {self.params.max_position_size:.1%}")
        
    def start(self):
        """Strategy start - called once at the beginning"""
        self.logger.info("🎬 Elite ML Strategy starting...")
        
    def prenext(self):
        """Called during warmup period"""
        self._collect_market_data()
        
    def next(self):
        """Main strategy logic - called on each bar"""
        try:
            current_date = self.data.datetime.date(0)
            
            # Collect market data
            self._collect_market_data()
            
            # Check if we have enough data to train models
            if not self.models_trained:
                self._check_and_train_models()
                
            # Skip if models not ready
            if not self.models_trained:
                return
            
            # Record daily portfolio value
            self._record_daily_value()
            
            # Check for rebalancing
            should_rebalance = self._should_rebalance(current_date)
            
            # Process existing positions
            self._manage_existing_positions()
            
            # Generate new signals and enter positions
            if should_rebalance:
                self._generate_signals_and_enter_positions()
                self.last_rebalance = current_date
                
        except Exception as e:
            self.logger.error(f"Error in strategy next(): {e}")
            
    def _collect_market_data(self):
        """Collect current market data for all symbols"""
        try:
            for symbol, data_feed in self.data_feeds.items():
                if len(data_feed) > 0:
                    bar_data = {
                        'Date': data_feed.datetime.date(0),
                        'Open': data_feed.open[0],
                        'High': data_feed.high[0],
                        'Low': data_feed.low[0],
                        'Close': data_feed.close[0],
                        'Volume': data_feed.volume[0] if hasattr(data_feed, 'volume') else 1000000
                    }
                    
                    if symbol not in self.market_data:
                        self.market_data[symbol] = []
                    
                    self.market_data[symbol].append(bar_data)
                    
                    # Keep only last 1000 bars to manage memory
                    if len(self.market_data[symbol]) > 1000:
                        self.market_data[symbol] = self.market_data[symbol][-1000:]
                        
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            
    def _check_and_train_models(self):
        """Check if we have enough data and train ML models"""
        try:
            # Check if we have enough data for any symbol
            symbols_ready = []
            for symbol, data_list in self.market_data.items():
                if len(data_list) >= self.training_data_length:
                    symbols_ready.append(symbol)
            
            if len(symbols_ready) < 3:  # Need at least 3 symbols with enough data
                return
                
            self.logger.info(f"🧠 Training ML models for {len(symbols_ready)} symbols...")
            
            models_trained = 0
            for symbol in symbols_ready[:10]:  # Train on first 10 symbols to save time
                try:
                    # Convert to DataFrame
                    df = pd.DataFrame(self.market_data[symbol])
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    
                    # Train models
                    success = self.ml_generator.train_ensemble_models(symbol, df)
                    if success:
                        models_trained += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to train model for {symbol}: {e}")
                    continue
            
            if models_trained >= 3:
                self.models_trained = True
                self.logger.info(f"✅ Successfully trained {models_trained} ML models")
                
                # Print model summary
                summary = self.ml_generator.get_model_summary()
                if 'average_performance' in summary:
                    avg_perf = summary['average_performance']
                    self.logger.info(f"📊 Average Model Performance:")
                    self.logger.info(f"   Strength R²: {avg_perf.get('strength_r2', 0):.3f}")
                    self.logger.info(f"   Direction Accuracy: {avg_perf.get('direction_accuracy', 0):.3f}")
            else:
                self.logger.warning(f"⚠️ Only trained {models_trained} models, need at least 3")
                
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            
    def _should_rebalance(self, current_date) -> bool:
        """Check if we should rebalance the portfolio"""
        if self.last_rebalance is None:
            return True
            
        days_since_rebalance = (current_date - self.last_rebalance).days
        return days_since_rebalance >= self.params.rebalance_frequency
        
    def _record_daily_value(self):
        """Record daily portfolio value"""
        try:
            current_date = self.data.datetime.date(0)
            portfolio_value = self.broker.getvalue()
            cash = self.broker.getcash()
            positions_value = portfolio_value - cash
            
            self.daily_values.append({
                'date': current_date,
                'total_value': portfolio_value,
                'cash': cash,
                'positions_value': positions_value,
                'num_positions': len([pos for pos in self.broker.positions.values() if pos.size != 0])
            })
            
        except Exception as e:
            self.logger.error(f"Error recording daily value: {e}")
            
    def _manage_existing_positions(self):
        """Manage existing positions - stop losses, profit targets, etc."""
        try:
            current_date = self.data.datetime.date(0)
            
            for symbol, data_feed in self.data_feeds.items():
                position = self.broker.getposition(data_feed)
                
                if position.size == 0:
                    continue
                    
                current_price = data_feed.close[0]
                
                # Get position data
                pos_data = self.positions_data.get(symbol, {})
                entry_price = pos_data.get('entry_price', current_price)
                entry_date = pos_data.get('entry_date', current_date)
                peak_price = pos_data.get('peak_price', entry_price)
                
                # Update peak price for trailing stop
                if position.size > 0:  # Long position
                    peak_price = max(peak_price, current_price)
                else:  # Short position  
                    peak_price = min(peak_price, current_price)
                
                self.positions_data[symbol]['peak_price'] = peak_price
                
                # Check holding period
                holding_days = (current_date - entry_date).days
                
                should_close = False
                close_reason = ""
                
                if position.size > 0:  # Long position
                    # Stop loss
                    if current_price < entry_price * (1 - self.params.stop_loss_pct):
                        should_close = True
                        close_reason = f"Stop loss: {self.params.stop_loss_pct:.1%}"
                    
                    # Profit target
                    elif current_price > entry_price * (1 + self.params.profit_target_pct):
                        should_close = True
                        close_reason = f"Profit target: {self.params.profit_target_pct:.1%}"
                    
                    # Trailing stop
                    elif current_price < peak_price * (1 - self.params.trailing_stop_pct):
                        should_close = True
                        close_reason = f"Trailing stop: {self.params.trailing_stop_pct:.1%}"
                        
                elif position.size < 0:  # Short position
                    # Stop loss
                    if current_price > entry_price * (1 + self.params.stop_loss_pct):
                        should_close = True
                        close_reason = f"Stop loss: {self.params.stop_loss_pct:.1%}"
                    
                    # Profit target
                    elif current_price < entry_price * (1 - self.params.profit_target_pct):
                        should_close = True
                        close_reason = f"Profit target: {self.params.profit_target_pct:.1%}"
                    
                    # Trailing stop
                    elif current_price > peak_price * (1 + self.params.trailing_stop_pct):
                        should_close = True
                        close_reason = f"Trailing stop: {self.params.trailing_stop_pct:.1%}"
                
                # Check minimum holding period
                if should_close and holding_days < self.params.min_hold_days:
                    should_close = False
                    close_reason = ""
                
                # Close position if needed
                if should_close:
                    if position.size > 0:
                        order = self.close(data=data_feed)
                    else:
                        order = self.close(data=data_feed)
                        
                    if order:
                        self._log_trade(symbol, 'CLOSE', abs(position.size), current_price, close_reason)
                        
                        # Update performance tracking
                        pnl = (current_price - entry_price) * abs(position.size)
                        if position.size < 0:  # Short position
                            pnl = -pnl
                            
                        if pnl > 0:
                            self.trades_won += 1
                            self.total_profit += pnl
                        else:
                            self.trades_lost += 1
                            self.total_loss += abs(pnl)
                        
                        # Remove position data
                        if symbol in self.positions_data:
                            del self.positions_data[symbol]
                            
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
            
    def _generate_signals_and_enter_positions(self):
        """Generate ML signals and enter new positions"""
        try:
            current_date = self.data.datetime.date(0)
            portfolio_value = self.broker.getvalue()
            current_positions = len([pos for pos in self.broker.positions.values() if pos.size != 0])
            
            # Skip if at maximum positions
            if current_positions >= self.params.max_positions:
                return
                
            # Calculate current portfolio heat
            current_heat = self._calculate_portfolio_heat()
            available_heat = self.params.max_portfolio_heat - current_heat
            
            if available_heat <= 0.05:  # Less than 5% heat available
                return
                
            # Generate signals for all symbols
            signals = []
            for symbol, data_feed in self.data_feeds.items():
                try:
                    # Skip if already have position
                    position = self.broker.getposition(data_feed)
                    if position.size != 0:
                        continue
                        
                    # Get market data for this symbol
                    if symbol not in self.market_data or len(self.market_data[symbol]) < 50:
                        continue
                        
                    # Convert to DataFrame for ML analysis
                    df = pd.DataFrame(self.market_data[symbol])
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                    
                    # Get ML signal
                    ml_signal = self.ml_generator.get_ml_signal(symbol, df)
                    
                    # Apply filters
                    if not self._passes_entry_filters(ml_signal, data_feed):
                        continue
                        
                    signals.append((symbol, data_feed, ml_signal))
                    
                except Exception as e:
                    self.logger.warning(f"Error generating signal for {symbol}: {e}")
                    continue
            
            # Sort signals by confidence * strength
            signals.sort(key=lambda x: x[2]['confidence'] * x[2]['predicted_strength'], reverse=True)
            
            # Enter positions for top signals
            positions_entered = 0
            max_new_positions = min(3, self.params.max_positions - current_positions)  # Max 3 new positions per rebalance
            
            for symbol, data_feed, ml_signal in signals[:max_new_positions]:
                try:
                    current_price = data_feed.close[0]
                    
                    # Calculate position size
                    shares = elite_position_size(ml_signal, portfolio_value, current_price, self.params.max_position_size)
                    
                    if shares < 1:
                        continue
                        
                    # Check if position value exceeds maximum
                    position_value = shares * current_price
                    max_position_value = portfolio_value * self.params.max_symbol_exposure
                    
                    if position_value > max_position_value:
                        shares = int(max_position_value / current_price)
                        
                    if shares < 1:
                        continue
                    
                    # Enter position based on signal
                    order = None
                    if ml_signal['signal'] == 'BUY':
                        order = self.buy(data=data_feed, size=shares)
                        action = 'BUY'
                    elif ml_signal['signal'] == 'SELL':
                        order = self.sell(data=data_feed, size=shares)
                        action = 'SELL'
                    
                    if order:
                        # Store position data
                        self.positions_data[symbol] = {
                            'entry_price': current_price,
                            'entry_date': current_date,
                            'peak_price': current_price,
                            'signal_data': ml_signal,
                            'shares': shares,
                            'action': action
                        }
                        
                        self._log_trade(symbol, action, shares, current_price, 
                                      f"ML Signal - Conf: {ml_signal['confidence']:.2f}")
                        
                        positions_entered += 1
                        
                        # Store signal for analysis
                        self.signals_history.append({
                            'date': current_date,
                            'symbol': symbol,
                            'signal': ml_signal,
                            'action': action,
                            'shares': shares,
                            'price': current_price
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error entering position for {symbol}: {e}")
                    continue
            
            if positions_entered > 0:
                self.logger.info(f"📈 Entered {positions_entered} new positions on {current_date}")
                
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            
    def _passes_entry_filters(self, ml_signal: Dict[str, Any], data_feed) -> bool:
        """Check if signal passes entry filters"""
        try:
            # Confidence filter
            if ml_signal['confidence'] < self.params.signal_threshold:
                return False
                
            # Signal strength filter
            if ml_signal['predicted_strength'] < 0.4:
                return False
                
            # Volatility filter
            if ml_signal['volatility'] > self.params.volatility_filter:
                return False
                
            # Volume filter (if available)
            if ml_signal['volume_ratio'] < self.params.volume_filter:
                return False
                
            # Probability filters
            if ml_signal['signal'] == 'BUY' and ml_signal['buy_probability'] < 0.6:
                return False
                
            if ml_signal['signal'] == 'SELL' and ml_signal['sell_probability'] < 0.6:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in entry filters: {e}")
            return False
            
    def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (risk exposure)"""
        try:
            portfolio_value = self.broker.getvalue()
            total_exposure = 0
            
            for data_feed in self.data_feeds.values():
                position = self.broker.getposition(data_feed)
                if position.size != 0:
                    current_price = data_feed.close[0]
                    position_value = abs(position.size) * current_price
                    total_exposure += position_value
                    
            return total_exposure / portfolio_value if portfolio_value > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio heat: {e}")
            return 0
            
    def _log_trade(self, symbol: str, action: str, shares: int, price: float, reason: str):
        """Log trade details"""
        try:
            trade_data = {
                'date': self.data.datetime.date(0),
                'symbol': symbol,
                'action': action,
                'shares': shares,
                'price': price,
                'value': shares * price,
                'reason': reason,
                'portfolio_value': self.broker.getvalue(),
                'cash': self.broker.getcash()
            }
            
            self.trade_log.append(trade_data)
            
            self.logger.info(f"💼 {action} {shares} {symbol} @ ${price:.2f} - {reason}")
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
            
    def stop(self):
        """Called when strategy stops"""
        try:
            self.logger.info("🏁 Elite ML Strategy stopping...")
            
            # Calculate final performance metrics
            final_value = self.broker.getvalue()
            total_trades = len(self.trade_log)
            win_rate = self.trades_won / (self.trades_won + self.trades_lost) if (self.trades_won + self.trades_lost) > 0 else 0
            
            self.logger.info(f"📊 Final Performance:")
            self.logger.info(f"   Final Value: ${final_value:,.2f}")
            self.logger.info(f"   Total Trades: {total_trades}")
            self.logger.info(f"   Win Rate: {win_rate:.1%}")
            self.logger.info(f"   Trades Won: {self.trades_won}, Lost: {self.trades_lost}")
            
            if self.total_profit > 0 and self.total_loss > 0:
                profit_factor = self.total_profit / self.total_loss
                self.logger.info(f"   Profit Factor: {profit_factor:.2f}")
                
            # ML model summary
            if self.models_trained:
                summary = self.ml_generator.get_model_summary()
                self.logger.info(f"🧠 ML Models Summary:")
                self.logger.info(f"   Models Trained: {summary['total_models']}")
                if 'average_performance' in summary:
                    avg_perf = summary['average_performance']
                    self.logger.info(f"   Avg Strength R²: {avg_perf.get('strength_r2', 0):.3f}")
                    self.logger.info(f"   Avg Direction Accuracy: {avg_perf.get('direction_accuracy', 0):.3f}")
                    
        except Exception as e:
            self.logger.error(f"Error in strategy stop: {e}")
