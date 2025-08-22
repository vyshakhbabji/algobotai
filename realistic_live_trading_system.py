"""
🚀 REALISTIC LIVE TRADING SIMULATOR
Daily ML Training + Human-like Trading Decisions

This simulates EXACTLY how the system would work in live trading:
- ML models retrained DAILY using only past data
- Human-like decisions: buy more, hold, partial sell, full sell
- Portfolio rebalancing based on fresh signals
- NO look-ahead bias - only uses data available up to trading day
- Tracks model performance evolution over time

Expected: True realistic performance with daily model updates
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any

# Set random seeds for determinism
np.random.seed(42)

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import lightgbm as lgb

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONPATH'] = str(Path(__file__).parent)
sys.path.append(str(Path(__file__).parent))

class RealisticLiveTradingSystem:
    """Realistic live trading simulation with daily ML retraining"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {symbol: shares}
        self.trades = []
        self.daily_performance = []
        self.daily_model_performance = []
        
        # Daily ML models (retrained each day)
        self.daily_models = {}  # {date: {symbol: models}}
        self.daily_scalers = {}  # {date: {symbol: scaler}}
        self.signal_strength_models = {}  # {date: {symbol: strength_model}}
        self.regime_models = {}  # {date: {symbol: regime_model}}
        
        # Configuration - SMART AGGRESSIVE: Quality over Quantity
        self.config = {
            # TIERED POSITION SIZING - Deploy capital based on signal confidence
            'max_position_size': 0.30,  # 30% for regular signals (more capital per idea)
            'max_position_size_exceptional': 0.50,  # 50% for exceptional signals
            'max_position_size_ultra': 0.70,  # 70% for ultra signals (new tier!)
            'exceptional_signal_threshold': 0.80,  # 80% for exceptional (was 0.75)
            'ultra_signal_threshold': 0.92,  # 92% for ultra confidence
            'min_position_size': 0.03,  # Min 3% per stock
            'stop_loss_mult': 2.0,      # 2x ATR stop loss
            'take_profit_mult': 3.0,    # 3x ATR take profit
            'rebalance_threshold': 0.10, # 10% threshold for rebalancing
            'signal_threshold': 0.50,    # Slightly lower threshold to capture more opportunities
            'max_positions': 6,         # Max 6 positions (was 8 - more concentrated)
            'partial_sell_threshold': 0.3,  # Sell 30% on weak signals (faster profit taking)
            'ml_retrain_days': 1,       # Retrain every day (realistic)
            'min_training_days': 100,   # Need 100 days to train
            'max_new_positions_per_day': 4,  # Limit entries (was 12 - QUALITY OVER QUANTITY!)
            'commission_bps': 5,        # 0.05% commission
            'slippage_bps': 2,          # 0.02% slippage
            'min_commission': 1.0,      # $1 minimum commission
            
            # SMART Capital Deployment - Reserve cash for amazing opportunities
            'target_invested_floor': 0.70,    # Keep 30% cash for opportunities (was 90%)
            'target_invested_ceiling': 0.85,  # Max 85% deployed normally (was 98%)
            'emergency_cash_reserve': 0.20,   # Always keep 20% for ultra signals
            'min_avg_strength_for_ceiling': 0.75,  # High bar for full deployment
            'min_individual_strength_for_extra': 0.70,  # High bar for extra allocation
            'cash_reserve_floor': 0.15,       # Minimum 15% cash (was 1% - need dry powder!)
            'max_utilization_topups_per_day': 3,  # Conservative top-ups
            'min_extra_position_size': 0.03,   # Minimum size for extra allocations

            # Dynamic trailing stop to protect profits
            'trailing_stop_mult': 1.5,  # Trail by 1.5x ATR from the high watermark
            
            # Portfolio-Level Exposure Guards (Enhancement #4)
            'max_net_exposure': 0.95,         # Max portfolio exposure (95%)
            'drawdown_throttle_threshold': 0.10,  # Drawdown threshold for throttling (10%)
            'drawdown_throttle_sessions': 10,     # Sessions to throttle after drawdown
            'drawdown_throttle_ceiling': 0.70,    # Reduced ceiling during throttling (70%)
            
            # Enhancement Tracking and Persistence (Enhancement #9)
            'enable_enhancement_logging': True,   # Track all enhancements
            'enable_state_persistence': True,     # Save/resume trading state
            'auto_save_frequency': 10             # Auto-save every N days
        }
        
        # Daily price cache for consistent portfolio valuation
        self.daily_prices = {}  # {date: {symbol: price}}
        
        # Position cost basis tracking
        self.position_cost_basis = {}  # {symbol: avg_entry_price}

        # Track highest price since entry for trailing stop logic
        self.position_high_watermark = {}  # {symbol: highest_price}
        
        # Pending orders queue for next-day execution (Fix #2)
        self.pending_orders = []  # Orders to execute next trading day
        
        # Model performance tracking (Fix #13)
        self.daily_model_performance = {}  # {date: {symbol: {accuracy, r2, samples}}}
        
        # Performance tracking
        self.start_date = None
        self.end_date = None
        self.model_evolution = {}  # Track how models evolve
        
        # Drawdown throttling state (Enhancement #4)
        self.peak_portfolio_value = initial_capital
        self.drawdown_throttle_remaining = 0  # Sessions remaining in throttle mode
        
        # Enhancement tracking (Enhancement #9)
        self.enhancement_metrics = {
            'rebalancing_actions': 0,
            'risk_orders_queued': 0,
            'capital_utilization_orders': 0,
            'drawdown_throttle_activations': 0,
            'trading_days_validated': 0,
            'data_hygiene_rejections': 0,
            'enhanced_slippage_calculations': 0,
            'end_of_day_summaries': 0,
            'state_saves': 0
        }
        
        self.logger = self._setup_logging()
        self.logger.info("🚀 Realistic Live Trading System initialized")
    
    def _portfolio_value(self, prices_today: Dict[str, float]) -> float:
        """Calculate total portfolio value using consistent daily prices"""
        total_value = self.current_capital
        for symbol, shares in self.positions.items():
            if shares > 0 and symbol in prices_today:
                total_value += shares * prices_today[symbol]
        return total_value
    
    def _apply_transaction_costs(self, price: float, shares: int, is_buy: bool, symbol: str = None, 
                               market_data: Dict = None) -> Tuple[float, float]:
        """Enhanced slippage and commission model (Enhancement #7) - SIMPLIFIED"""
        # Use simple slippage for now to debug performance issues
        slippage_factor = self.config['slippage_bps'] / 10000.0
        
        if is_buy:
            # Buy at higher price due to slippage
            execution_price = price * (1 + slippage_factor)
        else:
            # Sell at lower price due to slippage
            execution_price = price * (1 - slippage_factor)
        
        # Calculate commission
        notional = execution_price * shares
        commission = max(
            self.config['min_commission'],
            notional * self.config['commission_bps'] / 10000.0
        )
        
        return execution_price, commission
    
    def _calculate_enhanced_slippage(self, base_slippage: float, shares: int, price: float, 
                                   symbol: str, market_data: Dict, is_buy: bool) -> float:
        """Calculate enhanced slippage based on multiple factors (Enhancement #7)"""
        try:
            enhanced_slippage = base_slippage
            
            # Factor 1: Order size impact (larger orders = more slippage)
            notional = shares * price
            if notional > 100000:  # $100k+ orders
                size_multiplier = min(2.0, 1.0 + (notional - 100000) / 1000000)  # Up to 2x slippage
                enhanced_slippage *= size_multiplier
            
            # Factor 2: Volatility impact (from market data if available)
            if market_data and symbol:
                try:
                    # Use recent volatility as proxy for bid-ask spread
                    recent_prices = market_data.get('recent_prices', [])
                    if len(recent_prices) >= 5:
                        volatility = pd.Series(recent_prices).pct_change().std()
                        if volatility > 0.05:  # High volatility (5%+ daily moves)
                            volatility_multiplier = min(2.0, 1.0 + volatility * 10)
                            enhanced_slippage *= volatility_multiplier
                except:
                    pass
            
            # Factor 3: Market cap impact (smaller stocks = more slippage)
            if symbol:
                # Simplified market cap estimation based on price
                if price < 10:  # Small/penny stocks
                    enhanced_slippage *= 1.5
                elif price > 200:  # Large cap stocks
                    enhanced_slippage *= 0.8
            
            # Factor 4: Market conditions (simplified - could add VIX, market hours, etc.)
            # For now, apply random market microstructure noise
            import random
            microstructure_noise = random.uniform(0.8, 1.2)  # ±20% randomness
            enhanced_slippage *= microstructure_noise
            
            # Cap maximum slippage at 50 bps (0.5%)
            enhanced_slippage = min(enhanced_slippage, 0.005)
            
            return enhanced_slippage
            
        except Exception as e:
            self.logger.debug(f"Enhanced slippage calculation error: {e}")
            return base_slippage  # Fallback to base slippage
    

    def _apply_risk_rules(self, current_date: pd.Timestamp, prices_today: Dict[str, float], all_history: Dict) -> int:
        """Apply ATR-based stop loss, take profit, and trailing stops - queue as pending orders"""
        risk_orders_queued = 0

        for symbol, shares in list(self.positions.items()):
            if shares <= 0 or symbol not in prices_today or symbol not in all_history:
                continue

            current_price = prices_today[symbol]
            avg_entry = self.position_cost_basis.get(symbol, current_price)

            prev_high = self.position_high_watermark.get(symbol, avg_entry)
            if current_price > prev_high:
                self.position_high_watermark[symbol] = current_price
            high_water = self.position_high_watermark.get(symbol, current_price)

            symbol_data = all_history[symbol]
            if current_date not in symbol_data.index:
                continue
            data_up_to_today = symbol_data.loc[:current_date]
            if len(data_up_to_today) < 15:
                continue
            high = data_up_to_today['High']
            low = data_up_to_today['Low']
            close = data_up_to_today['Close']
            prev_close = close.shift(1)
            tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            if pd.isna(atr) or atr <= 0:
                continue

            stop_loss_price = avg_entry - atr * self.config['stop_loss_mult']
            take_profit_price = avg_entry + atr * self.config['take_profit_mult']
            trailing_stop_price = high_water - atr * self.config['trailing_stop_mult']

            pnl_pct = (current_price - avg_entry) / avg_entry

            if current_price <= stop_loss_price:
                stop_decision = {
                    'action': 'STOP_LOSS',
                    'shares': shares,
                    'reason': f'Stop loss: {pnl_pct:.1%} (exit @ ${stop_loss_price:.2f})',
                    'stop_price': stop_loss_price,
                    'order_type': 'STOP_LOSS',
                    'atr': atr
                }
                risk_signal = {
                    'strength': 0.0,
                    'price': current_price,
                    'base_strength': 0.0,
                    'ml_multiplier': 1.0,
                    'regime_boost': 1.0,
                    'total_enhancement': 1.0,
                    'ml_enhanced': False,
                    'risk_exit': True,
                    'exit_price': stop_loss_price,
                    'atr': atr
                }
                if self._queue_pending_order(symbol, stop_decision, risk_signal, current_date):
                    risk_orders_queued += 1
                    self.enhancement_metrics['risk_orders_queued'] += 1
                    self.logger.info(
                        f"   🛡️ Stop loss queued: {symbol} @ ${stop_loss_price:.2f} (P&L: {pnl_pct:.1%})")

            if current_price >= take_profit_price:
                tp_decision = {
                    'action': 'TAKE_PROFIT',
                    'shares': shares,
                    'reason': f'Take profit: {pnl_pct:.1%} (exit @ ${take_profit_price:.2f})',
                    'take_profit_price': take_profit_price,
                    'order_type': 'TAKE_PROFIT',
                    'atr': atr
                }
                risk_signal = {
                    'strength': 0.0,
                    'price': current_price,
                    'base_strength': 0.0,
                    'ml_multiplier': 1.0,
                    'regime_boost': 1.0,
                    'total_enhancement': 1.0,
                    'ml_enhanced': False,
                    'risk_exit': True,
                    'exit_price': take_profit_price,
                    'atr': atr
                }
                if self._queue_pending_order(symbol, tp_decision, risk_signal, current_date):
                    risk_orders_queued += 1
                    self.enhancement_metrics['risk_orders_queued'] += 1
                    self.logger.info(
                        f"   🎯 Take profit queued: {symbol} @ ${take_profit_price:.2f} (P&L: {pnl_pct:.1%})")

            if current_price > avg_entry and trailing_stop_price > avg_entry and current_price <= trailing_stop_price:
                ts_decision = {
                    'action': 'TRAILING_STOP',
                    'shares': shares,
                    'reason': f'Trailing stop: {pnl_pct:.1%} (exit @ ${trailing_stop_price:.2f})',
                    'stop_price': trailing_stop_price,
                    'order_type': 'TRAILING_STOP',
                    'atr': atr
                }
                risk_signal = {
                    'strength': 0.0,
                    'price': current_price,
                    'base_strength': 0.0,
                    'ml_multiplier': 1.0,
                    'regime_boost': 1.0,
                    'total_enhancement': 1.0,
                    'ml_enhanced': False,
                    'risk_exit': True,
                    'exit_price': trailing_stop_price,
                    'atr': atr
                }
                if self._queue_pending_order(symbol, ts_decision, risk_signal, current_date):
                    risk_orders_queued += 1
                    self.enhancement_metrics['risk_orders_queued'] += 1
                    self.logger.info(
                        f"   🏁 Trailing stop queued: {symbol} @ ${trailing_stop_price:.2f} (P&L: {pnl_pct:.1%})")

        return risk_orders_queued
    
    def _get_current_invested_ratio(self, current_date: pd.Timestamp, prices_today: Dict[str, float]) -> float:
        """Calculate current invested ratio (portfolio value in stocks ÷ total portfolio value)"""
        try:
            # Calculate current portfolio value in stocks
            portfolio_stock_value = 0.0
            for symbol, shares in self.positions.items():
                if symbol in prices_today:
                    portfolio_stock_value += shares * prices_today[symbol]
            
            # Total portfolio value = cash + stocks
            total_portfolio_value = self.current_capital + portfolio_stock_value
            
            if total_portfolio_value <= 0:
                return 0.0
            
            invested_ratio = portfolio_stock_value / total_portfolio_value
            return invested_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating invested ratio: {e}")
            return 0.0
    
    def _check_portfolio_exposure_limits(self, prices_today: Dict[str, float]) -> bool:
        """Check if portfolio is within exposure limits (Enhancement #4)"""
        try:
            current_invested_ratio = self._get_current_invested_ratio(None, prices_today)
            max_exposure = self.config['max_net_exposure']
            
            if current_invested_ratio >= max_exposure:
                self.logger.warning(f"   🚫 Portfolio exposure limit hit: {current_invested_ratio:.1%} >= {max_exposure:.1%}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking exposure limits: {e}")
            return True  # Allow trading if check fails
    
    def _update_drawdown_throttle(self, current_portfolio_value: float) -> None:
        """Update drawdown throttling state (Enhancement #4)"""
        try:
            # Update peak value
            if current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
                self.drawdown_throttle_remaining = 0  # Reset throttle when new peak
            else:
                # Check for drawdown
                drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
                
                if drawdown >= self.config['drawdown_throttle_threshold']:
                    if self.drawdown_throttle_remaining == 0:
                        # New throttle period triggered
                        self.drawdown_throttle_remaining = self.config['drawdown_throttle_sessions']
                        self.logger.warning(f"   🔴 Drawdown throttle activated: {drawdown:.1%} drawdown, throttling for {self.drawdown_throttle_remaining} sessions")
                
                # Decrement throttle counter
                if self.drawdown_throttle_remaining > 0:
                    self.drawdown_throttle_remaining -= 1
                    if self.drawdown_throttle_remaining == 0:
                        self.logger.info(f"   🟢 Drawdown throttle period ended")
            
        except Exception as e:
            self.logger.error(f"Error updating drawdown throttle: {e}")
    
    def _get_effective_target_ceiling(self) -> float:
        """Get effective target ceiling considering drawdown throttling (Enhancement #4)"""
        if self.drawdown_throttle_remaining > 0:
            throttled_ceiling = self.config['drawdown_throttle_ceiling']
            self.logger.debug(f"   ⚠️ Drawdown throttle active ({self.drawdown_throttle_remaining} sessions): ceiling {throttled_ceiling:.1%}")
            return throttled_ceiling
        else:
            return self.config['target_invested_ceiling']
    
    def _apply_drawdown_throttle(self, target_invested: float) -> float:
        """Apply drawdown throttling to target investment level (Enhancement #4)"""
        if self.drawdown_throttle_remaining > 0:
            # During throttle period, reduce target ceiling
            if target_invested >= self.config['target_invested_ceiling']:
                throttled_target = self.config['drawdown_throttle_ceiling']
                self.logger.debug(f"   ⚠️ Drawdown throttle: reducing target {target_invested:.1%} → {throttled_target:.1%}")
                return throttled_target
        return target_invested
    
    def _is_valid_trading_day(self, date: pd.Timestamp) -> bool:
        """Enhanced trading calendar validation (Enhancement #5)"""
        try:
            # Basic weekday check
            if date.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Extended US market holiday list
            us_holidays_2024_2025 = [
                '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
                '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
                '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
                '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
            ]
            
            date_str = date.strftime('%Y-%m-%d')
            if date_str in us_holidays_2024_2025:
                return False
            
            # Check for partial trading days (early close) - could add logic here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trading calendar validation error: {e}")
            return True  # Default to trading day on error
    
    def _perform_data_hygiene_checks(self, symbols: List[str], all_history: Dict, current_date: pd.Timestamp) -> List[str]:
        """Enhanced data hygiene and validation (Enhancement #6)"""
        valid_symbols = []
        
        try:
            for symbol in symbols:
                if symbol not in all_history:
                    continue
                
                # Check if we have data for current date
                if current_date not in all_history[symbol].index:
                    continue
                
                # Get recent data window for validation
                data = all_history[symbol]
                current_idx = data.index.get_loc(current_date)
                
                # Need at least 20 days of history for ML features
                if current_idx < 20:
                    continue
                
                # Get current day's data
                current_data = data.iloc[current_idx]
                
                # Data hygiene checks
                price_checks = [
                    current_data['Close'] > 0,
                    current_data['Volume'] > 0,
                    current_data['High'] >= current_data['Low'],
                    current_data['High'] >= current_data['Close'],
                    current_data['Low'] <= current_data['Close'],
                    not pd.isna(current_data['Close']),
                    not pd.isna(current_data['Volume'])
                ]
                
                if all(price_checks):
                    # Additional check: reasonable price movement (< 50% in one day)
                    if current_idx > 0:
                        prev_close = data.iloc[current_idx - 1]['Close']
                        price_change = abs(current_data['Close'] - prev_close) / prev_close
                        if price_change < 0.5:  # Less than 50% daily change
                            valid_symbols.append(symbol)
                    else:
                        valid_symbols.append(symbol)
            
            self.logger.debug(f"   ✅ Data hygiene: {len(valid_symbols)}/{len(symbols)} symbols passed validation")
            return valid_symbols
            
        except Exception as e:
            self.logger.error(f"Data hygiene check error: {e}")
            return symbols  # Return all symbols on error
    
    def _log_end_of_day_metrics(self, current_date: pd.Timestamp, day_idx: int, total_days: int,
                               total_value: float, daily_trades: int, daily_ml_enhancements: int,
                               daily_new_positions: int) -> None:
        """Comprehensive end-of-day logging and metrics (Enhancement #8)"""
        try:
            # Basic performance metrics
            daily_return = (total_value - self.initial_capital) / self.initial_capital
            daily_return_pct = daily_return * 100
            
            # Position metrics
            position_count = len(self.positions)
            total_invested = sum(shares * 0 for shares in self.positions.values())  # Will be calculated properly with prices
            
            # Calculate properly with prices
            invested_value = 0
            if self.positions and hasattr(self, '_last_prices'):
                for symbol, shares in self.positions.items():
                    if symbol in self._last_prices:
                        invested_value += shares * self._last_prices[symbol]
            
            invested_ratio = invested_value / total_value if total_value > 0 else 0
            
            # Risk metrics
            pending_count = len(self.pending_orders)
            risk_orders_today = len([o for o in self.pending_orders if o.get('order_type') in ['STOP_LOSS', 'TAKE_PROFIT']])
            
            # Model metrics
            active_models = len([m for models in self.daily_models.values() for m in models.values()])
            
            # Trading calendar progress
            progress_pct = (day_idx + 1) / total_days * 100
            
            # Drawdown tracking
            if hasattr(self, 'peak_portfolio_value'):
                current_drawdown = (self.peak_portfolio_value - total_value) / self.peak_portfolio_value
                throttle_status = f" | Throttle: {self.drawdown_throttle_remaining}d" if self.drawdown_throttle_remaining > 0 else ""
            else:
                current_drawdown = 0
                throttle_status = ""
            
            # Daily summary log
            self.logger.info(
                f"📊 EOD Summary [{progress_pct:.1f}%]: "
                f"${total_value:,.0f} ({daily_return_pct:+.2f}%) | "
                f"Pos: {position_count} ({invested_ratio:.1%}) | "
                f"Trades: {daily_trades} (ML: {daily_ml_enhancements}) | "
                f"Pending: {pending_count} (Risk: {risk_orders_today}) | "
                f"Cash: ${self.current_capital:,.0f}"
                f"{throttle_status}"
            )
            
            # Weekly detailed summary (every 5 days)
            if (day_idx + 1) % 5 == 0:
                # Recent performance
                recent_values = [d['total_value'] for d in self.daily_performance[-5:]]
                week_return = (recent_values[-1] - recent_values[0]) / recent_values[0] * 100 if len(recent_values) >= 2 else 0
                
                # Position analysis
                top_positions = []
                if self.positions and hasattr(self, '_last_prices'):
                    position_values = {}
                    for symbol, shares in self.positions.items():
                        if symbol in self._last_prices:
                            position_values[symbol] = shares * self._last_prices[symbol]
                    
                    top_positions = sorted(position_values.items(), key=lambda x: x[1], reverse=True)[:3]
                
                self.logger.info(
                    f"📈 Weekly Summary: {week_return:+.2f}% | "
                    f"Active Models: {active_models} | "
                    f"Drawdown: {current_drawdown:.2%} | "
                    f"Top Holdings: {', '.join([f'{s}:${v:,.0f}' for s, v in top_positions])}"
                )
            
        except Exception as e:
            self.logger.error(f"End-of-day logging error: {e}")
    
    def _execute_risk_action(self, action: Dict, current_date: pd.Timestamp) -> bool:
        """Execute a risk management action"""
        try:
            symbol = action['symbol']
            shares = action['shares']
            base_price = action['price']
            
            # Apply transaction costs
            execution_price, commission = self._apply_transaction_costs(base_price, shares, is_buy=False, symbol=symbol)
            gross_proceeds = shares * execution_price
            net_proceeds = gross_proceeds - commission
            
            # Calculate realized P&L for consistency with normal exits (Fix #5)
            cost_basis = self.position_cost_basis.get(symbol, execution_price)
            realized_pnl = (execution_price - cost_basis) * shares - commission
            
            # Update positions
            self.current_capital += net_proceeds
            if symbol in self.positions:
                del self.positions[symbol]
            if symbol in self.position_cost_basis:
                del self.position_cost_basis[symbol]
            
            # Record trade with consistent fields (Fix #5)
            self.trades.append({
                'date': current_date,
                'symbol': symbol,
                'action': action['action'],
                'shares': shares,
                'close_price_reference': base_price,
                'execution_price': execution_price,
                'open_price': base_price,
                'commission': commission,
                'gross_proceeds': gross_proceeds,
                'net_proceeds': net_proceeds,
                'cost_basis': cost_basis,
                'realized_pnl': realized_pnl,
                'pnl_pct': action['pnl_pct'],
                'reason': action['reason'],
                'risk_triggered': True,
                'strength': 0.0,  # Risk exits don't have signal strength
                'base_strength': 0.0,
                'ml_multiplier': 1.0,
                'regime_boost': 1.0,
                'total_enhancement': 1.0,
                'ml_enhanced': False,
                'order_date': current_date,
                'execution_date': current_date,
                'next_day_execution': False
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing risk action for {action['symbol']}: {e}")
            return False
    
    def _rank_buy_candidates(self, signals: Dict[str, Dict], prices_today: Dict[str, float]) -> List[Dict]:
        """Collect and rank BUY signals above minimum strength threshold"""
        try:
            candidates = []
            min_strength = self.config['min_individual_strength_for_extra']
            
            for symbol, signal in signals.items():
                # Check if signal strength meets minimum and we have price data
                if (signal['strength'] >= min_strength and
                    symbol in prices_today):
                    
                    # Only consider signals that would result in BUY actions
                    # Based on our signal strength threshold logic
                    if signal['strength'] >= self.config['signal_threshold']:
                        candidate = {
                            'symbol': symbol,
                            'strength': signal['strength'],
                            'signal': signal,
                            'price': prices_today[symbol],
                            'has_position': symbol in self.positions
                        }
                        candidates.append(candidate)
            
            # Sort by signal strength (strongest first)
            candidates.sort(key=lambda x: x['strength'], reverse=True)
            
            self.logger.debug(f"   📊 Found {len(candidates)} buy candidates above strength {min_strength:.2f}")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error ranking buy candidates: {e}")
            return []
    
    def _queue_pending_order(self, symbol: str, decision: Dict, signal: Dict, order_date: pd.Timestamp) -> bool:
        """Queue an order for execution on the next trading day at open price"""
        try:
            if decision['action'] == 'HOLD' or decision['shares'] == 0:
                return False
            
            order = {
                'symbol': symbol,
                'action': decision['action'],
                'shares': decision['shares'],
                'signal_strength': signal['strength'],
                'base_strength': signal.get('base_strength', signal['strength']),
                'ml_multiplier': signal.get('ml_multiplier', 1.0),
                'regime_boost': signal.get('regime_boost', 1.0),
                'total_enhancement': signal.get('total_enhancement', 1.0),
                'reason': decision['reason'],
                'ml_enhanced': signal.get('ml_enhanced', False),
                'order_date': order_date,
                'close_price_reference': signal['price']  # Reference price from close
            }

            # Include optional risk parameters (ATR-based exits)
            for key in ['order_type', 'stop_price', 'take_profit_price', 'trail_price', 'atr']:
                if key in decision:
                    order[key] = decision[key]
            if 'stop_price' in decision:
                order['exit_price'] = decision['stop_price']
            elif 'take_profit_price' in decision:
                order['exit_price'] = decision['take_profit_price']
            elif 'trail_price' in decision:
                order['exit_price'] = decision['trail_price']

            self.pending_orders.append(order)
            return True
            
        except Exception as e:
            self.logger.error(f"Error queuing order for {symbol}: {e}")
            return False
    
    def _execute_pending_orders(self, current_date: pd.Timestamp, all_history: Dict) -> int:
        """Execute pending orders with enhanced risk order gap handling (Enhancement #2)"""
        executed_orders = 0
        
        orders_to_remove = []
        for i, order in enumerate(self.pending_orders):
            try:
                symbol = order['symbol']
                
                # Get the OHLC data for current_date (next day after order was placed)
                if symbol in all_history:
                    symbol_data = all_history[symbol].loc[:current_date]
                    if not symbol_data.empty and current_date in symbol_data.index:
                        day_data = symbol_data.loc[current_date]
                        open_price = day_data['Open']
                        high_price = day_data['High']
                        low_price = day_data['Low']
                        
                        execution_price = None
                        should_execute = True
                        
                        # Enhanced execution logic for risk orders (Enhancement #2)
                        if order.get('order_type') in ['STOP_LOSS', 'TRAILING_STOP']:
                            stop_price = order.get('stop_price', open_price)
                            
                            # Gap down beyond stop -> fill at open
                            if open_price <= stop_price:
                                execution_price = open_price
                                self.logger.debug(f"   🔴 Stop loss gap fill: {symbol} @ ${execution_price:.2f} (open, gap below ${stop_price:.2f})")
                            
                            # Intraday hit stop -> fill at stop price
                            elif low_price <= stop_price:
                                execution_price = stop_price
                                self.logger.debug(f"   🔴 Stop loss hit: {symbol} @ ${execution_price:.2f} (stop level)")
                            
                            else:
                                # Stop not hit, keep order alive
                                should_execute = False
                                
                        elif order.get('order_type') == 'TAKE_PROFIT':
                            tp_price = order.get('take_profit_price', open_price)
                            
                            # Gap up beyond TP -> fill at open
                            if open_price >= tp_price:
                                execution_price = open_price
                                self.logger.debug(f"   🟢 Take profit gap fill: {symbol} @ ${execution_price:.2f} (open, gap above ${tp_price:.2f})")
                            
                            # Intraday hit TP -> fill at TP price
                            elif high_price >= tp_price:
                                execution_price = tp_price
                                self.logger.debug(f"   🟢 Take profit hit: {symbol} @ ${execution_price:.2f} (TP level)")
                            
                            else:
                                # TP not hit, keep order alive
                                should_execute = False
                        
                        else:
                            # Normal order -> execute at open
                            execution_price = open_price
                        
                        # Execute if conditions met
                        if should_execute and execution_price is not None:
                            if self._execute_order_at_price(order, execution_price, current_date):
                                executed_orders += 1
                                order_type_display = order.get('order_type', order['action'])
                                self.logger.info(f"   📋 Executed: {order_type_display} {order['shares']} {symbol} @ ${execution_price:.2f}")
                                orders_to_remove.append(i)
                            else:
                                # Order rejected (position limits, insufficient capital, etc.)
                                order_type_display = order.get('order_type', order['action'])
                                self.logger.info(f"   ❌ Rejected: {order_type_display} {order['shares']} {symbol} @ ${execution_price:.2f}")
                                orders_to_remove.append(i)
                        elif order.get('order_type') in ['STOP_LOSS', 'TAKE_PROFIT', 'TRAILING_STOP']:
                            # Keep risk orders alive if not triggered
                            pass
                        else:
                            # Remove non-risk orders that couldn't execute
                            orders_to_remove.append(i)
                    else:
                        # No data available, keep order for next day
                        self.logger.warning(f"   ⚠️ No OHLC data for {symbol} on {current_date.date()}")
                else:
                    # Remove order if no history available
                    orders_to_remove.append(i)
                    
            except Exception as e:
                self.logger.error(f"Error executing pending order for {order['symbol']}: {e}")
                orders_to_remove.append(i)
        
        # Remove executed/failed orders (in reverse order to maintain indices)
        for i in reversed(orders_to_remove):
            del self.pending_orders[i]
        
        return executed_orders
    
    def _rebalance_portfolio(self, prices_today: Dict[str, float], current_date: pd.Timestamp) -> int:
        """Rebalance overweight/underweight positions (Enhancement #1)"""
        try:
            rebalance_orders = 0
            portfolio_value = self._portfolio_value(prices_today)
            
            if portfolio_value <= 0:
                return 0
            
            rebalance_threshold = self.config['rebalance_threshold']  # 5%
            target_weight = self.config['max_position_size']  # 15% target for normal positions
            
            self.logger.debug(f"   ⚖️ Rebalancing portfolio (threshold: {rebalance_threshold:.1%})")
            
            for symbol, shares in list(self.positions.items()):
                if symbol not in prices_today or shares <= 0:
                    continue
                
                current_value = shares * prices_today[symbol]
                current_weight = current_value / portfolio_value
                
                # Check if position needs rebalancing
                if current_weight > target_weight + rebalance_threshold:
                    # Overweight - trim position
                    target_value = portfolio_value * target_weight
                    excess_value = current_value - target_value
                    trim_shares = int(excess_value / prices_today[symbol])
                    
                    if trim_shares > 0:
                        trim_decision = {
                            'action': 'SELL_PARTIAL',
                            'shares': trim_shares,
                            'reason': f'Rebalance trim: {current_weight:.1%} → {target_weight:.1%}'
                        }
                        
                        # Create dummy signal for rebalancing
                        rebalance_signal = {
                            'strength': 0.5,
                            'price': prices_today[symbol],
                            'base_strength': 0.5,
                            'ml_multiplier': 1.0,
                            'regime_boost': 1.0,
                            'total_enhancement': 1.0,
                            'ml_enhanced': False
                        }
                        
                        if self._queue_pending_order(symbol, trim_decision, rebalance_signal, current_date):
                            rebalance_orders += 1
                            self.enhancement_metrics['rebalancing_actions'] += 1
                            self.logger.debug(f"   📉 Rebalance trim: {symbol} {trim_shares} shares ({current_weight:.1%} → {target_weight:.1%})")
                
                elif current_weight < target_weight - rebalance_threshold and self.current_capital > 1000:
                    # Underweight - top up position (if cash available)
                    target_value = portfolio_value * target_weight
                    needed_value = target_value - current_value
                    available_cash = min(needed_value, self.current_capital * 0.8)  # Use max 80% of available cash
                    
                    if available_cash > 100:  # Minimum $100 top-up
                        topup_shares = int(available_cash / prices_today[symbol])
                        
                        if topup_shares > 0:
                            topup_decision = {
                                'action': 'BUY_MORE',
                                'shares': topup_shares,
                                'reason': f'Rebalance top-up: {current_weight:.1%} → {target_weight:.1%}'
                            }
                            
                            # Create dummy signal for rebalancing
                            rebalance_signal = {
                                'strength': 0.5,
                                'price': prices_today[symbol],
                                'base_strength': 0.5,
                                'ml_multiplier': 1.0,
                                'regime_boost': 1.0,
                                'total_enhancement': 1.0,
                                'ml_enhanced': False
                            }
                            
                            if self._queue_pending_order(symbol, topup_decision, rebalance_signal, current_date):
                                rebalance_orders += 1
                                self.enhancement_metrics['rebalancing_actions'] += 1
                                self.logger.debug(f"   📈 Rebalance top-up: {symbol} {topup_shares} shares ({current_weight:.1%} → {target_weight:.1%})")
            
            if rebalance_orders > 0:
                self.logger.info(f"   ⚖️ Portfolio rebalancing: {rebalance_orders} orders queued")
            
            return rebalance_orders
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return 0
    
    def _optimize_capital_utilization(self, signals: Dict[str, Dict], prices_today: Dict[str, float], 
                                    current_date: pd.Timestamp, daily_new_positions: int) -> Tuple[int, int]:
        """Optimize capital utilization by queuing extra BUY orders to reach target invested % (Enhancement #3)"""
        try:
            # Get current invested ratio
            current_invested_ratio = self._get_current_invested_ratio(current_date, prices_today)
            
            # Rank buy candidates
            candidates = self._rank_buy_candidates(signals, prices_today)
            
            if not candidates:
                self.logger.debug(f"   💰 No buy candidates for capital utilization")
                return 0, daily_new_positions
            
            # Decide target investment level based on signal strength and ultra opportunities
            avg_strength = sum(c['strength'] for c in candidates) / len(candidates)
            max_strength = max(c['strength'] for c in candidates)
            
            # EMERGENCY ULTRA SIGNAL HANDLING
            ultra_threshold = self.config.get('ultra_signal_threshold', 0.92)
            emergency_reserve = self.config.get('emergency_cash_reserve', 0.20)
            
            if max_strength >= ultra_threshold:
                # ULTRA SIGNAL DETECTED - Use emergency reserves!
                target_invested = 0.95  # Deploy almost everything for ultra signals
                self.logger.info(f"   🔥 ULTRA SIGNAL ({max_strength:.3f}) - DEPLOYING EMERGENCY RESERVES!")
            elif avg_strength >= self.config['min_avg_strength_for_ceiling']:
                target_invested = self.config['target_invested_ceiling']
                self.logger.debug(f"   🎯 Strong signals (avg: {avg_strength:.3f}) - targeting ceiling: {target_invested:.1%}")
            else:
                target_invested = self.config['target_invested_floor']
                self.logger.debug(f"   🎯 Moderate signals (avg: {avg_strength:.3f}) - targeting floor: {target_invested:.1%}")
            
            # Apply drawdown throttling (Enhancement #4)
            target_invested = self._apply_drawdown_throttle(target_invested)
            
            # Check if we need to increase investment
            if current_invested_ratio >= target_invested:
                self.logger.debug(f"   ✅ Already at target: {current_invested_ratio:.1%} >= {target_invested:.1%}")
                return 0, daily_new_positions
            
            # Calculate total portfolio value and available cash
            portfolio_stock_value = sum(shares * prices_today.get(symbol, 0) 
                                      for symbol, shares in self.positions.items())
            total_portfolio_value = self.current_capital + portfolio_stock_value
            
            # Calculate cash needed to reach target
            target_stock_value = total_portfolio_value * target_invested
            needed_investment = target_stock_value - portfolio_stock_value
            
            # Respect cash reserve floor, but allow breach for ceiling target with strong signals
            min_cash_reserve = total_portfolio_value * self.config['cash_reserve_floor']
            
            # Enhanced cash allocation logic: Allow reserve breach for ultra signals and strong signals targeting ceiling
            if max_strength >= ultra_threshold:
                # ULTRA SIGNAL: Deploy emergency reserves - use 95% of ALL available cash
                max_investable = max(0, self.current_capital * 0.95)
                self.logger.debug(f"   🔥 ULTRA SIGNAL → EMERGENCY DEPLOYMENT: Using 95% of available cash (${max_investable:.0f})")
            elif target_invested >= self.config['target_invested_ceiling'] and avg_strength >= self.config['min_avg_strength_for_ceiling']:
                # Strong signals targeting ceiling: allow using up to 95% of cash (minimal reserve)
                max_investable = max(0, self.current_capital * 0.95)
                self.logger.debug(f"   🎯 Strong signals → ceiling target: Allowing reserve breach (95% cash utilization)")
            else:
                # Normal operation: respect full cash reserve floor
                max_investable = max(0, self.current_capital - min_cash_reserve)
                self.logger.debug(f"   💰 Normal operation: Respecting cash reserve floor (${min_cash_reserve:.0f})")
            
            available_cash = min(needed_investment, max_investable)
            
            if available_cash < 100:  # Need at least $100 to invest
                self.logger.debug(f"   💸 Insufficient cash for utilization: ${available_cash:.0f}")
                return 0, daily_new_positions
            
            self.logger.debug(f"   💰 Capital utilization: {current_invested_ratio:.1%} → {target_invested:.1%}")
            self.logger.debug(f"   💸 Available for investment: ${available_cash:.0f}")
            
            # Allocate extra capital to candidates (strongest first)
            extra_orders = 0
            remaining_cash = available_cash
            
            for candidate in candidates:
                if extra_orders >= self.config['max_utilization_topups_per_day']:
                    self.logger.debug(f"   ⚠️ Hit daily top-up limit: {extra_orders}")
                    break
                
                if remaining_cash < 100:
                    break
                
                symbol = candidate['symbol']
                price = candidate['price']
                strength = candidate['strength']
                
                # Check position limits
                current_position = self.positions.get(symbol, 0)
                current_value = current_position * price
                position_weight = current_value / total_portfolio_value if total_portfolio_value > 0 else 0
                
                # Determine max allowed weight for this symbol
                if strength >= self.config['exceptional_signal_threshold']:
                    max_allowed = self.config['max_position_size_exceptional']
                else:
                    max_allowed = self.config['max_position_size']
                
                if candidate['has_position']:
                    # Top-up existing position
                    if position_weight >= max_allowed:
                        continue  # Already at max
                    
                    max_additional_weight = max_allowed - position_weight
                    max_additional_value = min(remaining_cash, total_portfolio_value * max_additional_weight)
                    
                else:
                    # New position - enforce daily cap (Enhancement #3)
                    if (len(self.positions) >= self.config['max_positions'] or
                        daily_new_positions >= self.config['max_new_positions_per_day']):
                        self.logger.debug(f"   ⚠️ New position cap hit: {daily_new_positions}/{self.config['max_new_positions_per_day']}")
                        continue  # Hit position limits
                    
                    # Use minimum of available cash and max position size
                    max_additional_value = min(remaining_cash, total_portfolio_value * max_allowed)
                    daily_new_positions += 1  # Increment counter for new position
                
                # Calculate shares (minimum position size check)
                min_value = total_portfolio_value * self.config['min_extra_position_size']
                if max_additional_value < min_value:
                    continue
                
                additional_shares = int(max_additional_value / price)
                if additional_shares <= 0:
                    continue
                
                actual_value = additional_shares * price
                
                # Create utilization order
                utilization_decision = {
                    'action': 'BUY_MORE' if candidate['has_position'] else 'BUY',
                    'shares': additional_shares,
                    'reason': f'Capital utilization (strength: {strength:.3f}, target: {target_invested:.1%})'
                }
                
                # Queue the order
                if self._queue_pending_order(symbol, utilization_decision, candidate['signal'], current_date):
                    remaining_cash -= actual_value
                    extra_orders += 1
                    self.logger.debug(f"   📈 Queued {symbol}: {additional_shares} shares @ ${price:.2f} (${actual_value:.0f})")
            
            self.logger.debug(f"   🎯 Capital utilization complete: {extra_orders} extra orders queued")
            return extra_orders, daily_new_positions
            
        except Exception as e:
            self.logger.error(f"Error optimizing capital utilization: {e}")
            return 0, daily_new_positions
    
    def _execute_order_at_price(self, order: Dict, execution_price: float, current_date: pd.Timestamp) -> bool:
        """Execute a specific order at given price with transaction costs"""
        try:
            symbol = order['symbol']
            action = order['action']
            shares = order['shares']
            
            current_position = self.positions.get(symbol, 0)
            
            if action in ['BUY', 'BUY_MORE']:
                # Double-check position size limits before execution (Fix #8 + Portfolio value bug fix)
                new_position = current_position + shares
                new_position_value = new_position * execution_price
                
                # CRITICAL FIX: Use proper daily prices for all positions, not just one symbol
                # Copy to avoid mutating stored dictionary by reference
                proper_prices = self.daily_prices.get(current_date, {}).copy()
                proper_prices[symbol] = execution_price  # Update with current execution price
                portfolio_value = self._portfolio_value(proper_prices)
                new_weight = new_position_value / portfolio_value if portfolio_value > 0 else 0
                
                # Check against appropriate limit (ultra vs exceptional vs normal)
                signal_strength = order.get('signal_strength', 0)
                if signal_strength >= self.config.get('ultra_signal_threshold', 0.92):
                    max_allowed = self.config.get('max_position_size_ultra', 0.65)
                elif signal_strength >= self.config['exceptional_signal_threshold']:
                    max_allowed = self.config['max_position_size_exceptional']
                else:
                    max_allowed = self.config['max_position_size']
                
                if new_weight > max_allowed:
                    # DEBUG: Track rejection details
                    rejection_info = {
                        'symbol': symbol,
                        'signal_strength': signal_strength,
                        'new_weight': new_weight,
                        'max_allowed': max_allowed,
                        'shares': shares,
                        'execution_price': execution_price,
                        'current_date': current_date,
                        'is_ultra_signal': signal_strength >= self.config.get('ultra_signal_threshold', 0.92)
                    }
                    self.debug_data['rejected_trades'].append(rejection_info)
                    self.debug_data['position_limit_violations'] += 1
                    
                    if signal_strength >= self.config.get('ultra_signal_threshold', 0.92):
                        self.logger.warning(f"   🚨 ULTRA SIGNAL REJECTED! {new_weight:.1%} > {max_allowed:.1%} for {symbol} (strength: {signal_strength:.3f})")
                    else:
                        self.logger.warning(f"   ⚠️ Order would exceed position size limit: {new_weight:.1%} > {max_allowed:.1%} for {symbol}")
                    
                    self.logger.info(f"   ❌ Rejected: {action} {shares} {symbol} @ ${execution_price:.2f}")
                    return False
                
                # Apply transaction costs to open price
                final_price, commission = self._apply_transaction_costs(execution_price, shares, is_buy=True, symbol=symbol)
                total_cost = (shares * final_price) + commission
                
                if total_cost <= self.current_capital:
                    # Update position cost basis (weighted average)
                    old_position = current_position
                    new_position = old_position + shares
                    
                    if old_position > 0:
                        old_basis = self.position_cost_basis.get(symbol, final_price)
                        new_basis = ((old_position * old_basis) + (shares * final_price)) / new_position
                        self.position_cost_basis[symbol] = new_basis
                    else:
                        self.position_cost_basis[symbol] = final_price
                    
                    self.positions[symbol] = new_position
                    self.current_capital -= total_cost

                    # Initialize or update high watermark for trailing stops
                    current_high = self.position_high_watermark.get(symbol, 0)
                    self.position_high_watermark[symbol] = max(current_high, final_price)
                    
                    self.trades.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': action,
                        'shares': shares,
                        'close_price_reference': order['close_price_reference'],
                        'execution_price': final_price,
                        'open_price': execution_price,
                        'commission': commission,
                        'total_cost': total_cost,
                        'strength': order['signal_strength'],
                        'base_strength': order['base_strength'],
                        'ml_multiplier': order['ml_multiplier'],
                        'regime_boost': order['regime_boost'],
                        'total_enhancement': order['total_enhancement'],
                        'reason': order['reason'],
                        'ml_enhanced': order['ml_enhanced'],
                        'order_date': order['order_date'],
                        'execution_date': current_date,
                        'next_day_execution': True
                    })
                    return True
            
            elif action in ['SELL_ALL', 'SELL_PARTIAL']:
                if shares <= current_position:
                    # Apply transaction costs to open price
                    final_price, commission = self._apply_transaction_costs(execution_price, shares, is_buy=False, symbol=symbol)
                    gross_proceeds = shares * final_price
                    net_proceeds = gross_proceeds - commission
                    
                    # Calculate realized P&L for normal exits (Fix #4)
                    cost_basis = self.position_cost_basis.get(symbol, final_price)
                    realized_pnl = (final_price - cost_basis) * shares - commission
                    
                    self.current_capital += net_proceeds
                    self.positions[symbol] = current_position - shares
                    
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                        if symbol in self.position_cost_basis:
                            del self.position_cost_basis[symbol]
                        if symbol in self.position_high_watermark:
                            del self.position_high_watermark[symbol]
                    
                    self.trades.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': action,
                        'shares': shares,
                        'close_price_reference': order['close_price_reference'],
                        'execution_price': final_price,
                        'open_price': execution_price,
                        'commission': commission,
                        'gross_proceeds': gross_proceeds,
                        'net_proceeds': net_proceeds,
                        'cost_basis': cost_basis,
                        'realized_pnl': realized_pnl,
                        'strength': order['signal_strength'],
                        'base_strength': order['base_strength'],
                        'ml_multiplier': order['ml_multiplier'],
                        'regime_boost': order['regime_boost'],
                        'total_enhancement': order['total_enhancement'],
                        'reason': order['reason'],
                        'ml_enhanced': order['ml_enhanced'],
                        'order_date': order['order_date'],
                        'execution_date': current_date,
                        'next_day_execution': True
                    })
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing order for {order['symbol']}: {e}")
            return False
    
    def _setup_logging(self):
        """Setup logging with file output for debugging"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"trading_debug_log_{timestamp}.txt"
        
        logger = logging.getLogger('RealisticLiveTrading')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for debugging
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Store debug info
        self.debug_log_file = log_filename
        self.debug_data = {
            'ultra_signals_detected': 0,
            'ultra_signals_executed': 0,
            'emergency_reserves_triggered': 0,
            'position_limit_violations': 0,
            'rejected_trades': [],
            'executed_trades': [],
            'signal_analysis': []
        }
        
        logger.info(f"🔍 DEBUG MODE: Logging to {log_filename}")
        return logger
    
    def get_elite_stocks(self) -> List[str]:
        """Same elite stock selection"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX',
            'CRM', 'SNOW', 'PLTR', 'COIN', 'UBER', 'DIS', 'JPM', 'BAC',
            'JNJ', 'PG', 'KO', 'WMT', 'HD', 'V', 'MA', 'PFE', 'VZ'
        ][:20]  # Limit to 20 for faster daily training
    
    def fetch_data_up_to_date(self, symbol: str, current_date: pd.Timestamp) -> pd.DataFrame:
        """Fetch data up to specific date (no look-ahead bias)"""
        try:
            ticker = yf.Ticker(symbol)
            # Get data from 2 years before trading period starts
            start_date = "2023-05-21"  # 2 years of training data
            end_date = current_date.strftime('%Y-%m-%d')
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Ensure no future data leakage
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Only return data up to and including current_date
            data = data[data.index <= current_date]
            
            return data.dropna()
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} up to {current_date.date()}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        try:
            df = df.copy()
            
            # Basic features
            df['returns'] = df['Close'].pct_change()
            df['volume_ma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
            
            # Moving averages
            df['ma_5'] = df['Close'].rolling(5).mean()
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['ma_50'] = df['Close'].rolling(50).mean()
            
            # Price relative positions
            df['price_vs_ma5'] = (df['Close'] - df['ma_5']) / df['ma_5']
            df['price_vs_ma20'] = (df['Close'] - df['ma_20']) / df['ma_20']
            df['price_vs_ma50'] = (df['Close'] - df['ma_50']) / df['ma_50']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
            
            # Volatility
            df['volatility_10d'] = df['returns'].rolling(10).std()
            df['volatility_20d'] = df['returns'].rolling(20).std()
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Fix divide-by-zero for Bollinger Bands
            bb_width = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / bb_width
            df['bb_position'] = df['bb_position'].fillna(0.5)  # Default to middle
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Trend features
            df['trend_5d'] = np.where(df['Close'] > df['Close'].shift(5), 1, -1)
            df['trend_10d'] = np.where(df['Close'] > df['Close'].shift(10), 1, -1)
            df['trend_consistency'] = (df['trend_5d'] + df['trend_10d']) / 2
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def create_ml_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML targets for prediction"""
        try:
            df = df.copy()
            
            # Market regime target (trending vs ranging)
            # More sophisticated regime detection
            price_momentum = df['Close'].pct_change(10).abs()
            volatility = df['volatility_10d']
            
            # Regime score: high momentum + low volatility = trending
            regime_score = price_momentum / (volatility + 0.001)
            regime_threshold = regime_score.rolling(50).quantile(0.6)
            df['regime_target'] = (regime_score > regime_threshold).astype(int)
            
            # Signal strength target (0.3-1.0) - ENHANCED VERSION
            # Based on future volatility-adjusted moves
            future_returns_3d = df['Close'].shift(-3) / df['Close'] - 1
            vol_adj_move = np.abs(future_returns_3d) / (df['volatility_10d'] + 0.001)
            
            # Normalize to 0.3-1.0 range (same as enhanced system)
            signal_strength_raw = np.clip(vol_adj_move * 1.5, 0, 1)
            df['ml_signal_strength'] = 0.3 + (signal_strength_raw * 0.7)
            
            # Also create the old target for compatibility
            df['signal_strength_target'] = df['ml_signal_strength']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating ML targets: {e}")
            return df
    
    def train_daily_models(self, symbol: str, data: pd.DataFrame, current_date: pd.Timestamp) -> bool:
        """Train ML models using only data up to current date - ENHANCED VERSION"""
        try:
            if len(data) < self.config['min_training_days']:
                return False
            
            # Calculate features and targets
            df = self.calculate_technical_indicators(data.copy())
            df = self.create_ml_targets(df)
            
            # Feature columns
            feature_cols = [
                'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50',
                'rsi_normalized', 'volatility_10d', 'volatility_20d',
                'volume_ratio', 'bb_position', 'macd_histogram',
                'trend_consistency'
            ]
            
            # Clean data (remove future-looking NaN values)
            clean_data = df[feature_cols + ['regime_target', 'ml_signal_strength']].dropna()
            
            # Fix target leakage: only train on data BEFORE current_date
            train_mask = clean_data.index < current_date
            clean_data = clean_data[train_mask]
            
            # Enforce min_training_days per symbol consistently (Fix #16)
            if len(clean_data) < self.config['min_training_days']:
                self.logger.debug(f"   ⚠️ {symbol}: Insufficient training data ({len(clean_data)} < {self.config['min_training_days']})")
                return False
            
            X = clean_data[feature_cols].values
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store scaler for this date
            if current_date not in self.daily_scalers:
                self.daily_scalers[current_date] = {}
            self.daily_scalers[current_date][symbol] = scaler
            
            # Initialize model storage for this date
            if current_date not in self.daily_models:
                self.daily_models[current_date] = {}
            if symbol not in self.daily_models[current_date]:
                self.daily_models[current_date][symbol] = {}
            
            models_trained = 0
            
            # 1. Train Signal Strength Model (ENHANCED - Like enhanced_kelly_ml_system)
            y_strength = clean_data['ml_signal_strength'].values
            
            try:
                strength_model = lgb.LGBMRegressor(
                    n_estimators=50,  # Faster for daily training
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                
                # Quick validation
                if len(clean_data) > 20:
                    tscv = TimeSeriesSplit(n_splits=2)
                    cv_scores = cross_val_score(strength_model, X_scaled, y_strength, cv=tscv, scoring='r2')
                    avg_r2 = cv_scores.mean()
                else:
                    avg_r2 = 0.0
                
                # Train on all available data
                strength_model.fit(X_scaled, y_strength)
                self.daily_models[current_date][symbol]['strength'] = strength_model
                models_trained += 1
                
            except Exception as e:
                self.logger.error(f"Signal strength model training failed for {symbol}: {e}")
            
            # 2. Train Regime Model (existing logic)
            y_regime = clean_data['regime_target'].values
            
            try:
                regime_model = RandomForestClassifier(
                    n_estimators=50,  # Faster for daily training
                    max_depth=6,
                    random_state=42
                )
                
                # Quick validation
                if len(clean_data) > 20:
                    tscv = TimeSeriesSplit(n_splits=2)
                    cv_scores = cross_val_score(regime_model, X_scaled, y_regime, cv=tscv, scoring='accuracy')
                    avg_accuracy = cv_scores.mean()
                else:
                    avg_accuracy = 0.5
                
                # Train on all available data
                regime_model.fit(X_scaled, y_regime)
                self.daily_models[current_date][symbol]['regime'] = regime_model
                models_trained += 1
                
                # Track model evolution
                if symbol not in self.model_evolution:
                    self.model_evolution[symbol] = []
                
                # Store daily model performance (Fix #13)
                if current_date not in self.daily_model_performance:
                    self.daily_model_performance[current_date] = {}
                
                self.daily_model_performance[current_date][symbol] = {
                    'regime_accuracy': avg_accuracy,
                    'strength_r2': avg_r2 if 'avg_r2' in locals() else 0.0,
                    'training_samples': len(clean_data),
                    'models_trained': models_trained
                }
                
                self.model_evolution[symbol].append({
                    'date': current_date,
                    'regime_accuracy': avg_accuracy,
                    'strength_r2': avg_r2 if 'avg_r2' in locals() else 0.0,
                    'training_samples': len(clean_data),
                    'models_trained': models_trained
                })
                
                return models_trained > 0
                
            except Exception as e:
                self.logger.error(f"Regime model training failed for {symbol} on {current_date.date()}: {e}")
                return models_trained > 0
                
        except Exception as e:
            self.logger.error(f"Error training daily models for {symbol}: {e}")
            return False
    
    def generate_base_signal(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate base technical signal (same proven logic)"""
        try:
            if len(df) < 50:
                return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
            
            latest = df.iloc[-1]
            
            # Technical conditions
            rsi = latest['rsi']
            price = latest['Close']
            price_vs_ma5 = latest['price_vs_ma5']
            price_vs_ma20 = latest['price_vs_ma20']
            volume_ratio = latest.get('volume_ratio', 1.0)
            macd_hist = latest.get('macd_histogram', 0)
            
            # Strong buy conditions
            strong_buy = (
                (rsi < 35 and price_vs_ma5 > -0.03) or  # Oversold with support
                (price_vs_ma5 > 0.02 and price_vs_ma20 > 0.01 and volume_ratio > 1.2 and macd_hist > 0)
            )
            
            # Strong sell conditions
            strong_sell = (
                (rsi > 75 and price_vs_ma5 < 0.02) or  # Overbought with resistance
                (price_vs_ma5 < -0.02 and price_vs_ma20 < -0.01 and macd_hist < 0)
            )
            
            # Calculate signal strength
            if strong_buy:
                strength = min(0.8, (1.2 + price_vs_ma5 + (80-rsi)/100 + (volume_ratio-1)))
                return {'signal': 'BUY', 'strength': max(0.4, strength), 'price': price}
            elif strong_sell:
                strength = min(0.8, (1.2 - price_vs_ma5 + (rsi-20)/100))
                return {'signal': 'SELL', 'strength': max(0.4, strength), 'price': price}
            else:
                return {'signal': 'HOLD', 'strength': 0.0, 'price': price}
                
        except Exception as e:
            self.logger.error(f"Error generating base signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
    
    def enhance_signal_with_daily_ml(self, symbol: str, base_signal: Dict, df: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, Any]:
        """Enhance signal using models trained up to current date - ENHANCED VERSION"""
        try:
            if base_signal['signal'] == 'HOLD':
                return base_signal
            
            # Check if we have models for this date and symbol
            if (current_date not in self.daily_models or 
                symbol not in self.daily_models[current_date] or
                current_date not in self.daily_scalers or
                symbol not in self.daily_scalers[current_date]):
                return base_signal
            
            # Prepare features
            feature_cols = [
                'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50',
                'rsi_normalized', 'volatility_10d', 'volatility_20d',
                'volume_ratio', 'bb_position', 'macd_histogram',
                'trend_consistency'
            ]
            
            latest_features = df[feature_cols].iloc[-1:].values
            scaler = self.daily_scalers[current_date][symbol]
            X_scaled = scaler.transform(latest_features)
            
            # Default enhancements
            ml_multiplier = 1.0
            regime_boost = 1.0
            
            # 1. Predict Signal Strength Enhancement (ENHANCED - Like enhanced_kelly_ml_system)
            if 'strength' in self.daily_models[current_date][symbol]:
                try:
                    strength_model = self.daily_models[current_date][symbol]['strength']
                    strength_pred = strength_model.predict(X_scaled)[0]
                    strength_pred = np.clip(strength_pred, 0.3, 1.0)
                    
                    # Convert prediction to multiplier
                    ml_multiplier = strength_pred
                    
                except Exception as e:
                    self.logger.error(f"Signal strength prediction error for {symbol}: {e}")
            
            # 2. Predict Market Regime Enhancement (existing logic)
            if 'regime' in self.daily_models[current_date][symbol]:
                try:
                    regime_model = self.daily_models[current_date][symbol]['regime']
                    regime_proba = regime_model.predict_proba(X_scaled)[0]
                    trending_prob = regime_proba[1] if len(regime_proba) > 1 else 0.5
                    
                    # Boost signals in trending markets
                    regime_boost = 1.0 + (trending_prob * 0.4)  # Up to 40% boost
                    
                except Exception as e:
                    self.logger.error(f"Regime prediction error for {symbol}: {e}")
            
            # 3. Combine Enhancements (ENHANCED - Like enhanced_kelly_ml_system)
            total_multiplier = ml_multiplier * regime_boost
            total_multiplier = np.clip(total_multiplier, 0.5, 1.8)  # Reasonable bounds
            
            # Apply enhancement
            enhanced_strength = base_signal['strength'] * total_multiplier
            enhanced_strength = np.clip(enhanced_strength, 0.0, 1.0)
            
            return {
                'signal': base_signal['signal'],
                'strength': enhanced_strength,
                'price': base_signal['price'],
                'base_strength': base_signal['strength'],
                'ml_multiplier': ml_multiplier,
                'regime_boost': regime_boost,
                'total_enhancement': total_multiplier,
                'ml_enhanced': True
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing signal for {symbol}: {e}")
            return base_signal
    
    def make_human_like_decision(self, symbol: str, signal: Dict, prices_today: Dict[str, float], daily_new_positions: int = 0) -> Dict[str, Any]:
        """Make human-like trading decisions: buy more, hold, partial sell, full sell"""
        try:
            action = signal['signal']
            strength = signal['strength']
            price = signal['price']
            
            current_position = self.positions.get(symbol, 0)
            current_value = current_position * price if current_position > 0 else 0
            
            # Fix portfolio valuation using consistent daily prices
            portfolio_value = self._portfolio_value(prices_today)
            position_weight = current_value / portfolio_value if portfolio_value > 0 else 0
            
            decision = {
                'action': 'HOLD',
                'shares': 0,
                'reason': 'No signal',
                'current_position': current_position,
                'current_weight': position_weight
            }
            
            if action == 'BUY' and strength >= self.config['signal_threshold']:
                # Check max positions limit
                if current_position == 0 and len(self.positions) >= self.config['max_positions']:
                    decision.update({
                        'reason': f'Max positions limit reached ({self.config["max_positions"]})'
                    })
                    return decision
                
                if current_position == 0:
                    # Check daily new position limit (Fix #20)
                    if daily_new_positions >= self.config['max_new_positions_per_day']:
                        decision.update({
                            'reason': f'Daily new position limit reached ({self.config["max_new_positions_per_day"]})'
                        })
                        return decision
                    
                    # SMART TIERED POSITION SIZING - Match position size to signal confidence
                    if strength >= self.config.get('ultra_signal_threshold', 0.92):
                        max_allowed = self.config.get('max_position_size_ultra', 0.65)
                        self.logger.info(f"   🔥 ULTRA signal ({strength:.3f}) - using maximum position limit {max_allowed:.1%}")
                    elif strength >= self.config['exceptional_signal_threshold']:
                        max_allowed = self.config['max_position_size_exceptional']
                        self.logger.info(f"   🚀 Exceptional signal ({strength:.3f}) - using enhanced position limit {max_allowed:.1%}")
                    else:
                        max_allowed = self.config['max_position_size']
                    
                    # SMART SCALING: More aggressive scaling for higher confidence
                    if strength >= 0.92:  # Ultra signals
                        target_weight = max_allowed * 1.0   # Use FULL ultra limit
                    elif strength >= 0.85:  # Very strong
                        target_weight = max_allowed * 0.90  
                    elif strength >= 0.80:  # Strong
                        target_weight = max_allowed * 0.80  
                    elif strength >= 0.70:  # Good
                        target_weight = max_allowed * 0.65  
                    elif strength >= 0.60:  # Decent
                        target_weight = max_allowed * 0.50  
                    else:  # Weak but acceptable
                        target_weight = max_allowed * 0.35  
                    
                    target_value = portfolio_value * target_weight
                    
                    # Enforce minimum position size
                    min_value = portfolio_value * self.config['min_position_size']
                    if target_value < min_value:
                        target_value = min_value
                    
                    shares = int(target_value / price)
                    
                    if shares > 0 and target_value <= self.current_capital:
                        decision.update({
                            'action': 'BUY',
                            'shares': shares,
                            'reason': f'New position (strength: {strength:.3f})'
                        })
                
                elif position_weight < self.config['max_position_size']:
                    # Add to existing position - MAXIMUM AGGRESSION ADD-ON LOGIC
                    if strength > 0.35:  # Very low threshold for adding (was 0.55 - DEPLOY CAPITAL!)
                        if strength >= self.config['exceptional_signal_threshold']:
                            max_allowed = self.config['max_position_size_exceptional']
                        else:
                            max_allowed = self.config['max_position_size']
                        
                        # MAXIMUM AGGRESSION: Large add-ons to deploy capital
                        if strength >= 0.8:
                            additional_weight = min(0.20, max_allowed - position_weight)  # Huge add-on
                        elif strength >= 0.7:
                            additional_weight = min(0.15, max_allowed - position_weight)  # Large add-on
                        elif strength >= 0.6:
                            additional_weight = min(0.12, max_allowed - position_weight)  # Good add-on
                        elif strength >= 0.5:
                            additional_weight = min(0.10, max_allowed - position_weight)  # Medium add-on
                        else:
                            additional_weight = min(0.08, max_allowed - position_weight)  # Small but meaningful add-on
                        
                        additional_value = portfolio_value * additional_weight
                        additional_shares = int(additional_value / price)
                        
                        # Check if post-trade weight would exceed limits
                        post_trade_shares = current_position + additional_shares
                        post_trade_value = post_trade_shares * price
                        post_trade_weight = post_trade_value / portfolio_value
                        
                        if post_trade_weight <= max_allowed and additional_shares > 0 and additional_value <= self.current_capital:
                            decision.update({
                                'action': 'BUY_MORE',
                                'shares': additional_shares,
                                'reason': f'Adding to position (strength: {strength:.3f})'
                            })
            
            elif action == 'SELL' and current_position > 0:
                # SMART PROFIT TAKING - Different thresholds based on original signal strength
                # Check if we have profit to protect
                current_value = current_position * price
                avg_entry = self.position_cost_basis.get(symbol, price)
                position_return = (price - avg_entry) / avg_entry
                
                if position_return > 0.15:  # If we have 15%+ profit, be more selective about selling
                    sell_threshold_adjustment = 0.1  # Require stronger sell signal to exit winners
                else:
                    sell_threshold_adjustment = 0.0  # Normal thresholds for break-even/loss positions
                
                if strength > (0.80 + sell_threshold_adjustment):  # Very strong sell signal
                    decision.update({
                        'action': 'SELL_ALL',
                        'shares': current_position,
                        'reason': f'Full exit (strength: {strength:.3f}, return: {position_return:.1%})'
                    })
                elif strength > (0.70 + sell_threshold_adjustment):  # Strong sell signal
                    # Take profits on winners, trim losers
                    if position_return > 0.10:  # If profitable, take partial profits
                        partial_shares = max(1, int(current_position * 0.4))  # Sell 40% of winners
                    else:
                        partial_shares = max(1, int(current_position * 0.6))  # Sell 60% of losers
                    
                    if partial_shares > 0 and partial_shares <= current_position:
                        decision.update({
                            'action': 'SELL_PARTIAL',
                            'shares': partial_shares,
                            'reason': f'Partial exit (strength: {strength:.3f})'
                        })
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making decision for {symbol}: {e}")
            return {'action': 'HOLD', 'shares': 0, 'reason': 'Error'}
    
    def execute_human_decision(self, symbol: str, decision: Dict, signal: Dict, current_date: pd.Timestamp) -> bool:
        """Execute the human-like trading decision with transaction costs"""
        try:
            action = decision['action']
            shares = decision['shares']
            base_price = signal['price']
            
            if action == 'HOLD' or shares == 0:
                return False
            
            current_position = self.positions.get(symbol, 0)
            
            if action in ['BUY', 'BUY_MORE']:
                # Apply transaction costs
                execution_price, commission = self._apply_transaction_costs(base_price, shares, is_buy=True, symbol=symbol)
                total_cost = (shares * execution_price) + commission
                
                if total_cost <= self.current_capital:
                    # Update position cost basis (weighted average)
                    old_position = current_position
                    new_position = old_position + shares
                    
                    if old_position > 0:
                        old_basis = self.position_cost_basis.get(symbol, execution_price)
                        new_basis = ((old_position * old_basis) + (shares * execution_price)) / new_position
                        self.position_cost_basis[symbol] = new_basis
                    else:
                        self.position_cost_basis[symbol] = execution_price
                    
                    self.positions[symbol] = new_position
                    self.current_capital -= total_cost
                    
                    self.trades.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': action,
                        'shares': shares,
                        'base_price': base_price,
                        'execution_price': execution_price,
                        'commission': commission,
                        'total_cost': total_cost,
                        'strength': signal['strength'],
                        'base_strength': signal.get('base_strength', signal['strength']),
                        'ml_multiplier': signal.get('ml_multiplier', 1.0),
                        'regime_boost': signal.get('regime_boost', 1.0),
                        'total_enhancement': signal.get('total_enhancement', 1.0),
                        'reason': decision['reason'],
                        'ml_enhanced': signal.get('ml_enhanced', False)
                    })
                    return True
            
            elif action in ['SELL_ALL', 'SELL_PARTIAL']:
                if shares <= current_position:
                    # Apply transaction costs
                    execution_price, commission = self._apply_transaction_costs(base_price, shares, is_buy=False, symbol=symbol)
                    gross_proceeds = shares * execution_price
                    net_proceeds = gross_proceeds - commission
                    
                    self.current_capital += net_proceeds
                    self.positions[symbol] = current_position - shares
                    
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                        if symbol in self.position_cost_basis:
                            del self.position_cost_basis[symbol]
                    
                    self.trades.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': action,
                        'shares': shares,
                        'base_price': base_price,
                        'execution_price': execution_price,
                        'commission': commission,
                        'gross_proceeds': gross_proceeds,
                        'net_proceeds': net_proceeds,
                        'strength': signal['strength'],
                        'base_strength': signal.get('base_strength', signal['strength']),
                        'ml_multiplier': signal.get('ml_multiplier', 1.0),
                        'regime_boost': signal.get('regime_boost', 1.0),
                        'total_enhancement': signal.get('total_enhancement', 1.0),
                        'reason': decision['reason'],
                        'ml_enhanced': signal.get('ml_enhanced', False)
                    })
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing decision for {symbol} on {current_date.date()}: {e}")
            return False
    
    def run_realistic_live_trading(self, start_date: str = "2025-05-22", end_date: str = "2025-08-21") -> Dict[str, Any]:
        """Run realistic live trading simulation with all fixes applied"""
        try:
            self.logger.info(f"🚀 Starting Realistic Live Trading: {start_date} to {end_date}")
            self.start_date = pd.to_datetime(start_date)
            self.end_date = pd.to_datetime(end_date)
            
            stocks = self.get_elite_stocks()
            
            # Create trading date range with improved calendar handling (Fix #8)
            try:
                # Use business day frequency for more robust calendar handling
                trading_dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
                
                # Convert to list for filtering
                trading_dates = trading_dates.tolist()
                
                # Additional filter for common US market holidays (basic)
                us_holidays = [
                    '2024-01-01',  # New Year's Day
                    '2024-01-15',  # MLK Day
                    '2024-02-19',  # Presidents Day
                    '2024-03-29',  # Good Friday
                    '2024-05-27',  # Memorial Day
                    '2024-06-19',  # Juneteenth
                    '2024-07-04',  # Independence Day
                    '2024-09-02',  # Labor Day
                    '2024-11-28',  # Thanksgiving
                    '2024-12-25',  # Christmas
                ]
                
                # Filter out holidays
                holiday_dates = [pd.to_datetime(h) for h in us_holidays]
                trading_dates = [d for d in trading_dates if d not in holiday_dates]
                
            except Exception as e:
                self.logger.warning(f"Calendar handling error, falling back to simple weekday filter: {e}")
                # Fallback to original method
                trading_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                trading_dates = [d for d in trading_dates if d.weekday() < 5]  # Trading days only
            
            total_days = len(trading_dates)
            
            # Pre-fetch all historical data to avoid refetching (Fix #3)
            self.logger.info("📊 Pre-fetching historical data for all symbols...")
            all_history = {}
            for symbol in stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    start_fetch = "2023-05-21"  # 2 years of training data
                    end_fetch = self.end_date.strftime('%Y-%m-%d')
                    
                    data = ticker.history(start=start_fetch, end=end_fetch)
                    if not data.empty:
                        if data.index.tz is not None:
                            data.index = data.index.tz_localize(None)
                        all_history[symbol] = data.dropna()
                        self.logger.info(f"   ✅ {symbol}: {len(data)} bars")
                    else:
                        self.logger.warning(f"   ❌ {symbol}: No data")
                except Exception as e:
                    self.logger.error(f"   ❌ {symbol}: {e}")
            
            # Filter trading dates to only those with available data (Fix #10 - Holiday handling)
            available_trading_dates = []
            for date in trading_dates:
                has_data = False
                for symbol in stocks:
                    if symbol in all_history and date in all_history[symbol].index:
                        has_data = True
                        break
                if has_data:
                    available_trading_dates.append(date)
            
            trading_dates = available_trading_dates
            total_days = len(trading_dates)
            self.logger.info(f"📅 Found {total_days} trading days with available data")
            
            for day_idx, current_date in enumerate(trading_dates):
                try:
                    self.logger.info(f"📅 Day {day_idx+1}/{total_days}: {current_date.date()}")
                    
                    # Enhancement #5: Trading Calendar Validation
                    if not self._is_valid_trading_day(current_date):
                        self.logger.info(f"   ⏭️ Skipping non-trading day: {current_date.date()}")
                        continue
                    
                    # Enhancement #6: Data Hygiene Checks
                    valid_symbols = self._perform_data_hygiene_checks(stocks, all_history, current_date)
                    if len(valid_symbols) < len(stocks) * 0.5:  # Less than 50% symbols have valid data
                        self.logger.warning(f"   ⚠️ Insufficient data quality: {len(valid_symbols)}/{len(stocks)} symbols valid")
                        continue
                    
                    daily_trades = 0
                    daily_ml_enhancements = 0
                    daily_new_positions = 0  # Track new positions today (Fix #20)
                    
                    # Execute pending orders from previous day at market open (Fix #2)
                    if self.pending_orders:
                        executed_orders = self._execute_pending_orders(current_date, all_history)
                        daily_trades += executed_orders
                        if executed_orders > 0:
                            self.logger.info(f"   📋 Executed {executed_orders} pending orders at market open")
                    
                    # Build daily price cache for consistent portfolio valuation (Fix #1)
                    prices_today = {}
                    for symbol in stocks:
                        if symbol in all_history:
                            symbol_data = all_history[symbol].loc[:current_date]
                            if not symbol_data.empty:
                                prices_today[symbol] = symbol_data['Close'].iloc[-1]
                    
                    self.daily_prices[current_date] = prices_today
                    
                    # Update drawdown throttling state (Enhancement #4)
                    current_portfolio_value = self._portfolio_value(prices_today)
                    self._update_drawdown_throttle(current_portfolio_value)
                    
                    # Apply enhanced risk management rules - queue as pending orders (Enhancement #2)
                    risk_orders_queued = self._apply_risk_rules(current_date, prices_today, all_history)
                    if risk_orders_queued > 0:
                        self.logger.info(f"   🛡️ Risk management: {risk_orders_queued} orders queued")
                    
                    # 1. Retrain ML models daily using only past data
                    for symbol in stocks:
                        if symbol in all_history:
                            symbol_data = all_history[symbol].loc[:current_date]
                            if len(symbol_data) >= self.config['min_training_days']:
                                self.train_daily_models(symbol, symbol_data, current_date)
                    
                    # 2. Generate signals and make trading decisions
                    for symbol in stocks:
                        try:
                            # Rate-limit & missing data resilience (Fix #18)
                            if symbol not in all_history:
                                self.logger.debug(f"   ⚠️ {symbol}: No historical data available")
                                continue
                            if symbol not in prices_today:
                                self.logger.debug(f"   ⚠️ {symbol}: No price data for today")
                                continue
                            
                            # Get data up to current date (no refetching)
                            symbol_data = all_history[symbol].loc[:current_date]
                            
                            if len(symbol_data) < 50:
                                continue
                            
                            # Calculate indicators
                            df = self.calculate_technical_indicators(symbol_data)
                            
                            # Generate base signal
                            base_signal = self.generate_base_signal(df)
                            
                            # Enhance with daily ML models
                            enhanced_signal = self.enhance_signal_with_daily_ml(symbol, base_signal, df, current_date)
                            
                            # Make human-like decision with correct portfolio valuation
                            decision = self.make_human_like_decision(symbol, enhanced_signal, prices_today, daily_new_positions)
                            
                            # Check if this would be a new position and increment counter
                            if decision['action'] == 'BUY' and self.positions.get(symbol, 0) == 0:
                                daily_new_positions += 1
                            
                            # Queue decision for next-day execution (Fix #2 - No same-bar trading)
                            if self._queue_pending_order(symbol, decision, enhanced_signal, current_date):
                                daily_trades += 1  # Count queued orders as "trades" for logging
                                if enhanced_signal.get('ml_enhanced', False):
                                    daily_ml_enhancements += 1
                                self.logger.info(f"   📝 Queued: {decision['action']} {decision['shares']} {symbol} @ ${enhanced_signal['price']:.2f} (close) → execute tomorrow @ open")
                            
                        except Exception as e:
                            self.logger.error(f"Error processing {symbol} on {current_date.date()}: {e}")
                            continue
                    
                    # 2.1 Portfolio rebalancing pass (Enhancement #1)
                    if not self._check_portfolio_exposure_limits(prices_today):
                        self.logger.warning(f"   🚫 Skipping rebalancing and utilization due to exposure limits")
                    else:
                        rebalance_orders = self._rebalance_portfolio(prices_today, current_date)
                        if rebalance_orders > 0:
                            self.logger.info(f"   ⚖️ Portfolio rebalancing: {rebalance_orders} orders queued")
                    
                    # 2.2 Capital utilization optimization (after rebalancing)
                    if self._check_portfolio_exposure_limits(prices_today):
                        signals_for_utilization = {}
                        for symbol in stocks:
                            try:
                                if symbol not in all_history or symbol not in prices_today:
                                    continue
                                
                                symbol_data = all_history[symbol].loc[:current_date]
                                if len(symbol_data) < 50:
                                    continue
                                
                                # Recalculate signals for utilization analysis
                                df = self.calculate_technical_indicators(symbol_data)
                                base_signal = self.generate_base_signal(df)
                                enhanced_signal = self.enhance_signal_with_daily_ml(symbol, base_signal, df, current_date)
                                signals_for_utilization[symbol] = enhanced_signal
                                
                            except Exception as e:
                                continue
                        
                        # Apply capital utilization optimization with enhanced position tracking
                        extra_orders, daily_new_positions = self._optimize_capital_utilization(
                            signals_for_utilization, prices_today, current_date, daily_new_positions
                        )
                        if extra_orders > 0:
                            self.logger.info(f"   💰 Capital utilization: +{extra_orders} extra orders queued")
                    
                    # 3. Track daily performance using consistent prices
                    total_value = self._portfolio_value(prices_today)
                    invested_ratio = self._get_current_invested_ratio(current_date, prices_today)
                    
                    self.daily_performance.append({
                        'date': current_date,
                        'total_value': total_value,
                        'cash': self.current_capital,
                        'positions': len(self.positions),
                        'daily_trades': daily_trades,
                        'ml_enhancements': daily_ml_enhancements,
                        'pending_orders': len(self.pending_orders),
                        'invested_ratio': invested_ratio,
                        'invested_ratio_pct': invested_ratio * 100
                    })
                    
                    # Memory cleanup: Remove old models to prevent growth (Fix #12)
                    if day_idx > 5:  # Keep last 5 days
                        old_date = trading_dates[day_idx - 5]
                        if old_date in self.daily_models:
                            del self.daily_models[old_date]
                        if old_date in self.daily_scalers:
                            del self.daily_scalers[old_date]
                    
                    # Memory cleanup: Prune model evolution tracking to prevent unbounded growth
                    max_evolution_records = 20  # Keep last 20 training records per symbol
                    for symbol in self.model_evolution:
                        if len(self.model_evolution[symbol]) > max_evolution_records:
                            # Keep only the most recent records
                            self.model_evolution[symbol] = self.model_evolution[symbol][-max_evolution_records:]
                    
                    # Enhancement #8: End-of-day comprehensive logging and metrics
                    self._log_end_of_day_metrics(current_date, day_idx, total_days, total_value, 
                                               daily_trades, daily_ml_enhancements, daily_new_positions)
                    
                    # Progress update
                    if (day_idx + 1) % 10 == 0:
                        daily_return = (total_value - self.initial_capital) / self.initial_capital * 100
                        self.logger.info(f"   Progress: {daily_return:+.1f}% | Positions: {len(self.positions)} | Queued orders: {len(self.pending_orders)} | Actions today: {daily_trades}")
                
                except Exception as e:
                    self.logger.error(f"Error on {current_date.date()}: {e}")
                    continue
            
            # Calculate final performance
            result = self.calculate_realistic_performance()
            
            # Save results to JSON for persistence (Fix #7)
            self.save_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in realistic live trading: {e}")
            return {"error": str(e)}
    
    def save_results(self, result: Dict[str, Any]) -> None:
        """Save results to JSON file for persistence (Fix #7)"""
        try:
            import json
            from datetime import datetime
            
            if 'error' in result:
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realistic_live_trading_results_{timestamp}.json"
            
            # Prepare data for JSON serialization
            save_data = {
                'timestamp': timestamp,
                'configuration': self.config,
                'performance_metrics': result,
                'trade_summary': {
                    'total_trades': len(self.trades),
                    'total_capital': self.current_capital,
                    'final_positions': len(self.positions),
                    'models_trained': len(self.model_evolution)
                },
                'daily_performance': self.daily_performance,
                'model_evolution': self.model_evolution
            }
            
            # Convert pandas timestamps to strings for JSON serialization
            for day_data in save_data['daily_performance']:
                if hasattr(day_data['date'], 'isoformat'):
                    day_data['date'] = day_data['date'].isoformat()
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def calculate_realistic_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.daily_performance:
                return {"error": "No performance data"}
            
            # Extract daily values
            daily_values = [d['total_value'] for d in self.daily_performance]
            dates = [d['date'] for d in self.daily_performance]
            
            # Final portfolio value
            final_value = daily_values[-1]
            
            # Calculate daily returns
            daily_returns = np.array([
                (daily_values[i] - daily_values[i-1]) / daily_values[i-1]
                for i in range(1, len(daily_values))
            ])
            
            # Performance metrics
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate period length
            start_date = dates[0]
            end_date = dates[-1]
            days = (end_date - start_date).days
            annual_return = (total_return + 1) ** (365.0 / days) - 1
            
            # Risk metrics - FIXED Sharpe calculation
            risk_free_rate = 0.02
            
            # Correct Sharpe calculation from daily returns
            daily_excess_returns = np.array(daily_returns) - (risk_free_rate / 252)  # Daily risk-free rate
            mu = np.mean(daily_excess_returns) * 252  # Annualized excess return
            sigma = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = mu / sigma if sigma > 0 else 0
            volatility = sigma  # Keep for compatibility
            
            # Maximum drawdown (calculate first for Calmar ratio)
            peak = self.initial_capital
            max_drawdown = 0
            for value in daily_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calmar ratio: Annualized return / Max drawdown (Fix #6)
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino ratio: Annualized return / Downside deviation (Fix #6)
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Trade statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.get('pnl_pct', 0) > 0])
            losing_trades = total_trades - winning_trades
            hit_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Average win/loss
            win_pnls = [t.get('pnl_pct', 0) for t in self.trades if t.get('pnl_pct', 0) > 0]
            loss_pnls = [t.get('pnl_pct', 0) for t in self.trades if t.get('pnl_pct', 0) < 0]
            
            avg_win = np.mean(win_pnls) if win_pnls else 0
            avg_loss = np.mean(loss_pnls) if loss_pnls else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # ML enhancement statistics
            ml_enhanced_trades = len([t for t in self.trades if t.get('ml_enhanced', False)])
            risk_triggered_trades = len([t for t in self.trades if t.get('risk_triggered', False)])
            
            # Enhanced ML statistics
            avg_ml_multiplier = 0.0
            avg_regime_boost = 0.0
            avg_total_enhancement = 0.0
            
            if ml_enhanced_trades > 0:
                ml_trades = [t for t in self.trades if t.get('ml_enhanced', False)]
                avg_ml_multiplier = np.mean([t.get('ml_multiplier', 1.0) for t in ml_trades])
                avg_regime_boost = np.mean([t.get('regime_boost', 1.0) for t in ml_trades])
                avg_total_enhancement = np.mean([t.get('total_enhancement', 1.0) for t in ml_trades])
            
            # Daily trading statistics
            avg_daily_trades = np.mean([d['daily_trades'] for d in self.daily_performance])
            avg_daily_ml = np.mean([d['ml_enhancements'] for d in self.daily_performance])
            
            # Model evolution statistics
            model_stats = {}
            for symbol in self.model_evolution:
                if self.model_evolution[symbol]:
                    accuracies = [m['regime_accuracy'] for m in self.model_evolution[symbol]]
                    strength_r2s = [m.get('strength_r2', 0.0) for m in self.model_evolution[symbol]]
                    model_stats[symbol] = {
                        'avg_accuracy': np.mean(accuracies),
                        'final_accuracy': accuracies[-1],
                        'avg_strength_r2': np.mean(strength_r2s),
                        'final_strength_r2': strength_r2s[-1] if strength_r2s else 0.0,
                        'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
                    }
            
            # Exposure metrics
            total_position_days = sum([d['positions'] for d in self.daily_performance])
            max_possible_days = len(self.daily_performance) * self.config['max_positions']
            portfolio_exposure = total_position_days / max_possible_days if max_possible_days > 0 else 0
            
            return {
                'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'days': days,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annual_return': annual_return,
                'annual_return_pct': annual_return * 100,
                'volatility': volatility,
                'volatility_pct': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'hit_rate': hit_rate,
                'hit_rate_pct': hit_rate * 100,
                'avg_win_pct': avg_win * 100,
                'avg_loss_pct': avg_loss * 100,
                'win_loss_ratio': win_loss_ratio,
                'ml_enhanced_trades': ml_enhanced_trades,
                'risk_triggered_trades': risk_triggered_trades,
                'ml_enhancement_rate': ml_enhanced_trades / total_trades * 100 if total_trades > 0 else 0,
                'avg_daily_trades': avg_daily_trades,
                'avg_daily_ml_enhancements': avg_daily_ml,
                'avg_ml_multiplier': avg_ml_multiplier,
                'avg_regime_boost': avg_regime_boost, 
                'avg_total_enhancement': avg_total_enhancement,
                'portfolio_exposure': portfolio_exposure,
                'portfolio_exposure_pct': portfolio_exposure * 100,
                'final_positions': len(self.positions),
                'models_trained': len(self.model_evolution),
                'model_stats': model_stats,
                
                # Capital Utilization Metrics
                'avg_invested_ratio': np.mean([d['invested_ratio'] for d in self.daily_performance]),
                'avg_invested_ratio_pct': np.mean([d['invested_ratio_pct'] for d in self.daily_performance]),
                'min_invested_ratio_pct': min([d['invested_ratio_pct'] for d in self.daily_performance]),
                'max_invested_ratio_pct': max([d['invested_ratio_pct'] for d in self.daily_performance]),
                'final_invested_ratio_pct': self.daily_performance[-1]['invested_ratio_pct'] if self.daily_performance else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}
    
    def generate_institutional_enhancement_summary(self) -> str:
        """Generate comprehensive summary of all 12 institutional enhancements (Enhancement #9)"""
        try:
            summary = []
            summary.append("=" * 80)
            summary.append("🏛️  INSTITUTIONAL-GRADE TRADING SYSTEM ENHANCEMENT SUMMARY")
            summary.append("=" * 80)
            
            # Enhancement #1: Portfolio Rebalancing System
            rebalance_count = self.enhancement_metrics.get('rebalancing_actions', 0)
            summary.append(f"\n✅ Enhancement #1: Portfolio Rebalancing System")
            summary.append(f"   • Threshold-based rebalancing with {rebalance_count} actions executed")
            summary.append(f"   • Target weight deviation threshold: ±5%")
            summary.append(f"   • Automatic portfolio optimization and drift correction")
            
            # Enhancement #2: Enhanced Risk Exits with Pending Orders
            risk_orders = self.enhancement_metrics.get('risk_orders_queued', 0)
            summary.append(f"\n✅ Enhancement #2: Enhanced Risk Exits with Pending Orders")
            summary.append(f"   • T+1 stop-loss and take-profit execution: {risk_orders} orders queued")
            summary.append(f"   • Gap detection and price validation for risk orders")
            summary.append(f"   • Institutional-grade order management system")
            
            # Enhancement #3: Daily New Position Cap Enforcement
            cap_orders = self.enhancement_metrics.get('capital_utilization_orders', 0)
            summary.append(f"\n✅ Enhancement #3: Daily New Position Cap Enforcement")
            summary.append(f"   • Dynamic position limits with {cap_orders} capital utilization orders")
            summary.append(f"   • Target invested ratio: {self.config['target_invested_floor']:.0%}-{self.config['target_invested_ceiling']:.0%}")
            summary.append(f"   • Exceptional signal position sizing (up to {self.config['max_position_size_exceptional']:.0%})")
            
            # Enhancement #4: Portfolio-Level Exposure Guards with Drawdown Throttling
            throttle_activations = self.enhancement_metrics.get('drawdown_throttle_activations', 0)
            summary.append(f"\n✅ Enhancement #4: Portfolio-Level Exposure Guards")
            summary.append(f"   • Maximum net exposure: {self.config['max_net_exposure']:.0%}")
            summary.append(f"   • Drawdown throttling: {throttle_activations} activations")
            summary.append(f"   • Adaptive position sizing during adverse conditions")
            
            # Enhancement #5: Trading Calendar Integration
            validated_days = self.enhancement_metrics.get('trading_days_validated', 0)
            summary.append(f"\n✅ Enhancement #5: Trading Calendar Integration")
            summary.append(f"   • Market holiday validation: {validated_days} trading days validated")
            summary.append(f"   • Extended US market holiday calendar (2024-2025)")
            summary.append(f"   • Automated non-trading day detection")
            
            # Enhancement #6: Data Hygiene and Validation
            hygiene_rejections = self.enhancement_metrics.get('data_hygiene_rejections', 0)
            summary.append(f"\n✅ Enhancement #6: Data Hygiene and Validation")
            summary.append(f"   • Comprehensive price data validation: {hygiene_rejections} rejections")
            summary.append(f"   • Volume, price range, and volatility checks")
            summary.append(f"   • Outlier detection and data quality assurance")
            
            # Enhancement #7: Enhanced Slippage Model
            slippage_calcs = self.enhancement_metrics.get('enhanced_slippage_calculations', 0)
            summary.append(f"\n✅ Enhancement #7: Enhanced Slippage Model")
            summary.append(f"   • Multi-factor slippage calculation: {slippage_calcs} calculations")
            summary.append(f"   • Order size, volatility, and market cap impacts")
            summary.append(f"   • Market microstructure noise simulation")
            
            # Enhancement #8: End-of-Day Comprehensive Logging
            eod_summaries = self.enhancement_metrics.get('end_of_day_summaries', 0)
            summary.append(f"\n✅ Enhancement #8: End-of-Day Comprehensive Logging")
            summary.append(f"   • Daily performance summaries: {eod_summaries} reports generated")
            summary.append(f"   • Weekly detailed analysis with position breakdowns")
            summary.append(f"   • Real-time progress tracking and risk monitoring")
            
            # Enhancement #9: State Persistence and Resumability
            state_saves = self.enhancement_metrics.get('state_saves', 0)
            summary.append(f"\n✅ Enhancement #9: State Persistence and Resumability")
            summary.append(f"   • Comprehensive state tracking: {state_saves} saves completed")
            summary.append(f"   • JSON serialization with timestamp metadata")
            summary.append(f"   • Full model evolution and performance persistence")
            
            # Enhancement #10-12: Additional Infrastructure
            summary.append(f"\n✅ Enhancements #10-12: Advanced Infrastructure")
            summary.append(f"   • Memory management and cleanup systems")
            summary.append(f"   • Model evolution tracking with performance metrics")
            summary.append(f"   • Institutional-grade error handling and recovery")
            
            # System Status Summary
            summary.append(f"\n🎯 INSTITUTIONAL SYSTEM STATUS:")
            summary.append(f"   • Total Enhancement Actions: {sum(self.enhancement_metrics.values())}")
            summary.append(f"   • Active Models: {len(self.daily_models)}")
            summary.append(f"   • Current Positions: {len(self.positions)}")
            summary.append(f"   • Pending Orders: {len(self.pending_orders)}")
            
            if self.drawdown_throttle_remaining > 0:
                summary.append(f"   • Risk Status: THROTTLE ACTIVE ({self.drawdown_throttle_remaining} sessions)")
            else:
                summary.append(f"   • Risk Status: NORMAL OPERATIONS")
            
            # Performance vs Baseline
            if self.daily_performance and len(self.daily_performance) > 1:
                total_return = (self.daily_performance[-1]['total_value'] - self.initial_capital) / self.initial_capital
                summary.append(f"   • Current Return: {total_return:+.2%}")
            
            summary.append("\n" + "=" * 80)
            summary.append("🏆 INSTITUTIONAL-GRADE TRADING SYSTEM FULLY OPERATIONAL")
            summary.append("=" * 80)
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error generating enhancement summary: {e}"

def main():
    """Test the Realistic Live Trading System"""
    print("🚀 ENHANCED REALISTIC LIVE TRADING SIMULATOR")
    print("Daily ML Training + Advanced Signal Enhancement + Human-like Decisions")
    print("=" * 60)
    
    # Initialize system
    system = RealisticLiveTradingSystem(initial_capital=100000.0)
    
    # Run 3-month realistic simulation
    print("📈 Running 3-month enhanced realistic live trading simulation...")
    print("• ML models retrained DAILY")
    print("• ENHANCED: LightGBM signal strength models")
    print("• ENHANCED: ML multiplier system (0.3-1.0x)")
    print("• Human-like decisions: buy more, hold, partial sell, full sell")
    print("• NO look-ahead bias - only past data used")
    print("• Real-time portfolio rebalancing")
    
    result = system.run_realistic_live_trading(
        start_date="2025-05-22",
        end_date="2025-08-21"
    )
    
    if "error" not in result:
        print(f"\n✅ Realistic Live Trading Results:")
        print(f"   Period:                 {result['period']}")
        print(f"   Initial Capital:        ${result['initial_capital']:,.0f}")
        print(f"   Final Value:            ${result['final_value']:,.0f}")
        print(f"   Total Return:           {result['total_return_pct']:+.1f}%")
        print(f"   Annual Return:          {result['annual_return_pct']:+.1f}%")
        print(f"   Volatility:             {result['volatility_pct']:.1f}%")
        print(f"   Sharpe Ratio:           {result['sharpe_ratio']:.2f}")
        print(f"   Calmar Ratio:           {result['calmar_ratio']:.2f}")
        print(f"   Sortino Ratio:          {result['sortino_ratio']:.2f}")
        print(f"   Max Drawdown:           {result['max_drawdown_pct']:.1f}%")
        print(f"   Total Trades:           {result['total_trades']}")
        print(f"   Hit Rate:               {result['hit_rate_pct']:.1f}%")
        print(f"   Win/Loss Ratio:         {result['win_loss_ratio']:.2f}")
        print(f"   Risk Triggered:         {result['risk_triggered_trades']}")
        print(f"   ML Enhanced Trades:     {result['ml_enhanced_trades']} ({result['ml_enhancement_rate']:.1f}%)")
        print(f"   Avg ML Multiplier:      {result['avg_ml_multiplier']:.2f}x")
        print(f"   Avg Regime Boost:       {result['avg_regime_boost']:.2f}x") 
        print(f"   Avg Total Enhancement:  {result['avg_total_enhancement']:.2f}x")
        print(f"   Portfolio Exposure:     {result['portfolio_exposure_pct']:.1f}%")
        print(f"   Final Positions:        {result['final_positions']}")
        print(f"   Models Trained:         {result['models_trained']}")
        
        print(f"\n💰 Capital Utilization:")
        print(f"   Average Invested:       {result['avg_invested_ratio_pct']:.1f}%")
        print(f"   Final Invested:         {result['final_invested_ratio_pct']:.1f}%")
        print(f"   Range:                  {result['min_invested_ratio_pct']:.1f}% - {result['max_invested_ratio_pct']:.1f}%")
        print(f"   Target Floor/Ceiling:   {system.config['target_invested_floor']*100:.0f}% - {system.config['target_invested_ceiling']*100:.0f}%")
        
        print(f"\n🤖 Model Performance (Top 5):")
        model_stats = result['model_stats']
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)[:5]
        for symbol, stats in sorted_models:
            print(f"   {symbol}: {stats['avg_accuracy']:.3f} regime acc | {stats['avg_strength_r2']:.3f} strength R²")
        
        # Compare to previous systems
        print(f"\n🎯 Performance Comparison:")
        print(f"   Baseline Kelly System:  49.8% annual")
        print(f"   Enhanced ML System:     91.2% annual")
        print(f"   Realistic Live System:  {result['annual_return_pct']:+.1f}% annual")
        
        if result['annual_return_pct'] > 60:
            print("🎉 Excellent! Realistic system maintains strong performance!")
        elif result['annual_return_pct'] > 30:
            print("✅ Good performance with realistic constraints")
        else:
            print("⚠️ Performance impacted by realistic daily training")
            
    else:
        print(f"❌ Error: {result['error']}")
    
    print("\n🔬 Enhanced Realistic Trading Features:")
    print("• Daily ML model retraining (like real trading)")
    print("• ENHANCED: LightGBM signal strength prediction")
    print("• ENHANCED: ML multiplier system (0.3-1.0x)")
    print("• NEW: Intelligent capital utilization (80-90% target)")
    print("• NEW: Dynamic position sizing with exceptional limits")
    print("• Human-like position management") 
    print("• No future data exposure")
    print("• Portfolio rebalancing")
    print("• Advanced signal enhancement")
    print("• Comprehensive risk management")

if __name__ == "__main__":
    main()
