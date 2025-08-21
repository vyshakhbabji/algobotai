#!/usr/bin/env python3
"""
COMPLETE AUTO TRADER SYSTEM
Elite AI v2.0 + Position Sizing + Risk Management + Alpaca Integration + Performance Monitoring
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Core AI Engine
from elite_ai_trader import EliteAITrader
from algobot.config import GLOBAL_CONFIG

# Alpaca Integration (will work when Alpaca is properly configured)
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️ Alpaca not available - using simulation mode")

class PositionSizer:
    """Advanced position sizing with multiple strategies"""
    
    def __init__(self, account_size: float = 100000):
        self.account_size = account_size
        
    def calculate_position_size(self, 
                              symbol: str,
                              signal: str, 
                              confidence: float,
                              current_price: float,
                              strategy: str = 'kelly') -> Dict:
        """Calculate position size based on multiple strategies"""
        
        if signal == 'HOLD':
            return {'shares': 0, 'value': 0, 'reason': 'HOLD signal'}
        
        # Risk per trade (2% max)
        max_risk_per_trade = self.account_size * 0.02
        
        # Confidence-based sizing
        confidence_multiplier = min(confidence / 0.6, 1.0)  # Scale confidence
        
        if strategy == 'fixed_percent':
            # Fixed percentage of account
            base_allocation = self.account_size * 0.1  # 10% base
            position_value = base_allocation * confidence_multiplier
            
        elif strategy == 'kelly':
            # Simplified Kelly Criterion
            # Assuming win rate = confidence, average win = 5%, average loss = -2%
            win_rate = confidence
            avg_win = 0.05
            avg_loss = 0.02
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            position_value = self.account_size * kelly_fraction
            
        elif strategy == 'volatility_adjusted':
            # Adjust for implied volatility (simplified)
            base_allocation = self.account_size * 0.1
            volatility_factor = 1.0  # Could calculate from price data
            position_value = base_allocation * confidence_multiplier / volatility_factor
            
        else:  # Conservative default
            position_value = self.account_size * 0.05 * confidence_multiplier
        
        # Apply maximum position limits (centralized)
        max_position = self.account_size * float(GLOBAL_CONFIG.risk.max_position_pct)  # e.g., 0.15 => 15%
        position_value = min(position_value, max_position)
        
        # Calculate shares
        shares = int(position_value / current_price)
        actual_value = shares * current_price
        
        return {
            'shares': shares,
            'value': actual_value,
            'percentage': (actual_value / self.account_size) * 100,
            'strategy': strategy,
            'confidence_multiplier': confidence_multiplier,
            'max_risk': max_risk_per_trade,
            'reason': f'{strategy} sizing with {confidence:.1%} confidence'
        }

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self):
        # Total gross exposure cap as a fraction of equity (e.g., 0.60 => 60%)
        self.max_portfolio_risk = float(GLOBAL_CONFIG.risk.max_portfolio_risk_pct)
        self.max_correlation_exposure = 0.40  # Max 40% in correlated assets
        self.max_sector_exposure = 0.30  # Max 30% per sector
        self.stop_loss_percent = float(GLOBAL_CONFIG.risk.stop_loss_pct)  # 8% stop loss
        self.take_profit_percent = float(GLOBAL_CONFIG.risk.take_profit_pct)  # 15% take profit
        
    def validate_trade(self, 
                      symbol: str,
                      position_size: Dict,
                      current_positions: Dict,
                      sector_map: Dict = None) -> Dict:
        """Validate if trade meets risk management criteria"""
        
        validation_result = {
            'approved': True,
            'reasons': [],
            'warnings': [],
            'adjustments': {}
        }
        
        # Check position size limits (percentage points)
        per_name_cap_pct = float(GLOBAL_CONFIG.risk.max_position_pct) * 100.0
        if position_size['percentage'] > per_name_cap_pct:
            validation_result['approved'] = False
            validation_result['reasons'].append(
                f"Position too large: {position_size['percentage']:.1f}% > {per_name_cap_pct:.0f}%"
            )
        
        # Check total portfolio exposure
        current_exposure = sum(pos.get('percentage', 0) for pos in current_positions.values())
        new_exposure = current_exposure + position_size['percentage']
        
        if new_exposure > self.max_portfolio_risk * 100:
            validation_result['approved'] = False
            validation_result['reasons'].append(f"Portfolio exposure too high: {new_exposure:.1f}%")
        
        # Check if we already have position in this stock
        if symbol in current_positions:
            validation_result['warnings'].append(f"Already have position in {symbol}")
        
        # Sector exposure check (if sector map provided)
        if sector_map and symbol in sector_map:
            sector = sector_map[symbol]
            sector_exposure = sum(
                pos.get('percentage', 0) 
                for sym, pos in current_positions.items() 
                if sector_map.get(sym) == sector
            )
            
            if sector_exposure + position_size['percentage'] > self.max_sector_exposure * 100:
                validation_result['warnings'].append(f"High {sector} sector exposure")
        
        return validation_result
    
    def calculate_stop_loss(self, entry_price: float, signal: str) -> float:
        """Calculate stop loss price"""
        if signal == 'BUY':
            return entry_price * (1 - self.stop_loss_percent)
        elif signal == 'SELL':
            return entry_price * (1 + self.stop_loss_percent)
        return entry_price
    
    def calculate_take_profit(self, entry_price: float, signal: str) -> float:
        """Calculate take profit price"""
        if signal == 'BUY':
            return entry_price * (1 + self.take_profit_percent)
        elif signal == 'SELL':
            return entry_price * (1 - self.take_profit_percent)
        return entry_price

class PerformanceMonitor:
    """Performance tracking and analysis"""
    
    def __init__(self, log_file: str = 'auto_trader_performance.json'):
        self.log_file = log_file
        self.trades = []
        self.positions = {}
        self.load_performance_data()
        
    def load_performance_data(self):
        """Load existing performance data"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.trades = data.get('trades', [])
                self.positions = data.get('positions', {})
        except FileNotFoundError:
            self.trades = []
            self.positions = {}
    
    def save_performance_data(self):
        """Save performance data"""
        data = {
            'trades': self.trades,
            'positions': self.positions,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_trade(self, trade_data: Dict):
        """Log a new trade"""
        trade_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_data['symbol'],
            'signal': trade_data['signal'],
            'shares': trade_data['shares'],
            'price': trade_data['price'],
            'value': trade_data['value'],
            'confidence': trade_data['confidence'],
            'ai_prediction': trade_data['prediction'],
            'stop_loss': trade_data.get('stop_loss'),
            'take_profit': trade_data.get('take_profit')
        }
        self.trades.append(trade_entry)
        self.save_performance_data()
    
    def update_position(self, symbol: str, position_data: Dict):
        """Update position data"""
        self.positions[symbol] = position_data
        self.save_performance_data()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'error': 'No trades to analyze'}
        
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        buy_trades = len(df[df['signal'] == 'BUY'])
        sell_trades = len(df[df['signal'] == 'SELL'])
        
        # P&L calculation (simplified - would need exit prices for real calculation)
        total_value_traded = df['value'].sum()
        avg_confidence = df['confidence'].mean()
        avg_prediction = df['ai_prediction'].mean()
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_value_traded': total_value_traded,
            'average_confidence': avg_confidence,
            'average_ai_prediction': avg_prediction,
            'trade_frequency': 'Daily' if total_trades > 0 else 'None'
        }

class CompleteAutoTrader:
    """Complete automated trading system"""
    
    def __init__(self, 
                 account_size: float = 100000,
                 paper_trading: bool = True,
                 alpaca_config: Dict = None):
        
        # Core components
        self.ai_engine = EliteAITrader()
        self.position_sizer = PositionSizer(account_size)
        self.risk_manager = RiskManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Trading parameters
        self.account_size = account_size
        self.paper_trading = paper_trading
        self.current_positions = {}
        
        # Alpaca integration
        self.alpaca_client = None
        if ALPACA_AVAILABLE and alpaca_config:
            try:
                self.alpaca_client = TradingClient(
                    api_key=alpaca_config['api_key'],
                    secret_key=alpaca_config['secret_key'],
                    paper=paper_trading
                )
                print("✅ Alpaca client initialized")
            except Exception as e:
                print(f"⚠️ Alpaca initialization failed: {e}")
        
        # Sector mapping for risk management
        self.sector_map = {
            'NVDA': 'Technology', 'AAPL': 'Technology', 'GOOGL': 'Technology',
            'MSFT': 'Technology', 'META': 'Technology', 'TSLA': 'Technology',
            'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance',
            'WMT': 'Consumer', 'KO': 'Consumer', 'PG': 'Consumer',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy'
        }
    
    def screen_stocks(self, stock_universe: List[str]) -> List[Dict]:
        """Screen stocks for trading opportunities"""
        
        print(f"🔍 SCREENING {len(stock_universe)} STOCKS FOR TRADING OPPORTUNITIES")
        print("=" * 60)
        
        opportunities = []
        
        for symbol in stock_universe:
            try:
                print(f"\n📊 Screening {symbol}...")
                
                # Train AI models
                training_success = self.ai_engine.train_simple_models(symbol)
                
                if training_success:
                    # Validate prediction quality
                    quality = self.ai_engine.validate_prediction_quality(symbol)
                    
                    if quality in ["🟢 GOOD", "🟡 FAIR"]:
                        # Get trading prediction
                        prediction = self.ai_engine.predict_stock(symbol)
                        
                        if prediction and prediction['signal'] in ['BUY', 'SELL']:
                            # Get current price (simplified)
                            import yfinance as yf
                            ticker = yf.Ticker(symbol)
                            current_price = ticker.info.get('currentPrice', 100)
                            
                            opportunity = {
                                'symbol': symbol,
                                'signal': prediction['signal'],
                                'predicted_return': prediction['predicted_return'],
                                'confidence': prediction['confidence'],
                                'quality': quality.split()[-1],
                                'current_price': current_price,
                                'ai_prediction': prediction
                            }
                            
                            opportunities.append(opportunity)
                            print(f"   ✅ {symbol}: {prediction['signal']} opportunity found!")
                        else:
                            print(f"   ⚪ {symbol}: HOLD signal")
                    else:
                        print(f"   ❌ {symbol}: Quality too poor")
                else:
                    print(f"   ❌ {symbol}: Training failed")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {str(e)}")
        
        print(f"\n🎯 SCREENING COMPLETE: {len(opportunities)} opportunities found")
        return opportunities
    
    def execute_trading_decision(self, opportunity: Dict) -> Dict:
        """Execute a trading decision with full risk management"""
        
        symbol = opportunity['symbol']
        signal = opportunity['signal']
        confidence = opportunity['confidence']
        current_price = opportunity['current_price']
        
        print(f"\n🚀 EXECUTING TRADE DECISION FOR {symbol}")
        print("-" * 40)
        
        # Step 1: Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            symbol, signal, confidence, current_price, strategy='kelly'
        )
        
        print(f"📊 Position Size: {position_size['shares']} shares (${position_size['value']:,.2f})")
        print(f"📈 Allocation: {position_size['percentage']:.1f}% of portfolio")
        
        # Step 2: Risk management validation
        risk_validation = self.risk_manager.validate_trade(
            symbol, position_size, self.current_positions, self.sector_map
        )
        
        if not risk_validation['approved']:
            print(f"❌ TRADE REJECTED: {', '.join(risk_validation['reasons'])}")
            return {'status': 'rejected', 'reasons': risk_validation['reasons']}
        
        if risk_validation['warnings']:
            print(f"⚠️ WARNINGS: {', '.join(risk_validation['warnings'])}")
        
        # Step 3: Calculate stop loss and take profit
        stop_loss = self.risk_manager.calculate_stop_loss(current_price, signal)
        take_profit = self.risk_manager.calculate_take_profit(current_price, signal)
        
        print(f"🛡️ Stop Loss: ${stop_loss:.2f}")
        print(f"🎯 Take Profit: ${take_profit:.2f}")
        
        # Step 4: Execute trade
        execution_result = self._execute_order(
            symbol, signal, position_size['shares'], current_price
        )
        
        if execution_result['status'] == 'executed':
            # Step 5: Log trade and update positions
            trade_data = {
                'symbol': symbol,
                'signal': signal,
                'shares': position_size['shares'],
                'price': current_price,
                'value': position_size['value'],
                'confidence': confidence,
                'prediction': opportunity['predicted_return'],
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            self.performance_monitor.log_trade(trade_data)
            
            # Update position tracking
            self.current_positions[symbol] = {
                'shares': position_size['shares'],
                'entry_price': current_price,
                'value': position_size['value'],
                'percentage': position_size['percentage'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'entry_date': datetime.now().isoformat()
            }
            
            self.performance_monitor.update_position(symbol, self.current_positions[symbol])
            
            print(f"✅ TRADE EXECUTED SUCCESSFULLY!")
            
        return execution_result
    
    def _execute_order(self, symbol: str, signal: str, shares: int, price: float) -> Dict:
        """Execute order via Alpaca or simulation"""
        
        if self.alpaca_client and not self.paper_trading:
            # Real Alpaca execution
            try:
                order_side = OrderSide.BUY if signal == 'BUY' else OrderSide.SELL
                
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.alpaca_client.submit_order(order_request)
                
                return {
                    'status': 'executed',
                    'order_id': order.id,
                    'method': 'alpaca_live'
                }
                
            except Exception as e:
                print(f"❌ Alpaca execution failed: {e}")
                return {'status': 'failed', 'error': str(e)}
        
        else:
            # Simulation mode
            print(f"📝 SIMULATED: {signal} {shares} shares of {symbol} at ${price:.2f}")
            return {
                'status': 'executed',
                'order_id': f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'method': 'simulation'
            }
    
    def run_trading_session(self, stock_universe: List[str]) -> Dict:
        """Run a complete trading session"""
        
        print("🤖 ELITE AI v2.0 AUTO TRADER - TRADING SESSION")
        print("=" * 60)
        print(f"📊 Account Size: ${self.account_size:,}")
        print(f"🎯 Universe: {len(stock_universe)} stocks")
        print(f"⚡ Mode: {'Paper Trading' if self.paper_trading else 'Live Trading'}")
        print("=" * 60)
        
        session_start = datetime.now()
        
        # Step 1: Screen for opportunities
        opportunities = self.screen_stocks(stock_universe)
        
        if not opportunities:
            print("\n📊 NO TRADING OPPORTUNITIES FOUND")
            return {'status': 'no_opportunities', 'opportunities': 0}
        
        # Step 2: Rank opportunities by confidence * predicted return
        opportunities.sort(
            key=lambda x: x['confidence'] * abs(x['predicted_return']), 
            reverse=True
        )
        
        print(f"\n🎯 TOP TRADING OPPORTUNITIES:")
        print("-" * 30)
        for i, opp in enumerate(opportunities[:5], 1):
            print(f"{i}. {opp['symbol']}: {opp['signal']} "
                  f"({opp['predicted_return']*100:+.1f}%, {opp['confidence']:.1%} confidence)")
        
        # Step 3: Execute trades for top opportunities
        executed_trades = []
        max_trades_per_session = 3  # Limit trades per session
        
        for opportunity in opportunities[:max_trades_per_session]:
            result = self.execute_trading_decision(opportunity)
            if result['status'] == 'executed':
                executed_trades.append(opportunity['symbol'])
        
        # Step 4: Session summary
        session_end = datetime.now()
        session_duration = session_end - session_start
        
        print(f"\n📊 TRADING SESSION COMPLETE")
        print("=" * 30)
        print(f"⏱️ Duration: {session_duration}")
        print(f"🎯 Opportunities Found: {len(opportunities)}")
        print(f"✅ Trades Executed: {len(executed_trades)}")
        print(f"📈 Stocks Traded: {', '.join(executed_trades) if executed_trades else 'None'}")
        
        # Performance metrics
        performance = self.performance_monitor.calculate_performance_metrics()
        print(f"📊 Total Trades: {performance.get('total_trades', 0)}")
        print(f"📈 Average Confidence: {performance.get('average_confidence', 0):.1%}")
        
        return {
            'status': 'completed',
            'opportunities_found': len(opportunities),
            'trades_executed': len(executed_trades),
            'symbols_traded': executed_trades,
            'session_duration': str(session_duration),
            'performance': performance
        }

def main():
    """Demo of complete auto trading system"""
    
    # Configuration
    DEMO_ACCOUNT_SIZE = 100000  # $100K demo account
    
    # Stock universe for testing
    tech_universe = ["NVDA", "AAPL", "GOOGL", "TSLA", "AMZN"]
    
    # Initialize auto trader
    auto_trader = CompleteAutoTrader(
        account_size=DEMO_ACCOUNT_SIZE,
        paper_trading=True  # Safe demo mode
    )
    
    # Run trading session
    session_result = auto_trader.run_trading_session(tech_universe)
    
    print(f"\n" + "="*60)
    print(f"🎯 AUTO TRADER DEMO COMPLETE!")
    print(f"   ✅ Elite AI v2.0 ensemble models")
    print(f"   ✅ Advanced position sizing (Kelly Criterion)")
    print(f"   ✅ Comprehensive risk management")
    print(f"   ✅ Performance monitoring & logging")
    print(f"   ✅ Alpaca API integration ready")
    print(f"   ✅ Fully automated trading pipeline")
    print(f"="*60)
    
    return session_result

if __name__ == "__main__":
    main()
