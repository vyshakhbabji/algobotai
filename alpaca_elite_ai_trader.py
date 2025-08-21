#!/usr/bin/env python3
"""
ALPACA ELITE AI INTEGRATION
Updated Alpaca system to use Elite AI v2.0 as core decision engine
"""

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca SDK not installed. Install with: pip install alpaca-py")

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from elite_ai_trader import EliteAITrader
from complete_auto_trader import CompleteAutoTrader

class AlpacaEliteAITrader:
    """
    Alpaca trading system powered by Elite AI v2.0
    - Uses research-grade AI for all trading decisions
    - Quality gates prevent bad trades
    - Ensemble consensus for reliability
    """
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        # Check if Alpaca is available
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca SDK not installed. Install with: pip install alpaca-py")
        
        # Initialize Alpaca API
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper_trading
        )
        
        # Initialize Elite AI v2.0 - Our research-grade brain
        self.ai_engine = EliteAITrader()
        
        # Initialize complete auto trader for position sizing and risk management
        account = self.trading_client.get_account()
        self.auto_trader = CompleteAutoTrader(
            account_size=float(account.buying_power),
            paper_trading=paper_trading
        )
        
        self.paper_trading = paper_trading
        self.trading_log = []
        
    def ai_powered_stock_screening(self, universe: List[str]) -> List[Dict]:
        """Screen stocks using Elite AI v2.0 quality assessment"""
        
        print("🤖 AI-POWERED ALPACA STOCK SCREENING")
        print("=" * 45)
        print("Using Elite AI v2.0 for intelligent trade selection")
        
        approved_trades = []
        
        for symbol in universe:
            print(f"\n📊 Screening {symbol} with Elite AI v2.0:")
            
            try:
                # Get AI prediction with quality assessment
                result = self.ai_engine.predict_stock(symbol)
                
                if result and result.get('action') != 'NO_ACTION':
                    confidence = result['confidence']
                    action = result['action']
                    
                    # Only proceed if AI is confident enough
                    if confidence > 0.55:  # Conservative threshold
                        # Get current price for position sizing
                        current_price = self.get_current_price(symbol)
                        
                        if current_price:
                            # Calculate position size using Kelly Criterion
                            position_size = self.auto_trader.position_sizer.calculate_position_size(
                                symbol=symbol,
                                signal_confidence=confidence,
                                current_price=current_price
                            )
                            
                            # Risk validation
                            risk_approved = self.auto_trader.risk_manager.validate_trade(
                                symbol=symbol,
                                action=action,
                                position_value=position_size * current_price
                            )
                            
                            if risk_approved:
                                approved_trades.append({
                                    'symbol': symbol,
                                    'action': action,
                                    'confidence': confidence,
                                    'current_price': current_price,
                                    'position_size': position_size,
                                    'position_value': position_size * current_price,
                                    'ai_quality': 'APPROVED'
                                })
                                
                                print(f"   ✅ TRADE APPROVED: {action} {position_size} shares")
                                print(f"      Confidence: {confidence:.1%}")
                                print(f"      Position Value: ${position_size * current_price:,.2f}")
                            else:
                                print(f"   ❌ RISK REJECTED: Position fails risk management")
                        else:
                            print(f"   ❌ PRICE ERROR: Cannot get current price")
                    else:
                        print(f"   ⚠️  AI CAUTIOUS: Confidence too low ({confidence:.1%})")
                else:
                    print(f"   ❌ AI REFUSES: Quality too poor for trading")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)[:50]}")
        
        print(f"\n🎯 AI SCREENING SUMMARY:")
        print(f"   Stocks Analyzed: {len(universe)}")
        print(f"   AI Approved: {len(approved_trades)}")
        print(f"   Approval Rate: {len(approved_trades)/len(universe)*100:.0f}%")
        
        return approved_trades
    
    def execute_ai_approved_trades(self, approved_trades: List[Dict]) -> List[Dict]:
        """Execute trades approved by Elite AI v2.0"""
        
        print(f"\n🚀 EXECUTING AI-APPROVED TRADES VIA ALPACA")
        print("=" * 50)
        
        executed_trades = []
        
        for trade in approved_trades:
            symbol = trade['symbol']
            action = trade['action']
            quantity = trade['position_size']
            
            try:
                print(f"\n📈 Executing {action} {quantity} {symbol}:")
                
                # Determine order side
                side = 'buy' if action == 'BUY' else 'sell'
                
                # Place market order
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                
                print(f"   ✅ Order Placed: {order.id}")
                print(f"   📊 Status: {order.status}")
                
                # Log the trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_id': order.id,
                    'ai_confidence': trade['confidence'],
                    'position_value': trade['position_value'],
                    'status': 'EXECUTED'
                }
                
                executed_trades.append(trade_record)
                self.trading_log.append(trade_record)
                
                # Update performance monitor
                self.auto_trader.performance_monitor.log_trade(trade_record)
                
            except Exception as e:
                print(f"   ❌ EXECUTION FAILED: {str(e)}")
                
                # Log failed trade
                failed_trade = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'error': str(e),
                    'status': 'FAILED'
                }
                
                executed_trades.append(failed_trade)
                self.trading_log.append(failed_trade)
        
        return executed_trades
    
    def run_ai_trading_session(self, universe: List[str]) -> Dict:
        """Run complete AI-powered trading session"""
        
        print("🔥 ALPACA + ELITE AI v2.0 TRADING SESSION")
        print("=" * 50)
        print("Research-grade AI powering real market execution")
        
        # Get account info
        account = self.api.get_account()
        print(f"\n💰 Account Status:")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Paper Trading: {self.paper_trading}")
        
        # Step 1: AI-powered screening
        approved_trades = self.ai_powered_stock_screening(universe)
        
        if not approved_trades:
            print(f"\n⚠️  NO TRADES APPROVED")
            print(f"Elite AI v2.0 found no suitable opportunities")
            print(f"This is GOOD - better to wait than trade poor setups!")
            return {'trades_executed': 0, 'message': 'AI_CONSERVATIVE'}
        
        # Step 2: Execute approved trades
        executed_trades = self.execute_ai_approved_trades(approved_trades)
        
        # Step 3: Summary
        successful_trades = [t for t in executed_trades if t['status'] == 'EXECUTED']
        failed_trades = [t for t in executed_trades if t['status'] == 'FAILED']
        
        print(f"\n🎯 TRADING SESSION SUMMARY:")
        print(f"   Successful Trades: {len(successful_trades)}")
        print(f"   Failed Trades: {len(failed_trades)}")
        print(f"   Total Value Traded: ${sum(t.get('position_value', 0) for t in successful_trades):,.2f}")
        
        # Performance update
        performance = self.auto_trader.performance_monitor.get_performance_summary()
        
        session_summary = {
            'timestamp': datetime.now(),
            'trades_executed': len(successful_trades),
            'trades_failed': len(failed_trades),
            'total_value': sum(t.get('position_value', 0) for t in successful_trades),
            'ai_engine': 'Elite_AI_v2.0',
            'performance': performance
        }
        
        return session_summary
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        try:
            quote = self.api.get_latest_quote(symbol)
            return float(quote.bid_price + quote.ask_price) / 2
        except:
            return None
    
    def get_positions(self) -> List[Dict]:
        """Get current positions with AI analysis"""
        
        positions = self.api.list_positions()
        
        ai_analyzed_positions = []
        
        for position in positions:
            symbol = position.symbol
            
            # Get AI analysis of current position
            ai_result = self.ai_engine.predict_stock(symbol)
            
            position_data = {
                'symbol': symbol,
                'quantity': float(position.qty),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'ai_current_signal': ai_result.get('action', 'NO_ACTION') if ai_result else 'NO_ACTION',
                'ai_confidence': ai_result.get('confidence', 0) if ai_result else 0
            }
            
            ai_analyzed_positions.append(position_data)
        
        return ai_analyzed_positions
    
    def ai_portfolio_rebalancing(self) -> Dict:
        """AI-powered portfolio rebalancing"""
        
        print("🔄 AI-POWERED PORTFOLIO REBALANCING")
        print("=" * 40)
        
        positions = self.get_positions()
        
        rebalancing_actions = []
        
        for position in positions:
            symbol = position['symbol']
            current_signal = position['ai_current_signal']
            confidence = position['ai_confidence']
            
            print(f"\n📊 {symbol}:")
            print(f"   Current P&L: {position['unrealized_plpc']:.1%}")
            print(f"   AI Signal: {current_signal} ({confidence:.1%})")
            
            # Rebalancing logic based on AI signals
            if current_signal == 'SELL' and confidence > 0.6:
                rebalancing_actions.append({
                    'symbol': symbol,
                    'action': 'REDUCE_POSITION',
                    'reason': f'AI bearish signal ({confidence:.1%})',
                    'current_value': position['market_value']
                })
                print(f"   🔴 Recommendation: REDUCE POSITION")
                
            elif current_signal == 'BUY' and confidence > 0.6:
                rebalancing_actions.append({
                    'symbol': symbol,
                    'action': 'INCREASE_POSITION',
                    'reason': f'AI bullish signal ({confidence:.1%})',
                    'current_value': position['market_value']
                })
                print(f"   🟢 Recommendation: INCREASE POSITION")
                
            else:
                print(f"   🟡 Recommendation: HOLD POSITION")
        
        return {
            'rebalancing_actions': rebalancing_actions,
            'positions_analyzed': len(positions)
        }

def demo_alpaca_elite_ai():
    """Demo Alpaca integration with Elite AI v2.0"""
    
    print("🚀 ALPACA + ELITE AI v2.0 INTEGRATION DEMO")
    print("=" * 50)
    
    # Note: This is a demo - you need real Alpaca API keys
    print("📝 Setup Instructions:")
    print("1. Get Alpaca API keys from alpaca.markets")
    print("2. Replace 'DEMO_KEY' with your actual keys")
    print("3. Start with paper_trading=True")
    print()
    
    print("💻 Integration Code:")
    print("```python")
    print("# Create AI-powered Alpaca trader")
    print("trader = AlpacaEliteAITrader(")
    print("    api_key='YOUR_ALPACA_API_KEY',")
    print("    secret_key='YOUR_ALPACA_SECRET_KEY',")
    print("    paper_trading=True  # Start safe!")
    print(")")
    print("")
    print("# Run AI-powered trading session")
    print("universe = ['AAPL', 'TSLA', 'GOOGL', 'NVDA', 'AMZN']")
    print("results = trader.run_ai_trading_session(universe)")
    print("")
    print("# AI will:")
    print("# 1. Screen all stocks with Elite AI v2.0")
    print("# 2. Only trade high-confidence signals")
    print("# 3. Use Kelly Criterion for position sizing")
    print("# 4. Execute via Alpaca API")
    print("# 5. Apply comprehensive risk management")
    print("```")
    
    print(f"\n🎯 KEY BENEFITS:")
    print(f"✅ Elite AI v2.0 prevents bad trades")
    print(f"✅ Quality gates protect your capital")
    print(f"✅ Ensemble consensus for reliability")
    print(f"✅ Real Alpaca execution")
    print(f"✅ Professional risk management")
    
    print(f"\n🔥 RESULT: Research-grade AI trading via Alpaca!")

if __name__ == "__main__":
    demo_alpaca_elite_ai()
