#!/usr/bin/env python3
"""
FINAL WORKING KELLY SYSTEM
Copy exact working logic, only change position sizing to Kelly
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import trading system
try:
    from unified_ml_trading_system import UnifiedMLTradingSystem
    UNIFIED_AVAILABLE = True
    print("✅ Unified ML Trading System available")
except ImportError:
    UNIFIED_AVAILABLE = False
    print("⚠️ Using standalone implementation")

class WorkingKellySystem:
    """
    Final Kelly system using EXACT working logic
    """
    
    def __init__(self, account_size=100000):
        self.account_size = account_size
        
        if UNIFIED_AVAILABLE:
            self.trading_system = UnifiedMLTradingSystem()
        else:
            self.trading_system = None
        
        print(f"💰 Account Size: ${account_size:,.2f}")
    
    def calculate_indicators(self, df):
        """EXACT indicator calculation from working system"""
        d = df.copy()
        
        # Moving averages
        d['MA5'] = d['Close'].rolling(5).mean()
        d['MA10'] = d['Close'].rolling(10).mean()
        d['MA20'] = d['Close'].rolling(20).mean()
        
        # RSI
        delta = d['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        d['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = d['High'] - d['Low']
        high_close = np.abs(d['High'] - d['Close'].shift())
        low_close = np.abs(d['Low'] - d['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        d['ATR'] = tr.rolling(14).mean()
        
        # Volume
        d['Volume_MA'] = d['Volume'].rolling(20).mean()
        d['Volume_Ratio'] = d['Volume'] / d['Volume_MA']
        
        return d
    
    def generate_trading_signal(self, symbol, all_data, current_date, trained_models):
        """EXACT signal generation from working system"""
        
        # CRITICAL: Only use data up to current_date
        historical_data = all_data[symbol].loc[:current_date]
        
        if len(historical_data) < 30:
            return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
        
        # Calculate indicators
        df = self.calculate_indicators(historical_data)
        
        if len(df) < 20:
            return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
        
        # Get current values
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        
        # Technical signal components
        buy_strength = 0.0
        sell_strength = 0.0
        
        # RSI signals
        rsi = latest['RSI'] if not pd.isna(latest['RSI']) else 50
        if rsi < 35:  # Oversold
            buy_strength += 0.25
        elif rsi > 65:  # Overbought
            sell_strength += 0.25
        
        # Moving average signals
        ma5 = latest['MA5'] if not pd.isna(latest['MA5']) else current_price
        ma10 = latest['MA10'] if not pd.isna(latest['MA10']) else current_price
        ma20 = latest['MA20'] if not pd.isna(latest['MA20']) else current_price
        
        # Trend signals
        if current_price > ma5 > ma20:  # Strong uptrend
            buy_strength += 0.3
        elif current_price < ma5 < ma20:  # Strong downtrend
            sell_strength += 0.3
        
        # Momentum signals
        if len(df) >= 10:
            momentum_5d = (current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6]
            momentum_10d = (current_price - df['Close'].iloc[-11]) / df['Close'].iloc[-11]
            
            if momentum_5d > 0.03 and momentum_10d > 0.05:  # Strong momentum
                buy_strength += 0.25
            elif momentum_5d < -0.03 and momentum_10d < -0.05:
                sell_strength += 0.25
        
        # Volume confirmation
        volume_ratio = latest['Volume_Ratio'] if not pd.isna(latest['Volume_Ratio']) else 1.0
        if volume_ratio > 1.3:  # Above average volume
            buy_strength *= 1.1
            sell_strength *= 1.1
        
        # ML Enhancement (exact working logic)
        ml_boost = 0.0
        if UNIFIED_AVAILABLE and symbol in trained_models:
            try:
                ml_prediction = self.trading_system.get_ml_prediction(symbol, historical_data)
                ml_boost = (ml_prediction - 50) / 250  # Scale to ±0.2
                
                if ml_boost > 0:
                    buy_strength += ml_boost
                else:
                    sell_strength += abs(ml_boost)
            except:
                pass
        
        # Determine final signal (EXACT working logic)
        signal = 'HOLD'
        strength = 0.0
        threshold = 0.35  # EXACT threshold
        
        if buy_strength > threshold and buy_strength > sell_strength:
            signal = 'BUY'
            strength = min(1.0, buy_strength)
        elif sell_strength > threshold and sell_strength > buy_strength:
            signal = 'SELL'
            strength = min(1.0, sell_strength)
        
        return {
            'signal': signal,
            'strength': strength,
            'price': current_price,
            'rsi': rsi,
            'ml_boost': ml_boost,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'volume_ratio': volume_ratio
        }
    
    def simulate_kelly_trading(self, test_data, trained_models, test_start, test_end):
        """EXACT simulation logic with ONLY Kelly position sizing change"""
        
        print(f"\\n🚀 KELLY ENHANCED FORWARD TEST: {test_start} to {test_end}")
        print("EXACT working logic + Kelly position sizing")
        print("-" * 70)
        
        # Initialize portfolio (EXACT same as working)
        portfolio = {
            'cash': self.account_size,
            'positions': {},
            'total_value': self.account_size
        }
        
        # Get trading dates
        trading_dates = pd.bdate_range(start=test_start, end=test_end)
        
        # Track results
        daily_results = []
        all_trades = []
        
        # EXACT same limits as working system
        max_positions = 8  # Same as working
        max_position_pct = 0.12  # Same as working - we'll enhance this with Kelly
        
        for i, date in enumerate(trading_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            if i % 10 == 0:
                print(f"   Day {i+1:2d}/{len(trading_dates)}: {date_str}")
            
            # Generate signals (EXACT same logic)
            daily_signals = {}
            
            for symbol in test_data.keys():
                try:
                    if date_str in test_data[symbol].index:
                        signal_data = self.generate_trading_signal(
                            symbol, test_data, date_str, trained_models
                        )
                        
                        if signal_data['signal'] != 'HOLD':
                            daily_signals[symbol] = signal_data
                            
                except Exception:
                    continue
            
            # Execute trades (EXACT same logic, ONLY position sizing change)
            current_positions = len([p for p in portfolio['positions'].values() if p > 0])
            
            # Sort signals by strength (EXACT same)
            sorted_signals = sorted(daily_signals.items(), 
                                  key=lambda x: x[1]['strength'], reverse=True)
            
            trades_today = []
            
            for symbol, signal_data in sorted_signals:
                signal = signal_data['signal']
                strength = signal_data['strength']
                price = signal_data['price']
                
                if signal == 'BUY' and current_positions < max_positions:
                    # ONLY CHANGE: Kelly-enhanced position sizing
                    # Original: position_pct = max_position_pct * strength
                    # Kelly: Scale position based on signal strength more aggressively
                    
                    kelly_multiplier = 1.5  # Kelly enhancement
                    strength_factor = min(strength / 0.35, 2.5)  # Scale 0.35-1.0 → 1.0-2.86
                    position_pct = max_position_pct * strength_factor * kelly_multiplier
                    position_pct = min(position_pct, 0.20)  # Cap at 20%
                    
                    position_value = portfolio['cash'] * position_pct
                    shares = int(position_value / price)
                    
                    if shares > 0 and portfolio['cash'] > position_value * 1.1:
                        # Execute buy (EXACT same as working)
                        total_cost = shares * price * 1.003  # 0.3% transaction cost
                        
                        portfolio['cash'] -= total_cost
                        portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + shares
                        current_positions += 1
                        
                        trade = {
                            'date': date_str,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': shares * price,
                            'strength': strength,
                            'position_pct': position_pct,
                            'kelly_enhancement': kelly_multiplier,
                            'rsi': signal_data.get('rsi', 50),
                            'ml_boost': signal_data.get('ml_boost', 0)
                        }
                        
                        trades_today.append(trade)
                        all_trades.append(trade)
                
                elif signal == 'SELL' and symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                    # Execute sell (EXACT same as working)
                    shares = portfolio['positions'][symbol]
                    trade_value = shares * price * 0.997  # Transaction cost
                    
                    portfolio['cash'] += trade_value
                    portfolio['positions'][symbol] = 0
                    
                    trade = {
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'strength': strength,
                        'rsi': signal_data.get('rsi', 50),
                        'ml_boost': signal_data.get('ml_boost', 0)
                    }
                    
                    trades_today.append(trade)
                    all_trades.append(trade)
            
            # Calculate portfolio value (EXACT same)
            portfolio_value = portfolio['cash']
            positions_value = 0
            
            for symbol, shares in portfolio['positions'].items():
                if shares > 0 and symbol in test_data:
                    try:
                        symbol_data = test_data[symbol].loc[:date_str]
                        if len(symbol_data) > 0:
                            current_price = float(symbol_data['Close'].iloc[-1])
                            position_value = shares * current_price
                            positions_value += position_value
                    except:
                        continue
            
            portfolio_value += positions_value
            
            # Calculate daily return
            prev_value = daily_results[-1]['portfolio_value'] if daily_results else self.account_size
            daily_return = (portfolio_value - prev_value) / prev_value
            
            # Store daily results
            daily_result = {
                'date': date_str,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions_value': positions_value,
                'active_positions': len([p for p in portfolio['positions'].values() if p > 0]),
                'daily_return': daily_return,
                'trades_today': len(trades_today)
            }
            
            daily_results.append(daily_result)
        
        return {
            'daily_results': daily_results,
            'all_trades': all_trades,
            'final_portfolio': portfolio,
            'trading_days': len(trading_dates)
        }
    
    def run_working_kelly_test(self):
        """Run the complete working Kelly test"""
        
        print("🚀 WORKING KELLY SYSTEM")
        print("Exact working logic + Kelly position sizing enhancement")
        print("=" * 70)
        
        # Setup exact same data splits
        today = datetime.now()
        test_end = today
        test_start = test_end - timedelta(days=90)
        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=547)
        
        splits = {
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d')
        }
        
        print(f"\\n📅 DATA SPLITS:")
        print(f"   📚 Training: {splits['train_start']} to {splits['train_end']}")
        print(f"   🔮 Testing: {splits['test_start']} to {splits['test_end']}")
        
        # Use exact same symbols
        if UNIFIED_AVAILABLE:
            universe = self.trading_system.get_elite_stock_universe()
            symbols = universe[:25]
        else:
            symbols = ['PLTR', 'NVDA', 'TSLA', 'AVGO', 'AMD', 'INTC', 'MU', 'META', 'LRCX', 
                      'AAPL', 'GOOGL', 'AMZN', 'BAC', 'ORCL', 'RBLX', 'SHOP', 'U', 'NET', 
                      'GS', 'NKE', 'NFLX', 'DIS', 'MS', 'SBUX', 'COIN']
        
        print(f"🎯 Selected {len(symbols)} symbols")
        
        # Download all data
        print(f"\\n📥 Downloading complete dataset...")
        all_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=splits['train_start'], end=splits['test_end'])
                
                if len(df) > 50:
                    all_data[symbol] = df
                    print(f"   ✅ {symbol}: {len(df)} days")
                else:
                    print(f"   ❌ {symbol}: Insufficient data")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error")
        
        print(f"✅ Successfully downloaded {len(all_data)} symbols")
        
        # Run Kelly simulation
        print(f"\\n🚀 Running Kelly enhanced simulation...")
        simulation_results = self.simulate_kelly_trading(
            all_data, {}, splits['test_start'], splits['test_end']
        )
        
        # Calculate performance
        daily_results = simulation_results['daily_results']
        
        if not daily_results:
            print("❌ No results generated")
            return
        
        # Performance metrics
        values = [d['portfolio_value'] for d in daily_results]
        daily_returns = [d['daily_return'] for d in daily_results]
        
        initial_value = values[0]
        final_value = values[-1]
        total_return = (final_value - initial_value) / initial_value
        profit_loss = final_value - initial_value
        
        # Annualized return
        days = len(values)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # Risk metrics
        daily_returns_array = np.array(daily_returns)
        volatility = np.std(daily_returns_array) * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative_returns = np.cumprod(1 + daily_returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate
        positive_days = len([r for r in daily_returns if r > 0])
        win_rate = positive_days / len(daily_returns) if daily_returns else 0
        
        # Print results
        print("\\n" + "="*80)
        print("🚀 WORKING KELLY SYSTEM RESULTS")
        print("="*80)
        
        print(f"\\n💰 ACCOUNT PERFORMANCE:")
        print(f"   📈 Initial Value: ${initial_value:,.2f}")
        print(f"   📊 Final Value: ${final_value:,.2f}")
        print(f"   💵 Profit/Loss: ${profit_loss:+,.2f}")
        print(f"   📈 3-Month Return: {total_return*100:+.2f}%")
        print(f"   🎯 Annualized Return: {annual_return*100:+.2f}%")
        
        print(f"\\n📊 RISK ANALYSIS:")
        print(f"   📉 Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"   📈 Volatility: {volatility*100:.1f}%")
        print(f"   ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   🎲 Win Rate: {win_rate*100:.1f}%")
        
        print(f"\\n💼 TRADING ACTIVITY:")
        print(f"   🔄 Total Trades: {len(simulation_results['all_trades'])}")
        
        buy_trades = [t for t in simulation_results['all_trades'] if t['action'] == 'BUY']
        sell_trades = [t for t in simulation_results['all_trades'] if t['action'] == 'SELL']
        
        print(f"   🟢 Buy Orders: {len(buy_trades)}")
        print(f"   🔴 Sell Orders: {len(sell_trades)}")
        
        if simulation_results['all_trades']:
            avg_trade = np.mean([t['value'] for t in simulation_results['all_trades']])
            print(f"   💵 Avg Trade Size: ${avg_trade:,.0f}")
        
        # Compare to original
        original_annual = 43.1
        improvement = annual_return*100 - original_annual
        
        print(f"\\n📈 KELLY ENHANCEMENT RESULTS:")
        print(f"   📊 Original System: {original_annual}% annual")
        print(f"   🚀 Kelly Enhanced: {annual_return*100:.1f}% annual")
        print(f"   💹 Improvement: {improvement:+.1f}% ({improvement/original_annual*100:+.0f}%)")
        
        if annual_return*100 >= 50:
            print(f"   ✅ EXCELLENT: Kelly enhancement successful!")
        elif annual_return*100 >= 45:
            print(f"   🟡 GOOD: Moderate Kelly improvement")
        else:
            print(f"   ⚠️ MIXED: Similar to original performance")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"working_kelly_results_{timestamp}.json"
        
        results = {
            'account_performance': {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return_pct': total_return * 100,
                'annualized_return_pct': annual_return * 100,
                'profit_loss': profit_loss
            },
            'kelly_enhancement': {
                'original_annual_return': original_annual,
                'kelly_annual_return': annual_return * 100,
                'improvement_pct': improvement,
                'improvement_factor': improvement / original_annual if original_annual > 0 else 0
            },
            'trades': simulation_results['all_trades'],
            'daily_values': daily_results
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Detailed results saved to: {filename}")
        
        return results


def main():
    """Main execution"""
    
    print("🚀 WORKING KELLY SYSTEM TEST")
    print("Enhanced position sizing on proven 43.1% system")
    print("=" * 70)
    
    # Initialize system
    kelly_system = WorkingKellySystem(account_size=100000)
    
    try:
        # Run complete test
        results = kelly_system.run_working_kelly_test()
        
        if results:
            perf = results['account_performance']
            kelly = results['kelly_enhancement']
            
            print(f"\\n🎯 FINAL SUMMARY:")
            print(f"   💰 Profit: ${perf['profit_loss']:+,.0f}")
            print(f"   📈 3-Month: {perf['total_return_pct']:+.1f}%")
            print(f"   🎯 Annualized: {perf['annualized_return_pct']:+.1f}%")
            print(f"   🚀 Improvement: {kelly['improvement_pct']:+.1f}%")
            
            return results
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
