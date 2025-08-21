#!/usr/bin/env python3
"""
Forward Test: 1-2 Years Training → 3 Months Unseen Testing
Exactly what you requested - proper prompt.yaml implementation

Shows how much your system would have made in 3 months using models
trained on 1-2 years of historical data with ZERO visibility to test period.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import your sophisticated components
try:
    from unified_ml_trading_system import UnifiedMLTradingSystem
    UNIFIED_AVAILABLE = True
    print("✅ Unified ML Trading System available")
except ImportError:
    UNIFIED_AVAILABLE = False
    print("⚠️ Using basic signals only")

class ProperForwardTester:
    """
    Proper forward test with complete data separation
    - Training: 1-2 years of historical data
    - Testing: 3 months of completely unseen data
    - Zero data leakage
    """
    
    def __init__(self, account_size: float = 100000):
        self.account_size = account_size
        
        # Initialize sophisticated trading system
        if UNIFIED_AVAILABLE:
            self.trading_system = UnifiedMLTradingSystem()
            print("🤖 Using ML + Technical + Options system")
        else:
            self.trading_system = None
            print("📊 Using basic technical signals")
        
        print(f"💰 Account Size: ${account_size:,.2f}")
    
    def setup_data_splits(self):
        """Setup proper train/test splits with NO data leakage"""
        
        # Current date
        today = datetime.now()
        
        # Test period: Last 3 months (UNSEEN data)
        test_end = today
        test_start = test_end - timedelta(days=90)
        
        # Training period: 18 months BEFORE test period starts
        train_end = test_start - timedelta(days=1)  # Day before test
        train_start = train_end - timedelta(days=547)  # ~18 months
        
        splits = {
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d')
        }
        
        train_days = (train_end - train_start).days
        test_days = (test_end - test_start).days
        
        print(f"\\n📅 DATA SEPARATION (NO LEAKAGE):")
        print(f"   📚 Training: {splits['train_start']} to {splits['train_end']} ({train_days} days)")
        print(f"   🔮 Testing: {splits['test_start']} to {splits['test_end']} ({test_days} days)")
        print(f"   ⚠️ CRITICAL: Test data completely unseen during training!")
        
        return splits
    
    def get_trading_universe(self):
        """Get universe of stocks for testing"""
        
        if UNIFIED_AVAILABLE:
            print("🔍 Using Elite AI Stock Selection...")
            universe = self.trading_system.get_elite_stock_universe()
            selected = universe[:25]  # Top 25 for focused testing
        else:
            print("📋 Using default high-quality universe...")
            selected = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                'CRM', 'ADBE', 'NFLX', 'AMD', 'PLTR', 'SNOW', 'COIN',
                'JPM', 'BAC', 'V', 'MA', 'DIS', 'COST', 'HD', 'INTC',
                'REGN', 'BIIB', 'AVGO'
            ]
        
        print(f"🎯 Selected {len(selected)} symbols for testing")
        return selected
    
    def download_data(self, symbols, start_date, end_date):
        """Download historical data for symbols"""
        
        print(f"📥 Downloading data from {start_date} to {end_date}...")
        
        data = {}
        successful = 0
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if len(df) > 50:  # Minimum data requirement
                    data[symbol] = df
                    successful += 1
                    print(f"   ✅ {symbol}: {len(df)} days")
                else:
                    print(f"   ❌ {symbol}: Insufficient data")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {str(e)[:50]}")
        
        print(f"✅ Successfully downloaded {successful}/{len(symbols)} symbols")
        return data
    
    def train_models_on_historical_data(self, symbols, train_data):
        """Train ML models ONLY on historical training data"""
        
        print(f"\\n🤖 Training models on HISTORICAL data only...")
        
        if not UNIFIED_AVAILABLE:
            print("⚠️ No ML system - using technical signals only")
            return {}
        
        trained_models = {}
        
        for symbol in symbols:
            if symbol in train_data:
                try:
                    print(f"   Training {symbol}...")
                    
                    # Train on historical data only
                    historical_data = train_data[symbol]
                    success = self.trading_system.train_ensemble_models(symbol, historical_data)
                    
                    if success:
                        trained_models[symbol] = True
                        print(f"   ✅ {symbol}: ML models trained")
                    else:
                        print(f"   ⚠️ {symbol}: Training failed")
                        
                except Exception as e:
                    print(f"   ❌ {symbol}: Error - {str(e)[:50]}")
        
        print(f"🎯 Successfully trained {len(trained_models)} ML models")
        return trained_models
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
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
        """Generate trading signal using historical data up to current_date ONLY"""
        
        # CRITICAL: Only use data up to current_date (no future leakage)
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
        
        # ML Enhancement (if available and trained)
        ml_boost = 0.0
        if UNIFIED_AVAILABLE and symbol in trained_models:
            try:
                ml_prediction = self.trading_system.get_ml_prediction(symbol, historical_data)
                # Convert ML prediction to boost (-20% to +20%)
                ml_boost = (ml_prediction - 50) / 250  # Scale to ±0.2
                
                if ml_boost > 0:
                    buy_strength += ml_boost
                else:
                    sell_strength += abs(ml_boost)
            except:
                pass
        
        # Determine final signal
        signal = 'HOLD'
        strength = 0.0
        threshold = 0.35  # Signal threshold
        
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
    
    def simulate_forward_test(self, test_data, trained_models, test_start, test_end):
        """Simulate trading during test period using only historical data"""
        
        print(f"\\n🔮 FORWARD TEST SIMULATION: {test_start} to {test_end}")
        print("⚠️ Using ONLY historical data for signals - NO FUTURE LEAKAGE!")
        print("-" * 70)
        
        # Initialize portfolio
        portfolio = {
            'cash': self.account_size,
            'positions': {},
            'total_value': self.account_size
        }
        
        # Get trading dates in test period
        trading_dates = pd.bdate_range(start=test_start, end=test_end)
        
        # Track results
        daily_results = []
        all_trades = []
        
        max_positions = 8  # Max concurrent positions
        max_position_pct = 0.12  # 12% max per position
        
        for i, date in enumerate(trading_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            if i % 10 == 0:  # Progress update
                print(f"   Day {i+1:2d}/{len(trading_dates)}: {date_str}")
            
            # Generate signals for available symbols
            daily_signals = {}
            
            for symbol in test_data.keys():
                try:
                    # Check if we have data for this date
                    if date_str in test_data[symbol].index:
                        signal_data = self.generate_trading_signal(
                            symbol, test_data, date_str, trained_models
                        )
                        
                        if signal_data['signal'] != 'HOLD':
                            daily_signals[symbol] = signal_data
                            
                except Exception:
                    continue
            
            # Execute trades based on signals
            current_positions = len([p for p in portfolio['positions'].values() if p > 0])
            
            # Sort signals by strength for prioritization
            sorted_signals = sorted(daily_signals.items(), 
                                  key=lambda x: x[1]['strength'], reverse=True)
            
            trades_today = []
            
            for symbol, signal_data in sorted_signals:
                signal = signal_data['signal']
                strength = signal_data['strength']
                price = signal_data['price']
                
                if signal == 'BUY' and current_positions < max_positions:
                    # Calculate position size
                    position_pct = max_position_pct * strength
                    position_value = portfolio['cash'] * position_pct
                    shares = int(position_value / price)
                    
                    if shares > 0 and portfolio['cash'] > position_value * 1.1:
                        # Execute buy with transaction costs
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
                            'rsi': signal_data.get('rsi', 50),
                            'ml_boost': signal_data.get('ml_boost', 0)
                        }
                        
                        trades_today.append(trade)
                        all_trades.append(trade)
                
                elif signal == 'SELL' and symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                    # Execute sell
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
            
            # Calculate portfolio value
            portfolio_value = portfolio['cash']
            positions_value = 0
            
            for symbol, shares in portfolio['positions'].items():
                if shares > 0 and symbol in test_data:
                    try:
                        # Get price for this date
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
    
    def calculate_performance_metrics(self, simulation_results):
        """Calculate comprehensive performance metrics"""
        
        daily_results = simulation_results['daily_results']
        
        if not daily_results:
            return {}
        
        # Extract values
        values = [d['portfolio_value'] for d in daily_results]
        daily_returns = [d['daily_return'] for d in daily_results]
        
        initial_value = values[0]
        final_value = values[-1]
        
        # Core metrics
        total_return = (final_value - initial_value) / initial_value
        profit_loss = final_value - initial_value
        
        # Annualized return (3 months to annual)
        days = len(values)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # Risk metrics
        daily_returns_array = np.array(daily_returns)
        volatility = np.std(daily_returns_array) * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + daily_returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate
        positive_days = len([r for r in daily_returns if r > 0])
        win_rate = positive_days / len(daily_returns) if daily_returns else 0
        
        # Trade analysis
        trades = simulation_results['all_trades']
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        return {
            'account_performance': {
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return_pct': total_return * 100,
                'annualized_return_pct': annual_return * 100,
                'profit_loss_amount': profit_loss,
                'trading_days': days
            },
            'risk_metrics': {
                'volatility_pct': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown * 100,
                'win_rate_pct': win_rate * 100
            },
            'trading_activity': {
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'avg_trade_value': np.mean([t['value'] for t in trades]) if trades else 0
            }
        }
    
    def run_complete_forward_test(self):
        """Run the complete forward test process"""
        
        print("🚀 COMPLETE FORWARD TEST")
        print("Training: 1-2 years | Testing: 3 months unseen")
        print("=" * 80)
        
        # Setup data splits
        splits = self.setup_data_splits()
        
        # Get trading universe
        symbols = self.get_trading_universe()
        
        # Step 1: Download training data ONLY
        print(f"\\n📥 STEP 1: Download training data...")
        train_data = self.download_data(symbols, splits['train_start'], splits['train_end'])
        
        # Step 2: Train models on historical data
        print(f"\\n🤖 STEP 2: Train models...")
        trained_models = self.train_models_on_historical_data(symbols, train_data)
        
        # Step 3: Download ALL data (train + test) for simulation
        print(f"\\n📥 STEP 3: Download complete dataset...")
        all_data = self.download_data(symbols, splits['train_start'], splits['test_end'])
        
        # Step 4: Run forward test simulation
        print(f"\\n🔮 STEP 4: Forward test simulation...")
        simulation_results = self.simulate_forward_test(
            all_data, trained_models, splits['test_start'], splits['test_end']
        )
        
        # Step 5: Calculate performance
        print(f"\\n📊 STEP 5: Calculate performance...")
        performance = self.calculate_performance_metrics(simulation_results)
        
        # Compile final results
        final_results = {
            'test_metadata': {
                'account_size': self.account_size,
                'data_splits': splits,
                'universe_size': len(symbols),
                'models_trained': len(trained_models),
                'symbols_with_data': len(all_data),
                'test_type': '3_month_forward_test',
                'ml_system_used': UNIFIED_AVAILABLE
            },
            'performance_metrics': performance,
            'daily_portfolio_values': simulation_results['daily_results'],
            'trade_history': simulation_results['all_trades'],
            'final_positions': simulation_results['final_portfolio']['positions'],
            'test_timestamp': datetime.now().isoformat()
        }
        
        return final_results
    
    def print_final_results(self, results):
        """Print comprehensive final results"""
        
        print("\\n" + "="*90)
        print("📊 3-MONTH FORWARD TEST RESULTS")
        print("="*90)
        
        # Test metadata
        meta = results['test_metadata']
        perf = results['performance_metrics']
        
        print(f"🔬 TEST CONFIGURATION:")
        print(f"   💰 Account Size: ${meta['account_size']:,.2f}")
        print(f"   📊 Universe: {meta['universe_size']} symbols")
        print(f"   🤖 Models Trained: {meta['models_trained']}")
        print(f"   📅 Test Period: {meta['data_splits']['test_start']} to {meta['data_splits']['test_end']}")
        print(f"   🧠 ML System: {'✅ Advanced' if meta['ml_system_used'] else '📊 Basic'}")
        
        # Performance results
        account = perf['account_performance']
        risk = perf['risk_metrics']
        trading = perf['trading_activity']
        
        print(f"\\n💰 ACCOUNT PERFORMANCE:")
        print(f"   📈 Initial Value: ${account['initial_value']:,.2f}")
        print(f"   📊 Final Value: ${account['final_value']:,.2f}")
        print(f"   💵 Profit/Loss: ${account['profit_loss_amount']:+,.2f}")
        print(f"   📈 3-Month Return: {account['total_return_pct']:+.2f}%")
        print(f"   🎯 Annualized Return: {account['annualized_return_pct']:+.2f}%")
        
        print(f"\\n📊 RISK ANALYSIS:")
        print(f"   📉 Max Drawdown: {risk['max_drawdown_pct']:.2f}%")
        print(f"   📈 Volatility: {risk['volatility_pct']:.1f}%")
        print(f"   ⚡ Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        print(f"   🎲 Win Rate: {risk['win_rate_pct']:.1f}%")
        
        print(f"\\n💼 TRADING ACTIVITY:")
        print(f"   🔄 Total Trades: {trading['total_trades']}")
        print(f"   🟢 Buy Orders: {trading['buy_trades']}")
        print(f"   🔴 Sell Orders: {trading['sell_trades']}")
        print(f"   💵 Avg Trade Size: ${trading['avg_trade_value']:,.0f}")
        
        # Target assessment
        annual_return = account['annualized_return_pct']
        target_return = 30.0  # 30% annual target
        
        print(f"\\n🎯 TARGET ASSESSMENT:")
        print(f"   🎯 Target Annual Return: {target_return:.0f}%")
        print(f"   📊 Achieved Return: {annual_return:+.1f}%")
        
        if annual_return >= target_return:
            print(f"   ✅ SUCCESS: Target exceeded by {annual_return - target_return:.1f}%!")
            print(f"   🎉 Your system would have made ${account['profit_loss_amount']:,.0f} in 3 months!")
        elif annual_return >= target_return * 0.8:  # Within 80% of target
            print(f"   🟡 CLOSE: {annual_return:.1f}% is close to {target_return:.0f}% target")
            print(f"   💡 Minor optimization needed")
        else:
            print(f"   ❌ BELOW TARGET: Needs improvement")
            print(f"   📈 Gap: {target_return - annual_return:.1f}% to reach target")
        
        # Show best trades if available
        trades = results.get('trade_history', [])
        if trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            if len(buy_trades) >= 3:
                print(f"\\n🏆 TOP TRADES (by signal strength):")
                top_trades = sorted(buy_trades, key=lambda x: x['strength'], reverse=True)[:3]
                for i, trade in enumerate(top_trades, 1):
                    ml_str = f" +ML({trade['ml_boost']:+.2f})" if abs(trade.get('ml_boost', 0)) > 0.01 else ""
                    print(f"   {i}. {trade['symbol']}: ${trade['price']:.2f} x{trade['shares']} - Strength {trade['strength']:.2f}{ml_str}")
        
        print("\\n" + "="*90)


def main():
    """Main execution function"""
    
    print("🔬 FORWARD TEST: 1-2 Years Training → 3 Months Unseen Testing")
    print("Shows exactly how much your system would have made")
    print("=" * 80)
    
    # Initialize tester
    tester = ProperForwardTester(account_size=100000)
    
    try:
        # Run complete forward test
        results = tester.run_complete_forward_test()
        
        # Print results
        tester.print_final_results(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forward_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Detailed results saved to: {filename}")
        
        # Return key metrics for easy access
        perf = results['performance_metrics']['account_performance']
        return {
            'profit_loss': perf['profit_loss_amount'],
            'return_pct': perf['total_return_pct'],
            'annual_return_pct': perf['annualized_return_pct'],
            'final_value': perf['final_value']
        }
        
    except Exception as e:
        print(f"❌ Forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\\n🎯 Quick Summary:")
        print(f"   💰 Would have made: ${results['profit_loss']:+,.0f}")
        print(f"   📈 3-Month return: {results['return_pct']:+.1f}%")
        print(f"   🎯 Annualized: {results['annual_return_pct']:+.1f}%")
