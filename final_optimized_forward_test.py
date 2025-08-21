#!/usr/bin/env python3
"""
FINAL 3-MONTH FORWARD TEST - Kelly Optimized System
Compare original vs optimized performance

This will show the impact of Kelly sizing + 99% deployment optimization
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
    print("⚠️ Using standalone implementation")

class FinalOptimizedForwardTest:
    """
    Final forward test with Kelly optimization
    Compare original vs optimized performance
    """
    
    def __init__(self, account_size: float = 100000):
        self.account_size = account_size
        
        # Initialize trading system
        if UNIFIED_AVAILABLE:
            self.trading_system = UnifiedMLTradingSystem()
            print("🤖 Using advanced ML + Technical system")
        else:
            self.trading_system = None
            print("📊 Using technical signals only")
        
        # Load Kelly optimized configuration
        try:
            with open('kelly_optimized_config.json', 'r') as f:
                self.kelly_config = json.load(f)
            print("🚀 Kelly optimized configuration loaded")
        except FileNotFoundError:
            print("⚠️ Using default configuration")
            self.kelly_config = {
                'portfolio': {
                    'max_positions': 15,
                    'max_position_size': 0.15,
                    'min_position_size': 0.02,
                    'target_deployment': 0.99,
                    'kelly_multiplier': 1.5
                },
                'signals': {
                    'buy_threshold': 0.30,
                    'sell_threshold': 0.25
                }
            }
        
        print(f"💰 Account Size: ${account_size:,.2f}")
        print(f"🎯 Max Positions: {self.kelly_config['portfolio']['max_positions']}")
        print(f"📊 Target Deployment: {self.kelly_config['portfolio']['target_deployment']*100:.0f}%")
    
    def setup_data_splits(self):
        """Setup proper train/test splits"""
        
        today = datetime.now()
        
        # Test period: Last 3 months
        test_end = today
        test_start = test_end - timedelta(days=90)
        
        # Training period: 18 months before test
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
        
        return splits
    
    def get_trading_universe(self):
        """Get elite trading universe"""
        
        if UNIFIED_AVAILABLE:
            print("🔍 Using Elite AI Stock Selection...")
            universe = self.trading_system.get_elite_stock_universe()
            selected = universe[:25]  # Use same 25 as working system
        else:
            print("📋 Using expanded high-quality universe...")
            selected = [
                # Mega Cap Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                'AMD', 'NFLX', 'CRM', 'ADBE', 'ORCL', 'INTC',
                
                # Growth Stocks
                'PLTR', 'SNOW', 'COIN', 'RBLX', 'U', 'NET', 'DDOG',
                'SHOP', 'ROKU', 'ZM', 'SQ',
                
                # Financials
                'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',
                
                # Other Quality
                'DIS', 'NKE', 'COST'
            ]
        
        print(f"🎯 Selected {len(selected)} symbols for testing")
        return selected
    
    def download_data(self, symbols, start_date, end_date):
        """Download data for symbols"""
        
        print(f"📥 Downloading data from {start_date} to {end_date}...")
        
        data = {}
        successful = 0
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if len(df) > 50:
                    data[symbol] = df
                    successful += 1
                    print(f"   ✅ {symbol}: {len(df)} days")
                else:
                    print(f"   ❌ {symbol}: Insufficient data")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {str(e)[:50]}")
        
        print(f"✅ Successfully downloaded {successful}/{len(symbols)} symbols")
        return data
    
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
    
    def calculate_kelly_position_size(self, signal_strength: float, base_kelly: float = 0.08) -> float:
        """Calculate Kelly position size based on signal strength"""
        
        # Scale Kelly based on signal strength
        strength_multiplier = min(signal_strength / 0.30, 2.5)  # 0.30-1.0 → 1.0-3.33
        kelly_multiplier = self.kelly_config['portfolio']['kelly_multiplier']
        
        position_size = base_kelly * strength_multiplier * kelly_multiplier
        
        # Apply bounds
        min_size = self.kelly_config['portfolio']['min_position_size']
        max_size = self.kelly_config['portfolio']['max_position_size']
        
        return max(min_size, min(position_size, max_size))
    
    def generate_enhanced_signal(self, symbol, all_data, current_date, trained_models):
        """Generate enhanced signal with Kelly sizing"""
        
        # Get historical data up to current date
        historical_data = all_data[symbol].loc[:current_date]
        
        if len(historical_data) < 30:
            return {'signal': 'HOLD', 'strength': 0.0, 'price': 0, 'position_size': 0}
        
        # Calculate indicators
        df = self.calculate_indicators(historical_data)
        
        if len(df) < 20:
            return {'signal': 'HOLD', 'strength': 0.0, 'price': 0, 'position_size': 0}
        
        # Get current values
        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        
        # Technical signal components (EXACT working logic)
        buy_strength = 0.0
        sell_strength = 0.0
        
        # RSI signals (exact working thresholds)
        rsi = latest['RSI'] if not pd.isna(latest['RSI']) else 50
        if rsi < 35:  # Oversold
            buy_strength += 0.25
        elif rsi > 65:  # Overbought
            sell_strength += 0.25
        
        # Moving average signals
        ma5 = latest['MA5'] if not pd.isna(latest['MA5']) else current_price
        ma10 = latest['MA10'] if not pd.isna(latest['MA10']) else current_price
        ma20 = latest['MA20'] if not pd.isna(latest['MA20']) else current_price
        
        # Trend signals (exact working logic)
        if current_price > ma5 > ma20:  # Strong uptrend
            buy_strength += 0.3
        elif current_price < ma5 < ma20:  # Strong downtrend
            sell_strength += 0.3
        
        # Momentum signals (exact working logic)
        if len(df) >= 10:
            momentum_5d = (current_price - df['Close'].iloc[-6]) / df['Close'].iloc[-6]
            momentum_10d = (current_price - df['Close'].iloc[-11]) / df['Close'].iloc[-11]
            
            if momentum_5d > 0.03 and momentum_10d > 0.05:  # Strong momentum
                buy_strength += 0.25
            elif momentum_5d < -0.03 and momentum_10d < -0.05:
                sell_strength += 0.25
        
        # Volume confirmation (exact working logic)
        volume_ratio = latest['Volume_Ratio'] if not pd.isna(latest['Volume_Ratio']) else 1.0
        if volume_ratio > 1.3:  # Above average volume
            buy_strength *= 1.1
            sell_strength *= 1.1
        
        # ML Enhancement (exact working logic)
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
        
        # Determine signal with EXACT working thresholds
        signal = 'HOLD'
        strength = 0.0
        position_size = 0.0
        
        # Use exact working threshold
        threshold = 0.35  # EXACT threshold from working system
        
        if buy_strength > threshold and buy_strength > sell_strength:
            signal = 'BUY'
            strength = min(1.0, buy_strength)
            position_size = self.calculate_kelly_position_size(strength)
        elif sell_strength > threshold and sell_strength > buy_strength:
            signal = 'SELL'
            strength = min(1.0, sell_strength)
            position_size = 0  # Sell is full position
        
        return {
            'signal': signal,
            'strength': strength,
            'price': current_price,
            'position_size': position_size,
            'rsi': rsi,
            'ml_boost': ml_boost,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'volume_ratio': volume_ratio
        }
    
    def simulate_kelly_optimized_trading(self, test_data, trained_models, test_start, test_end):
        """Simulate trading with Kelly optimization"""
        
        print(f"\\n🚀 KELLY OPTIMIZED FORWARD TEST: {test_start} to {test_end}")
        print("🎯 Using Kelly criterion + 99% deployment + 15 max positions")
        print("-" * 70)
        
        # Initialize portfolio
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
        
        max_positions = self.kelly_config['portfolio']['max_positions']  # 15
        target_deployment = self.kelly_config['portfolio']['target_deployment']  # 0.99
        
        for i, date in enumerate(trading_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            if i % 10 == 0:
                print(f"   Day {i+1:2d}/{len(trading_dates)}: {date_str}")
            
            # Generate signals for all symbols
            daily_signals = {}
            
            for symbol in test_data.keys():
                try:
                    if date_str in test_data[symbol].index:
                        signal_data = self.generate_enhanced_signal(
                            symbol, test_data, date_str, trained_models
                        )
                        
                        if signal_data['signal'] != 'HOLD':
                            daily_signals[symbol] = signal_data
                            
                except Exception:
                    continue
            
            # Execute Kelly-optimized trades
            current_positions = len([p for p in portfolio['positions'].values() if p > 0])
            current_deployment = 0
            
            # Calculate current deployment
            for symbol, shares in portfolio['positions'].items():
                if shares > 0 and symbol in test_data:
                    try:
                        symbol_data = test_data[symbol].loc[:date_str]
                        if len(symbol_data) > 0:
                            current_price = float(symbol_data['Close'].iloc[-1])
                            current_deployment += (shares * current_price) / self.account_size
                    except:
                        continue
            
            # Sort signals by strength for prioritization
            sorted_signals = sorted(daily_signals.items(), 
                                  key=lambda x: x[1]['strength'], reverse=True)
            
            trades_today = []
            
            for symbol, signal_data in sorted_signals:
                signal = signal_data['signal']
                strength = signal_data['strength']
                price = signal_data['price']
                position_size_pct = signal_data['position_size']
                
                if signal == 'BUY' and current_positions < max_positions:
                    # Check if we can deploy more capital
                    if current_deployment + position_size_pct <= target_deployment:
                        position_value = self.account_size * position_size_pct
                        shares = int(position_value / price)
                        
                        if shares > 0 and portfolio['cash'] > position_value * 1.003:
                            # Execute buy
                            total_cost = shares * price * 1.003  # 0.3% transaction cost
                            
                            portfolio['cash'] -= total_cost
                            portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + shares
                            current_positions += 1
                            current_deployment += position_size_pct
                            
                            trade = {
                                'date': date_str,
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price,
                                'value': shares * price,
                                'strength': strength,
                                'position_size_pct': position_size_pct,
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
            
            # Calculate actual deployment
            actual_deployment = positions_value / portfolio_value if portfolio_value > 0 else 0
            
            # Store daily results
            daily_result = {
                'date': date_str,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions_value': positions_value,
                'active_positions': len([p for p in portfolio['positions'].values() if p > 0]),
                'actual_deployment': actual_deployment,
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
        """Calculate performance metrics"""
        
        daily_results = simulation_results['daily_results']
        
        if not daily_results:
            return {}
        
        # Extract values
        values = [d['portfolio_value'] for d in daily_results]
        daily_returns = [d['daily_return'] for d in daily_results]
        deployments = [d['actual_deployment'] for d in daily_results]
        
        initial_value = values[0]
        final_value = values[-1]
        
        # Core metrics
        total_return = (final_value - initial_value) / initial_value
        profit_loss = final_value - initial_value
        
        # Annualized return
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
        
        # Deployment efficiency
        avg_deployment = np.mean(deployments)
        
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
            'deployment_metrics': {
                'average_deployment_pct': avg_deployment * 100,
                'target_deployment_pct': self.kelly_config['portfolio']['target_deployment'] * 100,
                'deployment_efficiency': avg_deployment / self.kelly_config['portfolio']['target_deployment']
            },
            'trading_activity': {
                'total_trades': len(simulation_results['all_trades']),
                'buy_trades': len([t for t in simulation_results['all_trades'] if t['action'] == 'BUY']),
                'sell_trades': len([t for t in simulation_results['all_trades'] if t['action'] == 'SELL']),
                'avg_trade_value': np.mean([t['value'] for t in simulation_results['all_trades']]) if simulation_results['all_trades'] else 0
            }
        }
    
    def run_final_forward_test(self):
        """Run the complete Kelly optimized forward test"""
        
        print("🚀 FINAL KELLY OPTIMIZED FORWARD TEST")
        print("Enhanced system with Kelly sizing + 99% deployment")
        print("=" * 80)
        
        # Setup data splits
        splits = self.setup_data_splits()
        
        # Get expanded universe
        symbols = self.get_trading_universe()
        
        # Download training data
        print(f"\\n📥 STEP 1: Download training data...")
        train_data = self.download_data(symbols, splits['train_start'], splits['train_end'])
        
        # Train models
        print(f"\\n🤖 STEP 2: Train models...")
        trained_models = {}
        if UNIFIED_AVAILABLE:
            for symbol in symbols:
                if symbol in train_data:
                    try:
                        success = self.trading_system.train_ensemble_models(symbol, train_data[symbol])
                        if success:
                            trained_models[symbol] = True
                    except:
                        pass
        print(f"🎯 Successfully trained {len(trained_models)} ML models")
        
        # Download complete dataset
        print(f"\\n📥 STEP 3: Download complete dataset...")
        all_data = self.download_data(symbols, splits['train_start'], splits['test_end'])
        
        # Run Kelly optimized simulation
        print(f"\\n🚀 STEP 4: Kelly optimized simulation...")
        simulation_results = self.simulate_kelly_optimized_trading(
            all_data, trained_models, splits['test_start'], splits['test_end']
        )
        
        # Calculate performance
        print(f"\\n📊 STEP 5: Calculate performance...")
        performance = self.calculate_performance_metrics(simulation_results)
        
        # Compile results
        final_results = {
            'test_metadata': {
                'account_size': self.account_size,
                'data_splits': splits,
                'universe_size': len(symbols),
                'models_trained': len(trained_models),
                'symbols_with_data': len(all_data),
                'test_type': 'kelly_optimized_forward_test',
                'optimization': 'Kelly sizing + 99% deployment + 15 positions',
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
        print("🚀 KELLY OPTIMIZED FORWARD TEST RESULTS")
        print("="*90)
        
        meta = results['test_metadata']
        perf = results['performance_metrics']
        
        print(f"🔬 TEST CONFIGURATION:")
        print(f"   💰 Account Size: ${meta['account_size']:,.2f}")
        print(f"   📊 Universe: {meta['universe_size']} symbols")
        print(f"   🤖 Models Trained: {meta['models_trained']}")
        print(f"   📅 Test Period: {meta['data_splits']['test_start']} to {meta['data_splits']['test_end']}")
        print(f"   🎯 Optimization: {meta['optimization']}")
        
        # Performance results
        account = perf['account_performance']
        risk = perf['risk_metrics']
        deployment = perf['deployment_metrics']
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
        
        print(f"\\n🎯 DEPLOYMENT EFFICIENCY:")
        print(f"   📊 Target Deployment: {deployment['target_deployment_pct']:.0f}%")
        print(f"   📈 Actual Deployment: {deployment['average_deployment_pct']:.1f}%")
        print(f"   ⚡ Efficiency: {deployment['deployment_efficiency']*100:.1f}%")
        
        print(f"\\n💼 TRADING ACTIVITY:")
        print(f"   🔄 Total Trades: {trading['total_trades']}")
        print(f"   🟢 Buy Orders: {trading['buy_trades']}")
        print(f"   🔴 Sell Orders: {trading['sell_trades']}")
        print(f"   💵 Avg Trade Size: ${trading['avg_trade_value']:,.0f}")
        
        # Compare to original system
        original_annual = 43.1
        current_annual = account['annualized_return_pct']
        improvement = current_annual - original_annual
        
        print(f"\\n📈 OPTIMIZATION RESULTS:")
        print(f"   📊 Original System: 43.1% annual")
        print(f"   🚀 Kelly Optimized: {current_annual:.1f}% annual")
        print(f"   💹 Improvement: {improvement:+.1f}% ({improvement/original_annual*100:+.0f}%)")
        
        if current_annual >= 50:
            print(f"   ✅ SUCCESS: Significant improvement achieved!")
        elif current_annual >= 45:
            print(f"   🟡 GOOD: Moderate improvement")
        else:
            print(f"   ⚠️ MIXED: Results similar to original")
        
        return results


def main():
    """Main execution"""
    
    print("🚀 FINAL KELLY OPTIMIZED FORWARD TEST")
    print("Testing enhanced system with Kelly criterion + 99% deployment")
    print("=" * 80)
    
    # Initialize tester
    tester = FinalOptimizedForwardTest(account_size=100000)
    
    try:
        # Run complete test
        results = tester.run_final_forward_test()
        
        # Print results
        final_results = tester.print_final_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kelly_optimized_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n💾 Detailed results saved to: {filename}")
        
        # Return summary
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
        print(f"\\n🎯 FINAL SUMMARY:")
        print(f"   💰 Profit: ${results['profit_loss']:+,.0f}")
        print(f"   📈 3-Month: {results['return_pct']:+.1f}%")
        print(f"   🎯 Annualized: {results['annual_return_pct']:+.1f}%")
        print(f"   📊 Final Value: ${results['final_value']:,.0f}")
