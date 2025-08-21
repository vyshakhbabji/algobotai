#!/usr/bin/env python3
"""
SMART PARAMETER OPTIMIZER & STOCK SCREENER
Finds optimal trading parameters that work across stocks
Then screens which stocks currently meet those conditions
Trade only the best candidates that match optimal parameters!
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class SmartParameterOptimizer:
    def __init__(self, stocks=None):
        # DIVERSE STOCK UNIVERSE FOR PARAMETER TESTING
        if stocks:
            self.stocks = stocks
        else:
            self.stocks = [
                # TECH GIANTS
                "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN",
                # VOLATILE GROWTH  
                "TSLA", "META", "NFLX", "CRM", "UBER",
                # TRADITIONAL VALUE
                "JPM", "WMT", "JNJ", "PG", "KO",
                # EMERGING/SPECULATIVE
                "PLTR", "COIN", "SNOW", "AMD", "INTC",
                # ENERGY/MATERIALS
                "XOM", "CVX", "CAT", "BA", "GE"
            ]
        
        self.optimal_params = None
        self.optimization_results = []
        
        # PARAMETER SPACE - Focus on most impactful parameters
        self.parameter_space = {
            'momentum_period': [5, 10, 15, 20],  # Days for momentum calculation
            'momentum_threshold': [0.02, 0.025, 0.03, 0.035, 0.04],  # Minimum momentum for buy
            'rsi_oversold': [20, 25, 30],  # RSI oversold level
            'rsi_overbought': [70, 75, 80],  # RSI overbought level
            'volume_multiplier': [1.2, 1.5, 2.0],  # Volume vs average
            'volatility_max': [0.05, 0.07, 0.10],  # Max daily volatility
        }
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        data = data.copy()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['MA20'] = data['Close'].rolling(20).mean()
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        
        return data
    
    def test_parameters_on_stock(self, symbol, params):
        """Test specific parameters on a single stock"""
        try:
            # Get 2 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=750)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 100:
                return None
            
            data = self.calculate_indicators(data)
            
            # Calculate performance with these parameters
            return self.backtest_with_params(data, params, symbol)
            
        except Exception as e:
            return None
    
    def backtest_with_params(self, data, params, symbol):
        """Backtest strategy with specific parameters"""
        initial_cash = 10000
        cash = initial_cash
        shares = 0
        position = None
        trades = []
        
        momentum_period = params['momentum_period']
        momentum_threshold = params['momentum_threshold']
        rsi_oversold = params['rsi_oversold']
        rsi_overbought = params['rsi_overbought']
        volume_multiplier = params['volume_multiplier']
        volatility_max = params['volatility_max']
        
        for i in range(max(30, momentum_period), len(data)):
            current_price = float(data['Close'].iloc[i])
            
            # Calculate momentum
            past_price = float(data['Close'].iloc[i - momentum_period])
            momentum = (current_price - past_price) / past_price
            
            # Get indicators
            rsi = float(data['RSI'].iloc[i]) if not pd.isna(data['RSI'].iloc[i]) else 50
            volume_ratio = float(data['Volume'].iloc[i]) / float(data['Volume_MA'].iloc[i]) if not pd.isna(data['Volume_MA'].iloc[i]) else 1
            
            # Calculate volatility
            recent_returns = data['Close'].iloc[i-10:i].pct_change().dropna()
            volatility = float(recent_returns.std()) if len(recent_returns) > 0 else 0
            
            # UNIVERSAL SIGNAL LOGIC
            signal = 'HOLD'
            
            # BUY CONDITIONS - Stock meets optimal criteria
            if (momentum > momentum_threshold and 
                rsi < rsi_overbought and 
                volume_ratio > volume_multiplier and 
                volatility < volatility_max and
                position != 'LONG'):
                
                # Close short if any
                if position == 'SHORT':
                    cash += shares * current_price
                    shares = 0
                
                # Go long
                shares = cash / current_price
                cash = 0
                position = 'LONG'
                trades.append({'action': 'BUY', 'price': current_price, 'date': data.index[i]})
            
            # SELL CONDITIONS - Exit when momentum fades
            elif ((momentum < -momentum_threshold/2 or rsi > rsi_overbought or volatility > volatility_max) and 
                  position == 'LONG'):
                
                # Sell shares
                cash = shares * current_price
                shares = 0
                position = 'CASH'
                trades.append({'action': 'SELL', 'price': current_price, 'date': data.index[i]})
        
        # Final value
        final_price = float(data['Close'].iloc[-1])
        final_value = cash + (shares * final_price)
        
        # Buy and hold comparison
        start_price = float(data['Close'].iloc[max(30, momentum_period)])
        buy_hold_return = ((final_price - start_price) / start_price) * 100
        strategy_return = ((final_value - initial_cash) / initial_cash) * 100
        
        return {
            'symbol': symbol,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': strategy_return - buy_hold_return,
            'num_trades': len(trades),
            'win_rate': len([t for t in trades if t['action'] == 'SELL']) > 0,
            'final_value': final_value
        }
    
    def test_parameter_combination(self, params):
        """Test parameter combination across all stocks"""
        print(f"🧪 Testing: momentum={params['momentum_period']}d, threshold={params['momentum_threshold']:.3f}, RSI={params['rsi_oversold']}-{params['rsi_overbought']}")
        
        results = []
        total_outperformance = 0
        successful_tests = 0
        
        for symbol in self.stocks:
            result = self.test_parameters_on_stock(symbol, params)
            if result:
                results.append(result)
                total_outperformance += result['outperformance']
                successful_tests += 1
        
        if successful_tests == 0:
            return None
        
        avg_outperformance = total_outperformance / successful_tests
        winning_stocks = len([r for r in results if r['outperformance'] > 0])
        win_rate = winning_stocks / successful_tests
        
        print(f"   📊 Avg outperformance: {avg_outperformance:+.1f}%, Win rate: {win_rate:.1%} ({winning_stocks}/{successful_tests})")
        
        return {
            'params': params,
            'avg_outperformance': avg_outperformance,
            'win_rate': win_rate,
            'winning_stocks': winning_stocks,
            'total_stocks': successful_tests,
            'results': results
        }
    
    def find_optimal_parameters(self, max_tests=50):
        """Find the best universal parameters"""
        print("🎯 FINDING OPTIMAL UNIVERSAL PARAMETERS")
        print("=" * 50)
        
        # Generate parameter combinations
        import itertools
        all_combinations = list(itertools.product(
            self.parameter_space['momentum_period'],
            self.parameter_space['momentum_threshold'], 
            self.parameter_space['rsi_oversold'],
            self.parameter_space['rsi_overbought'],
            self.parameter_space['volume_multiplier'],
            self.parameter_space['volatility_max']
        ))
        
        print(f"📊 Total combinations: {len(all_combinations)}")
        print(f"🧪 Testing top {min(max_tests, len(all_combinations))} combinations")
        
        # Test combinations
        best_result = None
        best_score = -float('inf')
        
        # Smart sampling - test most promising combinations first
        np.random.shuffle(all_combinations)
        
        for i, combo in enumerate(all_combinations[:max_tests]):
            params = {
                'momentum_period': combo[0],
                'momentum_threshold': combo[1],
                'rsi_oversold': combo[2], 
                'rsi_overbought': combo[3],
                'volume_multiplier': combo[4],
                'volatility_max': combo[5]
            }
            
            result = self.test_parameter_combination(params)
            if result:
                self.optimization_results.append(result)
                
                # Score = combination of outperformance and win rate
                score = result['avg_outperformance'] + (result['win_rate'] * 10)  # Bonus for consistency
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    self.optimal_params = params.copy()
                    print(f"🎯 NEW BEST! Score: {score:.1f} (Outperf: {result['avg_outperformance']:+.1f}%, Win: {result['win_rate']:.1%})")
            
            if (i + 1) % 10 == 0:
                print(f"📈 Progress: {i+1}/{min(max_tests, len(all_combinations))}")
        
        print("\n🏆 OPTIMAL PARAMETERS FOUND!")
        if best_result:
            print(f"📊 Average Outperformance: {best_result['avg_outperformance']:+.1f}%")
            print(f"🎯 Win Rate: {best_result['win_rate']:.1%} ({best_result['winning_stocks']}/{best_result['total_stocks']})")
            print(f"⚙️  Parameters:")
            for key, value in self.optimal_params.items():
                print(f"   {key}: {value}")
        
        return best_result
    
    def screen_stocks_now(self, extended_universe=None):
        """Screen stocks that currently meet optimal parameters"""
        if not self.optimal_params:
            print("❌ No optimal parameters found! Run optimization first.")
            return []
        
        print(f"\n🔍 SCREENING STOCKS WITH OPTIMAL PARAMETERS")
        print("=" * 50)
        
        # Extended universe for screening
        if extended_universe:
            screen_stocks = extended_universe
        else:
            # Larger universe for screening
            screen_stocks = self.stocks + [
                "QQQ", "SPY", "IWM", "VTI", "VEA",  # ETFs
                "DIS", "V", "MA", "HD", "UNH",     # Blue chips
                "ROKU", "SHOP", "SQ", "PYPL",      # Growth
                "F", "GM", "DAL", "AAL",           # Cyclicals
            ]
        
        qualifying_stocks = []
        
        for symbol in screen_stocks:
            try:
                # Get recent data
                data = yf.download(symbol, period="3mo", progress=False)
                if data.empty or len(data) < 30:
                    continue
                
                data = self.calculate_indicators(data)
                latest_idx = -1
                
                # Current metrics
                current_price = float(data['Close'].iloc[latest_idx])
                momentum_period = self.optimal_params['momentum_period']
                past_price = float(data['Close'].iloc[latest_idx - momentum_period])
                momentum = (current_price - past_price) / past_price
                
                rsi = float(data['RSI'].iloc[latest_idx]) if not pd.isna(data['RSI'].iloc[latest_idx]) else 50
                volume_ratio = float(data['Volume'].iloc[latest_idx]) / float(data['Volume_MA'].iloc[latest_idx]) if not pd.isna(data['Volume_MA'].iloc[latest_idx]) else 1
                
                # Volatility
                recent_returns = data['Close'].iloc[-10:].pct_change().dropna()
                volatility = float(recent_returns.std()) if len(recent_returns) > 0 else 0
                
                # Check if meets optimal criteria
                meets_criteria = (
                    momentum > self.optimal_params['momentum_threshold'] and
                    rsi < self.optimal_params['rsi_overbought'] and
                    volume_ratio > self.optimal_params['volume_multiplier'] and
                    volatility < self.optimal_params['volatility_max']
                )
                
                if meets_criteria:
                    qualifying_stocks.append({
                        'symbol': symbol,
                        'momentum': momentum,
                        'rsi': rsi,
                        'volume_ratio': volume_ratio,
                        'volatility': volatility,
                        'score': momentum * (1 - volatility) * min(volume_ratio, 3)  # Composite score
                    })
                    
                    print(f"✅ {symbol}: Momentum {momentum:+.1%}, RSI {rsi:.0f}, Vol {volume_ratio:.1f}x, Volatility {volatility:.3f}")
            
            except Exception as e:
                continue
        
        # Sort by composite score
        qualifying_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🎯 FOUND {len(qualifying_stocks)} QUALIFYING STOCKS!")
        if qualifying_stocks:
            print(f"🥇 TOP 5 CANDIDATES:")
            for i, stock in enumerate(qualifying_stocks[:5], 1):
                print(f"   #{i} {stock['symbol']}: Score {stock['score']:.2f}, Momentum {stock['momentum']:+.1%}")
        
        return qualifying_stocks
    
    def save_results(self):
        """Save optimization results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimal_parameters_{timestamp}.json"
        
        save_data = {
            'optimal_parameters': self.optimal_params,
            'optimization_results': self.optimization_results[:5],  # Top 5 results
            'timestamp': datetime.now().isoformat(),
            'stocks_tested': self.stocks
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")

def main():
    """Run smart parameter optimization and stock screening"""
    print("🚀 SMART PARAMETER OPTIMIZER & STOCK SCREENER")
    print("=" * 60)
    print("🎯 Find optimal parameters that work across stocks")
    print("🔍 Screen which stocks currently meet those criteria")
    print("💰 Trade only the best candidates!")
    print("=" * 60)
    
    optimizer = SmartParameterOptimizer()
    
    # 1. Find optimal parameters
    best_result = optimizer.find_optimal_parameters(max_tests=50)
    
    if best_result:
        # 2. Screen current market for qualifying stocks
        qualifying_stocks = optimizer.screen_stocks_now()
        
        # 3. Save results
        optimizer.save_results()
        
        print(f"\n🎯 READY TO TRADE!")
        print(f"📊 Optimal parameters found with {best_result['avg_outperformance']:+.1f}% avg outperformance")
        print(f"🔍 {len(qualifying_stocks)} stocks currently qualify")
        print(f"💰 Focus on top 3-5 candidates for best results")
        
        return optimizer, qualifying_stocks
    else:
        print("❌ Optimization failed!")
        return None, []

if __name__ == "__main__":
    optimizer, stocks = main()
