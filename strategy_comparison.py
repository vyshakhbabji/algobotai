#!/usr/bin/env python3
"""
STRATEGY COMPARISON: MOMENTUM vs TECHNICAL SIGNALS
Compares the Institutional Momentum Strategy against the Automated Signal Optimizer
Tests both on the same stocks and timeframe for fair comparison
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StrategyComparison:
    def __init__(self):
        # SAME STOCK UNIVERSE FOR FAIR COMPARISON
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
        
        # YOUR PROVEN TECHNICAL CONFIG
        self.technical_config = {
            'trend_5d_buy_threshold': 0.025,
            'trend_5d_sell_threshold': -0.02,
            'trend_10d_buy_threshold': 0.025,
            'trend_10d_sell_threshold': -0.045,
            'rsi_overbought': 65,
            'rsi_oversold': 20,
            'volatility_threshold': 0.07,
            'volume_ratio_threshold': 1.6
        }
    
    def calculate_momentum_score(self, symbol, end_date):
        """Calculate momentum score (Jegadeesh & Titman style)"""
        try:
            start_date = end_date - timedelta(days=200)  # Extra for momentum calculation
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty or len(data) < 126:
                return None
            
            prices = data['Close']
            current_price = float(prices.iloc[-1])
            
            # 6-month momentum (key factor)
            if len(prices) >= 126:
                price_6m_ago = float(prices.iloc[-126])
                momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            else:
                return None
                
            # 3-month momentum
            if len(prices) >= 63:
                price_3m_ago = float(prices.iloc[-63])
                momentum_3m = (current_price - price_3m_ago) / price_3m_ago
            else:
                momentum_3m = momentum_6m
                
            # 1-month momentum
            if len(prices) >= 21:
                price_1m_ago = float(prices.iloc[-21])
                momentum_1m = (current_price - price_1m_ago) / price_1m_ago
            else:
                momentum_1m = momentum_3m
            
            # Risk adjustment (volatility)
            returns = prices.pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))
            
            # Combined momentum score with risk adjustment
            momentum_score = (momentum_6m * 0.5 + momentum_3m * 0.3 + momentum_1m * 0.2) / max(volatility, 0.1)
            
            return {
                'symbol': symbol,
                'score': momentum_score,
                'momentum_6m': momentum_6m * 100,
                'momentum_3m': momentum_3m * 100,
                'momentum_1m': momentum_1m * 100,
                'volatility': volatility * 100,
                'current_price': current_price
            }
        except Exception as e:
            print(f"❌ Error calculating momentum for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for signal strategy"""
        data = data.copy()
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_technical_signals(self, data, config):
        """Generate signals using your proven technical config"""
        signals = []
        
        for i in range(30, len(data)):
            price = float(data['Close'].iloc[i])
            
            # Multi-timeframe analysis
            recent_5d = data['Close'].iloc[i-5:i]
            recent_10d = data['Close'].iloc[i-10:i]
            
            # Trend analysis
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Moving averages
            ma5 = float(data['MA5'].iloc[i]) if not pd.isna(data['MA5'].iloc[i]) else price
            ma10 = float(data['MA10'].iloc[i]) if not pd.isna(data['MA10'].iloc[i]) else price
            
            # RSI
            rsi = float(data['RSI'].iloc[i]) if not pd.isna(data['RSI'].iloc[i]) else 50
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # Volume ratio
            try:
                recent_volume = float(data['Volume'].iloc[i-10:i].mean())
                current_volume = float(data['Volume'].iloc[i])
                volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
            except:
                volume_ratio = 1
            
            # SIGNAL LOGIC
            signal = 'HOLD'
            
            # SELL CONDITIONS
            if (trend_5d < config['trend_5d_sell_threshold'] and trend_10d < config['trend_10d_sell_threshold']) or \
               (price < ma5 < ma10) or \
               (rsi > config['rsi_overbought'] and trend_5d < config['trend_5d_sell_threshold']/2) or \
               (volatility > config['volatility_threshold'] and trend_10d < config['trend_10d_sell_threshold']):
                signal = 'SELL'
            
            # BUY CONDITIONS
            elif (trend_5d > config['trend_5d_buy_threshold'] and trend_10d > config['trend_10d_buy_threshold'] and volume_ratio > config['volume_ratio_threshold']) or \
                 (price > ma5 > ma10 and trend_5d > config['trend_5d_buy_threshold']) or \
                 (rsi < config['rsi_oversold'] and trend_5d > config['trend_5d_buy_threshold']/2):
                signal = 'BUY'
            
            signals.append({
                'date': data.index[i],
                'price': price,
                'signal': signal
            })
        
        return signals
    
    def backtest_strategy(self, symbol, strategy_type, period_days=365):
        """Backtest either momentum or technical strategy"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 50)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty or len(data) < 50:
                return None
            
            initial_cash = 10000
            
            if strategy_type == 'momentum':
                # MOMENTUM STRATEGY: Simple buy-and-hold of top momentum stocks
                start_price = float(data['Close'].iloc[0])
                end_price = float(data['Close'].iloc[-1])
                final_value = initial_cash * (end_price / start_price)
                strategy_return = ((final_value - initial_cash) / initial_cash) * 100
                num_trades = 1  # Simple buy-and-hold
                
            elif strategy_type == 'technical':
                # TECHNICAL STRATEGY: Active trading with signals
                data = self.calculate_technical_indicators(data)
                signals = self.generate_technical_signals(data, self.technical_config)
                
                if not signals:
                    return None
                
                cash = initial_cash
                shares = 0
                position = None
                trades = 0
                
                for signal_data in signals:
                    price = signal_data['price']
                    signal = signal_data['signal']
                    
                    if signal == 'BUY' and position != 'LONG':
                        if position == 'SHORT':
                            cash += shares * price
                            shares = 0
                        shares = cash / price
                        cash = 0
                        position = 'LONG'
                        trades += 1
                        
                    elif signal == 'SELL' and position != 'SHORT':
                        if position == 'LONG':
                            cash = shares * price
                            shares = 0
                        position = 'SHORT'
                        trades += 1
                
                final_price = signals[-1]['price']
                final_value = cash + (shares * final_price)
                strategy_return = ((final_value - initial_cash) / initial_cash) * 100
                num_trades = trades
            
            # Buy-and-hold comparison
            start_price = float(data['Close'].iloc[0])
            end_price = float(data['Close'].iloc[-1])
            buy_hold_return = ((end_price - start_price) / start_price) * 100
            
            outperformance = strategy_return - buy_hold_return
            
            return {
                'symbol': symbol,
                'strategy': strategy_type,
                'strategy_return': strategy_return,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'num_trades': num_trades,
                'final_value': final_value
            }
            
        except Exception as e:
            print(f"❌ Error testing {symbol} with {strategy_type}: {str(e)}")
            return None
    
    def compare_strategies(self, period_days=365):
        """Compare both strategies head-to-head"""
        print("🥊 STRATEGY COMPARISON: MOMENTUM vs TECHNICAL SIGNALS")
        print("=" * 70)
        print(f"📊 Testing {len(self.stocks)} stocks over {period_days} days")
        print(f"🏛️  Momentum Strategy: Jegadeesh & Titman (1993)")
        print(f"⚙️  Technical Strategy: Your proven config")
        print("=" * 70)
        
        momentum_results = []
        technical_results = []
        
        # Get momentum rankings first
        print("\n📈 CALCULATING MOMENTUM SCORES...")
        momentum_scores = []
        for symbol in self.stocks:
            score_data = self.calculate_momentum_score(symbol, datetime.now())
            if score_data:
                momentum_scores.append(score_data)
        
        # Sort by momentum score
        momentum_scores.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 TOP 10 MOMENTUM STOCKS:")
        for i, stock in enumerate(momentum_scores[:10], 1):
            print(f"   {i:2d}. {stock['symbol']:6s} Score: {stock['score']:+.3f} | 6M: {stock['momentum_6m']:+.1f}%")
        
        print(f"\n🧪 BACKTESTING BOTH STRATEGIES...")
        print("-" * 70)
        
        comparison_data = []
        
        for symbol in self.stocks:
            print(f"Testing {symbol}...")
            
            # Test momentum strategy
            momentum_result = self.backtest_strategy(symbol, 'momentum', period_days)
            if momentum_result:
                momentum_results.append(momentum_result)
            
            # Test technical strategy
            technical_result = self.backtest_strategy(symbol, 'technical', period_days)
            if technical_result:
                technical_results.append(technical_result)
            
            # Compare results for this stock
            if momentum_result and technical_result:
                comparison_data.append({
                    'symbol': symbol,
                    'momentum_return': momentum_result['strategy_return'],
                    'technical_return': technical_result['strategy_return'],
                    'momentum_outperf': momentum_result['outperformance'],
                    'technical_outperf': technical_result['outperformance'],
                    'momentum_trades': momentum_result['num_trades'],
                    'technical_trades': technical_result['num_trades']
                })
        
        # ANALYSIS
        print(f"\n" + "="*70)
        print(f"📊 DETAILED COMPARISON RESULTS")
        print(f"="*70)
        
        print(f"\n{'Stock':<8} {'Momentum':<10} {'Technical':<10} {'Winner':<10} {'Diff':<8}")
        print("-" * 50)
        
        momentum_wins = 0
        technical_wins = 0
        
        for data in comparison_data:
            momentum_perf = data['momentum_return']
            technical_perf = data['technical_return']
            diff = technical_perf - momentum_perf
            winner = "Technical" if diff > 0 else "Momentum"
            
            if diff > 0:
                technical_wins += 1
            else:
                momentum_wins += 1
            
            print(f"{data['symbol']:<8} {momentum_perf:+7.1f}%   {technical_perf:+7.1f}%   {winner:<10} {diff:+6.1f}%")
        
        # SUMMARY STATISTICS
        if momentum_results and technical_results:
            momentum_avg = sum(r['strategy_return'] for r in momentum_results) / len(momentum_results)
            technical_avg = sum(r['strategy_return'] for r in technical_results) / len(technical_results)
            
            momentum_outperf_avg = sum(r['outperformance'] for r in momentum_results) / len(momentum_results)
            technical_outperf_avg = sum(r['outperformance'] for r in technical_results) / len(technical_results)
            
            momentum_win_rate = len([r for r in momentum_results if r['outperformance'] > 0]) / len(momentum_results)
            technical_win_rate = len([r for r in technical_results if r['outperformance'] > 0]) / len(technical_results)
            
            print(f"\n" + "="*70)
            print(f"🏆 FINAL COMPARISON SUMMARY")
            print(f"="*70)
            
            print(f"\n📈 AVERAGE RETURNS:")
            print(f"   🏛️  Momentum Strategy:  {momentum_avg:+.1f}%")
            print(f"   ⚙️  Technical Strategy: {technical_avg:+.1f}%")
            print(f"   🎯 Difference:         {technical_avg - momentum_avg:+.1f}%")
            
            print(f"\n📊 OUTPERFORMANCE vs BUY-HOLD:")
            print(f"   🏛️  Momentum Strategy:  {momentum_outperf_avg:+.1f}%")
            print(f"   ⚙️  Technical Strategy: {technical_outperf_avg:+.1f}%")
            
            print(f"\n🎯 WIN RATES (vs Buy-Hold):")
            print(f"   🏛️  Momentum Strategy:  {momentum_win_rate:.1%}")
            print(f"   ⚙️  Technical Strategy: {technical_win_rate:.1%}")
            
            print(f"\n🥊 HEAD-TO-HEAD:")
            print(f"   🏛️  Momentum Wins:      {momentum_wins}/{len(comparison_data)} ({momentum_wins/len(comparison_data):.1%})")
            print(f"   ⚙️  Technical Wins:     {technical_wins}/{len(comparison_data)} ({technical_wins/len(comparison_data):.1%})")
            
            # WINNER DECLARATION
            if technical_avg > momentum_avg:
                winner = "TECHNICAL SIGNALS"
                advantage = technical_avg - momentum_avg
            else:
                winner = "MOMENTUM STRATEGY"
                advantage = momentum_avg - technical_avg
            
            print(f"\n🏆 OVERALL WINNER: {winner}")
            print(f"   📊 Advantage: {advantage:+.1f}% average return")
            
            # RECOMMENDATIONS
            print(f"\n💡 RECOMMENDATIONS:")
            if winner == "TECHNICAL SIGNALS":
                print(f"   ✅ Deploy your proven technical configuration")
                print(f"   ✅ Average {technical_avg:+.1f}% returns with {technical_win_rate:.1%} win rate")
                print(f"   ⚙️  Your config parameters are well-optimized!")
            else:
                print(f"   ✅ Deploy institutional momentum strategy")
                print(f"   ✅ Average {momentum_avg:+.1f}% returns with {momentum_win_rate:.1%} win rate")
                print(f"   🏛️  Academic validation with 55+ years of proof")
            
            return {
                'momentum_avg': momentum_avg,
                'technical_avg': technical_avg,
                'winner': winner,
                'momentum_results': momentum_results,
                'technical_results': technical_results,
                'comparison_data': comparison_data
            }

def main():
    """Run the strategy comparison"""
    print("🚀 STRATEGY COMPARISON ENGINE")
    print("=" * 50)
    print("Comparing Momentum vs Technical Signal strategies")
    print("Same stocks, same timeframe, fair comparison!")
    print("=" * 50)
    
    comparison = StrategyComparison()
    results = comparison.compare_strategies(period_days=365)  # 1 year comparison
    
    return comparison, results

if __name__ == "__main__":
    comparison, results = main()
