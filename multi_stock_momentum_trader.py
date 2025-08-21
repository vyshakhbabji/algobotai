#!/usr/bin/env python3
"""
MULTI-STOCK MOMENTUM TRADER
Test our optimized momentum strategy across all stocks with $10K each
Shows which stocks work best with momentum signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class MultiStockMomentumTrader:
    def __init__(self, starting_capital=10000):
        self.starting_capital = starting_capital
        
        # Stock universe from our optimizer
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
        
        # MOMENTUM-OPTIMIZED THRESHOLDS (from our successful NVDA test)
        self.config = {
            'trend_5d_buy_threshold': 0.015,    # 1.5% - Catches early momentum
            'trend_5d_sell_threshold': -0.025,  # -2.5% - Protects profits
            'trend_10d_buy_threshold': 0.015,   # 1.5% - Confirms momentum
            'trend_10d_sell_threshold': -0.045, # -4.5% - Strict exit
            'rsi_overbought': 85,               # 85 - Allow momentum runs
            'rsi_oversold': 30,                 # 30 - Realistic oversold
            'volatility_threshold': 0.10,       # 10% - Higher tolerance
            'volume_ratio_threshold': 1.1       # 1.1x - Easy to trigger
        }
        
        print(f"📈 MULTI-STOCK MOMENTUM TRADER")
        print(f"💰 Capital per stock: ${starting_capital:,}")
        print(f"📊 Testing {len(self.stocks)} stocks")
        print(f"💎 Total capital deployed: ${starting_capital * len(self.stocks):,}")
        
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        data = data.copy()
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        
        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def get_signal(self, data, idx):
        """Get trading signal for current day"""
        try:
            price = float(data['Close'].iloc[idx])
            
            # Trend calculations
            recent_5d = data['Close'].iloc[idx-5:idx]
            recent_10d = data['Close'].iloc[idx-10:idx]
            
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Technical indicators
            rsi = float(data['RSI'].iloc[idx]) if not pd.isna(data['RSI'].iloc[idx]) else 50
            
            # Volume analysis
            recent_vol = float(data['Volume'].iloc[idx-10:idx].mean())
            current_vol = float(data['Volume'].iloc[idx])
            vol_ratio = current_vol / recent_vol if recent_vol > 0 else 1
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # SIGNAL DETERMINATION
            signal = 'HOLD'
            
            # BUY: Strong dual-trend momentum
            if (trend_5d > self.config['trend_5d_buy_threshold'] and 
                trend_10d > self.config['trend_10d_buy_threshold'] and
                rsi < self.config['rsi_overbought'] and
                volatility < self.config['volatility_threshold']):
                signal = 'BUY'
            
            # SELL: Trend breakdown or extreme overbought
            elif (trend_5d < self.config['trend_5d_sell_threshold'] and 
                  trend_10d < self.config['trend_10d_sell_threshold']) or \
                 (rsi > self.config['rsi_overbought'] and trend_5d < 0):
                signal = 'SELL'
            
            return {
                'signal': signal,
                'price': price,
                'trend_5d': trend_5d,
                'trend_10d': trend_10d,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': vol_ratio
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'price': price if 'price' in locals() else 0}
    
    def trade_single_stock(self, symbol):
        """Trade a single stock with momentum strategy"""
        try:
            print(f"\n📈 Trading {symbol}")
            print("-" * 30)
            
            # Download 4 months of data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=120)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 20:
                print(f"❌ {symbol}: Insufficient data")
                return None
                
            data = self.calculate_indicators(data)
            
            # Portfolio state
            cash = self.starting_capital
            shares = 0
            position = None
            trades = []
            
            # Trading simulation
            for i in range(15, len(data)):
                date = data.index[i]
                signal_data = self.get_signal(data, i)
                price = signal_data['price']
                signal = signal_data['signal']
                
                # Execute trades
                if signal == 'BUY' and position != 'LONG' and cash > 0:
                    # Buy with full available cash
                    shares_to_buy = cash / price
                    amount = shares_to_buy * price
                    
                    shares += shares_to_buy
                    cash = 0
                    position = 'LONG'
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'amount': amount
                    })
                    
                    print(f"📈 BUY:  {date.strftime('%m/%d')} | ${price:6.2f} | {shares_to_buy:5.1f} shares")
                    
                elif signal == 'SELL' and position == 'LONG' and shares > 0:
                    # Sell all shares
                    amount = shares * price
                    sold_shares = shares
                    
                    # Calculate trade profit
                    last_buy = None
                    for t in reversed(trades):
                        if t['action'] == 'BUY':
                            last_buy = t
                            break
                    
                    profit = amount - last_buy['amount'] if last_buy else 0
                    profit_pct = (profit / last_buy['amount']) * 100 if last_buy else 0
                    
                    cash = amount
                    shares = 0
                    position = None
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': sold_shares,
                        'amount': amount,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                    
                    print(f"📉 SELL: {date.strftime('%m/%d')} | ${price:6.2f} | Profit: {profit_pct:+5.1f}%")
            
            # Final calculations
            final_price = float(data['Close'].iloc[-1])
            final_value = cash + (shares * final_price)
            
            start_price = float(data['Close'].iloc[15])
            buy_hold_value = self.starting_capital * (final_price / start_price)
            
            strategy_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
            buy_hold_return = ((buy_hold_value - self.starting_capital) / self.starting_capital) * 100
            outperformance = strategy_return - buy_hold_return
            
            result = {
                'symbol': symbol,
                'final_value': final_value,
                'strategy_return': strategy_return,
                'buy_hold_value': buy_hold_value,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'trades': trades,
                'final_cash': cash,
                'final_shares': shares,
                'price_change': ((final_price - start_price) / start_price) * 100
            }
            
            # Display results
            status = "✅" if outperformance > 0 else "❌"
            print(f"{status} Final: ${final_value:7,.0f} ({strategy_return:+5.1f}%) vs BH: {buy_hold_return:+5.1f}% | Diff: {outperformance:+5.1f}% | Trades: {len(trades)}")
            
            return result
            
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")
            return None
    
    def run_all_stocks(self):
        """Run momentum strategy on all stocks"""
        print(f"\n🚀 RUNNING MOMENTUM STRATEGY ON ALL {len(self.stocks)} STOCKS")
        print("=" * 80)
        
        results = []
        total_deployed = 0
        total_final_value = 0
        
        for symbol in self.stocks:
            result = self.trade_single_stock(symbol)
            if result:
                results.append(result)
                total_deployed += self.starting_capital
                total_final_value += result['final_value']
        
        # Overall portfolio analysis
        print(f"\n" + "="*80)
        print(f"🏆 MULTI-STOCK MOMENTUM PORTFOLIO RESULTS")
        print("="*80)
        
        if results:
            # Sort by performance
            results.sort(key=lambda x: x['outperformance'], reverse=True)
            
            # Portfolio totals
            total_strategy_return = ((total_final_value - total_deployed) / total_deployed) * 100
            total_buy_hold_value = sum(r['buy_hold_value'] for r in results)
            total_buy_hold_return = ((total_buy_hold_value - total_deployed) / total_deployed) * 100
            total_outperformance = total_strategy_return - total_buy_hold_return
            
            print(f"💰 PORTFOLIO TOTALS:")
            print(f"   🏦 Capital Deployed:  ${total_deployed:10,.0f}")
            print(f"   📈 Strategy Value:    ${total_final_value:10,.0f} ({total_strategy_return:+6.1f}%)")
            print(f"   🎯 Buy-Hold Value:    ${total_buy_hold_value:10,.0f} ({total_buy_hold_return:+6.1f}%)")
            print(f"   🏆 Total Outperform:  ${total_final_value - total_buy_hold_value:10,.0f} ({total_outperformance:+6.1f}%)")
            
            # Performance breakdown
            winners = [r for r in results if r['outperformance'] > 0]
            losers = [r for r in results if r['outperformance'] <= 0]
            
            print(f"\n📊 PERFORMANCE BREAKDOWN:")
            print(f"   ✅ Winners: {len(winners)}/{len(results)} ({len(winners)/len(results):.1%})")
            print(f"   ❌ Losers:  {len(losers)}/{len(results)} ({len(losers)/len(results):.1%})")
            
            if winners:
                avg_winner_outperform = sum(r['outperformance'] for r in winners) / len(winners)
                best_winner = max(winners, key=lambda x: x['outperformance'])
                print(f"   🥇 Best Winner: {best_winner['symbol']} ({best_winner['outperformance']:+.1f}%)")
                print(f"   📈 Avg Winner Outperform: {avg_winner_outperform:+.1f}%")
            
            if losers:
                avg_loser_underperform = sum(r['outperformance'] for r in losers) / len(losers)
                worst_loser = min(losers, key=lambda x: x['outperformance'])
                print(f"   📉 Worst Loser: {worst_loser['symbol']} ({worst_loser['outperformance']:+.1f}%)")
                print(f"   🔻 Avg Loser Underperform: {avg_loser_underperform:+.1f}%")
            
            # Top performers
            print(f"\n🏅 TOP 10 PERFORMERS:")
            for i, result in enumerate(results[:10], 1):
                status = "✅" if result['outperformance'] > 0 else "❌"
                print(f"   #{i:2d} {status} {result['symbol']:4s}: ${result['final_value']:7,.0f} ({result['strategy_return']:+6.1f}%) | "
                      f"vs BH: {result['outperformance']:+6.1f}% | Trades: {len(result['trades'])}")
            
            # Trading activity analysis
            total_trades = sum(len(r['trades']) for r in results)
            active_traders = len([r for r in results if len(r['trades']) > 0])
            
            print(f"\n🔄 TRADING ACTIVITY:")
            print(f"   📈 Total Trades: {total_trades}")
            print(f"   🎯 Active Stocks: {active_traders}/{len(results)}")
            print(f"   📊 Avg Trades per Stock: {total_trades/len(results):.1f}")
            
            # Stock category analysis
            tech_stocks = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX']
            tech_results = [r for r in results if r['symbol'] in tech_stocks]
            
            if tech_results:
                tech_outperform = sum(r['outperformance'] for r in tech_results) / len(tech_results)
                print(f"\n🖥️  TECH SECTOR ANALYSIS:")
                print(f"   📊 Tech Stocks: {len(tech_results)}")
                print(f"   📈 Avg Tech Outperformance: {tech_outperform:+.1f}%")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"multi_stock_momentum_results_{timestamp}.json"
            
            portfolio_summary = {
                'total_deployed': total_deployed,
                'total_final_value': total_final_value,
                'total_strategy_return': total_strategy_return,
                'total_buy_hold_return': total_buy_hold_return,
                'total_outperformance': total_outperformance,
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': len(winners)/len(results),
                'individual_results': results,
                'config_used': self.config
            }
            
            with open(filename, 'w') as f:
                json.dump(portfolio_summary, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to {filename}")
            
            return portfolio_summary
        
        else:
            print("❌ No successful trades!")
            return None

def main():
    """Run multi-stock momentum trader"""
    trader = MultiStockMomentumTrader(starting_capital=10000)
    results = trader.run_all_stocks()
    
    if results:
        print(f"\n🎯 SUMMARY:")
        print(f"   💰 Deployed ${results['total_deployed']:,} across {len(trader.stocks)} stocks")
        print(f"   📈 Returned ${results['total_final_value']:,.0f} ({results['total_strategy_return']:+.1f}%)")
        print(f"   🏆 Beat buy-and-hold by {results['total_outperformance']:+.1f}%")
        print(f"   🎯 Win rate: {results['win_rate']:.1%} ({results['winners']}/{results['winners'] + results['losers']})")
        
        # Key insights
        if results['total_outperformance'] > 0:
            print(f"   ✅ MOMENTUM STRATEGY SUCCESSFUL!")
        else:
            print(f"   ❌ Strategy underperformed buy-and-hold")
    
    return results

if __name__ == "__main__":
    results = main()
