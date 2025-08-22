"""
🎯 ENHANCED PARAMETER OPTIMIZATION (DEBUGGED)
Quick Win: Target 5-10% Performance Boost

Fix the optimization issues and focus on the most impactful parameters
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

class SmartParameterOptimizer:
    """Simplified but effective parameter optimization"""
    
    def __init__(self):
        self.baseline_return = 57.6
        self.results = []
        
    def get_stock_data(self, symbol: str, period: str = '6mo') -> pd.DataFrame:
        """Get clean stock data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            return df
            
        except Exception as e:
            print(f"❌ Data error for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading features"""
        if df.empty or len(df) < 20:
            return df
            
        df = df.copy()
        
        # Basic features
        df['returns'] = df['Close'].pct_change()
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        
        # Moving averages
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['price_sma_ratio'] = df['Close'] / df['sma_20']
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volume
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Momentum
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, df: pd.DataFrame, params: dict) -> float:
        """Generate trading signal with parameters"""
        if df.empty or len(df) < 20:
            return 0.0
        
        latest = df.iloc[-1]
        
        # Get indicators
        rsi = latest.get('rsi', 50)
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        price_sma_ratio = latest.get('price_sma_ratio', 1.0)
        volume_ratio = latest.get('volume_ratio', 1.0)
        momentum_5 = latest.get('momentum_5', 0)
        momentum_10 = latest.get('momentum_10', 0)
        
        # Calculate signal components
        signal = 0.0
        
        # RSI component (optimized thresholds)
        rsi_buy_threshold = params.get('rsi_buy_threshold', 35)
        rsi_sell_threshold = params.get('rsi_sell_threshold', 65)
        
        if rsi < rsi_buy_threshold:
            signal += params.get('rsi_weight', 0.3)
        elif rsi > rsi_sell_threshold:
            signal -= params.get('rsi_weight', 0.3)
        
        # MACD component
        if macd > macd_signal and macd > 0:
            signal += params.get('macd_weight', 0.25)
        elif macd < macd_signal and macd < 0:
            signal -= params.get('macd_weight', 0.25)
        
        # Price momentum component
        momentum_threshold = params.get('momentum_threshold', 0.02)
        if momentum_5 > momentum_threshold and momentum_10 > 0:
            signal += params.get('momentum_weight', 0.2)
        elif momentum_5 < -momentum_threshold and momentum_10 < 0:
            signal -= params.get('momentum_weight', 0.2)
        
        # Volume confirmation
        volume_threshold = params.get('volume_threshold', 1.2)
        if volume_ratio > volume_threshold:
            signal *= params.get('volume_multiplier', 1.1)
        
        # Price trend component
        if price_sma_ratio > params.get('trend_threshold', 1.02):
            signal += params.get('trend_weight', 0.15)
        elif price_sma_ratio < (2 - params.get('trend_threshold', 1.02)):
            signal -= params.get('trend_weight', 0.15)
        
        return np.clip(signal, -1.0, 1.0)
    
    def backtest_strategy(self, params: dict, stocks: list, start_date: str = "2024-05-20", 
                         end_date: str = "2024-08-20") -> dict:
        """Backtest strategy with given parameters"""
        
        # Portfolio setup
        initial_cash = 50000
        cash = initial_cash
        positions = {}
        portfolio_values = []
        
        # Trading parameters
        position_size_pct = params.get('position_size_pct', 0.15)
        min_signal_strength = params.get('min_signal_strength', 0.4)
        max_positions = params.get('max_positions', 8)
        
        # Date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        current_date = start_dt
        
        trade_count = 0
        daily_values = []
        
        # Get all stock data upfront
        stock_data = {}
        for symbol in stocks:
            df = self.get_stock_data(symbol, '1y')
            if not df.empty:
                stock_data[symbol] = self.calculate_features(df)
        
        # Trading simulation
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Trading days only
                
                # Calculate current portfolio value
                portfolio_value = cash
                for symbol, quantity in positions.items():
                    if symbol in stock_data:
                        try:
                            # Get price for current date
                            df = stock_data[symbol]
                            date_mask = pd.to_datetime(df['Date']).dt.date <= current_date.date()
                            if date_mask.any():
                                current_price = df[date_mask]['Close'].iloc[-1]
                                portfolio_value += quantity * current_price
                        except:
                            pass
                
                daily_values.append(portfolio_value)
                
                # Generate signals and trade
                for symbol in stocks:
                    if symbol not in stock_data:
                        continue
                    
                    try:
                        df = stock_data[symbol]
                        
                        # Get data up to current date
                        date_mask = pd.to_datetime(df['Date']).dt.date <= current_date.date()
                        current_df = df[date_mask]
                        
                        if len(current_df) < 20:
                            continue
                        
                        current_price = current_df['Close'].iloc[-1]
                        signal = self.generate_signal(current_df, params)
                        
                        current_position = positions.get(symbol, 0)
                        position_value = current_position * current_price
                        
                        # Buy signal
                        if (signal > min_signal_strength and 
                            current_position == 0 and 
                            len(positions) < max_positions):
                            
                            position_size = portfolio_value * position_size_pct
                            if position_size <= cash:
                                quantity = int(position_size / current_price)
                                if quantity > 0:
                                    cost = quantity * current_price
                                    positions[symbol] = quantity
                                    cash -= cost
                                    trade_count += 1
                        
                        # Sell signal
                        elif signal < -min_signal_strength and current_position > 0:
                            proceeds = current_position * current_price
                            cash += proceeds
                            del positions[symbol]
                            trade_count += 1
                    
                    except Exception as e:
                        continue
            
            current_date += timedelta(days=1)
        
        # Final calculation
        final_value = daily_values[-1] if daily_values else initial_cash
        total_return = (final_value / initial_cash - 1) * 100
        
        # Annualize
        trading_days = len([d for d in pd.date_range(start_dt, end_dt) if d.weekday() < 5])
        if trading_days > 0:
            annualized_return = ((final_value / initial_cash) ** (252 / trading_days) - 1) * 100
        else:
            annualized_return = 0
        
        return {
            'params': params,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'total_trades': trade_count,
            'final_positions': len(positions)
        }
    
    def optimize_parameters(self) -> dict:
        """Run focused parameter optimization"""
        
        print("🎯 SMART PARAMETER OPTIMIZATION")
        print("Enhancement C: Focused on high-impact parameters")
        print("=" * 50)
        
        # Top performing stocks
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        
        # Parameter test sets (focused on most impactful)
        test_configs = [
            # Baseline configuration
            {
                'name': 'Baseline',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.4,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'rsi_weight': 0.3,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.2,
                'volume_multiplier': 1.1,
                'trend_threshold': 1.02,
                'trend_weight': 0.15,
                'max_positions': 8
            },
            
            # More aggressive position sizing
            {
                'name': 'Aggressive Sizing',
                'position_size_pct': 0.18,
                'min_signal_strength': 0.35,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'rsi_weight': 0.3,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.2,
                'volume_multiplier': 1.1,
                'trend_threshold': 1.02,
                'trend_weight': 0.15,
                'max_positions': 8
            },
            
            # Lower signal thresholds
            {
                'name': 'Lower Thresholds',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.3,
                'rsi_buy_threshold': 40,
                'rsi_sell_threshold': 60,
                'rsi_weight': 0.35,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.015,
                'volume_threshold': 1.15,
                'volume_multiplier': 1.15,
                'trend_threshold': 1.015,
                'trend_weight': 0.2,
                'max_positions': 8
            },
            
            # Momentum focused
            {
                'name': 'Momentum Focus',
                'position_size_pct': 0.16,
                'min_signal_strength': 0.4,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'rsi_weight': 0.25,
                'macd_weight': 0.3,
                'momentum_weight': 0.3,
                'momentum_threshold': 0.025,
                'volume_threshold': 1.25,
                'volume_multiplier': 1.2,
                'trend_threshold': 1.025,
                'trend_weight': 0.15,
                'max_positions': 8
            },
            
            # Volume confirmation enhanced
            {
                'name': 'Volume Enhanced',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.4,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'rsi_weight': 0.3,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.3,
                'volume_multiplier': 1.25,
                'trend_threshold': 1.02,
                'trend_weight': 0.15,
                'max_positions': 8
            }
        ]
        
        best_result = None
        best_return = self.baseline_return
        
        print(f"🧪 Testing {len(test_configs)} optimized configurations...")
        
        for i, config in enumerate(test_configs):
            try:
                print(f"\n🔬 Test {i+1}/{len(test_configs)}: {config['name']}")
                
                result = self.backtest_strategy(config, stocks)
                annual_return = result['annualized_return_pct']
                improvement = annual_return - self.baseline_return
                
                print(f"   📊 Return: {annual_return:.1f}% ({improvement:+.1f}%)")
                print(f"   💼 Trades: {result['total_trades']}, Positions: {result['final_positions']}")
                
                self.results.append({
                    'config_name': config['name'],
                    'result': result
                })
                
                if annual_return > best_return:
                    best_return = annual_return
                    best_result = result
                    print(f"   🎉 NEW BEST: {annual_return:.1f}% (+{improvement:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return best_result
    
    def run_optimization(self):
        """Main optimization runner"""
        
        print("🎯 ENHANCEMENT C: SMART PARAMETER OPTIMIZATION")
        print("Target: 5-10% performance boost (62-67% annual returns)")
        print("=" * 60)
        
        best_result = self.optimize_parameters()
        
        if best_result:
            baseline = self.baseline_return
            optimized = best_result['annualized_return_pct']
            improvement = optimized - baseline
            
            print(f"\n" + "=" * 60)
            print("🏆 OPTIMIZATION RESULTS")
            print("=" * 60)
            print(f"📊 Baseline Return: {baseline:.1f}%")
            print(f"🎯 Optimized Return: {optimized:.1f}%")
            print(f"📈 Improvement: {improvement:+.1f}%")
            print(f"💼 Total Trades: {best_result['total_trades']}")
            print(f"📈 Final Value: ${best_result['final_value']:,.0f}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"smart_optimization_results_{timestamp}.json"
            
            summary = {
                'baseline_return': baseline,
                'best_result': best_result,
                'improvement_pct': improvement,
                'all_results': self.results,
                'success': improvement >= 5
            }
            
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {results_file}")
            
            if improvement >= 5:
                print(f"\n🎉 SUCCESS! Enhancement C delivered {improvement:.1f}% boost!")
                print("✅ Ready for Enhancement D (Advanced ML Models)")
                return True
            elif improvement >= 2:
                print(f"\n✅ Good improvement: {improvement:.1f}% boost")
                print("💡 Proceed to Enhancement D for additional gains")
                return True
            else:
                print(f"\n⚠️ Modest improvement: {improvement:.1f}%")
                return False
        
        else:
            print("\n❌ Optimization failed - no valid results")
            return False

def main():
    """Run smart parameter optimization"""
    optimizer = SmartParameterOptimizer()
    return optimizer.run_optimization()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to proceed with Enhancement D!")
    else:
        print("\n🔧 Need to adjust optimization approach")
