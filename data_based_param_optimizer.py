"""
🎯 ENHANCEMENT C: PARAMETER OPTIMIZATION (Using Existing Data)
Quick Win: Target 5-10% Performance Boost

Use existing clean data files to avoid rate limits
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

class DataBasedParameterOptimizer:
    """Parameter optimization using existing clean data files"""
    
    def __init__(self):
        self.baseline_return = 57.6
        self.results = []
        self.data_cache = {}
        
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load stock data from existing clean files"""
        try:
            # Try to find clean data file
            filename = f"clean_data_{symbol}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                # Ensure standard column names
                if 'Unnamed: 0' in df.columns:
                    df = df.drop('Unnamed: 0', axis=1)
                
                # Standardize column names
                column_mapping = {
                    'date': 'Date',
                    'open': 'Open', 
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df[old_col] = new_col
                
                # Ensure Date column is datetime
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                elif df.index.name == 'Date' or isinstance(df.index[0], (pd.Timestamp, str)):
                    df = df.reset_index()
                    df['Date'] = pd.to_datetime(df['Date'])
                
                return df
            else:
                print(f"⚠️ No clean data file found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")
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
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['price_sma_ratio'] = df['Close'] / df['sma_20']
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume features
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Momentum features
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Bollinger Bands
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + (bb_std * 2)
        df['bb_lower'] = df['sma_20'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_optimized_signal(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Generate trading signals with optimized parameters"""
        if df.empty or len(df) < 50:
            return pd.Series(index=df.index, data=0.0)
        
        signals = pd.Series(index=df.index, data=0.0)
        
        # Enhanced signal logic with parameters
        for i in range(50, len(df)):
            current = df.iloc[i]
            
            signal_strength = 0.0
            
            # RSI component (optimized)
            rsi = current['rsi']
            rsi_buy = params.get('rsi_buy_threshold', 35)
            rsi_sell = params.get('rsi_sell_threshold', 65)
            rsi_weight = params.get('rsi_weight', 0.3)
            
            if rsi < rsi_buy:
                signal_strength += rsi_weight
            elif rsi > rsi_sell:
                signal_strength -= rsi_weight
            
            # MACD component
            macd = current['macd']
            macd_signal = current['macd_signal']
            macd_hist = current['macd_histogram']
            macd_weight = params.get('macd_weight', 0.25)
            
            if macd > macd_signal and macd_hist > 0:
                signal_strength += macd_weight
            elif macd < macd_signal and macd_hist < 0:
                signal_strength -= macd_weight
            
            # Momentum component
            momentum_5 = current['momentum_5']
            momentum_10 = current['momentum_10']
            momentum_threshold = params.get('momentum_threshold', 0.02)
            momentum_weight = params.get('momentum_weight', 0.2)
            
            if momentum_5 > momentum_threshold and momentum_10 > 0:
                signal_strength += momentum_weight
            elif momentum_5 < -momentum_threshold and momentum_10 < 0:
                signal_strength -= momentum_weight
            
            # Volume confirmation
            volume_ratio = current['volume_ratio']
            volume_threshold = params.get('volume_threshold', 1.2)
            volume_multiplier = params.get('volume_multiplier', 1.1)
            
            if volume_ratio > volume_threshold:
                signal_strength *= volume_multiplier
            elif volume_ratio < 0.8:
                signal_strength *= 0.9
            
            # Price trend component
            price_sma_ratio = current['price_sma_ratio']
            trend_threshold = params.get('trend_threshold', 1.02)
            trend_weight = params.get('trend_weight', 0.15)
            
            if price_sma_ratio > trend_threshold:
                signal_strength += trend_weight
            elif price_sma_ratio < (2 - trend_threshold):
                signal_strength -= trend_weight
            
            # Bollinger Band component
            bb_position = current['bb_position']
            if bb_position < 0.2:  # Near lower band
                signal_strength += 0.1
            elif bb_position > 0.8:  # Near upper band
                signal_strength -= 0.1
            
            signals.iloc[i] = np.clip(signal_strength, -1.0, 1.0)
        
        return signals
    
    def backtest_optimized_strategy(self, params: dict, symbols: list, 
                                   start_date: str = "2024-05-20", 
                                   end_date: str = "2024-08-20") -> dict:
        """Backtest strategy with optimized parameters"""
        
        # Portfolio setup
        initial_cash = 50000
        cash = initial_cash
        positions = {}
        portfolio_values = []
        trades = []
        
        # Parameters
        position_size_pct = params.get('position_size_pct', 0.15)
        min_signal_strength = params.get('min_signal_strength', 0.4)
        max_positions = params.get('max_positions', 8)
        stop_loss_pct = params.get('stop_loss_pct', 0.08)
        take_profit_pct = params.get('take_profit_pct', 0.15)
        
        # Load and prepare all stock data
        stock_signals = {}
        stock_prices = {}
        
        for symbol in symbols:
            df = self.load_stock_data(symbol)
            if not df.empty and len(df) > 100:
                df_featured = self.calculate_features(df)
                if not df_featured.empty:
                    # Filter date range
                    df_featured = df_featured[
                        (df_featured['Date'] >= start_date) & 
                        (df_featured['Date'] <= end_date)
                    ].copy()
                    
                    if len(df_featured) > 20:
                        signals = self.generate_optimized_signal(df_featured, params)
                        stock_signals[symbol] = signals
                        stock_prices[symbol] = df_featured.set_index('Date')['Close']
        
        if not stock_signals:
            return {'annualized_return_pct': 0, 'total_trades': 0, 'final_value': initial_cash}
        
        # Get common date range
        all_dates = set()
        for symbol in stock_signals:
            all_dates.update(stock_signals[symbol].index)
        
        trading_dates = sorted([d for d in all_dates 
                              if pd.to_datetime(start_date) <= pd.to_datetime(d) <= pd.to_datetime(end_date)])
        
        # Trading simulation
        for date in trading_dates:
            try:
                current_portfolio_value = cash
                
                # Update portfolio value
                for symbol, quantity in positions.items():
                    if symbol in stock_prices and date in stock_prices[symbol].index:
                        current_price = stock_prices[symbol][date]
                        current_portfolio_value += quantity * current_price
                
                portfolio_values.append(current_portfolio_value)
                
                # Process each stock
                for symbol in stock_signals.keys():
                    if (symbol in stock_signals and 
                        date in stock_signals[symbol].index and
                        symbol in stock_prices and 
                        date in stock_prices[symbol].index):
                        
                        signal = stock_signals[symbol][date]
                        current_price = stock_prices[symbol][date]
                        current_position = positions.get(symbol, 0)
                        
                        # Buy logic
                        if (signal > min_signal_strength and 
                            current_position == 0 and 
                            len(positions) < max_positions):
                            
                            position_value = current_portfolio_value * position_size_pct
                            if position_value <= cash:
                                quantity = int(position_value / current_price)
                                if quantity > 0:
                                    cost = quantity * current_price
                                    positions[symbol] = quantity
                                    cash -= cost
                                    trades.append({
                                        'date': date,
                                        'symbol': symbol,
                                        'action': 'buy',
                                        'quantity': quantity,
                                        'price': current_price,
                                        'signal': signal
                                    })
                        
                        # Sell logic
                        elif signal < -min_signal_strength and current_position > 0:
                            proceeds = current_position * current_price
                            cash += proceeds
                            del positions[symbol]
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'sell',
                                'quantity': current_position,
                                'price': current_price,
                                'signal': signal
                            })
            
            except Exception as e:
                continue
        
        # Final portfolio value
        final_value = cash
        for symbol, quantity in positions.items():
            if symbol in stock_prices:
                try:
                    final_price = stock_prices[symbol].iloc[-1]
                    final_value += quantity * final_price
                except:
                    pass
        
        # Performance calculation
        total_return = (final_value / initial_cash - 1) * 100
        days = len(trading_dates)
        if days > 0:
            annualized_return = ((final_value / initial_cash) ** (252 / days) - 1) * 100
        else:
            annualized_return = 0
        
        return {
            'params': params,
            'final_value': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'total_trades': len(trades),
            'final_positions': len(positions),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values) if portfolio_values else 0
        }
    
    def _calculate_max_drawdown(self, portfolio_values: list) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values:
            return 0
        
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100
    
    def run_parameter_optimization(self):
        """Run comprehensive parameter optimization"""
        
        print("🎯 ENHANCEMENT C: DATA-BASED PARAMETER OPTIMIZATION")
        print("Target: 5-10% performance boost (62-67% annual returns)")
        print("=" * 60)
        
        # Check available data files
        available_symbols = []
        for file in os.listdir('.'):
            if file.startswith('clean_data_') and file.endswith('.csv'):
                symbol = file.replace('clean_data_', '').replace('.csv', '').upper()
                available_symbols.append(symbol)
        
        print(f"📊 Found data for {len(available_symbols)} symbols: {', '.join(available_symbols[:8])}")
        
        if len(available_symbols) < 4:
            print("❌ Insufficient data files found")
            return None
        
        # Use top symbols (limit to avoid overload)
        symbols = available_symbols[:8]
        
        # Parameter configurations to test
        test_configs = [
            {
                'name': 'Baseline',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.4,
                'max_positions': 8,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'rsi_weight': 0.3,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.2,
                'volume_multiplier': 1.1,
                'trend_threshold': 1.02,
                'trend_weight': 0.15
            },
            {
                'name': 'Aggressive Position Sizing',
                'position_size_pct': 0.18,
                'min_signal_strength': 0.35,
                'max_positions': 8,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'rsi_weight': 0.3,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.2,
                'volume_multiplier': 1.1,
                'trend_threshold': 1.02,
                'trend_weight': 0.15
            },
            {
                'name': 'Lower Signal Thresholds',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.3,
                'max_positions': 8,
                'rsi_buy_threshold': 40,
                'rsi_sell_threshold': 60,
                'rsi_weight': 0.35,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.015,
                'volume_threshold': 1.15,
                'volume_multiplier': 1.15,
                'trend_threshold': 1.015,
                'trend_weight': 0.2
            },
            {
                'name': 'Momentum Focused',
                'position_size_pct': 0.16,
                'min_signal_strength': 0.4,
                'max_positions': 8,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'rsi_weight': 0.25,
                'macd_weight': 0.3,
                'momentum_weight': 0.3,
                'momentum_threshold': 0.025,
                'volume_threshold': 1.25,
                'volume_multiplier': 1.2,
                'trend_threshold': 1.025,
                'trend_weight': 0.15
            },
            {
                'name': 'Volume Confirmation Enhanced',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.4,
                'max_positions': 8,
                'rsi_buy_threshold': 32,
                'rsi_sell_threshold': 68,
                'rsi_weight': 0.3,
                'macd_weight': 0.25,
                'momentum_weight': 0.2,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.3,
                'volume_multiplier': 1.25,
                'trend_threshold': 1.02,
                'trend_weight': 0.15
            },
            {
                'name': 'Conservative High Confidence',
                'position_size_pct': 0.12,
                'min_signal_strength': 0.5,
                'max_positions': 6,
                'rsi_buy_threshold': 30,
                'rsi_sell_threshold': 70,
                'rsi_weight': 0.35,
                'macd_weight': 0.3,
                'momentum_weight': 0.25,
                'momentum_threshold': 0.03,
                'volume_threshold': 1.3,
                'volume_multiplier': 1.3,
                'trend_threshold': 1.03,
                'trend_weight': 0.2
            }
        ]
        
        print(f"🧪 Testing {len(test_configs)} parameter configurations...")
        
        best_result = None
        best_return = self.baseline_return
        
        for i, config in enumerate(test_configs):
            try:
                print(f"\n🔬 Test {i+1}/{len(test_configs)}: {config['name']}")
                
                result = self.backtest_optimized_strategy(config, symbols)
                annual_return = result['annualized_return_pct']
                improvement = annual_return - self.baseline_return
                
                print(f"   📊 Return: {annual_return:.1f}% ({improvement:+.1f}%)")
                print(f"   💼 Trades: {result['total_trades']}, Final Value: ${result['final_value']:,.0f}")
                print(f"   📉 Max Drawdown: {result['max_drawdown']:.1f}%")
                
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
                import traceback
                traceback.print_exc()
        
        # Results summary
        if best_result:
            baseline = self.baseline_return
            optimized = best_result['annualized_return_pct']
            improvement = optimized - baseline
            
            print(f"\n" + "=" * 60)
            print("🏆 PARAMETER OPTIMIZATION RESULTS")
            print("=" * 60)
            print(f"📊 Baseline Return: {baseline:.1f}%")
            print(f"🎯 Optimized Return: {optimized:.1f}%")
            print(f"📈 Improvement: {improvement:+.1f}%")
            print(f"💼 Total Trades: {best_result['total_trades']}")
            print(f"📈 Final Value: ${best_result['final_value']:,.0f}")
            print(f"📉 Max Drawdown: {best_result['max_drawdown']:.1f}%")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"parameter_optimization_results_{timestamp}.json"
            
            summary = {
                'baseline_return': baseline,
                'best_result': best_result,
                'improvement_pct': improvement,
                'all_results': self.results,
                'symbols_tested': symbols,
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
                print("🔧 Consider additional parameter tuning")
                return False
        else:
            print("\n❌ No valid optimization results")
            return False

def main():
    """Run parameter optimization"""
    optimizer = DataBasedParameterOptimizer()
    success = optimizer.run_parameter_optimization()
    
    if success:
        print("\n🚀 Ready to proceed with Enhancement D!")
    else:
        print("\n🔧 Enhancement C needs refinement")
    
    return success

if __name__ == "__main__":
    main()
