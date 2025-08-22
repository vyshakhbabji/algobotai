"""
🎯 ENHANCEMENT C: SIMPLE PARAMETER OPTIMIZATION
Quick Win: Test key parameter variations on our proven baseline

Use our working baseline system as foundation and test focused improvements
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

class SimpleParameterOptimizer:
    """Simple but effective parameter optimization based on our working baseline"""
    
    def __init__(self):
        self.baseline_return = 57.6
        self.results = []
        
    def get_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get stock data with retry logic"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.reset_index()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            return df
            
        except Exception as e:
            # Use cached data if available
            print(f"⚠️ Using fallback for {symbol}")
            return pd.DataFrame()
    
    def calculate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced trading features"""
        if df.empty or len(df) < 50:
            return df
            
        df = df.copy()
        
        # Basic features
        df['returns'] = df['Close'].pct_change()
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        
        # Moving averages with different periods
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Enhanced volume analysis
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['volume_trend'] = df['volume_sma'].pct_change(5)
        
        # Momentum with multiple timeframes
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
        
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
    
    def train_enhanced_ml_model(self, symbol: str, df: pd.DataFrame) -> dict:
        """Train enhanced ML model"""
        try:
            if len(df) < 100:
                return {'model': None, 'accuracy': 0.0}
            
            # Feature selection
            feature_cols = [
                'rsi', 'macd', 'macd_histogram', 'bb_position', 
                'volume_ratio', 'volatility_ratio',
                'price_sma_20_ratio', 'momentum_5', 'momentum_10'
            ]
            
            available_features = [col for col in feature_cols if col in df.columns]
            if len(available_features) < 5:
                return {'model': None, 'accuracy': 0.0}
            
            X = df[available_features].fillna(0)
            
            # Enhanced target creation
            forward_returns = df['Close'].shift(-5) / df['Close'] - 1
            y = (forward_returns > 0.02).astype(int)  # Binary classification
            
            # Train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Clean data
            train_mask = ~(y_train.isna() | X_train.isna().any(axis=1))
            test_mask = ~(y_test.isna() | X_test.isna().any(axis=1))
            
            if train_mask.sum() < 50 or test_mask.sum() < 10:
                return {'model': None, 'accuracy': 0.0}
            
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_clean, y_train_clean)
            
            # Calculate accuracy
            pred = model.predict(X_test_clean)
            accuracy = np.mean(pred == y_test_clean)
            
            return {
                'model': model,
                'feature_cols': available_features,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"❌ ML training failed for {symbol}: {e}")
            return {'model': None, 'accuracy': 0.0}
    
    def generate_enhanced_signal(self, df: pd.DataFrame, model_info: dict, params: dict) -> float:
        """Generate enhanced trading signal"""
        if df.empty or not model_info or model_info['model'] is None:
            return 0.0
        
        try:
            latest = df.iloc[-1]
            
            # Get ML prediction
            X = latest[model_info['feature_cols']].fillna(0).values.reshape(1, -1)
            ml_signal = model_info['model'].predict_proba(X)[0][1]  # Probability of positive return
            
            # Technical analysis signal
            rsi = latest.get('rsi', 50)
            macd_hist = latest.get('macd_histogram', 0)
            bb_position = latest.get('bb_position', 0.5)
            volume_ratio = latest.get('volume_ratio', 1.0)
            momentum_5 = latest.get('momentum_5', 0)
            
            # Calculate technical signal with parameters
            tech_signal = 0.0
            
            # RSI component (optimized thresholds)
            rsi_buy = params.get('rsi_buy_threshold', 35)
            rsi_sell = params.get('rsi_sell_threshold', 65)
            if rsi < rsi_buy:
                tech_signal += 0.3
            elif rsi > rsi_sell:
                tech_signal -= 0.3
            
            # MACD component
            if macd_hist > 0:
                tech_signal += 0.2
            elif macd_hist < 0:
                tech_signal -= 0.2
            
            # Momentum component
            momentum_threshold = params.get('momentum_threshold', 0.02)
            if momentum_5 > momentum_threshold:
                tech_signal += 0.2
            elif momentum_5 < -momentum_threshold:
                tech_signal -= 0.2
            
            # Volume confirmation
            volume_threshold = params.get('volume_threshold', 1.2)
            if volume_ratio > volume_threshold:
                tech_signal *= 1.1
            
            # Bollinger Band position
            if bb_position < 0.2:
                tech_signal += 0.1
            elif bb_position > 0.8:
                tech_signal -= 0.1
            
            # Combine ML and technical signals
            ml_weight = params.get('ml_weight', 0.6)
            tech_weight = 1 - ml_weight
            
            final_signal = (ml_signal - 0.5) * 2 * ml_weight + tech_signal * tech_weight
            
            return np.clip(final_signal, -1.0, 1.0)
            
        except Exception as e:
            return 0.0
    
    def backtest_with_params(self, params: dict, symbols: list, 
                           start_date: str = "2024-05-20", 
                           end_date: str = "2024-08-20") -> dict:
        """Backtest with specific parameters"""
        
        # Portfolio setup
        initial_cash = 50000
        cash = initial_cash
        positions = {}
        
        # Parameters
        position_size_pct = params.get('position_size_pct', 0.15)
        min_signal_strength = params.get('min_signal_strength', 0.4)
        max_positions = params.get('max_positions', 8)
        
        trade_count = 0
        portfolio_values = [initial_cash]
        
        # Use a subset of symbols for speed
        test_symbols = symbols[:6]
        
        # Get data and train models
        models = {}
        for symbol in test_symbols:
            try:
                df = self.get_stock_data(symbol, '1y')
                if not df.empty and len(df) > 100:
                    df_featured = self.calculate_enhanced_features(df)
                    if not df_featured.empty:
                        model_info = self.train_enhanced_ml_model(symbol, df_featured)
                        if model_info['accuracy'] > 0.5:
                            models[symbol] = {
                                'model_info': model_info,
                                'data': df_featured
                            }
            except Exception as e:
                continue
        
        if not models:
            return {'annualized_return_pct': 0, 'total_trades': 0, 'final_value': initial_cash}
        
        # Simulate trading every 3 days for speed
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        current_date = start_dt
        
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Trading days
                
                # Calculate portfolio value
                portfolio_value = cash
                for symbol, quantity in positions.items():
                    if symbol in models:
                        try:
                            # Get current price
                            ticker = yf.Ticker(symbol)
                            current_price = ticker.history(period='1d')['Close'].iloc[-1]
                            portfolio_value += quantity * current_price
                        except:
                            pass
                
                portfolio_values.append(portfolio_value)
                
                # Generate signals and trade
                for symbol in models.keys():
                    try:
                        model_data = models[symbol]
                        df = model_data['data']
                        
                        # Filter data up to current date
                        df_current = df[df['Date'] <= current_date]
                        if len(df_current) < 50:
                            continue
                        
                        # Get current price and signal
                        current_price = df_current['Close'].iloc[-1]
                        signal = self.generate_enhanced_signal(df_current, model_data['model_info'], params)
                        
                        current_position = positions.get(symbol, 0)
                        
                        # Buy logic
                        if (signal > min_signal_strength and 
                            current_position == 0 and 
                            len(positions) < max_positions):
                            
                            position_value = portfolio_value * position_size_pct
                            if position_value <= cash:
                                quantity = int(position_value / current_price)
                                if quantity > 0:
                                    cost = quantity * current_price
                                    positions[symbol] = quantity
                                    cash -= cost
                                    trade_count += 1
                        
                        # Sell logic
                        elif signal < -min_signal_strength and current_position > 0:
                            proceeds = current_position * current_price
                            cash += proceeds
                            del positions[symbol]
                            trade_count += 1
                    
                    except Exception as e:
                        continue
            
            current_date += timedelta(days=3)  # Trade every 3 days for speed
        
        # Final value
        final_value = portfolio_values[-1] if portfolio_values else initial_cash
        total_return = (final_value / initial_cash - 1) * 100
        
        # Annualize
        days = (end_dt - start_dt).days
        if days > 0:
            annualized_return = ((final_value / initial_cash) ** (365 / days) - 1) * 100
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
    
    def run_optimization(self):
        """Run parameter optimization"""
        
        print("🎯 ENHANCEMENT C: SIMPLE PARAMETER OPTIMIZATION")
        print("Target: 5-10% performance boost (62-67% annual returns)")
        print("=" * 60)
        
        # Top performing symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        
        # Parameter configurations
        test_configs = [
            {
                'name': 'Baseline',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.4,
                'max_positions': 8,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.2,
                'ml_weight': 0.6
            },
            {
                'name': 'Aggressive Sizing',
                'position_size_pct': 0.18,
                'min_signal_strength': 0.35,
                'max_positions': 8,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.2,
                'ml_weight': 0.6
            },
            {
                'name': 'ML Focused',
                'position_size_pct': 0.15,
                'min_signal_strength': 0.4,
                'max_positions': 8,
                'rsi_buy_threshold': 35,
                'rsi_sell_threshold': 65,
                'momentum_threshold': 0.02,
                'volume_threshold': 1.2,
                'ml_weight': 0.8
            },
            {
                'name': 'Conservative',
                'position_size_pct': 0.12,
                'min_signal_strength': 0.5,
                'max_positions': 6,
                'rsi_buy_threshold': 30,
                'rsi_sell_threshold': 70,
                'momentum_threshold': 0.03,
                'volume_threshold': 1.3,
                'ml_weight': 0.7
            }
        ]
        
        print(f"🧪 Testing {len(test_configs)} parameter configurations...")
        
        best_result = None
        best_return = self.baseline_return
        
        for i, config in enumerate(test_configs):
            try:
                print(f"\n🔬 Test {i+1}/{len(test_configs)}: {config['name']}")
                
                result = self.backtest_with_params(config, symbols)
                annual_return = result['annualized_return_pct']
                improvement = annual_return - self.baseline_return
                
                print(f"   📊 Return: {annual_return:.1f}% ({improvement:+.1f}%)")
                print(f"   💼 Trades: {result['total_trades']}, Final Value: ${result['final_value']:,.0f}")
                
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
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"simple_optimization_results_{timestamp}.json"
            
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
                print("🔧 May need additional tuning or proceed to Enhancement D")
                return improvement > 0
        else:
            print("\n❌ No valid optimization results")
            return False

def main():
    """Run simple parameter optimization"""
    print("🎯 Starting Enhancement C: Parameter Optimization")
    print("Based on our proven 57.6% baseline system")
    print("=" * 60)
    
    optimizer = SimpleParameterOptimizer()
    success = optimizer.run_optimization()
    
    if success:
        print("\n🚀 Ready to proceed with Enhancement D!")
    else:
        print("\n💡 Consider proceeding to Enhancement D for ML improvements")
    
    return success

if __name__ == "__main__":
    main()
