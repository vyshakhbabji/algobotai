#!/usr/bin/env python3
"""
Improved AI Portfolio Manager
Fixed version with better feature engineering and model performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Dynamic stock universe - loaded from portfolio_universe.json
def load_stock_universe():
    """Load current stock universe from portfolio manager"""
    import json
    import os
    
    portfolio_file = "portfolio_universe.json"
    if os.path.exists(portfolio_file):
        with open(portfolio_file, 'r') as f:
            data = json.load(f)
            return data.get('stocks', [])
    else:
        # Fallback to default stocks
        return [
            'GOOG', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'AVGO', 'PLTR', 
            'NFLX', 'TSM', 'PANW', 'NOW', 'XLK', 'QQQ', 'COST'
        ]

STOCK_UNIVERSE = load_stock_universe()
INITIAL_CAPITAL = 10000

class ImprovedAIPortfolioManager:
    def __init__(self, capital=10000):
        self.initial_capital = capital
        self.current_capital = capital
        self.positions = {}
        self.stock_universe = load_stock_universe()  # Dynamic universe
        self.cash = capital
        self.portfolio_history = []
        self.trades = []
        self.models = {}
        self.scalers = {}
        
    def calculate_improved_features(self, df):
        """Calculate improved AI features with better signal-to-noise ratio"""
        try:
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Trend features (more stable)
            df['sma_10'] = df['Close'].rolling(window=10).mean()
            df['sma_30'] = df['Close'].rolling(window=30).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            
            # Price position relative to moving averages
            df['price_vs_sma10'] = (df['Close'] - df['sma_10']) / df['sma_10']
            df['price_vs_sma30'] = (df['Close'] - df['sma_30']) / df['sma_30']
            df['price_vs_sma50'] = (df['Close'] - df['sma_50']) / df['sma_50']
            
            # Momentum features (improved)
            df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
            df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
            
            # Volatility features
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()
            
            # Volume features (normalized)
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            df['volume_momentum'] = df['volume_ratio'] * df['momentum_5']
            
            # RSI (fixed calculation)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize RSI
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
            df['bb_std'] = df['Close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * bb_std)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * bb_std)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_squeeze'] = df['bb_std'] / df['bb_middle']  # Volatility squeeze indicator
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Support/Resistance levels
            df['high_20'] = df['High'].rolling(window=20).max()
            df['low_20'] = df['Low'].rolling(window=20).min()
            df['price_position'] = (df['Close'] - df['low_20']) / (df['high_20'] - df['low_20'])
            
            # Target: predict next 5-day return (classification approach)
            df['future_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
            
            # Create classification target (better for ML)
            df['target_class'] = pd.cut(df['future_return_5d'], 
                                      bins=[-np.inf, -0.02, 0.02, np.inf], 
                                      labels=[0, 1, 2])  # 0=down, 1=neutral, 2=up
            
            return df
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return df
    
    def train_improved_model(self, symbol, period='2y'):
        """Train improved AI model with cross-validation"""
        try:
            print(f"🤖 Training improved model for {symbol}...")
            
            # Get more data for better training
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 100:
                print(f"  ❌ Insufficient data for {symbol}")
                return None, None, None
            
            # Calculate improved features
            df = self.calculate_improved_features(df)
            
            # Select most predictive features
            feature_cols = [
                'price_vs_sma10', 'price_vs_sma30', 'price_vs_sma50',
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_30',
                'volume_ratio', 'volume_momentum',
                'rsi_normalized', 'bb_position', 'bb_squeeze',
                'macd', 'macd_histogram', 'price_position'
            ]
            
            # Prepare clean data
            clean_data = df[feature_cols + ['future_return_5d']].dropna()
            
            if len(clean_data) < 100:
                print(f"  ❌ Insufficient clean data for {symbol}")
                return None, None, None
            
            # Features and target
            X = clean_data[feature_cols].values
            y = clean_data['future_return_5d'].values
            
            # Scale features
            scaler = RobustScaler()  # More robust to outliers
            X_scaled = scaler.fit_transform(X)
            
            # Use ensemble of models
            models = {
                'rf': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            best_model = None
            best_score = -np.inf
            best_model_name = None
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            for name, model in models.items():
                try:
                    # Cross-validation scores
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                              scoring='r2', n_jobs=-1)
                    avg_score = cv_scores.mean()
                    
                    print(f"  📊 {name.upper()}: CV R² = {avg_score:.3f} (±{cv_scores.std():.3f})")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_model_name = name
                        
                except Exception as e:
                    print(f"  ⚠️ {name} failed: {e}")
                    continue
            
            if best_model is None:
                print(f"  ❌ All models failed for {symbol}")
                return None, None, None
            
            # Train best model on full data
            best_model.fit(X_scaled, y)
            
            # Final validation
            y_pred = best_model.predict(X_scaled)
            final_r2 = r2_score(y, y_pred)
            final_mse = mean_squared_error(y, y_pred)
            
            print(f"  ✅ Best model: {best_model_name.upper()}")
            print(f"  📈 Final R² = {final_r2:.3f}")
            print(f"  📉 MSE = {final_mse:.6f}")
            
            # Only accept models with positive R²
            if final_r2 > 0.01:  # At least 1% better than mean
                return best_model, scaler, final_r2
            else:
                print(f"  ⚠️ Model performance too low (R² = {final_r2:.3f})")
                return None, None, None
                
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            return None, None, None
    
    def get_prediction_strength(self, symbol, current_data):
        """Get AI prediction strength for a symbol"""
        try:
            if symbol not in self.models or self.models[symbol] is None:
                return 0
            
            model, scaler, r2_score = self.models[symbol]
            
            # Calculate features for current data
            df = current_data.copy()
            df = self.calculate_improved_features(df)
            
            # Feature columns (same as training)
            feature_cols = [
                'price_vs_sma10', 'price_vs_sma30', 'price_vs_sma50',
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_30',
                'volume_ratio', 'volume_momentum',
                'rsi_normalized', 'bb_position', 'bb_squeeze',
                'macd', 'macd_histogram', 'price_position'
            ]
            
            # Get latest features
            latest_features = df[feature_cols].iloc[-1:].values
            
            if np.isnan(latest_features).any():
                return 0
            
            # Scale and predict
            features_scaled = scaler.transform(latest_features)
            predicted_return = model.predict(features_scaled)[0]
            
            # Convert to strength score (0-100)
            # Weight by model quality (R²)
            base_strength = max(0, min(100, (predicted_return + 0.1) * 500))
            weighted_strength = base_strength * (0.5 + r2_score)  # Boost good models
            
            return max(0, min(100, weighted_strength))
            
        except Exception as e:
            print(f"  ❌ Prediction error for {symbol}: {e}")
            return 0
    
    def train_all_models(self):
        """Train improved models for all stocks in current universe"""
        print("🧠 TRAINING IMPROVED AI MODELS")
        print("=" * 60)
        
        # Refresh stock universe
        self.stock_universe = load_stock_universe()
        
        successful_models = 0
        
        for symbol in self.stock_universe:
            model, scaler, r2_score = self.train_improved_model(symbol, period='2y')
            
            if model is not None:
                self.models[symbol] = (model, scaler, r2_score)
                successful_models += 1
            else:
                self.models[symbol] = None
        
        print(f"\n✅ Trained {successful_models}/{len(STOCK_UNIVERSE)} successful models")
        
        if successful_models == 0:
            print("❌ No models trained successfully!")
            return False
        
        return True
    
    def backtest_improved_portfolio(self, start_date, end_date):
        """Backtest with improved models"""
        print(f"🚀 IMPROVED AI PORTFOLIO BACKTEST: {start_date} to {end_date}")
        print("=" * 80)
        
        # Train models first
        if not self.train_all_models():
            return None
        
        # Reset portfolio
        self.positions = {}
        self.cash = self.initial_capital
        self.portfolio_history = []
        self.trades = []
        
        # Generate monthly rebalancing dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        rebalance_dates = pd.date_range(start=start, end=end, freq='MS')
        if rebalance_dates[-1] < end:
            rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([end]))
        
        for rebalance_date in rebalance_dates:
            try:
                self.rebalance_portfolio_improved(rebalance_date.strftime('%Y-%m-%d'))
                
                # Record portfolio value
                portfolio_value = self.cash
                for symbol, shares in self.positions.items():
                    if shares > 0:
                        stock = yf.Ticker(symbol)
                        current_price = stock.history(period='1d')['Close'].iloc[-1]
                        portfolio_value += shares * current_price
                
                self.portfolio_history.append(portfolio_value)
                
            except Exception as e:
                print(f"  ❌ Rebalancing error on {rebalance_date}: {e}")
                continue
        
        # Final results
        if self.portfolio_history:
            final_value = self.portfolio_history[-1]
            total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
            print(f"\n📊 IMPROVED BACKTEST RESULTS:")
            print(f"   Initial Value: ${self.initial_capital:,.2f}")
            print(f"   Final Value: ${final_value:,.2f}")
            print(f"   Total Return: {total_return:.2f}%")
            print(f"   Total Trades: {len(self.trades)}")
            
            return {
                'final_portfolio_value': final_value,
                'total_return': total_return,
                'trades': self.trades,
                'portfolio_history': self.portfolio_history
            }
        
        return None
    
    def rebalance_portfolio_improved(self, date):
        """Rebalance portfolio using improved AI predictions"""
        print(f"\n📅 Rebalancing on {date}")
        
        # Refresh stock universe in case it was updated
        self.stock_universe = load_stock_universe()
        
        # Get predictions for all stocks in current universe
        predictions = {}
        
        for symbol in self.stock_universe:
            try:
                # Get recent data for prediction
                stock = yf.Ticker(symbol)
                df = stock.history(period='3mo', end=date)
                
                if len(df) < 50:  # Need sufficient data
                    continue
                
                strength = self.get_prediction_strength(symbol, df)
                current_price = df['Close'].iloc[-1]
                
                predictions[symbol] = {
                    'strength': strength,
                    'price': current_price
                }
                
            except Exception as e:
                print(f"  ⚠️ Error getting prediction for {symbol}: {e}")
                continue
        
        if not predictions:
            print("  ❌ No predictions available")
            return
        
        # Clear current positions
        for symbol in list(self.positions.keys()):
            if symbol in predictions:
                # Sell current position
                shares = self.positions[symbol]
                if shares > 0:
                    sell_value = shares * predictions[symbol]['price']
                    self.cash += sell_value
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares,
                        'price': predictions[symbol]['price']
                    })
            self.positions[symbol] = 0
        
        # Sort by prediction strength (only positive predictions)
        strong_predictions = {k: v for k, v in predictions.items() if v['strength'] > 55}
        
        if not strong_predictions:
            print("  ⚠️ No strong predictions, staying in cash")
            return
        
        sorted_predictions = sorted(strong_predictions.items(), 
                                  key=lambda x: x[1]['strength'], reverse=True)
        
        # Allocate to top 5 predictions
        top_picks = sorted_predictions[:5]
        allocation_per_stock = self.cash / len(top_picks)
        
        for symbol, pred_data in top_picks:
            try:
                shares_to_buy = int(allocation_per_stock / pred_data['price'])
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * pred_data['price']
                    
                    if cost <= self.cash:
                        self.positions[symbol] = shares_to_buy
                        self.cash -= cost
                        
                        self.trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': pred_data['price']
                        })
                        
                        print(f"  🟢 BUY {shares_to_buy} shares of {symbol} at ${pred_data['price']:.2f} "
                              f"(Strength: {pred_data['strength']:.1f})")
                        
            except Exception as e:
                print(f"  ❌ Error buying {symbol}: {e}")

def run_improved_backtest():
    """Run the improved backtest"""
    manager = ImprovedAIPortfolioManager(10000)
    
    # Test on recent period
    result = manager.backtest_improved_portfolio('2025-05-01', '2025-08-05')
    
    if result:
        print(f"\n🎯 VALIDATION: Improved model shows {result['total_return']:.2f}% return")
    else:
        print(f"\n❌ Improved backtest failed")

if __name__ == "__main__":
    try:
        run_improved_backtest()
        
    except KeyboardInterrupt:
        print("\n🛑 Backtest interrupted")
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
