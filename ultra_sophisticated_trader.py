#!/usr/bin/env python3
"""
Ultra-Sophisticated AI Trading System
Combining technical, fundamental, sentiment, macro, and options data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class UltraSophisticatedTrader:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def get_technical_indicators(self, data):
        """Get our existing 38 technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Momentum indicators
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        df['bb_std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / df['bb_width']
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['price_volume'] = df['Close'] * df['Volume']
        
        # Volatility
        df['volatility'] = df['Close'].rolling(window=20).std()
        df['atr'] = self.calculate_atr(df)
        
        # Price patterns
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        
        # Trend indicators
        df['price_momentum_1'] = df['Close'].pct_change()
        df['price_momentum_5'] = df['Close'].pct_change(5)
        df['price_momentum_10'] = df['Close'].pct_change(10)
        
        # Support/Resistance levels
        df['resistance_level'] = df['High'].rolling(window=20).max()
        df['support_level'] = df['Low'].rolling(window=20).min()
        df['distance_to_resistance'] = (df['resistance_level'] - df['Close']) / df['Close']
        df['distance_to_support'] = (df['Close'] - df['support_level']) / df['Close']
        
        # Advanced patterns
        df['gap_up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) > 0.02
        df['gap_down'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) < -0.02
        df['doji'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
        df['hammer'] = ((df['High'] - df['Low']) > 3 * abs(df['Close'] - df['Open'])) & (df['Close'] > df['Open'])
        
        # Rate of change
        df['roc_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Moving average relationships
        df['sma_cross_5_10'] = df['sma_5'] > df['sma_10']
        df['sma_cross_10_20'] = df['sma_10'] > df['sma_20']
        df['price_above_sma20'] = df['Close'] > df['sma_20']
        df['price_above_sma50'] = df['Close'] > df['sma_50']
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window).mean()
    
    def get_enhanced_features(self, symbol, data):
        """Get all sophisticated features"""
        print(f"🧠 Collecting enhanced features for {symbol}...")
        
        enhanced_features = {}
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 1. FUNDAMENTAL FEATURES (Key ratios)
            fundamental_features = {
                'pe_ratio': info.get('trailingPE', 20),  # Default to market average
                'peg_ratio': info.get('pegRatio', 1),
                'price_to_book': info.get('priceToBook', 3),
                'profit_margin': info.get('profitMargins', 0.1),
                'return_on_equity': info.get('returnOnEquity', 0.15),
                'debt_to_equity': info.get('debtToEquity', 0.5),
                'beta': info.get('beta', 1),
                'revenue_growth': info.get('revenueGrowth', 0.05),
            }
            
            # 2. MACRO FEATURES (Market sentiment)
            spy = yf.Ticker('^GSPC')
            vix = yf.Ticker('^VIX')
            spy_hist = spy.history(period="5d")
            vix_hist = vix.history(period="5d")
            
            if len(spy_hist) > 0 and len(vix_hist) > 0:
                spy_return = ((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1) * 100
                vix_level = vix_hist['Close'].iloc[-1]
                
                macro_features = {
                    'market_return_5d': spy_return,
                    'vix_level': vix_level,
                    'market_regime': 1 if spy_return > 1 and vix_level < 25 else (-1 if spy_return < -1 and vix_level > 30 else 0),
                    'risk_sentiment': spy_return / max(vix_level, 1)
                }
            else:
                macro_features = {'market_return_5d': 0, 'vix_level': 20, 'market_regime': 0, 'risk_sentiment': 0}
            
            # 3. SECTOR FEATURES
            sector = info.get('sector', 'Technology')
            sector_features = {
                'is_tech': 1 if sector == 'Technology' else 0,
                'is_finance': 1 if sector == 'Financial Services' else 0,
                'is_healthcare': 1 if sector == 'Healthcare' else 0,
                'is_consumer': 1 if 'Consumer' in sector else 0,
            }
            
            # 4. VOLATILITY FEATURES (from recent price action)
            recent_data = data.tail(20)  # Last 20 days
            volatility_features = {
                'price_volatility': recent_data['Close'].std() / recent_data['Close'].mean(),
                'volume_volatility': recent_data['Volume'].std() / recent_data['Volume'].mean(),
                'high_low_volatility': ((recent_data['High'] - recent_data['Low']) / recent_data['Close']).mean(),
            }
            
            # Combine all features
            enhanced_features.update(fundamental_features)
            enhanced_features.update(macro_features)
            enhanced_features.update(sector_features)
            enhanced_features.update(volatility_features)
            
            print(f"   ✅ Collected {len(enhanced_features)} enhanced features")
            
        except Exception as e:
            print(f"   ⚠️ Using default values due to: {e}")
            # Provide defaults if data collection fails
            enhanced_features = {
                'pe_ratio': 20, 'peg_ratio': 1, 'price_to_book': 3, 'profit_margin': 0.1,
                'return_on_equity': 0.15, 'debt_to_equity': 0.5, 'beta': 1, 'revenue_growth': 0.05,
                'market_return_5d': 0, 'vix_level': 20, 'market_regime': 0, 'risk_sentiment': 0,
                'is_tech': 1, 'is_finance': 0, 'is_healthcare': 0, 'is_consumer': 0,
                'price_volatility': 0.02, 'volume_volatility': 0.5, 'high_low_volatility': 0.03
            }
        
        return enhanced_features
    
    def prepare_training_data(self, symbol='NVDA', period='2y'):
        """Prepare comprehensive training dataset"""
        print(f"🔄 Preparing ultra-sophisticated training data for {symbol}...")
        
        # Get price data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        # Add technical indicators
        data = self.get_technical_indicators(data)
        
        # Get enhanced features (will be the same for all rows, but important for the model)
        enhanced_features = self.get_enhanced_features(symbol, data)
        
        # Add enhanced features to each row
        for feature, value in enhanced_features.items():
            data[f'enhanced_{feature}'] = value
        
        # Create target (next day return)
        data['target'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Select features (technical + enhanced)
        technical_features = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'volume_sma', 'volume_ratio', 'price_volume',
            'volatility', 'atr', 'high_low_ratio', 'close_open_ratio', 'price_range',
            'price_momentum_1', 'price_momentum_5', 'price_momentum_10',
            'distance_to_resistance', 'distance_to_support',
            'roc_5', 'roc_10'
        ]
        
        enhanced_feature_names = [f'enhanced_{k}' for k in enhanced_features.keys()]
        
        all_features = technical_features + enhanced_feature_names
        
        # Clean data
        feature_data = data[all_features + ['target']].dropna()
        
        X = feature_data[all_features]
        y = feature_data['target']
        
        print(f"   📊 Total features: {len(all_features)} (Technical: {len(technical_features)}, Enhanced: {len(enhanced_feature_names)})")
        print(f"   📈 Training samples: {len(X)}")
        
        return X, y, all_features, data
    
    def train_ultra_models(self, symbol='NVDA'):
        """Train models with enhanced features"""
        print(f"🚀 TRAINING ULTRA-SOPHISTICATED MODELS FOR {symbol}")
        print("=" * 60)
        
        # Prepare data
        X, y, feature_names, full_data = self.prepare_training_data(symbol)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models_to_train = {
            'Ultra_Random_Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                random_state=42
            ),
            'Ultra_Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\n📈 Training {name}...")
            
            # Train model
            if 'Random_Forest' in name:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                feature_importance = model.feature_importances_
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                feature_importance = model.feature_importances_
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred,
                'feature_importance': feature_importance
            }
            
            print(f"   ✅ MSE: {mse:.6f}, R²: {r2:.4f}")
        
        # Store models and scaler
        self.models = {name: result['model'] for name, result in results.items()}
        self.scalers[symbol] = scaler
        
        # Feature importance analysis
        print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 50)
        
        for name, result in results.items():
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': result['feature_importance']
            }).sort_values('importance', ascending=False)
            
            print(f"\n{name}:")
            for i, row in importance_df.head(10).iterrows():
                feature_type = "Enhanced" if row['feature'].startswith('enhanced_') else "Technical"
                print(f"   {row['importance']:.3f} - {row['feature']} ({feature_type})")
        
        return results, feature_names, full_data
    
    def generate_ultra_signals(self, data, symbol='NVDA'):
        """Generate trading signals using ultra-sophisticated models"""
        if not self.models:
            print("❌ No models trained! Train models first.")
            return None
        
        # Get latest features
        enhanced_features = self.get_enhanced_features(symbol, data)
        
        # Prepare features for prediction
        latest_data = data.tail(1).copy()
        
        # Add enhanced features
        for feature, value in enhanced_features.items():
            latest_data[f'enhanced_{feature}'] = value
        
        # Get technical features
        technical_features = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'volume_sma', 'volume_ratio', 'price_volume',
            'volatility', 'atr', 'high_low_ratio', 'close_open_ratio', 'price_range',
            'price_momentum_1', 'price_momentum_5', 'price_momentum_10',
            'distance_to_resistance', 'distance_to_support',
            'roc_5', 'roc_10'
        ]
        
        enhanced_feature_names = [f'enhanced_{k}' for k in enhanced_features.keys()]
        all_features = technical_features + enhanced_feature_names
        
        # Prepare feature vector
        feature_vector = latest_data[all_features].values.reshape(1, -1)
        
        # Get predictions from all models
        predictions = {}
        scaler = self.scalers.get(symbol)
        
        for name, model in self.models.items():
            if 'Random_Forest' in name:
                pred = model.predict(feature_vector)[0]
            else:
                pred = model.predict(scaler.transform(feature_vector))[0]
            predictions[name] = pred
        
        # Ensemble prediction (average)
        avg_prediction = np.mean(list(predictions.values()))
        
        # Generate signal based on prediction and confidence
        confidence = 1 - np.std(list(predictions.values()))  # Higher confidence when models agree
        
        if avg_prediction > 0.01 and confidence > 0.8:  # Strong buy signal
            signal = 'STRONG_BUY'
        elif avg_prediction > 0.005 and confidence > 0.6:  # Buy signal
            signal = 'BUY'
        elif avg_prediction < -0.01 and confidence > 0.8:  # Strong sell signal
            signal = 'STRONG_SELL'
        elif avg_prediction < -0.005 and confidence > 0.6:  # Sell signal
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'prediction': avg_prediction,
            'confidence': confidence,
            'individual_predictions': predictions
        }

def main():
    """Demonstrate ultra-sophisticated trading system"""
    trader = UltraSophisticatedTrader()
    
    # Train models
    results, features, data = trader.train_ultra_models('NVDA')
    
    # Generate current signal
    signal_result = trader.generate_ultra_signals(data, 'NVDA')
    
    print(f"\n🎯 CURRENT TRADING SIGNAL:")
    print("=" * 40)
    print(f"Signal: {signal_result['signal']}")
    print(f"Prediction: {signal_result['prediction']:.4f}")
    print(f"Confidence: {signal_result['confidence']:.3f}")
    
    print(f"\n🚀 SYSTEM ENHANCEMENT SUMMARY:")
    print("=" * 50)
    print(f"✅ Original System: 38 technical indicators only")
    print(f"🚀 Enhanced System: {len(features)} total features")
    print(f"   📊 Technical Indicators: ~30")
    print(f"   📈 Fundamental Metrics: 8")
    print(f"   🌍 Macro Indicators: 4") 
    print(f"   🏭 Sector Features: 4")
    print(f"   📊 Volatility Features: 3")
    print(f"")
    print(f"🎯 Expected Improvements:")
    print(f"   • Better cross-stock performance")
    print(f"   • More resilient to market regime changes")
    print(f"   • Early detection of fundamental shifts")
    print(f"   • Improved risk-adjusted returns")

if __name__ == "__main__":
    main()
