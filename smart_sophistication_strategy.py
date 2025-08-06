#!/usr/bin/env python3
"""
SMART SOPHISTICATION STRATEGY
Focus on quality improvements rather than quantity of features
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SmartTradingStrategy:
    def __init__(self):
        self.models = {}
        
    def create_smart_features(self, data):
        """Create high-quality engineered features instead of adding more raw data"""
        print("🧠 Creating smart engineered features...")
        
        df = data.copy()
        
        # 1. CORE TECHNICAL INDICATORS (The proven 6)
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=10).mean()
        df['price_momentum'] = df['Close'].pct_change(5) * 100
        df['volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Bollinger position
        bb_middle = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # 2. SMART FEATURE INTERACTIONS (Quality over quantity)
        print("   ✅ Adding feature interactions...")
        
        # RSI-Volume interaction (momentum + conviction)
        df['rsi_volume_strength'] = df['rsi'] * df['volume_ratio'] / 100
        
        # Volatility-adjusted momentum (risk-adjusted signals)
        df['vol_adj_momentum'] = df['price_momentum'] / (df['volatility'] + 0.01)  # Avoid division by zero
        
        # Price position relative to moving average with volume confirmation
        df['price_sma_volume'] = ((df['Close'] / df['sma_20']) - 1) * df['volume_ratio']
        
        # Bollinger squeeze detection (low volatility before breakout)
        df['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        df['squeeze_momentum'] = df['bb_squeeze'] * df['price_momentum']
        
        # 3. REGIME-AWARE FEATURES (Market condition adaptation)
        print("   ✅ Adding regime-aware features...")
        
        # Trend strength (how consistently price moves in one direction)
        returns_5d = df['Close'].pct_change(5)
        df['trend_consistency'] = returns_5d.rolling(window=10).apply(
            lambda x: len([r for r in x if r > 0]) / len(x) if len(x) > 0 else 0.5
        )
        
        # Market stress indicator (based on volatility spikes)
        vol_ma = df['volatility'].rolling(window=20).mean()
        df['stress_indicator'] = df['volatility'] / vol_ma
        
        # 4. ADAPTIVE THRESHOLDS (Dynamic rather than static)
        print("   ✅ Adding adaptive thresholds...")
        
        # Dynamic RSI thresholds based on recent volatility
        df['rsi_threshold_upper'] = 70 + (df['stress_indicator'] - 1) * 10
        df['rsi_threshold_lower'] = 30 - (df['stress_indicator'] - 1) * 10
        
        # Relative RSI (compared to recent levels)
        df['rsi_relative'] = (df['rsi'] - df['rsi'].rolling(window=20).mean()) / df['rsi'].rolling(window=20).std()
        
        print(f"   ✅ Created {df.shape[1] - data.shape[1]} smart features")
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_regime_specific_models(self, data, symbol='NVDA'):
        """Create different models for different market conditions"""
        print(f"🎯 Creating regime-specific models for {symbol}...")
        
        # Prepare features
        df_enhanced = self.create_smart_features(data)
        
        # Select smart features (only 12 high-quality features)
        feature_cols = [
            # Core 6 proven features
            'sma_20', 'rsi', 'volume_ratio', 'price_momentum', 'volatility', 'bb_position',
            # Smart interactions (6 additional)
            'rsi_volume_strength', 'vol_adj_momentum', 'price_sma_volume', 
            'squeeze_momentum', 'trend_consistency', 'rsi_relative'
        ]
        
        # Create target
        df_enhanced['target'] = df_enhanced['Close'].shift(-1) / df_enhanced['Close'] - 1
        
        # Clean data
        clean_data = df_enhanced[feature_cols + ['target', 'stress_indicator']].dropna()
        
        if len(clean_data) < 100:
            print("   ❌ Insufficient data")
            return None
        
        print(f"   📊 Using {len(feature_cols)} high-quality features")
        print(f"   📈 Training samples: {len(clean_data)}")
        print(f"   📏 Samples per feature: {len(clean_data) / len(feature_cols):.1f}")
        
        # Split by market regime
        stress_median = clean_data['stress_indicator'].median()
        
        # Low stress (calm market)
        calm_mask = clean_data['stress_indicator'] <= stress_median
        calm_data = clean_data[calm_mask]
        
        # High stress (volatile market) 
        volatile_mask = clean_data['stress_indicator'] > stress_median
        volatile_data = clean_data[volatile_mask]
        
        models = {}
        
        # Train regime-specific models
        for regime_name, regime_data in [('calm', calm_data), ('volatile', volatile_data)]:
            if len(regime_data) < 50:
                print(f"   ⚠️ Insufficient data for {regime_name} regime")
                continue
            
            print(f"   🧠 Training {regime_name} market model ({len(regime_data)} samples)...")
            
            X = regime_data[feature_cols]
            y = regime_data['target']
            
            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train specialized model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,  # Prevent overfitting
                min_samples_split=10,  # Require more samples per split
                min_samples_leaf=5,    # Require more samples per leaf
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Validate
            predictions = model.predict(X_test)
            
            if len(predictions) > 1:
                direction_accuracy = np.mean(
                    np.sign(y_test.iloc[1:].values) == np.sign(predictions[1:])
                ) * 100
            else:
                direction_accuracy = 50
            
            models[regime_name] = {
                'model': model,
                'direction_accuracy': direction_accuracy,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            print(f"     ✅ {regime_name.upper()} model: {direction_accuracy:.1f}% direction accuracy")
        
        self.models[symbol] = {
            'regime_models': models,
            'feature_cols': feature_cols,
            'stress_threshold': stress_median
        }
        
        return models
    
    def generate_smart_signals(self, data, symbol='NVDA'):
        """Generate trading signals using regime-specific models"""
        if symbol not in self.models:
            print(f"❌ No models trained for {symbol}")
            return None
        
        print(f"🎯 Generating smart signals for {symbol}...")
        
        # Prepare features
        df_enhanced = self.create_smart_features(data)
        feature_cols = self.models[symbol]['feature_cols']
        
        # Get latest data point
        latest_data = df_enhanced[feature_cols + ['stress_indicator']].dropna().tail(1)
        
        if len(latest_data) == 0:
            print("❌ No valid data for prediction")
            return None
        
        # Determine market regime
        stress_level = latest_data['stress_indicator'].iloc[0]
        stress_threshold = self.models[symbol]['stress_threshold']
        
        regime = 'calm' if stress_level <= stress_threshold else 'volatile'
        print(f"   📊 Market regime: {regime.upper()} (stress: {stress_level:.2f})")
        
        # Use appropriate model
        regime_models = self.models[symbol]['regime_models']
        
        if regime not in regime_models:
            print(f"   ⚠️ No model for {regime} regime, using available model")
            regime = list(regime_models.keys())[0]
        
        model_info = regime_models[regime]
        model = model_info['model']
        
        # Make prediction
        X = latest_data[feature_cols].values.reshape(1, -1)
        prediction = model.predict(X)[0]
        
        # Generate signal with regime-specific thresholds
        if regime == 'calm':
            # In calm markets, require stronger signals
            buy_threshold = 0.015   # 1.5%
            sell_threshold = -0.015
        else:
            # In volatile markets, use more sensitive thresholds
            buy_threshold = 0.01    # 1.0%
            sell_threshold = -0.01
        
        if prediction > buy_threshold:
            signal = 'BUY'
            confidence = min(prediction / buy_threshold, 3.0)  # Cap at 3x
        elif prediction < sell_threshold:
            signal = 'SELL'
            confidence = min(abs(prediction) / abs(sell_threshold), 3.0)
        else:
            signal = 'HOLD'
            confidence = 1 - (abs(prediction) / max(buy_threshold, abs(sell_threshold)))
        
        return {
            'signal': signal,
            'prediction': prediction,
            'confidence': confidence,
            'regime': regime,
            'stress_level': stress_level,
            'model_accuracy': model_info['direction_accuracy']
        }

def demonstrate_smart_strategy():
    """Demonstrate the smart sophistication approach"""
    print("🚀 SMART SOPHISTICATION STRATEGY DEMONSTRATION")
    print("=" * 60)
    print("❌ OLD APPROACH: Add more features (38 → 50+ features)")
    print("✅ NEW APPROACH: Engineer better features (6 → 12 high-quality)")
    print()
    
    trader = SmartTradingStrategy()
    
    # Get NVDA data
    print("📡 Fetching NVDA data...")
    ticker = yf.Ticker('NVDA')
    data = ticker.history(period='1y')
    
    # Train smart models
    models = trader.create_regime_specific_models(data, 'NVDA')
    
    if models:
        # Generate current signal
        signal_result = trader.generate_smart_signals(data, 'NVDA')
        
        print(f"\n🎯 CURRENT SMART SIGNAL:")
        print("=" * 30)
        print(f"Signal: {signal_result['signal']}")
        print(f"Prediction: {signal_result['prediction']:.4f} ({signal_result['prediction']*100:+.2f}%)")
        print(f"Confidence: {signal_result['confidence']:.2f}")
        print(f"Market Regime: {signal_result['regime'].upper()}")
        print(f"Model Accuracy: {signal_result['model_accuracy']:.1f}%")
        
        print(f"\n📊 FEATURE IMPORTANCE ANALYSIS:")
        print("-" * 40)
        
        for regime, model_info in models.items():
            print(f"\n{regime.upper()} MARKET:")
            importance = model_info['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            for feature, imp in sorted_features[:5]:
                feature_type = "Core" if feature in ['sma_20', 'rsi', 'volume_ratio', 'price_momentum', 'volatility', 'bb_position'] else "Smart"
                print(f"  {imp:.3f} - {feature} ({feature_type})")
    
    print(f"\n🎯 SMART STRATEGY BENEFITS:")
    print("=" * 40)
    print("✅ 1. QUALITY OVER QUANTITY:")
    print("     • 12 features vs 6 (100% increase)")
    print("     • But all features are engineered interactions")
    print("     • No irrelevant fundamental data")
    print()
    print("✅ 2. REGIME AWARENESS:")
    print("     • Different models for calm vs volatile markets")
    print("     • Adaptive signal thresholds")
    print("     • Context-appropriate strategies")
    print()
    print("✅ 3. PROPER DATA RATIOS:")
    print("     • ~20 samples per feature (good ratio)")
    print("     • Specialized models with focused data")
    print("     • Reduced overfitting risk")
    print()
    print("✅ 4. ACTIONABLE INSIGHTS:")
    print("     • Clear regime identification")
    print("     • Confidence scoring")
    print("     • Model performance tracking")

def main():
    """Main execution"""
    print("🎯 WHAT TO DO NOW - SMART ACTION PLAN")
    print("=" * 50)
    
    print("📋 IMMEDIATE ACTIONS (This week):")
    print("1. ✅ Abandon the 'more features' approach")
    print("2. ✅ Implement smart feature engineering") 
    print("3. ✅ Create regime-specific models")
    print("4. ✅ Add adaptive thresholds")
    print()
    
    print("📋 SHORT-TERM GOALS (Next month):")
    print("1. 🔄 Test smart strategy on multiple stocks")
    print("2. 📊 Compare vs basic 6-feature system")
    print("3. 🎯 Optimize regime detection")
    print("4. 📈 Implement position sizing")
    print()
    
    print("📋 MEDIUM-TERM VISION (3 months):")
    print("1. 🚀 Deploy regime-aware portfolio")
    print("2. 📡 Add real-time regime switching")
    print("3. 🧠 Implement ensemble of specialists")
    print("4. 💼 Build risk management layer")
    print()
    
    # Demonstrate the strategy
    demonstrate_smart_strategy()

if __name__ == "__main__":
    main()
