#!/usr/bin/env python3
"""
ENHANCED ML MODEL WITH INSTITUTIONAL MOMENTUM FEATURES
Integrates proven momentum features from our deployed portfolio
Uses same academic foundation as institutional momentum strategy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MomentumEnhancedMLModel:
    def __init__(self, symbols=None):
        # Use same stock universe as momentum portfolio
        if symbols:
            self.symbols = symbols
        else:
            self.symbols = [
                # From our deployed momentum portfolio
                "AMD", "GE", "PLTR", "MSFT", "NVDA", "JNJ", "CAT", "GOOGL"
            ]
        
        self.scaler = StandardScaler()
        self.classification_model = None
        self.regression_model = None
        
    def calculate_institutional_momentum_features(self, data):
        """Calculate same momentum features as our deployed portfolio"""
        
        # CORE MOMENTUM METRICS (from institutional research)
        # 6-month momentum (126 trading days)
        data['momentum_6m'] = data['Close'].pct_change(126) * 100
        
        # 3-month momentum (63 trading days) 
        data['momentum_3m'] = data['Close'].pct_change(63) * 100
        
        # 1-month momentum (21 trading days)
        data['momentum_1m'] = data['Close'].pct_change(21) * 100
        
        # RISK-ADJUSTED MOMENTUM
        # 20-day volatility
        data['volatility_20d'] = data['Close'].rolling(20).std() / data['Close'].rolling(20).mean() * 100
        
        # Risk-adjusted momentum (return/volatility)
        data['momentum_6m_risk_adj'] = data['momentum_6m'] / (data['volatility_20d'] + 0.1)
        data['momentum_3m_risk_adj'] = data['momentum_3m'] / (data['volatility_20d'] + 0.1)
        data['momentum_1m_risk_adj'] = data['momentum_1m'] / (data['volatility_20d'] + 0.1)
        
        # TECHNICAL MOMENTUM INDICATORS
        # Moving average slopes (trend strength)
        ma20 = data['Close'].rolling(20).mean()
        ma50 = data['Close'].rolling(50).mean()
        data['ma_slope_20'] = ma20.pct_change(5) * 100
        data['ma_slope_50'] = ma50.pct_change(10) * 100
        
        # Price relative to moving averages
        data['price_vs_ma20'] = (data['Close'] / ma20 - 1) * 100
        data['price_vs_ma50'] = (data['Close'] / ma50 - 1) * 100
        
        # VOLUME-PRICE MOMENTUM
        # Volume-weighted momentum
        volume_ma = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / volume_ma
        data['volume_momentum'] = data['momentum_1m'] * np.log(data['volume_ratio'] + 1)
        
        # MOMENTUM ACCELERATION (second derivative)
        data['momentum_acceleration_3m'] = data['momentum_3m'].diff(5)
        data['momentum_acceleration_1m'] = data['momentum_1m'].diff(5)
        
        # MOMENTUM CONSISTENCY (lower volatility = more reliable)
        data['momentum_consistency'] = data['momentum_1m'].rolling(20).std()
        
        # RELATIVE STRENGTH (vs SPY - market benchmark)
        try:
            spy_data = yf.download('SPY', start=data.index[0], end=data.index[-1], progress=False)
            if not spy_data.empty:
                spy_momentum_3m = spy_data['Close'].pct_change(63) * 100
                data['relative_strength_3m'] = data['momentum_3m'] - spy_momentum_3m.reindex(data.index)
            else:
                data['relative_strength_3m'] = data['momentum_3m']  # Fallback
        except:
            data['relative_strength_3m'] = data['momentum_3m']  # Fallback
        
        return data
    
    def create_ml_targets(self, data):
        """Create ML targets based on momentum signals"""
        
        # CLASSIFICATION TARGET: Strong momentum signal (like our portfolio)
        # Strong buy: momentum_6m > 10% AND momentum_3m > 5% AND risk_adj > 0.5
        strong_buy_condition = (
            (data['momentum_6m'] > 10) & 
            (data['momentum_3m'] > 5) & 
            (data['momentum_6m_risk_adj'] > 0.5)
        )
        
        # Strong sell: momentum_6m < -5% AND momentum_3m < -3% AND risk_adj < -0.3
        strong_sell_condition = (
            (data['momentum_6m'] < -5) & 
            (data['momentum_3m'] < -3) & 
            (data['momentum_6m_risk_adj'] < -0.3)
        )
        
        # Create classification target
        data['signal_class'] = 1  # HOLD (default)
        data.loc[strong_buy_condition, 'signal_class'] = 2  # STRONG BUY
        data.loc[strong_sell_condition, 'signal_class'] = 0  # STRONG SELL
        
        # REGRESSION TARGET: Next month return
        data['target_return_1m'] = data['Close'].pct_change(21).shift(-21) * 100
        
        return data
    
    def prepare_features(self, data):
        """Prepare feature matrix for ML models"""
        
        feature_columns = [
            # Core momentum features
            'momentum_6m', 'momentum_3m', 'momentum_1m',
            'momentum_6m_risk_adj', 'momentum_3m_risk_adj', 'momentum_1m_risk_adj',
            
            # Technical features
            'ma_slope_20', 'ma_slope_50',
            'price_vs_ma20', 'price_vs_ma50',
            
            # Volume features
            'volume_ratio', 'volume_momentum',
            
            # Advanced momentum
            'momentum_acceleration_3m', 'momentum_acceleration_1m',
            'momentum_consistency', 'relative_strength_3m',
            
            # Risk features
            'volatility_20d'
        ]
        
        # Select features that exist
        available_features = [col for col in feature_columns if col in data.columns]
        
        return data[available_features].dropna()
    
    def train_classification_model(self, symbol):
        """Train model to predict momentum signals"""
        
        print(f"🤖 Training Classification Model for {symbol}")
        print("=" * 40)
        
        # Get 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty or len(data) < 200:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        # Calculate momentum features
        data = self.calculate_institutional_momentum_features(data)
        data = self.create_ml_targets(data)
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Align with targets
        target_class = data['signal_class'].reindex(features.index).dropna()
        features = features.reindex(target_class.index)
        
        if len(features) < 100:
            print(f"❌ Insufficient clean data for {symbol}")
            return None
        
        # Split data (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            features, target_class, test_size=0.2, random_state=42, stratify=target_class
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.classification_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.classification_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classification_model, X_train_scaled, y_train, cv=5)
        
        print(f"✅ {symbol} Classification Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.classification_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 5 Features:")
        for _, row in feature_importance.head().iterrows():
            print(f"      {row['feature']}: {row['importance']:.3f}")
        
        return {
            'symbol': symbol,
            'model': self.classification_model,
            'scaler': self.scaler,
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'feature_importance': feature_importance,
            'features': X_train.columns.tolist()
        }
    
    def train_regression_model(self, symbol):
        """Train model to predict returns"""
        
        print(f"📈 Training Regression Model for {symbol}")
        print("=" * 35)
        
        # Get 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty or len(data) < 200:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        # Calculate momentum features
        data = self.calculate_institutional_momentum_features(data)
        data = self.create_ml_targets(data)
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Align with targets
        target_return = data['target_return_1m'].reindex(features.index).dropna()
        features = features.reindex(target_return.index)
        
        if len(features) < 100:
            print(f"❌ Insufficient clean data for {symbol}")
            return None
        
        # Split data (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            features, target_return, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler_reg = StandardScaler()
        X_train_scaled = scaler_reg.fit_transform(X_train)
        X_test_scaled = scaler_reg.transform(X_test)
        
        # Train Random Forest Regressor
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.regression_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.regression_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.regression_model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        print(f"✅ {symbol} Regression Results:")
        print(f"   R² Score: {r2:.3f}")
        print(f"   CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.regression_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   Top 5 Features:")
        for _, row in feature_importance.head().iterrows():
            print(f"      {row['feature']}: {row['importance']:.3f}")
        
        return {
            'symbol': symbol,
            'model': self.regression_model,
            'scaler': scaler_reg,
            'r2_score': r2,
            'cv_score': cv_scores.mean(),
            'feature_importance': feature_importance,
            'features': X_train.columns.tolist()
        }
    
    def generate_ml_signals(self, symbol):
        """Generate current ML signals for a symbol"""
        
        if not self.classification_model:
            print(f"❌ No trained classification model for {symbol}")
            return None
        
        # Get recent data
        data = yf.download(symbol, period='1y', progress=False, auto_adjust=True)
        
        # Calculate features
        data = self.calculate_institutional_momentum_features(data)
        features = self.prepare_features(data)
        
        if features.empty:
            print(f"❌ No features available for {symbol}")
            return None
        
        # Get latest features
        latest_features = features.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Predict signal
        signal_prob = self.classification_model.predict_proba(latest_features_scaled)[0]
        signal_class = self.classification_model.predict(latest_features_scaled)[0]
        
        signal_names = ['STRONG_SELL', 'HOLD', 'STRONG_BUY']
        
        return {
            'symbol': symbol,
            'signal': signal_names[signal_class],
            'signal_probability': signal_prob[signal_class],
            'probabilities': dict(zip(signal_names, signal_prob)),
            'momentum_6m': features['momentum_6m'].iloc[-1],
            'momentum_3m': features['momentum_3m'].iloc[-1],
            'risk_adj_momentum': features['momentum_6m_risk_adj'].iloc[-1]
        }

def main():
    """Train momentum-enhanced ML models"""
    
    print("🚀 MOMENTUM-ENHANCED ML MODEL TRAINING")
    print("=" * 45)
    print("🏛️ Using institutional momentum features")
    print("📊 Training on momentum portfolio stocks")
    print("🤖 Building classification + regression models")
    print("=" * 45)
    
    # Initialize ML system
    ml_model = MomentumEnhancedMLModel()
    
    # Train models on momentum portfolio stocks
    classification_results = []
    regression_results = []
    
    for symbol in ml_model.symbols[:3]:  # Test on first 3 stocks
        print(f"\n🎯 TRAINING MODELS FOR {symbol}")
        print("-" * 30)
        
        # Train classification model
        class_result = ml_model.train_classification_model(symbol)
        if class_result:
            classification_results.append(class_result)
        
        # Train regression model
        reg_result = ml_model.train_regression_model(symbol)
        if reg_result:
            regression_results.append(reg_result)
    
    # Summary
    if classification_results:
        avg_accuracy = np.mean([r['accuracy'] for r in classification_results])
        print(f"\n📊 CLASSIFICATION SUMMARY:")
        print(f"   Average Accuracy: {avg_accuracy:.3f}")
        
    if regression_results:
        avg_r2 = np.mean([r['r2_score'] for r in regression_results])
        print(f"\n📈 REGRESSION SUMMARY:")
        print(f"   Average R² Score: {avg_r2:.3f}")
    
    print(f"\n✅ ML MODELS ENHANCED WITH MOMENTUM FEATURES!")
    return ml_model

if __name__ == "__main__":
    ml_model = main()
