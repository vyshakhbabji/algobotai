#!/usr/bin/env python3
"""
🔧 DEBUG LIVE TRADING BACKTEST
Debug why our AI models are returning 0 strength predictions
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Import our live trading system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from improved_ai_portfolio_manager import ImprovedAIPortfolioManager

def debug_prediction_system():
    """Debug the prediction strength calculation"""
    print("🔧 DEBUGGING LIVE TRADING PREDICTION SYSTEM")
    print("=" * 50)
    
    # Initialize AI manager
    ai_manager = ImprovedAIPortfolioManager()
    
    # Train a single model for debugging
    symbol = "NVDA"
    print(f"🧠 Training model for {symbol}...")
    
    # Get training data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock = yf.Ticker(symbol)
    training_data = stock.history(start=start_date, end=end_date)
    
    print(f"📊 Training data shape: {training_data.shape}")
    
    # Train model
    model, scaler, r2_score = ai_manager.train_improved_model(symbol, data=training_data)
    
    if model is not None:
        ai_manager.models[symbol] = (model, scaler, r2_score)
        print(f"✅ Model trained successfully with R² = {r2_score:.3f}")
        
        # Get recent data for prediction
        recent_start = end_date - timedelta(days=60)
        recent_data = stock.history(start=recent_start, end=end_date)
        print(f"📊 Recent data shape: {recent_data.shape}")
        
        # Debug feature calculation
        print(f"\n🔍 DEBUGGING FEATURE CALCULATION:")
        df = recent_data.copy()
        df = ai_manager.calculate_improved_features(df)
        
        feature_cols = [
            'price_vs_sma10', 'price_vs_sma30', 'price_vs_sma50',
            'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_10', 'volatility_30',
            'volume_ratio', 'volume_momentum',
            'rsi_normalized', 'bb_position', 'bb_squeeze',
            'macd', 'macd_histogram', 'price_position'
        ]
        
        print(f"📋 Available features: {list(df.columns)}")
        
        # Check latest features
        latest_features = df[feature_cols].iloc[-1:]
        print(f"\n📊 Latest features:")
        for col in feature_cols:
            if col in df.columns:
                value = latest_features[col].iloc[0]
                print(f"   {col}: {value:.6f} (NaN: {pd.isna(value)})")
            else:
                print(f"   {col}: MISSING")
        
        # Test prediction
        prediction_strength = ai_manager.get_prediction_strength(symbol, recent_data)
        print(f"\n🎯 Prediction strength: {prediction_strength}")
        
        # Manual prediction calculation
        if not latest_features.isna().any().any():
            features_array = latest_features.values
            print(f"\n🧮 Manual prediction calculation:")
            print(f"   Features shape: {features_array.shape}")
            print(f"   Features: {features_array[0][:5]}... (first 5)")
            
            # Scale features
            features_scaled = scaler.transform(features_array)
            print(f"   Scaled features: {features_scaled[0][:5]}... (first 5)")
            
            # Predict
            predicted_return = model.predict(features_scaled)[0]
            print(f"   Raw prediction: {predicted_return}")
            
            # Calculate strength score
            base_strength = max(0, min(100, (predicted_return + 0.1) * 500))
            weighted_strength = base_strength * (0.5 + r2_score)
            final_strength = max(0, min(100, weighted_strength))
            
            print(f"   Base strength: {base_strength}")
            print(f"   Weighted strength: {weighted_strength}")
            print(f"   Final strength: {final_strength}")
        else:
            print(f"\n❌ Features contain NaN values!")
            nan_features = latest_features.columns[latest_features.isna().iloc[0]]
            print(f"   NaN features: {list(nan_features)}")
            
    else:
        print(f"❌ Model training failed!")
    
    # Test with different stocks
    print(f"\n🧪 TESTING MULTIPLE STOCKS:")
    test_symbols = ["AAPL", "GOOGL", "TSLA", "AMD"]
    
    for symbol in test_symbols:
        try:
            print(f"\n   Testing {symbol}...")
            
            # Get data
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            recent_data = stock.history(start=recent_start, end=end_date)
            
            # Train model
            model, scaler, r2_score = ai_manager.train_improved_model(symbol, data=data)
            
            if model is not None:
                ai_manager.models[symbol] = (model, scaler, r2_score)
                strength = ai_manager.get_prediction_strength(symbol, recent_data)
                print(f"      ✅ {symbol}: R² = {r2_score:.3f}, Strength = {strength:.1f}")
            else:
                print(f"      ❌ {symbol}: Training failed")
                
        except Exception as e:
            print(f"      ❌ {symbol}: Error - {e}")
    
    # Test with simpler threshold
    print(f"\n🔧 TESTING WITH LOWER STRENGTH THRESHOLD:")
    for symbol, model_data in ai_manager.models.items():
        if model_data is not None:
            recent_data = yf.Ticker(symbol).history(start=recent_start, end=end_date)
            strength = ai_manager.get_prediction_strength(symbol, recent_data)
            print(f"   {symbol}: Strength = {strength:.1f} ({'✅' if strength > 10 else '❌'} above 10)")

if __name__ == "__main__":
    debug_prediction_system()
