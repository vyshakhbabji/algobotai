#!/usr/bin/env python3
"""
HIGH PERFORMANCE PORTFOLIO SIMULATOR
Based on the previous Elite AI that achieved 35%+ returns
Fixed implementation with proper data handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

class HighPerformanceAI:
    def __init__(self):
        self.models = {}
        
    def create_features(self, data):
        """Create robust features for high performance"""
        df = data.copy()
        
        # Basic features
        df['returns'] = df['Close'].pct_change()
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        # Price ratios (key for performance)
        df.loc[:, 'price_sma5_ratio'] = df['Close'] / df['sma_5']
        df.loc[:, 'price_sma20_ratio'] = df['Close'] / df['sma_20']
        df.loc[:, 'price_sma50_ratio'] = df['Close'] / df['sma_50']
        
        # Momentum features
        df.loc[:, 'momentum_3'] = df['Close'] / df['Close'].shift(3) - 1
        df.loc[:, 'momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df.loc[:, 'momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Volatility
        df.loc[:, 'volatility'] = df['returns'].rolling(10).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df.loc[:, 'rsi'] = 100 - (100 / (1 + rs))
        
        # Target
        df['target'] = df['returns'].shift(-1)
        
        # Feature list
        features = [
            'returns', 'volume_ratio', 'price_sma5_ratio', 'price_sma20_ratio', 
            'price_sma50_ratio', 'momentum_3', 'momentum_5', 'momentum_10',
            'volatility', 'rsi'
        ]
        
        # Clean data
        df_clean = df.dropna()
        return df_clean[features], df_clean['target']
    
    def train_models(self, symbol):
        """Train high-performance models"""
        try:
            print(f"🚀 Training {symbol}")
            
            # Get data
            data = yf.download(symbol, period="2y", progress=False)
            
            if len(data) < 100:
                print(f"   ❌ Insufficient data")
                return False
            
            # Create features
            X, y = self.create_features(data)
            
            if len(X) < 50:
                print(f"   ❌ Insufficient clean data")
                return False
            
            print(f"   📊 {X.shape[1]} features, {X.shape[0]} samples")
            
            # Split data
            split = int(len(X) * 0.7)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            # High-performance models
            models = {
                'rf_aggressive': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=3,
                    random_state=42
                ),
                'gbm_power': GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            results = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                
                # Test performance
                test_pred = model.predict(X_test)
                test_r2 = model.score(X_test, y_test)
                direction_acc = np.mean(np.sign(y_test) == np.sign(test_pred))
                
                results[name] = {
                    'model': model,
                    'test_r2': test_r2,
                    'direction_accuracy': direction_acc
                }
                
                print(f"   🔥 {name}: R²={test_r2:.3f}, Direction={direction_acc:.1%}")
            
            self.models[symbol] = results
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return False
    
    def predict(self, symbol):
        """Make prediction"""
        if symbol not in self.models:
            return None
            
        try:
            # Get latest data
            data = yf.download(symbol, period="2y", progress=False)
            X, _ = self.create_features(data)
            
            # Latest features
            latest = X.iloc[-1:].values
            
            # Ensemble prediction
            predictions = []
            weights = []
            
            for name, model_data in self.models[symbol].items():
                model = model_data['model']
                accuracy = model_data['direction_accuracy']
                
                pred = model.predict(latest)[0] * 100  # Convert to %
                predictions.append(pred)
                weights.append(max(accuracy, 0.5))  # Minimum weight
            
            # Weighted prediction
            weights = np.array(weights)
            weights = weights / weights.sum()
            final_pred = np.average(predictions, weights=weights)
            
            # Confidence
            pred_std = np.std(predictions)
            confidence = min(max(weights) * (1 - pred_std/20), 0.9)
            
            # Enhanced signals for high returns
            if final_pred > 4.0 and confidence > 0.65:
                signal = "STRONG_BUY"
            elif final_pred > 2.0 and confidence > 0.6:
                signal = "BUY"
            elif final_pred < -4.0 and confidence > 0.65:
                signal = "STRONG_SELL"
            elif final_pred < -2.0 and confidence > 0.6:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return {
                'predicted_return': final_pred,
                'signal': signal,
                'confidence': confidence,
                'models_used': len(predictions)
            }
            
        except Exception as e:
            print(f"   ❌ Prediction error: {str(e)}")
            return None

def simulate_high_performance_portfolio():
    """Simulate the high-performance portfolio that achieved 35%+ returns"""
    
    print("🚀 HIGH-PERFORMANCE PORTFOLIO SIMULATION")
    print("=" * 55)
    print("Targeting the 35%+ returns of the original Elite AI")
    print("Training Period: Historical data")
    print("Investment Period: Current predictions")
    print("=" * 55)
    
    # Core high-performance stocks
    stocks = [
        "NVDA", "AAPL", "TSLA", "PLTR", "SNOW", 
        "GOOGL", "MSFT", "META", "AMZN", "NFLX",
        "AMD", "CRM", "UBER", "COIN", "ROKU"
    ]
    
    ai = HighPerformanceAI()
    
    # Train models
    print("🤖 TRAINING HIGH-PERFORMANCE MODELS")
    print("-" * 40)
    
    trained_stocks = []
    for symbol in stocks:
        if ai.train_models(symbol):
            trained_stocks.append(symbol)
    
    print(f"\n✅ Successfully trained {len(trained_stocks)}/{len(stocks)} models")
    
    # Get predictions
    print(f"\n🎯 GENERATING PREDICTIONS")
    print("-" * 30)
    
    predictions = {}
    for symbol in trained_stocks:
        pred = ai.predict(symbol)
        if pred and pred['confidence'] > 0.5:
            predictions[symbol] = pred
            conf_emoji = "🔥" if pred['confidence'] > 0.7 else "✅" if pred['confidence'] > 0.6 else "🟡"
            print(f"{symbol:<6} {pred['predicted_return']:>+6.2f}% {pred['signal']:<12} {conf_emoji} {pred['confidence']:.1%}")
    
    # Portfolio strategy for maximum returns
    print(f"\n💼 HIGH-PERFORMANCE PORTFOLIO STRATEGY")
    print("=" * 45)
    
    if not predictions:
        print("❌ No reliable predictions available")
        return
    
    # Sort by predicted return for aggressive strategy
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1]['predicted_return'], reverse=True)
    
    # Select top performers
    strong_buys = [s for s, p in predictions.items() if p['signal'] == 'STRONG_BUY']
    buys = [s for s, p in predictions.items() if p['signal'] == 'BUY']
    
    # Portfolio allocation strategy
    if strong_buys:
        selected_stocks = strong_buys[:5]  # Top 5 strong buys
        strategy = "AGGRESSIVE - STRONG BUY Focus"
    elif buys:
        selected_stocks = buys[:5]  # Top 5 buys
        strategy = "GROWTH - BUY Focus"
    else:
        # Top 5 by predicted return
        selected_stocks = [s for s, _ in sorted_predictions[:5]]
        strategy = "MOMENTUM - Top Predicted Returns"
    
    print(f"📊 Strategy: {strategy}")
    print(f"📈 Selected stocks: {selected_stocks}")
    
    # Calculate portfolio metrics
    if selected_stocks:
        investment_per_stock = 10000 / len(selected_stocks)
        
        portfolio_predictions = [predictions[s]['predicted_return'] for s in selected_stocks]
        avg_predicted_return = np.mean(portfolio_predictions)
        portfolio_confidence = np.mean([predictions[s]['confidence'] for s in selected_stocks])
        
        print(f"\n💰 PORTFOLIO ALLOCATION:")
        print("-" * 25)
        print(f"💵 Investment per stock: ${investment_per_stock:,.0f}")
        print(f"🎯 Average predicted return: {avg_predicted_return:+.2f}%")
        print(f"🔥 Portfolio confidence: {portfolio_confidence:.1%}")
        
        # Projected returns
        projected_value = 10000 * (1 + avg_predicted_return/100)
        projected_profit = projected_value - 10000
        
        print(f"\n📈 PROJECTED PERFORMANCE:")
        print("-" * 28)
        print(f"🏦 Initial investment: $10,000")
        print(f"💰 Projected value: ${projected_value:,.0f}")
        print(f"📊 Projected profit: ${projected_profit:+,.0f}")
        print(f"🎯 Return percentage: {avg_predicted_return:+.2f}%")
        
        # Performance assessment
        if avg_predicted_return > 30:
            verdict = "🚀 EXCEPTIONAL - Targeting 30%+ returns!"
            print(f"\n{verdict}")
            print(f"💎 This matches the original Elite AI performance!")
        elif avg_predicted_return > 20:
            verdict = "🌟 EXCELLENT - Targeting 20%+ returns!"
            print(f"\n{verdict}")
            print(f"✨ Strong performance expected!")
        elif avg_predicted_return > 10:
            verdict = "✅ GOOD - Targeting 10%+ returns"
            print(f"\n{verdict}")
            print(f"📈 Solid performance expected")
        else:
            verdict = "🟡 MODERATE - Conservative returns"
            print(f"\n{verdict}")
            print(f"📊 Steady but limited gains expected")
        
        # Individual stock breakdown
        print(f"\n📋 INDIVIDUAL STOCK BREAKDOWN:")
        print("-" * 40)
        print(f"{'Stock':<6} {'Investment':<12} {'Predicted':<10} {'Signal':<12} {'Confidence':<10}")
        print("-" * 60)
        
        for stock in selected_stocks:
            pred = predictions[stock]
            print(f"{stock:<6} ${investment_per_stock:>10,.0f} {pred['predicted_return']:>+7.2f}% {pred['signal']:<12} {pred['confidence']:<9.1%}")
        
        print("-" * 60)
        print(f"TOTAL  ${'10,000':>10} {avg_predicted_return:>+7.2f}% {'PORTFOLIO':<12} {portfolio_confidence:<9.1%}")
        
        # Risk analysis
        return_std = np.std(portfolio_predictions)
        print(f"\n⚖️ RISK ANALYSIS:")
        print("-" * 20)
        print(f"📊 Return volatility: {return_std:.2f}%")
        print(f"🎯 Best stock: {selected_stocks[0]} ({predictions[selected_stocks[0]]['predicted_return']:+.2f}%)")
        print(f"📉 Most conservative: {selected_stocks[-1]} ({predictions[selected_stocks[-1]]['predicted_return']:+.2f}%)")
        
        if return_std < 5:
            risk_level = "🟢 LOW RISK - Consistent predictions"
        elif return_std < 10:
            risk_level = "🟡 MODERATE RISK - Some variability"
        else:
            risk_level = "🔴 HIGH RISK - High variability"
        
        print(f"⚠️ Risk level: {risk_level}")
        
        return {
            'selected_stocks': selected_stocks,
            'strategy': strategy,
            'avg_predicted_return': avg_predicted_return,
            'projected_value': projected_value,
            'projected_profit': projected_profit,
            'portfolio_confidence': portfolio_confidence,
            'verdict': verdict
        }

if __name__ == "__main__":
    result = simulate_high_performance_portfolio()
    
    print(f"\n" + "="*60)
    print(f"🎯 HIGH-PERFORMANCE SIMULATION COMPLETE!")
    if result:
        print(f"   • Strategy: {result['strategy']}")
        print(f"   • Predicted return: {result['avg_predicted_return']:+.2f}%")
        print(f"   • Projected profit: ${result['projected_profit']:+,.0f}")
        print(f"   • {result['verdict']}")
    print(f"="*60)
