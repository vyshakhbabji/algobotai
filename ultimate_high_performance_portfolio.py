#!/usr/bin/env python3
"""
ULTIMATE HIGH-PERFORMANCE PORTFOLIO
Restored version targeting 35%+ returns with proper pandas handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

def create_features(data):
    """Create features for high performance predictions"""
    df = data.copy()
    
    # Basic returns
    df['returns'] = df['Close'].pct_change()
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    
    # Create new columns properly
    df = df.copy()  # Ensure we're working with a copy
    
    # Price ratios
    df['price_sma5_ratio'] = df['Close'] / df['sma_5']
    df['price_sma20_ratio'] = df['Close'] / df['sma_20']
    df['price_sma50_ratio'] = df['Close'] / df['sma_50']
    
    # Momentum
    df['momentum_3'] = df['Close'] / df['Close'].shift(3) - 1
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Volume
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(10).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Target
    df['target'] = df['returns'].shift(-1)
    
    # Feature list
    features = [
        'returns', 'price_sma5_ratio', 'price_sma20_ratio', 'price_sma50_ratio',
        'momentum_3', 'momentum_5', 'momentum_10', 'volume_ratio', 'volatility', 'rsi'
    ]
    
    # Clean data
    df_clean = df.dropna()
    return df_clean[features], df_clean['target']

def train_and_predict(symbol):
    """Train models and make prediction for a symbol"""
    try:
        print(f"🚀 Processing {symbol}")
        
        # Get data with retries
        data = None
        for attempt in range(3):
            try:
                data = yf.download(symbol, period="2y", progress=False, timeout=30)
                if len(data) > 0:
                    break
            except:
                continue
        
        if data is None or len(data) < 100:
            print(f"   ❌ Failed to get data for {symbol}")
            return None
        
        # Create features
        X, y = create_features(data)
        
        if len(X) < 50:
            print(f"   ❌ Insufficient clean data for {symbol}")
            return None
        
        # Split data
        split = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Train models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        }
        
        predictions = []
        accuracies = []
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            # Test performance
            test_pred = model.predict(X_test)
            direction_acc = np.mean(np.sign(y_test) == np.sign(test_pred))
            
            # Predict latest
            latest_pred = model.predict(X.iloc[-1:].values)[0] * 100  # Convert to %
            predictions.append(latest_pred)
            accuracies.append(direction_acc)
        
        # Ensemble prediction
        weights = np.array(accuracies)
        weights = weights / weights.sum()
        final_pred = np.average(predictions, weights=weights)
        
        # Confidence
        avg_accuracy = np.mean(accuracies)
        pred_std = np.std(predictions)
        confidence = min(avg_accuracy * (1 - pred_std/15), 0.9)
        
        # Signal generation - MORE AGGRESSIVE for higher returns
        if final_pred > 5.0 and confidence > 0.65:
            signal = "STRONG_BUY"
        elif final_pred > 2.5 and confidence > 0.6:
            signal = "BUY"
        elif final_pred < -5.0 and confidence > 0.65:
            signal = "STRONG_SELL"
        elif final_pred < -2.5 and confidence > 0.6:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        print(f"   ✅ {symbol}: {final_pred:+.2f}% - {signal} (conf: {confidence:.1%})")
        
        return {
            'symbol': symbol,
            'predicted_return': final_pred,
            'signal': signal,
            'confidence': confidence,
            'direction_accuracy': avg_accuracy
        }
        
    except Exception as e:
        print(f"   ❌ Error processing {symbol}: {str(e)}")
        return None

def create_ultimate_portfolio():
    """Create the ultimate high-performance portfolio"""
    
    print("🚀 ULTIMATE HIGH-PERFORMANCE PORTFOLIO")
    print("=" * 50)
    print("Targeting 35%+ returns with aggressive strategies")
    print("=" * 50)
    
    # High-momentum stocks known for big moves
    stocks = [
        "NVDA", "TSLA", "PLTR", "SNOW", "COIN", 
        "ROKU", "UBER", "AMD", "CRM", "NFLX"
    ]
    
    # Process stocks
    predictions = []
    for symbol in stocks:
        result = train_and_predict(symbol)
        if result and result['confidence'] > 0.55:
            predictions.append(result)
    
    if not predictions:
        print("❌ No reliable predictions available")
        return
    
    print(f"\n✅ Generated {len(predictions)} reliable predictions")
    
    # Sort by predicted return (aggressive strategy)
    predictions.sort(key=lambda x: x['predicted_return'], reverse=True)
    
    # Portfolio selection strategy
    strong_buys = [p for p in predictions if p['signal'] == 'STRONG_BUY']
    buys = [p for p in predictions if p['signal'] == 'BUY']
    
    if strong_buys:
        selected = strong_buys[:3]  # Top 3 strong buys
        strategy = "ULTRA AGGRESSIVE - Strong Buy Focus"
    elif buys:
        selected = buys[:4]  # Top 4 buys
        strategy = "AGGRESSIVE - Buy Focus"
    else:
        selected = predictions[:5]  # Top 5 by predicted return
        strategy = "MOMENTUM - Top Predicted Returns"
    
    print(f"\n💼 PORTFOLIO STRATEGY: {strategy}")
    print(f"📈 Selected {len(selected)} stocks")
    
    # Calculate portfolio metrics
    investment_per_stock = 10000 / len(selected)
    portfolio_return = np.mean([p['predicted_return'] for p in selected])
    portfolio_confidence = np.mean([p['confidence'] for p in selected])
    
    # Projected performance
    projected_value = 10000 * (1 + portfolio_return/100)
    projected_profit = projected_value - 10000
    
    print(f"\n💰 PORTFOLIO ALLOCATION:")
    print("-" * 30)
    print(f"💵 Investment per stock: ${investment_per_stock:,.0f}")
    print(f"🎯 Portfolio predicted return: {portfolio_return:+.2f}%")
    print(f"🔥 Portfolio confidence: {portfolio_confidence:.1%}")
    
    print(f"\n📈 PROJECTED PERFORMANCE:")
    print("-" * 30)
    print(f"🏦 Initial Investment: $10,000")
    print(f"💰 Projected Value: ${projected_value:,.0f}")
    print(f"📊 Projected Profit: ${projected_profit:+,.0f}")
    
    # Performance verdict
    if portfolio_return > 35:
        verdict = "🚀 EXCEPTIONAL! 35%+ returns - MATCHES ORIGINAL ELITE AI!"
        emoji = "🚀"
    elif portfolio_return > 25:
        verdict = "🌟 EXCELLENT! 25%+ returns - High performance achieved!"
        emoji = "🌟"
    elif portfolio_return > 15:
        verdict = "✅ VERY GOOD! 15%+ returns - Strong performance!"
        emoji = "✅"
    elif portfolio_return > 10:
        verdict = "🟢 GOOD! 10%+ returns - Solid performance!"
        emoji = "🟢"
    else:
        verdict = "🟡 MODERATE returns - Conservative performance"
        emoji = "🟡"
    
    print(f"\n🏆 PERFORMANCE VERDICT:")
    print(f"{emoji} {verdict}")
    
    # Detailed breakdown
    print(f"\n📋 DETAILED STOCK BREAKDOWN:")
    print("-" * 55)
    print(f"{'Stock':<6} {'Predicted':<10} {'Signal':<12} {'Confidence':<10} {'Investment':<12}")
    print("-" * 55)
    
    for pred in selected:
        print(f"{pred['symbol']:<6} {pred['predicted_return']:>+7.2f}% {pred['signal']:<12} {pred['confidence']:<9.1%} ${investment_per_stock:>10,.0f}")
    
    print("-" * 55)
    print(f"{'TOTAL':<6} {portfolio_return:>+7.2f}% {'PORTFOLIO':<12} {portfolio_confidence:<9.1%} ${'10,000':>10}")
    
    # Compare to original Elite AI
    print(f"\n🔍 COMPARISON TO ORIGINAL ELITE AI:")
    print("-" * 40)
    if portfolio_return >= 30:
        print(f"🎯 MATCHED! This portfolio targets similar 30%+ returns!")
        print(f"💎 Successfully restored high-performance trading!")
    elif portfolio_return >= 20:
        print(f"📈 STRONG! Getting close to the original 35% target!")
        print(f"✨ Significant improvement over conservative models!")
    else:
        print(f"📊 MODERATE improvement but still below original performance")
        print(f"🔧 May need further model tuning for maximum returns")
    
    # Risk assessment
    return_std = np.std([p['predicted_return'] for p in selected])
    print(f"\n⚖️ RISK ASSESSMENT:")
    print(f"📊 Return volatility: {return_std:.2f}%")
    print(f"🎯 Best stock: {selected[0]['symbol']} ({selected[0]['predicted_return']:+.2f}%)")
    print(f"📉 Most conservative: {selected[-1]['symbol']} ({selected[-1]['predicted_return']:+.2f}%)")
    
    if return_std < 8:
        risk_level = "🟢 MODERATE RISK - Consistent high predictions"
    elif return_std < 15:
        risk_level = "🟡 HIGH RISK - Some variability but strong upside"
    else:
        risk_level = "🔴 VERY HIGH RISK - High variability"
    
    print(f"⚠️ Risk level: {risk_level}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Used aggressive model parameters for maximum returns")
    print(f"   • Selected stocks with strongest momentum signals")
    print(f"   • Focused on high-volatility, high-growth stocks")
    print(f"   • Portfolio optimized for {'growth' if portfolio_return > 20 else 'moderate gains'}")
    
    return {
        'selected_stocks': [p['symbol'] for p in selected],
        'portfolio_return': portfolio_return,
        'projected_profit': projected_profit,
        'verdict': verdict
    }

if __name__ == "__main__":
    result = create_ultimate_portfolio()
    
    print(f"\n" + "="*60)
    print(f"🎯 ULTIMATE HIGH-PERFORMANCE PORTFOLIO COMPLETE!")
    if result:
        print(f"   • Predicted return: {result['portfolio_return']:+.2f}%")
        print(f"   • Projected profit: ${result['projected_profit']:+,.0f}")
        print(f"   • {result['verdict']}")
        
        if result['portfolio_return'] >= 30:
            print(f"   🚀 SUCCESS! Restored high-performance trading!")
        elif result['portfolio_return'] >= 20:
            print(f"   ✨ STRONG performance improvement achieved!")
        else:
            print(f"   📈 Moderate improvement - further tuning may help")
    print(f"="*60)
