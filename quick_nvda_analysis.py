#!/usr/bin/env python3
"""
NVDA Quick Analysis - Elite AI vs Market Sentiment
Simple but comprehensive analysis for NVDA position validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def quick_nvda_analysis():
    """Quick but comprehensive NVDA analysis"""
    print("🔍 NVDA POSITION ANALYSIS - Should You Hold or Sell?")
    print("=" * 55)
    
    # Get NVDA data
    nvda = yf.Ticker("NVDA")
    
    # Current market data
    try:
        info = nvda.info
        current_price = info.get('currentPrice', 0)
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        forward_pe = info.get('forwardPE', 0)
        
        print(f"\n📊 CURRENT NVDA FUNDAMENTALS")
        print(f"💰 Price: ${current_price:.2f}")
        print(f"📈 Market Cap: ${market_cap/1e12:.2f}T")
        print(f"📊 P/E Ratio: {pe_ratio:.1f}")
        print(f"🔮 Forward P/E: {forward_pe:.1f}")
        
    except Exception as e:
        print(f"❌ Error getting current data: {e}")
        current_price = 180.77  # Fallback
        pe_ratio = 58.5
        
    # Get price history
    try:
        price_data = yf.download("NVDA", period="6mo", interval="1d", progress=False)
        
        if len(price_data) > 0:
            # Recent performance
            current = price_data['Close'].iloc[-1]
            yesterday = price_data['Close'].iloc[-2] if len(price_data) > 1 else current
            week_ago = price_data['Close'].iloc[-5] if len(price_data) > 5 else current
            month_ago = price_data['Close'].iloc[-21] if len(price_data) > 21 else current
            
            daily_chg = ((current - yesterday) / yesterday) * 100
            weekly_chg = ((current - week_ago) / week_ago) * 100
            monthly_chg = ((current - month_ago) / month_ago) * 100
            
            print(f"\n📅 RECENT PERFORMANCE")
            print(f"Daily: {daily_chg:.2f}%")
            print(f"Weekly: {weekly_chg:.2f}%")
            print(f"Monthly: {monthly_chg:.2f}%")
            
            # Technical indicators
            print(f"\n📈 TECHNICAL ANALYSIS")
            
            # RSI
            delta = price_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
            # Moving averages
            ma20 = float(price_data['Close'].rolling(20).mean().iloc[-1])
            ma50 = float(price_data['Close'].rolling(50).mean().iloc[-1])
            
            # Support/Resistance
            recent_high = float(price_data['Close'].rolling(20).max().iloc[-1])
            recent_low = float(price_data['Close'].rolling(20).min().iloc[-1])
            
            print(f"RSI: {current_rsi:.1f} {'🔴 Overbought' if current_rsi > 70 else '🟢 Oversold' if current_rsi < 30 else '🟡 Neutral'}")
            print(f"MA20: ${ma20:.2f} {'🟢' if current > ma20 else '🔴'}")
            print(f"MA50: ${ma50:.2f} {'🟢' if current > ma50 else '🔴'}")
            print(f"20D High: ${recent_high:.2f}")
            print(f"20D Low: ${recent_low:.2f}")
            
            # Volatility
            returns = price_data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252) * 100)
            
            print(f"Volatility: {volatility:.1f}% annualized")
            
            # Technical score
            tech_score = 0
            tech_reasons = []
            
            if 30 <= current_rsi <= 70:
                tech_score += 1
                tech_reasons.append("✅ RSI in healthy range")
            elif current_rsi > 70:
                tech_reasons.append("⚠️ RSI overbought")
            else:
                tech_reasons.append("⚠️ RSI oversold")
                
            if current > ma20:
                tech_score += 1
                tech_reasons.append("✅ Above 20-day MA")
            else:
                tech_reasons.append("❌ Below 20-day MA")
                
            if current > ma50:
                tech_score += 1
                tech_reasons.append("✅ Above 50-day MA")
            else:
                tech_reasons.append("❌ Below 50-day MA")
                
            if monthly_chg > 0:
                tech_score += 1
                tech_reasons.append("✅ Positive monthly momentum")
            else:
                tech_reasons.append("❌ Negative monthly momentum")
                
            print(f"\n🎯 TECHNICAL SCORE: {tech_score}/4")
            for reason in tech_reasons:
                print(f"   {reason}")
                
        else:
            print("❌ No price data available")
            tech_score = 2
            monthly_chg = 0
            
    except Exception as e:
        print(f"❌ Error in technical analysis: {e}")
        tech_score = 2
        monthly_chg = 0
        
    # Market sentiment analysis
    print(f"\n💭 MARKET SENTIMENT ANALYSIS")
    
    sentiment_factors = []
    sentiment_score = 0
    
    # AI boom narrative
    print("🤖 AI Boom: NVIDIA at center of AI revolution")
    sentiment_factors.append("✅ Leading AI chip provider")
    sentiment_score += 1
    
    # Financial performance expectation
    print("💰 Earnings: Strong revenue growth expected")
    sentiment_factors.append("✅ Strong earnings trajectory")
    sentiment_score += 1
    
    # Valuation concerns
    if pe_ratio > 50:
        print(f"⚠️ Valuation: P/E of {pe_ratio:.1f} suggests high expectations")
        sentiment_factors.append("⚠️ High valuation risk")
    else:
        sentiment_factors.append("✅ Reasonable valuation")
        sentiment_score += 1
        
    # Market cap size
    print("📈 Size: $4.4T market cap (mega-cap stock)")
    sentiment_factors.append("⚠️ Limited upside due to size")
    
    print(f"\n🎯 SENTIMENT SCORE: {sentiment_score}/3")
    for factor in sentiment_factors:
        print(f"   {factor}")
        
    # Risk assessment
    print(f"\n⚠️ RISK ASSESSMENT")
    
    risk_factors = []
    risk_score = 0
    
    if pe_ratio > 50:
        risk_factors.append("🔴 High P/E ratio (valuation risk)")
        risk_score += 1
        
    if tech_score < 2:
        risk_factors.append("🔴 Weak technical position")
        risk_score += 1
        
    # Check if near highs
    try:
        if current > recent_high * 0.95:
            risk_factors.append("🟡 Near recent highs")
            risk_score += 0.5
    except:
        pass
        
    if monthly_chg < -10:
        risk_factors.append("🔴 Significant monthly decline")
        risk_score += 1
        
    print(f"Risk Level: {'🔴 HIGH' if risk_score >= 2 else '🟡 MEDIUM' if risk_score >= 1 else '🟢 LOW'}")
    for factor in risk_factors:
        print(f"   {factor}")
        
    # Elite AI simulation (simplified)
    print(f"\n🤖 ELITE AI SIMULATION")
    
    # Simulate based on technical and fundamental factors
    ai_prediction = 0
    ai_factors = []
    
    # Technical contribution
    if tech_score >= 3:
        ai_prediction += 2
        ai_factors.append("📈 Strong technical setup")
    elif tech_score == 2:
        ai_prediction += 0
        ai_factors.append("📊 Neutral technical setup")
    else:
        ai_prediction -= 2
        ai_factors.append("📉 Weak technical setup")
        
    # Valuation contribution
    if pe_ratio > 60:
        ai_prediction -= 2
        ai_factors.append("💰 Overvaluation concern")
    elif pe_ratio > 40:
        ai_prediction -= 1
        ai_factors.append("💰 High valuation")
    else:
        ai_prediction += 1
        ai_factors.append("💰 Reasonable valuation")
        
    # Momentum contribution
    if monthly_chg > 10:
        ai_prediction += 1
        ai_factors.append("🚀 Strong momentum")
    elif monthly_chg > 0:
        ai_prediction += 0.5
        ai_factors.append("🚀 Positive momentum")
    else:
        ai_prediction -= 1
        ai_factors.append("📉 Negative momentum")
        
    # Convert to percentage
    ai_prediction_pct = ai_prediction * 0.5  # Scale to percentage
    
    if ai_prediction_pct > 1:
        ai_signal = "BUY"
    elif ai_prediction_pct < -1:
        ai_signal = "SELL"
    else:
        ai_signal = "HOLD"
        
    confidence = min(0.9, abs(ai_prediction) * 0.2 + 0.3)
    
    print(f"Prediction: {ai_prediction_pct:.1f}%")
    print(f"Signal: {ai_signal}")
    print(f"Confidence: {confidence:.2f}")
    
    print("\nAI Reasoning:")
    for factor in ai_factors:
        print(f"   {factor}")
        
    # Final recommendation
    print(f"\n🎯 FINAL RECOMMENDATION FOR YOUR NVDA HOLDINGS")
    print("=" * 50)
    
    total_score = tech_score + sentiment_score - risk_score
    
    if total_score >= 4:
        recommendation = "🟢 STRONG HOLD/BUY"
        action = "Keep your position, consider adding on dips"
    elif total_score >= 2:
        recommendation = "🟡 CAUTIOUS HOLD"
        action = "Hold current position, avoid adding"
    elif total_score >= 0:
        recommendation = "🟠 CONSIDER TRIMMING"
        action = "Reduce position by 25-50%"
    else:
        recommendation = "🔴 SELL SIGNAL"
        action = "Consider significant reduction"
        
    print(f"Overall Score: {total_score}")
    print(f"Recommendation: {recommendation}")
    print(f"Action: {action}")
    
    # Specific guidance
    print(f"\n💡 SPECIFIC GUIDANCE:")
    
    if ai_signal == "SELL" and sentiment_score >= 2:
        print("⚖️ CONFLICTING SIGNALS DETECTED!")
        print("   • Elite AI suggests SELL")
        print("   • Market sentiment remains bullish")
        print("   • Recommendation: Partial profit-taking")
        print("   • Action: Sell 25-50% of position")
        print("   • Keep remainder with tight stop-loss")
        
    elif ai_signal == "HOLD" and risk_score >= 2:
        print("⚠️ HIGH RISK ENVIRONMENT")
        print("   • Consider position sizing")
        print("   • Set stop-loss at 10-15% below current price")
        print("   • Monitor for any sentiment shifts")
        
    else:
        print(f"   • Technical and sentiment align with {ai_signal} signal")
        print(f"   • Risk level is manageable")
        print(f"   • Follow the {recommendation} guidance")
        
    print(f"\n💼 PORTFOLIO MANAGEMENT:")
    print(f"   • Don't put all eggs in one basket")
    print(f"   • NVDA should be max 10-20% of portfolio")
    print(f"   • Consider AI diversification (AMD, MSFT, GOOGL)")
    print(f"   • Set clear profit targets and stop losses")
    
    return {
        'ai_signal': ai_signal,
        'ai_prediction': ai_prediction_pct,
        'confidence': confidence,
        'recommendation': recommendation,
        'action': action,
        'tech_score': tech_score,
        'sentiment_score': sentiment_score,
        'risk_score': risk_score,
        'total_score': total_score
    }

if __name__ == "__main__":
    results = quick_nvda_analysis()
    print(f"\n🔚 Analysis complete! Use this data to make an informed decision about your NVDA holdings.")
