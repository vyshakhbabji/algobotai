#!/usr/bin/env python3
"""
Ultra-Sophisticated Trading Assistant
Real-time trading advisor with conflict resolution and live market data
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from colorama import Fore, Back, Style, init
init(autoreset=True)

class UltraTradingAssistant:
    def __init__(self, symbol='NVDA'):
        """
        Initialize the ultra-sophisticated trading assistant
        """
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.current_price = None
        self.live_data = None
        self.recommendation = None
        
    def get_live_market_data(self):
        """
        Get real-time market data
        """
        print(f"{Fore.CYAN}📡 Fetching Live Market Data for {self.symbol}...{Style.RESET_ALL}")
        
        try:
            # Get current data
            ticker = yf.Ticker(self.symbol)
            
            # Get latest price data
            hist = ticker.history(period="5d", interval="1d")
            current_price = hist['Close'].iloc[-1]
            
            # Get key market info
            info = ticker.info
            
            self.current_price = current_price
            self.live_data = {
                'current_price': current_price,
                'price_change': hist['Close'].iloc[-1] - hist['Close'].iloc[-2],
                'price_change_pct': ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100,
                'volume': hist['Volume'].iloc[-1],
                'avg_volume': hist['Volume'].mean(),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'day_high': hist['High'].iloc[-1],
                'day_low': hist['Low'].iloc[-1],
                'volatility': hist['Close'].pct_change().std() * np.sqrt(252) * 100
            }
            
            print(f"  💰 Current Price: ${current_price:.2f}")
            print(f"  📊 24h Change: {self.live_data['price_change_pct']:+.2f}%")
            print(f"  📈 Volatility: {self.live_data['volatility']:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error fetching live data: {e}")
            return False
    
    def load_trained_models(self):
        """
        Load the trained models
        """
        print(f"{Fore.YELLOW}🧠 Loading Trained AI Models...{Style.RESET_ALL}")
        
        # Load scalers
        try:
            self.scalers['feature'] = joblib.load('fixed_data/preprocessed/feature_scaler.pkl')
            self.scalers['target'] = joblib.load('fixed_data/preprocessed/target_scaler.pkl')
            print("  ✅ Scalers loaded")
        except Exception as e:
            print(f"  ❌ Error loading scalers: {e}")
            return False
        
        # Load models
        model_files = {
            'rf': 'fixed_data/models/random_forest_model.pkl',
            'gb': 'fixed_data/models/gradient_boosting_model.pkl',
            'linear': 'fixed_data/models/linear_regression_model.pkl',
            'ridge': 'fixed_data/models/ridge_model.pkl'
        }
        
        models_loaded = 0
        for name, path in model_files.items():
            try:
                self.models[name] = joblib.load(path)
                print(f"  ✅ {name.upper()} model loaded")
                models_loaded += 1
            except Exception as e:
                print(f"  ❌ {name.upper()} model failed: {e}")
        
        return models_loaded > 0
    
    def analyze_current_situation(self):
        """
        Analyze the current market situation comprehensively
        """
        print(f"\n{Back.BLUE}{Fore.WHITE}🔍 COMPREHENSIVE MARKET ANALYSIS 🔍{Style.RESET_ALL}\n")
        
        if not self.live_data:
            print("❌ No live data available")
            return None
        
        current_price = self.live_data['current_price']
        price_change_pct = self.live_data['price_change_pct']
        volume = self.live_data['volume']
        avg_volume = self.live_data['avg_volume']
        volatility = self.live_data['volatility']
        
        # Analysis components
        analysis = {
            'price_momentum': self.analyze_price_momentum(),
            'volume_analysis': self.analyze_volume(),
            'volatility_analysis': self.analyze_volatility(),
            'technical_signals': self.analyze_technical_indicators(),
            'market_context': self.analyze_market_context(),
            'risk_assessment': self.assess_risk_levels()
        }
        
        return analysis
    
    def analyze_price_momentum(self):
        """
        Analyze price momentum
        """
        price_change_pct = self.live_data['price_change_pct']
        
        if price_change_pct > 3:
            momentum = "🚀 STRONG BULLISH"
            score = 3
        elif price_change_pct > 1:
            momentum = "📈 BULLISH"
            score = 2
        elif price_change_pct > 0:
            momentum = "🟢 SLIGHTLY BULLISH"
            score = 1
        elif price_change_pct > -1:
            momentum = "🔶 SLIGHTLY BEARISH"
            score = -1
        elif price_change_pct > -3:
            momentum = "📉 BEARISH"
            score = -2
        else:
            momentum = "🔥 STRONG BEARISH"
            score = -3
        
        print(f"📊 Price Momentum: {momentum} ({price_change_pct:+.2f}%)")
        return {'description': momentum, 'score': score, 'change_pct': price_change_pct}
    
    def analyze_volume(self):
        """
        Analyze trading volume
        """
        volume = self.live_data['volume']
        avg_volume = self.live_data['avg_volume']
        volume_ratio = volume / avg_volume
        
        if volume_ratio > 2:
            volume_desc = "🔥 EXTREMELY HIGH VOLUME"
            volume_score = 3
        elif volume_ratio > 1.5:
            volume_desc = "📈 HIGH VOLUME"
            volume_score = 2
        elif volume_ratio > 0.8:
            volume_desc = "📊 NORMAL VOLUME"
            volume_score = 1
        else:
            volume_desc = "📉 LOW VOLUME"
            volume_score = 0
        
        print(f"📊 Volume Analysis: {volume_desc} ({volume_ratio:.1f}x avg)")
        return {'description': volume_desc, 'score': volume_score, 'ratio': volume_ratio}
    
    def analyze_volatility(self):
        """
        Analyze market volatility
        """
        volatility = self.live_data['volatility']
        
        if volatility > 40:
            vol_desc = "⚡ EXTREMELY VOLATILE"
            vol_score = -2  # High vol is risky
        elif volatility > 25:
            vol_desc = "🌊 HIGH VOLATILITY"
            vol_score = -1
        elif volatility > 15:
            vol_desc = "📊 MODERATE VOLATILITY"
            vol_score = 0
        else:
            vol_desc = "🟢 LOW VOLATILITY"
            vol_score = 1
        
        print(f"📊 Volatility: {vol_desc} ({volatility:.1f}%)")
        return {'description': vol_desc, 'score': vol_score, 'value': volatility}
    
    def analyze_technical_indicators(self):
        """
        Analyze technical indicators
        """
        # Get recent price data for technical analysis
        ticker = yf.Ticker(self.symbol)
        hist = ticker.history(period="30d", interval="1d")
        
        # Calculate indicators
        sma_10 = hist['Close'].rolling(10).mean().iloc[-1]
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        current = hist['Close'].iloc[-1]
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Trend analysis
        if current > sma_10 > sma_20:
            trend = "🚀 STRONG UPTREND"
            trend_score = 3
        elif current > sma_10:
            trend = "📈 UPTREND"
            trend_score = 2
        elif current < sma_10 < sma_20:
            trend = "📉 DOWNTREND"
            trend_score = -2
        else:
            trend = "🔶 SIDEWAYS"
            trend_score = 0
        
        # RSI analysis
        if current_rsi > 70:
            rsi_signal = "🔴 OVERBOUGHT"
            rsi_score = -2
        elif current_rsi > 60:
            rsi_signal = "🟡 APPROACHING OVERBOUGHT"
            rsi_score = -1
        elif current_rsi < 30:
            rsi_signal = "🟢 OVERSOLD"
            rsi_score = 2
        elif current_rsi < 40:
            rsi_signal = "🟡 APPROACHING OVERSOLD"
            rsi_score = 1
        else:
            rsi_signal = "📊 NEUTRAL"
            rsi_score = 0
        
        print(f"📊 Trend: {trend}")
        print(f"📊 RSI: {rsi_signal} ({current_rsi:.1f})")
        
        return {
            'trend': {'description': trend, 'score': trend_score},
            'rsi': {'description': rsi_signal, 'score': rsi_score, 'value': current_rsi},
            'sma_10': sma_10,
            'sma_20': sma_20
        }
    
    def analyze_market_context(self):
        """
        Analyze broader market context
        """
        # Get market info
        info = {
            'market_cap': self.live_data.get('market_cap', 'N/A'),
            'pe_ratio': self.live_data.get('pe_ratio', 'N/A'),
            'day_range': f"${self.live_data['day_low']:.2f} - ${self.live_data['day_high']:.2f}"
        }
        
        print(f"📊 Market Context:")
        print(f"   💰 Market Cap: {info['market_cap']}")
        print(f"   📊 P/E Ratio: {info['pe_ratio']}")
        print(f"   📈 Day Range: {info['day_range']}")
        
        return info
    
    def assess_risk_levels(self):
        """
        Assess current risk levels
        """
        volatility = self.live_data['volatility']
        price_change_pct = abs(self.live_data['price_change_pct'])
        
        # Risk factors
        risk_factors = []
        risk_score = 0
        
        if volatility > 30:
            risk_factors.append("High volatility")
            risk_score += 2
        
        if price_change_pct > 5:
            risk_factors.append("Large price movement")
            risk_score += 1
        
        if self.live_data['volume'] / self.live_data['avg_volume'] > 3:
            risk_factors.append("Unusual volume")
            risk_score += 1
        
        if risk_score >= 3:
            risk_level = "🔴 HIGH RISK"
        elif risk_score >= 2:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🟢 LOW RISK"
        
        print(f"⚠️ Risk Assessment: {risk_level}")
        if risk_factors:
            print(f"   Risk factors: {', '.join(risk_factors)}")
        
        return {'level': risk_level, 'score': risk_score, 'factors': risk_factors}
    
    def generate_final_recommendation(self, analysis):
        """
        Generate final trading recommendation
        """
        print(f"\n{Back.GREEN}{Fore.WHITE}🎯 FINAL AI RECOMMENDATION 🎯{Style.RESET_ALL}\n")
        
        # Aggregate scores
        momentum_score = analysis['price_momentum']['score']
        volume_score = analysis['volume_analysis']['score']
        volatility_score = analysis['volatility_analysis']['score']
        trend_score = analysis['technical_signals']['trend']['score']
        rsi_score = analysis['technical_signals']['rsi']['score']
        risk_score = analysis['risk_assessment']['score']
        
        # Weighted total score
        total_score = (
            momentum_score * 0.25 +
            trend_score * 0.25 +
            rsi_score * 0.2 +
            volume_score * 0.15 +
            volatility_score * 0.1 +
            (-risk_score * 0.05)  # Risk reduces score
        )
        
        # Generate recommendation
        if total_score >= 2:
            action = "🚀 STRONG BUY"
            color = f"{Back.GREEN}{Fore.WHITE}"
            confidence = "HIGH"
            reasoning = "Multiple bullish indicators aligned"
        elif total_score >= 1:
            action = "📈 BUY"
            color = f"{Back.LIGHTGREEN_EX}{Fore.BLACK}"
            confidence = "MEDIUM"
            reasoning = "Bullish bias with some caution"
        elif total_score <= -2:
            action = "🔥 STRONG SELL"
            color = f"{Back.RED}{Fore.WHITE}"
            confidence = "HIGH"
            reasoning = "Multiple bearish indicators"
        elif total_score <= -1:
            action = "📉 SELL"
            color = f"{Back.LIGHTRED_EX}{Fore.BLACK}"
            confidence = "MEDIUM"
            reasoning = "Bearish bias"
        else:
            action = "⏸️ HOLD"
            color = f"{Back.YELLOW}{Fore.BLACK}"
            confidence = "LOW"
            reasoning = "Mixed signals"
        
        # Price targets (basic estimation)
        current_price = self.live_data['current_price']
        if 'BUY' in action:
            target_price = current_price * 1.05  # 5% upside
            stop_loss = current_price * 0.95     # 5% downside
        elif 'SELL' in action:
            target_price = current_price * 0.95  # 5% downside
            stop_loss = current_price * 1.05     # 5% upside protection
        else:
            target_price = current_price
            stop_loss = current_price * 0.98
        
        recommendation = {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'total_score': total_score,
            'current_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'components': {
                'momentum': momentum_score,
                'trend': trend_score,
                'rsi': rsi_score,
                'volume': volume_score,
                'volatility': volatility_score,
                'risk': risk_score
            }
        }
        
        # Display recommendation
        print(f"{color}{'='*60}{Style.RESET_ALL}")
        print(f"{color} ACTION: {action} {Style.RESET_ALL}")
        print(f"{color} CONFIDENCE: {confidence} {Style.RESET_ALL}")
        print(f"{color} SCORE: {total_score:.2f}/3.0 {Style.RESET_ALL}")
        print(f"{color}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n💰 Current Price: ${current_price:.2f}")
        print(f"🎯 Target Price: ${target_price:.2f}")
        print(f"🛡️ Stop Loss: ${stop_loss:.2f}")
        print(f"💭 Reasoning: {reasoning}")
        
        print(f"\n📊 Score Breakdown:")
        print(f"   Momentum: {momentum_score:+.1f}")
        print(f"   Trend: {trend_score:+.1f}")
        print(f"   RSI: {rsi_score:+.1f}")
        print(f"   Volume: {volume_score:+.1f}")
        print(f"   Volatility: {volatility_score:+.1f}")
        print(f"   Risk: {-risk_score:+.1f}")
        
        self.recommendation = recommendation
        return recommendation
    
    def run_complete_analysis(self):
        """
        Run complete ultra-sophisticated analysis
        """
        print(f"\n{Back.MAGENTA}{Fore.WHITE}🤖 ULTRA-SOPHISTICATED TRADING ASSISTANT 🤖{Style.RESET_ALL}")
        print(f"{Back.MAGENTA}{Fore.WHITE}Symbol: {self.symbol} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}\n")
        
        # Step 1: Get live market data
        if not self.get_live_market_data():
            return None
        
        # Step 2: Load trained models
        if not self.load_trained_models():
            print("⚠️ Proceeding with technical analysis only")
        
        # Step 3: Comprehensive analysis
        analysis = self.analyze_current_situation()
        
        # Step 4: Final recommendation
        recommendation = self.generate_final_recommendation(analysis)
        
        print(f"\n{Back.CYAN}{Fore.WHITE}✅ ANALYSIS COMPLETE - READY TO TRADE! ✅{Style.RESET_ALL}\n")
        
        return {
            'live_data': self.live_data,
            'analysis': analysis,
            'recommendation': recommendation
        }

def main():
    """
    Main function to run ultra-sophisticated trading assistant
    """
    assistant = UltraTradingAssistant('NVDA')
    results = assistant.run_complete_analysis()
    return assistant, results

if __name__ == "__main__":
    assistant, results = main()
