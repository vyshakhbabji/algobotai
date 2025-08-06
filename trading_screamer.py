#!/usr/bin/env python3
"""
Personal Trading Screamer
Your personal AI that will scream buy/sell/hold signals with clear reasoning
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Back, Style, init
import time
import os
init(autoreset=True)

class TradingScreener:
    def __init__(self, symbol='NVDA'):
        self.symbol = symbol
        
    def scream_alert(self, message, alert_type='INFO'):
        """Make it impossible to miss the alert"""
        if alert_type == 'BUY':
            color = f"{Back.GREEN}{Fore.WHITE}"
            icon = "🚀💰🚀"
        elif alert_type == 'SELL':
            color = f"{Back.RED}{Fore.WHITE}"
            icon = "🔥⚠️🔥"
        elif alert_type == 'HOLD':
            color = f"{Back.YELLOW}{Fore.BLACK}"
            icon = "⏸️📊⏸️"
        else:
            color = f"{Back.BLUE}{Fore.WHITE}"
            icon = "📢📢📢"
        
        print(f"\n{color}{'='*80}{Style.RESET_ALL}")
        print(f"{color} {icon} {message} {icon} {Style.RESET_ALL}")
        print(f"{color}{'='*80}{Style.RESET_ALL}\n")
    
    def analyze_and_scream(self):
        """Run analysis and scream the results"""
        
        print(f"{Back.MAGENTA}{Fore.WHITE}🤖 YOUR PERSONAL TRADING SCREAMER IS ANALYZING... 🤖{Style.RESET_ALL}\n")
        
        # Get live data
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period="30d")
            info = ticker.info
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change_pct = ((current_price / prev_price) - 1) * 100
            
            # Technical indicators
            sma_5 = hist['Close'].rolling(5).mean().iloc[-1]
            sma_10 = hist['Close'].rolling(10).mean().iloc[-1]
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            
            # Volume analysis
            avg_volume = hist['Volume'].mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Volatility
            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
            
        except Exception as e:
            self.scream_alert(f"ERROR GETTING DATA: {e}", "SELL")
            return
        
        # Analysis
        print(f"📊 {self.symbol} Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"📈 24h Change: {change_pct:+.2f}%")
        print(f"📊 Volume: {volume_ratio:.1f}x average")
        print(f"🎯 RSI: {rsi:.1f}")
        print(f"⚡ Volatility: {volatility:.1f}%")
        print()
        
        # Decision logic
        signals = []
        
        # Trend signals
        if current_price > sma_5 > sma_10 > sma_20:
            signals.append(("STRONG UPTREND", 3))
        elif current_price > sma_5 > sma_10:
            signals.append(("UPTREND", 2))
        elif current_price < sma_5 < sma_10 < sma_20:
            signals.append(("STRONG DOWNTREND", -3))
        elif current_price < sma_5 < sma_10:
            signals.append(("DOWNTREND", -2))
        else:
            signals.append(("SIDEWAYS", 0))
        
        # Momentum signals
        if change_pct > 3:
            signals.append(("STRONG MOMENTUM UP", 2))
        elif change_pct > 1:
            signals.append(("MOMENTUM UP", 1))
        elif change_pct < -3:
            signals.append(("STRONG MOMENTUM DOWN", -2))
        elif change_pct < -1:
            signals.append(("MOMENTUM DOWN", -1))
        
        # RSI signals
        if rsi < 30:
            signals.append(("OVERSOLD - BUY OPPORTUNITY", 2))
        elif rsi > 70:
            signals.append(("OVERBOUGHT - SELL SIGNAL", -2))
        elif rsi < 40:
            signals.append(("APPROACHING OVERSOLD", 1))
        elif rsi > 60:
            signals.append(("APPROACHING OVERBOUGHT", -1))
        
        # Volume signals
        if volume_ratio > 2:
            signals.append(("UNUSUAL HIGH VOLUME", 1))
        elif volume_ratio < 0.5:
            signals.append(("LOW VOLUME - WEAK SIGNAL", -1))
        
        # Risk signals
        if volatility > 40:
            signals.append(("EXTREMELY HIGH RISK", -2))
        elif volatility > 25:
            signals.append(("HIGH RISK", -1))
        
        # Calculate total score
        total_score = sum(score for _, score in signals)
        
        # Generate screaming recommendation
        print("🔍 Signal Analysis:")
        for signal, score in signals:
            color = Fore.GREEN if score > 0 else Fore.RED if score < 0 else Fore.YELLOW
            print(f"   {color}{signal}: {score:+d}{Style.RESET_ALL}")
        
        print(f"\n🎯 Total Score: {total_score}")
        
        # Final decision with screaming
        if total_score >= 4:
            self.scream_alert("🚨 URGENT: STRONG BUY SIGNAL! 🚨", "BUY")
            print(f"💡 Why: Strong bullish signals aligned")
            print(f"🎯 Target: ${current_price * 1.05:.2f} (+5%)")
            print(f"🛡️ Stop Loss: ${current_price * 0.95:.2f} (-5%)")
            action = "BUY"
            
        elif total_score >= 2:
            self.scream_alert("📈 BUY SIGNAL DETECTED!", "BUY")
            print(f"💡 Why: Bullish bias with good momentum")
            print(f"🎯 Target: ${current_price * 1.03:.2f} (+3%)")
            print(f"🛡️ Stop Loss: ${current_price * 0.97:.2f} (-3%)")
            action = "BUY"
            
        elif total_score <= -4:
            self.scream_alert("🚨 URGENT: STRONG SELL SIGNAL! 🚨", "SELL")
            print(f"💡 Why: Multiple bearish indicators")
            print(f"🎯 Target: ${current_price * 0.95:.2f} (-5%)")
            print(f"🛡️ Stop Loss: ${current_price * 1.05:.2f} (+5%)")
            action = "SELL"
            
        elif total_score <= -2:
            self.scream_alert("📉 SELL SIGNAL DETECTED!", "SELL")
            print(f"💡 Why: Bearish momentum building")
            print(f"🎯 Target: ${current_price * 0.97:.2f} (-3%)")
            print(f"🛡️ Stop Loss: ${current_price * 1.03:.2f} (+3%)")
            action = "SELL"
            
        else:
            self.scream_alert("⏸️ HOLD POSITION - MIXED SIGNALS", "HOLD")
            print(f"💡 Why: Conflicting signals, wait for clarity")
            print(f"👀 Watch for: Clear trend direction")
            print(f"🛡️ Stop Loss: ${current_price * 0.98:.2f} (-2%)")
            action = "HOLD"
        
        # Risk warning
        if volatility > 30:
            print(f"\n{Back.YELLOW}{Fore.RED}⚠️ HIGH VOLATILITY WARNING: {volatility:.1f}% ⚠️{Style.RESET_ALL}")
            print("💭 Consider smaller position sizes due to high risk")
        
        # Final summary
        print(f"\n{Back.CYAN}{Fore.WHITE}📋 TRADE SUMMARY{Style.RESET_ALL}")
        print(f"Action: {action}")
        print(f"Price: ${current_price:.2f}")
        print(f"Confidence: {'HIGH' if abs(total_score) >= 4 else 'MEDIUM' if abs(total_score) >= 2 else 'LOW'}")
        print(f"Risk Level: {'HIGH' if volatility > 30 else 'MEDIUM' if volatility > 20 else 'LOW'}")
        
        return {
            'action': action,
            'price': current_price,
            'score': total_score,
            'signals': signals,
            'volatility': volatility
        }

def main():
    """Run the trading screamer"""
    screamer = TradingScreener('NVDA')
    result = screamer.analyze_and_scream()
    
    # Ask if user wants continuous monitoring
    print(f"\n{Fore.CYAN}Would you like continuous monitoring? (y/n): {Style.RESET_ALL}", end="")
    
    return screamer, result

if __name__ == "__main__":
    screamer, result = main()
