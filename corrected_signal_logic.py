#!/usr/bin/env python3
"""
CORRECTED TRADING SIGNAL LOGIC
Fixes the backwards buy-high/sell-low problem in the original optimizer
Implements proper buy-low/sell-high logic with trend reversal detection
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CorrectedSignalGenerator:
    def __init__(self):
        # CORRECTED CONFIGURATION - Buy low, sell high
        self.config = {
            # Trend reversal thresholds (smaller values for early detection)
            'trend_5d_buy_threshold': 0.01,    # Buy when 5d trend improves from negative
            'trend_5d_sell_threshold': -0.01,  # Sell when 5d trend deteriorates from positive
            'trend_10d_buy_threshold': 0.015,  # Confirm with 10d trend improvement
            'trend_10d_sell_threshold': -0.015, # Confirm with 10d trend deterioration
            
            # RSI - Buy oversold, sell overbought
            'rsi_overbought': 70,  # Standard overbought level
            'rsi_oversold': 30,    # Standard oversold level
            'rsi_extreme_oversold': 25,  # Extreme oversold for strong buys
            'rsi_extreme_overbought': 75, # Extreme overbought for strong sells
            
            # Volatility and volume
            'volatility_threshold': 0.08,  # Higher volatility = more opportunity
            'volume_ratio_threshold': 1.3,  # Volume confirmation
            
            # Mean reversion parameters
            'price_deviation_buy': -0.05,   # Buy when 5% below MA20
            'price_deviation_sell': 0.05,   # Sell when 5% above MA20
        }
        
    def calculate_technical_indicators(self, data):
        """Calculate all technical indicators"""
        data = data.copy()
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Middle'] = bb_middle
        data['BB_Upper'] = bb_middle + (bb_std * 2)
        data['BB_Lower'] = bb_middle - (bb_std * 2)
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        return data
    
    def generate_corrected_signals(self, data):
        """Generate CORRECTED trading signals - Buy low, sell high"""
        signals = []
        signal_points = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        for i in range(30, len(data)):
            price = float(data['Close'].iloc[i])
            date = data.index[i]
            
            # Multi-timeframe analysis
            recent_5d = data['Close'].iloc[i-5:i]
            recent_10d = data['Close'].iloc[i-10:i]
            recent_20d = data['Close'].iloc[i-20:i]
            
            # Trend analysis
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Moving averages
            ma5 = float(data['MA5'].iloc[i]) if not pd.isna(data['MA5'].iloc[i]) else price
            ma10 = float(data['MA10'].iloc[i]) if not pd.isna(data['MA10'].iloc[i]) else price
            ma20 = float(data['MA20'].iloc[i]) if not pd.isna(data['MA20'].iloc[i]) else price
            
            # RSI
            rsi = float(data['RSI'].iloc[i]) if not pd.isna(data['RSI'].iloc[i]) else 50
            
            # Price deviation from MA20
            price_deviation = (price - ma20) / ma20 if ma20 > 0 else 0
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # Volume ratio
            try:
                recent_volume = float(data['Volume'].iloc[i-10:i].mean())
                current_volume = float(data['Volume'].iloc[i])
                volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
            except:
                volume_ratio = 1
            
            # MACD signals
            macd = float(data['MACD'].iloc[i]) if not pd.isna(data['MACD'].iloc[i]) else 0
            macd_signal = float(data['MACD_Signal'].iloc[i]) if not pd.isna(data['MACD_Signal'].iloc[i]) else 0
            macd_histogram = macd - macd_signal
            
            # Previous MACD for crossover detection
            prev_macd = float(data['MACD'].iloc[i-1]) if i > 0 and not pd.isna(data['MACD'].iloc[i-1]) else macd
            prev_macd_signal = float(data['MACD_Signal'].iloc[i-1]) if i > 0 and not pd.isna(data['MACD_Signal'].iloc[i-1]) else macd_signal
            
            # Trend direction changes
            trend_5d_improving = trend_5d > self.config['trend_5d_buy_threshold']
            trend_5d_deteriorating = trend_5d < self.config['trend_5d_sell_threshold']
            trend_10d_improving = trend_10d > self.config['trend_10d_buy_threshold']
            trend_10d_deteriorating = trend_10d < self.config['trend_10d_sell_threshold']
            
            # CORRECTED SIGNAL LOGIC - BUY LOW, SELL HIGH
            signal = 'HOLD'  # Default
            
            # ========================
            # STRONG BUY CONDITIONS (Buy at bottoms/dips)
            # ========================
            strong_buy_conditions = [
                # Extreme oversold with trend improvement
                (rsi <= self.config['rsi_extreme_oversold'] and trend_5d_improving),
                
                # Price significantly below MA20 with RSI oversold and volume
                (price_deviation <= self.config['price_deviation_buy'] and 
                 rsi <= self.config['rsi_oversold'] and 
                 volume_ratio > self.config['volume_ratio_threshold']),
                
                # MACD bullish crossover while oversold
                (macd > macd_signal and prev_macd <= prev_macd_signal and 
                 rsi <= self.config['rsi_oversold']),
                
                # Price below BB lower band with RSI oversold
                (price < float(data['BB_Lower'].iloc[i]) if not pd.isna(data['BB_Lower'].iloc[i]) else price and
                 rsi <= self.config['rsi_oversold']),
            ]
            
            # Regular BUY conditions (Buy on dips)
            buy_conditions = [
                # RSI oversold with positive volume
                (rsi <= self.config['rsi_oversold'] and volume_ratio > 1.1),
                
                # Price below MA5 and MA10 but showing signs of reversal
                (price < ma5 < ma10 and trend_5d_improving and rsi < 50),
                
                # MACD histogram turning positive while price is below MA20
                (macd_histogram > 0 and price < ma20 and rsi < 60),
            ]
            
            # ========================
            # STRONG SELL CONDITIONS (Sell at tops)
            # ========================
            strong_sell_conditions = [
                # Extreme overbought with trend deteriorating
                (rsi >= self.config['rsi_extreme_overbought'] and trend_5d_deteriorating),
                
                # Price significantly above MA20 with RSI overbought
                (price_deviation >= self.config['price_deviation_sell'] and 
                 rsi >= self.config['rsi_overbought']),
                
                # MACD bearish crossover while overbought
                (macd < macd_signal and prev_macd >= prev_macd_signal and 
                 rsi >= self.config['rsi_overbought']),
                
                # Price above BB upper band with RSI overbought
                (price > float(data['BB_Upper'].iloc[i]) if not pd.isna(data['BB_Upper'].iloc[i]) else price and
                 rsi >= self.config['rsi_overbought']),
            ]
            
            # Regular SELL conditions
            sell_conditions = [
                # RSI overbought with declining trend
                (rsi >= self.config['rsi_overbought'] and trend_5d_deteriorating),
                
                # Price above MA5 > MA10 but losing momentum
                (price > ma5 > ma10 and trend_5d_deteriorating and rsi > 50),
                
                # High volatility with negative trend
                (volatility > self.config['volatility_threshold'] and 
                 trend_10d_deteriorating),
            ]
            
            # Determine signal
            if any(strong_buy_conditions):
                signal = 'STRONG_BUY'
            elif any(buy_conditions):
                signal = 'BUY'
            elif any(strong_sell_conditions):
                signal = 'STRONG_SELL'
            elif any(sell_conditions):
                signal = 'SELL'
            
            signals.append({
                'date': date,
                'price': price,
                'signal': signal,
                'rsi': rsi,
                'trend_5d': trend_5d * 100,
                'trend_10d': trend_10d * 100,
                'price_deviation': price_deviation * 100,
                'volatility': volatility * 100,
                'volume_ratio': volume_ratio,
                'macd_histogram': macd_histogram
            })
            
            # Store signal points for plotting
            if signal in ['BUY', 'STRONG_BUY']:
                signal_points['BUY'].append((date, price))
            elif signal in ['SELL', 'STRONG_SELL']:
                signal_points['SELL'].append((date, price))
            else:
                signal_points['HOLD'].append((date, price))
        
        return signals, signal_points
    
    def create_corrected_chart(self, symbol='NVDA', period_days=180):
        """Create chart with corrected buy-low/sell-high signals"""
        print(f"📊 GENERATING CORRECTED {symbol} SIGNALS CHART")
        print("=" * 50)
        print("🎯 CORRECTED LOGIC: Buy low, sell high")
        print("✅ Buy on dips, oversold conditions, support levels")
        print("✅ Sell on peaks, overbought conditions, resistance levels")
        print("=" * 50)
        
        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)
        
        print(f"📈 Downloading {symbol} data ({period_days} days)...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            print(f"❌ Failed to download {symbol} data")
            return None
        
        print(f"✅ Downloaded {len(data)} days of data")
        
        # Calculate indicators
        data = self.calculate_technical_indicators(data)
        
        # Generate corrected signals
        signals, signal_points = self.generate_corrected_signals(data)
        
        # Filter data for display period
        display_start = end_date - timedelta(days=period_days)
        display_data = data[data.index >= display_start]
        
        # Create the chart
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), 
                                                gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Main price chart
        ax1.plot(display_data.index, display_data['Close'], 'white', linewidth=2, label=f'{symbol} Price')
        ax1.plot(display_data.index, display_data['MA5'], 'cyan', linewidth=1, alpha=0.7, label='MA5')
        ax1.plot(display_data.index, display_data['MA10'], 'yellow', linewidth=1, alpha=0.7, label='MA10')
        ax1.plot(display_data.index, display_data['MA20'], 'orange', linewidth=1, alpha=0.7, label='MA20')
        ax1.plot(display_data.index, display_data['MA50'], 'red', linewidth=1, alpha=0.7, label='MA50')
        
        # Bollinger Bands
        ax1.fill_between(display_data.index, display_data['BB_Upper'], display_data['BB_Lower'], 
                        alpha=0.1, color='gray', label='Bollinger Bands')
        ax1.plot(display_data.index, display_data['BB_Upper'], 'gray', linewidth=0.5, alpha=0.5)
        ax1.plot(display_data.index, display_data['BB_Lower'], 'gray', linewidth=0.5, alpha=0.5)
        
        # Plot corrected signals
        for signal_type, points in signal_points.items():
            if not points:
                continue
                
            # Filter points for display period
            filtered_points = [(date, price) for date, price in points if date >= display_start]
            
            if filtered_points:
                dates, prices = zip(*filtered_points)
                
                if signal_type == 'BUY':
                    ax1.scatter(dates, prices, color='lime', marker='^', s=120, 
                              label=f'BUY ({len(filtered_points)})', zorder=5, edgecolors='darkgreen', linewidth=2)
                elif signal_type == 'SELL':
                    ax1.scatter(dates, prices, color='red', marker='v', s=120, 
                              label=f'SELL ({len(filtered_points)})', zorder=5, edgecolors='darkred', linewidth=2)
        
        ax1.set_title(f'{symbol} - CORRECTED Trading Signals (Buy Low, Sell High)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # RSI with better levels
        display_signals = [s for s in signals if s['date'] >= display_start]
        if display_signals:
            rsi_dates = [s['date'] for s in display_signals]
            rsi_values = [s['rsi'] for s in display_signals]
            
            ax2.plot(rsi_dates, rsi_values, 'purple', linewidth=2)
            ax2.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='Extreme OB (75)')
            ax2.axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color='lime', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.axhline(y=25, color='green', linestyle='--', alpha=0.7, label='Extreme OS (25)')
            ax2.fill_between(rsi_dates, 0, rsi_values, alpha=0.2, color='purple')
        
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Price deviation from MA20
        if display_signals:
            price_dev = [s['price_deviation'] for s in display_signals]
            
            ax3.plot(rsi_dates, price_dev, 'gold', linewidth=2, label='Price vs MA20 %')
            ax3.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Sell Level (+5%)')
            ax3.axhline(y=-5, color='green', linestyle='--', alpha=0.7, label='Buy Level (-5%)')
            ax3.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            ax3.fill_between(rsi_dates, 0, price_dev, alpha=0.3, 
                           color=['red' if x > 0 else 'green' for x in price_dev])
        
        ax3.set_ylabel('Price Dev %', fontsize=12)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # MACD
        if display_signals:
            macd_histogram = [s['macd_histogram'] for s in display_signals]
            
            ax4.bar(rsi_dates, macd_histogram, color=['green' if x > 0 else 'red' for x in macd_histogram], 
                   alpha=0.7, label='MACD Histogram')
            ax4.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        
        ax4.set_ylabel('MACD Hist', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save the chart
        filename = f"{symbol.lower()}_corrected_signals_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"💾 Chart saved: {filename}")
        
        # Show statistics
        self.show_corrected_statistics(signals, display_start, symbol)
        
        plt.show()
        
        return signals, signal_points
    
    def show_corrected_statistics(self, signals, start_date, symbol):
        """Show corrected signal statistics"""
        display_signals = [s for s in signals if s['date'] >= start_date]
        
        if not display_signals:
            return
        
        buy_signals = [s for s in display_signals if s['signal'] in ['BUY', 'STRONG_BUY']]
        sell_signals = [s for s in display_signals if s['signal'] in ['SELL', 'STRONG_SELL']]
        hold_signals = [s for s in display_signals if s['signal'] == 'HOLD']
        
        print(f"\n📊 {symbol} CORRECTED SIGNAL STATISTICS ({len(display_signals)} days)")
        print("=" * 50)
        print(f"🟢 BUY Signals:    {len(buy_signals):3d} ({len(buy_signals)/len(display_signals)*100:5.1f}%)")
        print(f"🔴 SELL Signals:   {len(sell_signals):3d} ({len(sell_signals)/len(display_signals)*100:5.1f}%)")
        print(f"🟡 HOLD Signals:   {len(hold_signals):3d} ({len(hold_signals)/len(display_signals)*100:5.1f}%)")
        
        # Analyze signal quality
        if buy_signals and sell_signals:
            avg_buy_rsi = sum(s['rsi'] for s in buy_signals) / len(buy_signals)
            avg_sell_rsi = sum(s['rsi'] for s in sell_signals) / len(sell_signals)
            
            avg_buy_deviation = sum(s['price_deviation'] for s in buy_signals) / len(buy_signals)
            avg_sell_deviation = sum(s['price_deviation'] for s in sell_signals) / len(sell_signals)
            
            print(f"\n📈 SIGNAL QUALITY ANALYSIS:")
            print(f"   BUY signals - Avg RSI: {avg_buy_rsi:.1f}, Avg Price Dev: {avg_buy_deviation:+.1f}%")
            print(f"   SELL signals - Avg RSI: {avg_sell_rsi:.1f}, Avg Price Dev: {avg_sell_deviation:+.1f}%")
            
            if avg_buy_rsi < avg_sell_rsi:
                print(f"   ✅ GOOD: Buying when RSI lower than selling ({avg_buy_rsi:.1f} vs {avg_sell_rsi:.1f})")
            else:
                print(f"   ❌ BAD: Buying when RSI higher than selling ({avg_buy_rsi:.1f} vs {avg_sell_rsi:.1f})")
        
        # Current signal
        latest_signal = display_signals[-1]
        print(f"\n🎯 CURRENT SIGNAL: {latest_signal['signal']}")
        print(f"   Price:      ${latest_signal['price']:.2f}")
        print(f"   RSI:        {latest_signal['rsi']:.1f}")
        print(f"   5D Trend:   {latest_signal['trend_5d']:+.1f}%")
        print(f"   Price Dev:  {latest_signal['price_deviation']:+.1f}%")
        print(f"   MACD Hist:  {latest_signal['macd_histogram']:+.3f}")

def main():
    """Run corrected signal visualization"""
    print("🚀 CORRECTED TRADING SIGNALS GENERATOR")
    print("=" * 40)
    print("✅ FIXED: Buy-low/sell-high logic")
    print("✅ Buy on: Oversold, dips, support breaks")
    print("✅ Sell on: Overbought, peaks, resistance breaks")
    print("=" * 40)
    
    generator = CorrectedSignalGenerator()
    
    # Create corrected chart for NVDA
    signals, signal_points = generator.create_corrected_chart('NVDA', period_days=180)
    
    return generator, signals, signal_points

if __name__ == "__main__":
    generator, signals, signal_points = main()
