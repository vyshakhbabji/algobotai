#!/usr/bin/env python3
"""
NVDA TRADING SIGNALS VISUALIZER
Shows buy/sell/hold signals on NVDA chart using proven technical configuration
Displays price action with signal overlays and technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NVDASignalVisualizer:
    def __init__(self):
        # YOUR PROVEN TECHNICAL CONFIG
        self.config = {
            'trend_5d_buy_threshold': 0.025,
            'trend_5d_sell_threshold': -0.02,
            'trend_10d_buy_threshold': 0.025,
            'trend_10d_sell_threshold': -0.045,
            'rsi_overbought': 65,
            'rsi_oversold': 20,
            'volatility_threshold': 0.07,
            'volume_ratio_threshold': 1.6
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
        
        return data
    
    def generate_signals(self, data):
        """Generate trading signals using proven config"""
        signals = []
        signal_points = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        for i in range(30, len(data)):
            price = float(data['Close'].iloc[i])
            date = data.index[i]
            
            # Multi-timeframe analysis
            recent_5d = data['Close'].iloc[i-5:i]
            recent_10d = data['Close'].iloc[i-10:i]
            
            # Trend analysis
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Moving averages
            ma5 = float(data['MA5'].iloc[i]) if not pd.isna(data['MA5'].iloc[i]) else price
            ma10 = float(data['MA10'].iloc[i]) if not pd.isna(data['MA10'].iloc[i]) else price
            
            # RSI
            rsi = float(data['RSI'].iloc[i]) if not pd.isna(data['RSI'].iloc[i]) else 50
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # Volume ratio
            try:
                recent_volume = float(data['Volume'].iloc[i-10:i].mean())
                current_volume = float(data['Volume'].iloc[i])
                volume_ratio = current_volume / recent_volume if recent_volume > 0 else 1
            except:
                volume_ratio = 1
            
            # SIGNAL LOGIC
            signal = 'HOLD'
            
            # SELL CONDITIONS
            if (trend_5d < self.config['trend_5d_sell_threshold'] and 
                trend_10d < self.config['trend_10d_sell_threshold']) or \
               (price < ma5 < ma10) or \
               (rsi > self.config['rsi_overbought'] and 
                trend_5d < self.config['trend_5d_sell_threshold']/2) or \
               (volatility > self.config['volatility_threshold'] and 
                trend_10d < self.config['trend_10d_sell_threshold']):
                signal = 'SELL'
            
            # BUY CONDITIONS
            elif (trend_5d > self.config['trend_5d_buy_threshold'] and 
                  trend_10d > self.config['trend_10d_buy_threshold'] and 
                  volume_ratio > self.config['volume_ratio_threshold']) or \
                 (price > ma5 > ma10 and trend_5d > self.config['trend_5d_buy_threshold']) or \
                 (rsi < self.config['rsi_oversold'] and 
                  trend_5d > self.config['trend_5d_buy_threshold']/2):
                signal = 'BUY'
            
            signals.append({
                'date': date,
                'price': price,
                'signal': signal,
                'rsi': rsi,
                'trend_5d': trend_5d * 100,
                'trend_10d': trend_10d * 100,
                'volatility': volatility * 100,
                'volume_ratio': volume_ratio
            })
            
            # Store signal points for plotting
            signal_points[signal].append((date, price))
        
        return signals, signal_points
    
    def create_comprehensive_chart(self, period_days=180):
        """Create comprehensive NVDA chart with signals"""
        print("📊 GENERATING NVDA TRADING SIGNALS CHART")
        print("=" * 50)
        
        # Download NVDA data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)
        
        print(f"📈 Downloading NVDA data ({period_days} days)...")
        data = yf.download('NVDA', start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            print("❌ Failed to download NVDA data")
            return None
        
        print(f"✅ Downloaded {len(data)} days of data")
        
        # Calculate indicators
        data = self.calculate_technical_indicators(data)
        
        # Generate signals
        signals, signal_points = self.generate_signals(data)
        
        # Filter data for display period
        display_start = end_date - timedelta(days=period_days)
        display_data = data[data.index >= display_start]
        
        # Create the chart
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), 
                                                gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # Main price chart
        ax1.plot(display_data.index, display_data['Close'], 'white', linewidth=2, label='NVDA Price')
        ax1.plot(display_data.index, display_data['MA5'], 'cyan', linewidth=1, alpha=0.7, label='MA5')
        ax1.plot(display_data.index, display_data['MA10'], 'yellow', linewidth=1, alpha=0.7, label='MA10')
        ax1.plot(display_data.index, display_data['MA20'], 'orange', linewidth=1, alpha=0.7, label='MA20')
        ax1.plot(display_data.index, display_data['MA50'], 'red', linewidth=1, alpha=0.7, label='MA50')
        
        # Bollinger Bands
        ax1.fill_between(display_data.index, display_data['BB_Upper'], display_data['BB_Lower'], 
                        alpha=0.1, color='gray', label='Bollinger Bands')
        ax1.plot(display_data.index, display_data['BB_Upper'], 'gray', linewidth=0.5, alpha=0.5)
        ax1.plot(display_data.index, display_data['BB_Lower'], 'gray', linewidth=0.5, alpha=0.5)
        
        # Plot signals
        for signal_type, points in signal_points.items():
            if not points:
                continue
                
            # Filter points for display period
            filtered_points = [(date, price) for date, price in points if date >= display_start]
            
            if filtered_points:
                dates, prices = zip(*filtered_points)
                
                if signal_type == 'BUY':
                    ax1.scatter(dates, prices, color='lime', marker='^', s=100, 
                              label=f'BUY ({len(filtered_points)})', zorder=5, edgecolors='black')
                elif signal_type == 'SELL':
                    ax1.scatter(dates, prices, color='red', marker='v', s=100, 
                              label=f'SELL ({len(filtered_points)})', zorder=5, edgecolors='black')
        
        ax1.set_title('NVDA - Trading Signals with Technical Analysis', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # RSI
        display_signals = [s for s in signals if s['date'] >= display_start]
        if display_signals:
            rsi_dates = [s['date'] for s in display_signals]
            rsi_values = [s['rsi'] for s in display_signals]
            
            ax2.plot(rsi_dates, rsi_values, 'purple', linewidth=2)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax2.axhline(y=self.config['rsi_overbought'], color='orange', linestyle=':', alpha=0.7, 
                       label=f'Config OB ({self.config["rsi_overbought"]})')
            ax2.axhline(y=self.config['rsi_oversold'], color='cyan', linestyle=':', alpha=0.7, 
                       label=f'Config OS ({self.config["rsi_oversold"]})')
            ax2.fill_between(rsi_dates, 0, rsi_values, alpha=0.2, color='purple')
        
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Trend Analysis
        if display_signals:
            trend_5d = [s['trend_5d'] for s in display_signals]
            trend_10d = [s['trend_10d'] for s in display_signals]
            
            ax3.plot(rsi_dates, trend_5d, 'cyan', linewidth=2, label='5D Trend %')
            ax3.plot(rsi_dates, trend_10d, 'yellow', linewidth=2, label='10D Trend %')
            ax3.axhline(y=self.config['trend_5d_buy_threshold']*100, color='green', 
                       linestyle='--', alpha=0.7, label='Buy Threshold')
            ax3.axhline(y=self.config['trend_5d_sell_threshold']*100, color='red', 
                       linestyle='--', alpha=0.7, label='Sell Threshold')
            ax3.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        
        ax3.set_ylabel('Trend %', fontsize=12)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Volume and Volatility
        if display_signals:
            volatility = [s['volatility'] for s in display_signals]
            volume_ratio = [s['volume_ratio'] for s in display_signals]
            
            ax4_twin = ax4.twinx()
            
            ax4.plot(rsi_dates, volatility, 'orange', linewidth=2, label='Volatility %')
            ax4.axhline(y=self.config['volatility_threshold']*100, color='red', 
                       linestyle='--', alpha=0.7, label='Vol Threshold')
            
            ax4_twin.plot(rsi_dates, volume_ratio, 'lime', linewidth=2, label='Volume Ratio')
            ax4_twin.axhline(y=self.config['volume_ratio_threshold'], color='green', 
                           linestyle='--', alpha=0.7, label='Vol Ratio Threshold')
        
        ax4.set_ylabel('Volatility %', fontsize=12, color='orange')
        ax4_twin.set_ylabel('Volume Ratio', fontsize=12, color='lime')
        ax4.set_xlabel('Date', fontsize=12)
        ax4.legend(loc='upper left', fontsize=8)
        ax4_twin.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save the chart
        filename = f"nvda_signals_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"💾 Chart saved: {filename}")
        
        # Show statistics
        self.show_signal_statistics(signals, display_start)
        
        plt.show()
        
        return signals, signal_points
    
    def show_signal_statistics(self, signals, start_date):
        """Show signal statistics"""
        display_signals = [s for s in signals if s['date'] >= start_date]
        
        if not display_signals:
            return
        
        buy_signals = [s for s in display_signals if s['signal'] == 'BUY']
        sell_signals = [s for s in display_signals if s['signal'] == 'SELL']
        hold_signals = [s for s in display_signals if s['signal'] == 'HOLD']
        
        print(f"\n📊 NVDA SIGNAL STATISTICS ({len(display_signals)} days)")
        print("=" * 50)
        print(f"🟢 BUY Signals:    {len(buy_signals):3d} ({len(buy_signals)/len(display_signals)*100:5.1f}%)")
        print(f"🔴 SELL Signals:   {len(sell_signals):3d} ({len(sell_signals)/len(display_signals)*100:5.1f}%)")
        print(f"🟡 HOLD Signals:   {len(hold_signals):3d} ({len(hold_signals)/len(display_signals)*100:5.1f}%)")
        
        # Current signal
        latest_signal = display_signals[-1]
        print(f"\n🎯 CURRENT SIGNAL: {latest_signal['signal']}")
        print(f"   Price:      ${latest_signal['price']:.2f}")
        print(f"   RSI:        {latest_signal['rsi']:.1f}")
        print(f"   5D Trend:   {latest_signal['trend_5d']:+.1f}%")
        print(f"   10D Trend:  {latest_signal['trend_10d']:+.1f}%")
        print(f"   Volatility: {latest_signal['volatility']:.1f}%")
        print(f"   Vol Ratio:  {latest_signal['volume_ratio']:.1f}x")
        
        # Recent signals
        print(f"\n📅 RECENT SIGNALS (Last 10):")
        print("-" * 40)
        recent_signals = display_signals[-10:]
        for signal in recent_signals:
            date_str = signal['date'].strftime('%m/%d')
            signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "🟡"
            print(f"   {date_str} {signal_emoji} {signal['signal']:<4} ${signal['price']:6.2f} RSI:{signal['rsi']:4.1f}")

def main():
    """Run NVDA signal visualization"""
    print("🚀 NVDA TRADING SIGNALS VISUALIZER")
    print("=" * 40)
    print("📈 Using your proven technical configuration")
    print("🎯 Generating comprehensive signal chart")
    print("=" * 40)
    
    visualizer = NVDASignalVisualizer()
    
    # Create 6-month chart (you can adjust the period)
    signals, signal_points = visualizer.create_comprehensive_chart(period_days=180)
    
    return visualizer, signals, signal_points

if __name__ == "__main__":
    visualizer, signals, signal_points = main()
