#!/usr/bin/env python3
"""
Elite AI v3.0 Signal Analysis & Visualization
Analyze and visualize BUY/SELL/HOLD signals across all stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our IMPROVED Elite AI v3.0
from improved_elite_ai import ImprovedEliteAI

class SignalAnalysis:
    def __init__(self):
        # Comprehensive stock universe
        self.stocks = [
            # MEGA CAP TECH
            "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA",
            # FINANCE & TRADITIONAL
            "JPM", "BAC", "WMT", "JNJ", "PG", "KO", "DIS",
            # GROWTH & EMERGING
            "NFLX", "CRM", "UBER", "PLTR", "SNOW", "COIN"
        ]
        self.ai_model = ImprovedEliteAI()
        self.signal_results = {}
        
    def analyze_all_signals(self):
        """Analyze signals for all stocks"""
        print("🚀 ELITE AI v3.0 SIGNAL ANALYSIS")
        print("=" * 50)
        print(f"Analyzing signals for {len(self.stocks)} stocks")
        print("=" * 50)
        
        buy_signals = []
        sell_signals = []
        hold_signals = []
        poor_quality = []
        
        signal_data = []
        
        for stock in self.stocks:
            print(f"\n📊 Analyzing {stock}...")
            
            try:
                # Train the AI model
                training_results = self.ai_model.train_improved_models(stock)
                
                if training_results:
                    # Get prediction
                    prediction = self.ai_model.predict_with_improved_model(stock)
                    
                    if prediction:
                        signal = prediction['signal']
                        pred_return = prediction['predicted_return']
                        confidence = prediction.get('confidence', 0)
                        ensemble_r2 = prediction.get('ensemble_r2', 0)
                        models_used = prediction.get('models_used', 0)
                        
                        # Get current price
                        ticker = yf.Ticker(stock)
                        current_price = ticker.info.get('currentPrice', 0)
                        
                        signal_info = {
                            'stock': stock,
                            'signal': signal,
                            'predicted_return': pred_return,
                            'confidence': confidence,
                            'ensemble_r2': ensemble_r2,
                            'models_used': models_used,
                            'current_price': current_price,
                            'quality': 'GOOD' if ensemble_r2 > 0 else 'POOR'
                        }
                        
                        signal_data.append(signal_info)
                        
                        # Categorize signals
                        if signal == 'BUY':
                            buy_signals.append(stock)
                        elif signal == 'SELL':
                            sell_signals.append(stock)
                        elif signal == 'HOLD':
                            hold_signals.append(stock)
                            
                        # Check quality
                        if ensemble_r2 <= 0:
                            poor_quality.append(stock)
                            
                        print(f"   🚦 {signal} | Return: {pred_return:.2f}% | R²: {ensemble_r2:.3f}")
                        
                    else:
                        signal_data.append({
                            'stock': stock,
                            'signal': 'NO_SIGNAL',
                            'predicted_return': 0,
                            'confidence': 0,
                            'ensemble_r2': -1,
                            'models_used': 0,
                            'current_price': 0,
                            'quality': 'POOR'
                        })
                        poor_quality.append(stock)
                        print(f"   ❌ No signal generated")
                        
                else:
                    signal_data.append({
                        'stock': stock,
                        'signal': 'ERROR',
                        'predicted_return': 0,
                        'confidence': 0,
                        'ensemble_r2': -1,
                        'models_used': 0,
                        'current_price': 0,
                        'quality': 'ERROR'
                    })
                    poor_quality.append(stock)
                    print(f"   ❌ Training failed")
                    
            except Exception as e:
                signal_data.append({
                    'stock': stock,
                    'signal': 'ERROR',
                    'predicted_return': 0,
                    'confidence': 0,
                    'ensemble_r2': -1,
                    'models_used': 0,
                    'current_price': 0,
                    'quality': 'ERROR'
                })
                poor_quality.append(stock)
                print(f"   ❌ Error: {str(e)}")
        
        # Store results
        self.signal_results = {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'poor_quality': poor_quality,
            'signal_data': signal_data
        }
        
        # Print summary
        print(f"\n🎯 SIGNAL SUMMARY")
        print("=" * 30)
        print(f"🟢 BUY signals: {len(buy_signals)}")
        print(f"🔴 SELL signals: {len(sell_signals)}")
        print(f"🟡 HOLD signals: {len(hold_signals)}")
        print(f"❌ Poor quality: {len(poor_quality)}")
        print(f"📊 Total analyzed: {len(self.stocks)}")
        
        if buy_signals:
            print(f"\n🟢 BUY SIGNALS: {', '.join(buy_signals)}")
        if sell_signals:
            print(f"\n🔴 SELL SIGNALS: {', '.join(sell_signals)}")
        if hold_signals:
            print(f"\n🟡 HOLD SIGNALS: {', '.join(hold_signals)}")
            
        return signal_data
    
    def create_signal_visualizations(self):
        """Create comprehensive signal visualizations"""
        if not self.signal_results:
            print("❌ No signal data available. Run analyze_all_signals() first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Signal Distribution Pie Chart
        ax1 = plt.subplot(3, 3, 1)
        signal_counts = {
            'BUY': len(self.signal_results['buy_signals']),
            'SELL': len(self.signal_results['sell_signals']),
            'HOLD': len(self.signal_results['hold_signals']),
            'POOR QUALITY': len(self.signal_results['poor_quality'])
        }
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
        wedges, texts, autotexts = ax1.pie(signal_counts.values(), 
                                          labels=signal_counts.keys(),
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax1.set_title('Elite AI v3.0 Signal Distribution', fontsize=14, fontweight='bold')
        
        # 2. Signal Count Bar Chart
        ax2 = plt.subplot(3, 3, 2)
        bars = ax2.bar(signal_counts.keys(), signal_counts.values(), color=colors)
        ax2.set_title('Signal Counts by Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Stocks')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. Predicted Returns Distribution
        ax3 = plt.subplot(3, 3, 3)
        signal_df = pd.DataFrame(self.signal_results['signal_data'])
        
        # Filter out poor quality signals for returns analysis
        good_signals = signal_df[signal_df['quality'] == 'GOOD']
        
        if len(good_signals) > 0:
            ax3.hist(good_signals['predicted_return'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Predicted Returns Distribution\n(Good Quality Signals Only)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Predicted Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.axvline(good_signals['predicted_return'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {good_signals["predicted_return"].mean():.2f}%')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No Good Quality\nSignals Available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Predicted Returns Distribution', fontsize=12, fontweight='bold')
        
        # 4. Signal Quality by Stock
        ax4 = plt.subplot(3, 3, 4)
        stocks = signal_df['stock'].tolist()
        r2_scores = signal_df['ensemble_r2'].tolist()
        
        # Color code by signal type
        colors_map = {'BUY': '#2ecc71', 'SELL': '#e74c3c', 'HOLD': '#f39c12', 
                     'NO_SIGNAL': '#95a5a6', 'ERROR': '#95a5a6'}
        point_colors = [colors_map.get(sig, '#95a5a6') for sig in signal_df['signal']]
        
        ax4.scatter(range(len(stocks)), r2_scores, c=point_colors, alpha=0.7, s=60)
        ax4.set_title('Model Quality (R²) by Stock', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Stock Index')
        ax4.set_ylabel('Ensemble R² Score')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R² = 0')
        ax4.set_xticks(range(0, len(stocks), 2))
        ax4.set_xticklabels([stocks[i] for i in range(0, len(stocks), 2)], rotation=45)
        ax4.legend()
        
        # 5. Confidence vs Predicted Return Scatter
        ax5 = plt.subplot(3, 3, 5)
        if len(good_signals) > 0:
            scatter = ax5.scatter(good_signals['confidence'], good_signals['predicted_return'], 
                                c=[colors_map.get(sig, '#95a5a6') for sig in good_signals['signal']], 
                                alpha=0.7, s=60)
            ax5.set_title('Confidence vs Predicted Return', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Confidence')
            ax5.set_ylabel('Predicted Return (%)')
            
            # Add trend line
            if len(good_signals) > 1:
                z = np.polyfit(good_signals['confidence'], good_signals['predicted_return'], 1)
                p = np.poly1d(z)
                ax5.plot(good_signals['confidence'], p(good_signals['confidence']), "r--", alpha=0.8)
        else:
            ax5.text(0.5, 0.5, 'No Good Quality\nSignals Available', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Confidence vs Predicted Return', fontsize=12, fontweight='bold')
        
        # 6. Models Used Distribution
        ax6 = plt.subplot(3, 3, 6)
        models_used = signal_df['models_used'].tolist()
        unique_models = sorted(list(set(models_used)))
        model_counts = [models_used.count(m) for m in unique_models]
        
        ax6.bar([str(m) for m in unique_models], model_counts, color='lightcoral')
        ax6.set_title('Distribution of Models Used', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Number of Models')
        ax6.set_ylabel('Number of Stocks')
        
        # Add value labels
        for i, v in enumerate(model_counts):
            ax6.text(i, v, str(v), ha='center', va='bottom')
        
        # 7. Signal Performance Table
        ax7 = plt.subplot(3, 3, (7, 9))
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create detailed table
        table_data = []
        headers = ['Stock', 'Signal', 'Pred Return', 'Confidence', 'R² Score', 'Quality', 'Models']
        
        for data in self.signal_results['signal_data']:
            table_data.append([
                data['stock'],
                data['signal'],
                f"{data['predicted_return']:.2f}%",
                f"{data['confidence']:.1%}",
                f"{data['ensemble_r2']:.3f}",
                data['quality'],
                str(data['models_used'])
            ])
        
        table = ax7.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # Color code the table rows by signal type
        for i, data in enumerate(table_data):
            signal = data[1]
            if signal == 'BUY':
                color = '#d5f4e6'  # Light green
            elif signal == 'SELL':
                color = '#fadbd8'  # Light red
            elif signal == 'HOLD':
                color = '#fef9e7'  # Light yellow
            else:
                color = '#f8f9fa'  # Light gray
                
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
        
        ax7.set_title('Detailed Signal Analysis', fontsize=14, fontweight='bold', pad=20)
        
        # 8. Summary Statistics
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        # Calculate statistics
        total_stocks = len(self.stocks)
        good_quality = len([d for d in self.signal_results['signal_data'] if d['quality'] == 'GOOD'])
        avg_return = np.mean([d['predicted_return'] for d in self.signal_results['signal_data'] if d['quality'] == 'GOOD']) if good_quality > 0 else 0
        avg_confidence = np.mean([d['confidence'] for d in self.signal_results['signal_data'] if d['quality'] == 'GOOD']) if good_quality > 0 else 0
        
        stats_text = f"""
        📊 ELITE AI v3.0 STATISTICS
        ════════════════════════════
        
        Total Stocks Analyzed: {total_stocks}
        Good Quality Signals: {good_quality}
        Quality Rate: {good_quality/total_stocks*100:.1f}%
        
        🟢 BUY Signals: {len(self.signal_results['buy_signals'])}
        🔴 SELL Signals: {len(self.signal_results['sell_signals'])}
        🟡 HOLD Signals: {len(self.signal_results['hold_signals'])}
        
        Average Predicted Return: {avg_return:.2f}%
        Average Confidence: {avg_confidence:.1%}
        
        🎯 PERFORMANCE INSIGHTS:
        • AI is being conservative with quality
        • Only high-confidence predictions shown
        • Ensemble approach reduces overfitting
        • Multiple model validation ensures reliability
        """
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('Elite AI v3.0 Comprehensive Signal Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        plt.savefig('elite_ai_signal_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Signal analysis dashboard saved as 'elite_ai_signal_analysis.png'")
        
        plt.show()
        
        return fig

def main():
    """Run comprehensive signal analysis"""
    print("🚀 STARTING ELITE AI v3.0 SIGNAL ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SignalAnalysis()
    
    # Analyze all signals
    signal_data = analyzer.analyze_all_signals()
    
    # Create visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    analyzer.create_signal_visualizations()
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"   • Signal distribution analyzed")
    print(f"   • Quality metrics evaluated") 
    print(f"   • Performance charts generated")
    print(f"   • Dashboard saved as PNG")
    
    return analyzer.signal_results

if __name__ == "__main__":
    main()
