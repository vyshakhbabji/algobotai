#!/usr/bin/env python3
"""
Time Series Validation - Elite AI v2.0 Backtesting
Train on 1 year of data, test on next 3 months (NO FUTURE DATA!)
This is proper research-grade validation showing real historical performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our Elite AI v2.0
from elite_ai_trader import EliteAITrader

class TimeSeriesValidator:
    def __init__(self, stocks=None):
        if stocks:
            self.stocks = stocks
        else:
            # Default 20-stock universe for comprehensive testing
            self.stocks = [
                # MEGA CAP TECH
                "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA",
                # FINANCE & TRADITIONAL  
                "JPM", "BAC", "WMT", "JNJ", "PG", "KO", "DIS",
                # GROWTH & EMERGING
                "NFLX", "CRM", "UBER", "PLTR", "SNOW", "COIN"
            ]
        
        self.ai_model = EliteAITrader()
        self.validation_results = {}
        
    def run_time_series_validation(self):
        """
        Run proper time series validation:
        - Train on 1 year of data
        - Test predictions on next 3 months
        - NO FUTURE DATA LEAKAGE!
        """
        print("🕒 TIME SERIES VALIDATION - ELITE AI v2.0")
        print("=" * 55)
        print("Training Period: 1 Year Historical Data")
        print("Testing Period: Following 3 Months")
        print("NO FUTURE DATA - Proper Research Validation!")
        print(f"Testing {len(self.stocks)} stocks")
        print("=" * 55)
        
        # Define time periods (using historical dates to ensure we have real outcomes)
        train_end = datetime(2024, 6, 30)  # End of training period
        train_start = train_end - timedelta(days=365)  # 1 year before
        test_end = datetime(2024, 9, 30)   # End of testing period (3 months later)
        
        print(f"📅 Training Period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"📅 Testing Period: {train_end.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        print()
        
        all_results = []
        
        for stock in self.stocks:
            print(f"\n📈 VALIDATING {stock}")
            print("-" * 30)
            result = self.validate_single_stock(stock, train_start, train_end, test_end)
            if result:
                all_results.append(result)
                self.validation_results[stock] = result
        
        self.generate_validation_summary(all_results)
        return self.validation_results
    
    def validate_single_stock(self, symbol, train_start, train_end, test_end):
        """Validate a single stock with time series split"""
        try:
            # Download historical data
            print(f"📊 Downloading data for {symbol}...")
            data = yf.download(symbol, start=train_start - timedelta(days=30), end=test_end + timedelta(days=7))
            
            if len(data) < 100:
                print(f"❌ Insufficient data for {symbol}")
                return None
            
            # Split data properly (NO FUTURE LEAKAGE!)
            train_data = data[data.index <= train_end].copy()
            test_data = data[(data.index > train_end) & (data.index <= test_end)].copy()
            
            print(f"📈 Training samples: {len(train_data)}")
            print(f"📊 Testing samples: {len(test_data)}")
            
            if len(train_data) < 50 or len(test_data) < 10:
                print(f"❌ Insufficient split data for {symbol}")
                return None
            
            # Save training data temporarily for AI model
            train_filename = f"clean_data_{symbol.lower()}.csv"
            train_data.to_csv(train_filename)  # Save with index for proper loading
            
            # Train AI on historical data only
            print(f"🤖 Training Elite AI v2.0 on historical data...")
            training_results = self.ai_model.train_simple_models(symbol)
            
            if not training_results:
                print(f"❌ Training failed for {symbol}")
                return None
            
            # Validate model quality on training data
            quality = self.ai_model.validate_prediction_quality(symbol)
            print(f"📊 Model Quality: {quality}")
            
            if "POOR" in quality:
                print(f"❌ Model quality too poor for {symbol}")
                return None
            
            # Make prediction at end of training period
            prediction_result = self.ai_model.make_simple_prediction(symbol)
            
            if not prediction_result:
                print(f"❌ Prediction failed for {symbol}")
                return None
            
            # Calculate actual performance over next 3 months
            price_at_prediction = train_data['Close'].iloc[-1]
            price_at_end = test_data['Close'].iloc[-1]
            actual_return = ((price_at_end - price_at_prediction) / price_at_prediction) * 100
            
            # Extract prediction details
            predicted_return = prediction_result['predicted_return']
            signal = prediction_result['signal']
            confidence = prediction_result.get('confidence', 0)
            
            # Calculate prediction accuracy
            prediction_error = abs(predicted_return - actual_return)
            direction_correct = (
                (predicted_return > 0 and actual_return > 0) or
                (predicted_return < 0 and actual_return < 0) or
                (abs(predicted_return) < 2 and abs(actual_return) < 2)  # Both near zero
            )
            
            print(f"🎯 AI Predicted: {predicted_return:+.2f}%")
            print(f"📈 Actual Result: {actual_return:+.2f}%")
            print(f"🎪 Prediction Error: {prediction_error:.2f}%")
            print(f"🎯 Direction Correct: {'✅' if direction_correct else '❌'}")
            print(f"📊 Signal: {signal}")
            print(f"🔥 Confidence: {confidence:.1%}")
            
            # Clean up temporary file
            import os
            if os.path.exists(train_filename):
                os.remove(train_filename)
            
            return {
                'symbol': symbol,
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'prediction_error': prediction_error,
                'direction_correct': direction_correct,
                'signal': signal,
                'confidence': confidence,
                'quality': quality,
                'price_start': price_at_prediction,
                'price_end': price_at_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data)
            }
            
        except Exception as e:
            print(f"❌ Error validating {symbol}: {str(e)}")
            return None
    
    def generate_validation_summary(self, results):
        """Generate comprehensive validation summary"""
        if not results:
            print("❌ No validation results to summarize")
            return
        
        print(f"\n🎯 TIME SERIES VALIDATION SUMMARY")
        print("=" * 45)
        
        # Overall statistics
        total_predictions = len(results)
        correct_directions = sum(1 for r in results if r['direction_correct'])
        direction_accuracy = correct_directions / total_predictions if total_predictions > 0 else 0
        
        avg_error = np.mean([r['prediction_error'] for r in results])
        median_error = np.median([r['prediction_error'] for r in results])
        
        print(f"📊 Total Predictions: {total_predictions}")
        print(f"🎯 Direction Accuracy: {direction_accuracy:.1%} ({correct_directions}/{total_predictions})")
        print(f"📈 Average Error: {avg_error:.2f}%")
        print(f"📊 Median Error: {median_error:.2f}%")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 80)
        print(f"{'Stock':<6} {'Predicted':<10} {'Actual':<10} {'Error':<8} {'Direction':<9} {'Signal':<6} {'Quality':<8}")
        print("-" * 80)
        
        for r in results:
            direction_symbol = "✅" if r['direction_correct'] else "❌"
            quality_short = r['quality'].split()[-1] if r['quality'] else "N/A"
            
            print(f"{r['symbol']:<6} {r['predicted_return']:>+7.2f}% {r['actual_return']:>+7.2f}% "
                  f"{r['prediction_error']:>5.2f}% {direction_symbol:<9} {r['signal']:<6} {quality_short:<8}")
        
        print("-" * 80)
        
        # Performance by signal type
        buy_signals = [r for r in results if r['signal'] == 'BUY']
        sell_signals = [r for r in results if r['signal'] == 'SELL']
        hold_signals = [r for r in results if r['signal'] == 'HOLD']
        
        print(f"\n📊 PERFORMANCE BY SIGNAL:")
        print("-" * 30)
        
        if buy_signals:
            buy_accuracy = sum(1 for r in buy_signals if r['direction_correct']) / len(buy_signals)
            avg_buy_actual = np.mean([r['actual_return'] for r in buy_signals])
            print(f"🟢 BUY Signals: {len(buy_signals)} (Accuracy: {buy_accuracy:.1%}, Avg Return: {avg_buy_actual:+.2f}%)")
        
        if sell_signals:
            sell_accuracy = sum(1 for r in sell_signals if r['direction_correct']) / len(sell_signals)
            avg_sell_actual = np.mean([r['actual_return'] for r in sell_signals])
            print(f"🔴 SELL Signals: {len(sell_signals)} (Accuracy: {sell_accuracy:.1%}, Avg Return: {avg_sell_actual:+.2f}%)")
        
        if hold_signals:
            hold_accuracy = sum(1 for r in hold_signals if r['direction_correct']) / len(hold_signals)
            avg_hold_actual = np.mean([r['actual_return'] for r in hold_signals])
            print(f"🟡 HOLD Signals: {len(hold_signals)} (Accuracy: {hold_accuracy:.1%}, Avg Return: {avg_hold_actual:+.2f}%)")
        
        # Model quality analysis
        print(f"\n🔍 MODEL QUALITY ANALYSIS:")
        print("-" * 35)
        
        good_models = [r for r in results if 'GOOD' in r['quality']]
        fair_models = [r for r in results if 'FAIR' in r['quality']]
        
        if good_models:
            good_accuracy = sum(1 for r in good_models if r['direction_correct']) / len(good_models)
            print(f"🟢 GOOD Quality Models: {len(good_models)} (Accuracy: {good_accuracy:.1%})")
        
        if fair_models:
            fair_accuracy = sum(1 for r in fair_models if r['direction_correct']) / len(fair_models)
            print(f"🟡 FAIR Quality Models: {len(fair_models)} (Accuracy: {fair_accuracy:.1%})")
        
        # Statistical significance
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print("-" * 30)
        
        if total_predictions >= 10:
            # Simple binomial test approximation
            expected_random = 0.5  # Random chance
            z_score = (direction_accuracy - expected_random) / np.sqrt(expected_random * (1 - expected_random) / total_predictions)
            
            print(f"🎲 Random Chance: 50%")
            print(f"🎯 Our Accuracy: {direction_accuracy:.1%}")
            print(f"📊 Z-Score: {z_score:.2f}")
            
            if z_score > 1.96:
                print(f"✅ Statistically significant improvement over random!")
            elif z_score > 1.64:
                print(f"🟡 Marginally significant improvement")
            else:
                print(f"❌ Not significantly better than random")
        
        # Final assessment
        print(f"\n🏆 ELITE AI v2.0 VALIDATION VERDICT:")
        print("-" * 40)
        
        if direction_accuracy >= 0.65:
            verdict = "🌟 EXCELLENT"
        elif direction_accuracy >= 0.55:
            verdict = "✅ GOOD"
        elif direction_accuracy >= 0.45:
            verdict = "🟡 FAIR"
        else:
            verdict = "❌ POOR"
        
        print(f"📊 Overall Performance: {verdict}")
        print(f"💡 Direction Accuracy: {direction_accuracy:.1%}")
        print(f"🎯 Average Error: {avg_error:.2f}%")
        
        print(f"\n🔬 RESEARCH CONCLUSION:")
        print(f"   • Elite AI v2.0 tested on historical data (NO FUTURE LEAKAGE)")
        print(f"   • Trained on 1 year, validated on next 3 months")
        print(f"   • {verdict.split()[1]} performance vs random chance")
        print(f"   • Model quality gates protect against poor predictions")
        
        return {
            'total_predictions': total_predictions,
            'direction_accuracy': direction_accuracy,
            'average_error': avg_error,
            'verdict': verdict
        }

def main():
    """Run time series validation on multiple stock universes"""
    
    # Full 20-stock comprehensive validation
    print("🌟 COMPREHENSIVE 20-STOCK TIME SERIES VALIDATION")
    validator = TimeSeriesValidator()
    results = validator.run_time_series_validation()
    
    # Quick tech-focused validation
    print(f"\n\n🎯 TECH-FOCUSED VALIDATION")
    print("=" * 40)
    tech_stocks = ["NVDA", "AAPL", "GOOGL", "MSFT", "TSLA"]
    tech_validator = TimeSeriesValidator(stocks=tech_stocks)
    tech_results = tech_validator.run_time_series_validation()
    
    print(f"\n" + "="*60)
    print(f"🎯 TIME SERIES VALIDATION COMPLETE!")
    print(f"   • NO FUTURE DATA used in training")
    print(f"   • Proper 1-year train / 3-month test split")
    print(f"   • Research-grade historical validation")
    print(f"   • Elite AI v2.0 tested under real conditions")
    print(f"="*60)
    
    return {
        'comprehensive': results,
        'tech_focused': tech_results
    }

if __name__ == "__main__":
    main()
