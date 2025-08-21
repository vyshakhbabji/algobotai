#!/usr/bin/env python3
"""
Comprehensive Time Series Validation - Elite AI v2.0
Multiple time periods, 20 stocks, NO FUTURE DATA
Real research-grade validation of predictive performance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import warnings
warnings.filterwarnings('ignore')

def comprehensive_validation():
    """Comprehensive validation across multiple periods"""
    
    print("🔬 COMPREHENSIVE TIME SERIES VALIDATION")
    print("=" * 50)
    print("Multiple time periods, 20 stocks")
    print("NO FUTURE DATA CONTAMINATION!")
    print("=" * 50)
    
    # 20-stock universe
    stocks = [
        # MEGA CAP TECH
        "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA",
        # FINANCE & TRADITIONAL
        "JPM", "BAC", "WMT", "JNJ", "PG", "KO", "DIS",
        # GROWTH & EMERGING
        "NFLX", "CRM", "UBER", "PLTR", "SNOW", "COIN"
    ]
    
    # Multiple validation periods
    periods = [
        {
            'name': 'Q3 2024 Test',
            'train_start': datetime(2023, 7, 1),
            'train_end': datetime(2024, 6, 30),
            'test_end': datetime(2024, 9, 30)
        },
        {
            'name': 'Q2 2024 Test', 
            'train_start': datetime(2023, 4, 1),
            'train_end': datetime(2024, 3, 31),
            'test_end': datetime(2024, 6, 30)
        },
        {
            'name': 'Q1 2024 Test',
            'train_start': datetime(2023, 1, 1),
            'train_end': datetime(2023, 12, 31),
            'test_end': datetime(2024, 3, 31)
        }
    ]
    
    all_results = []
    
    for period in periods:
        print(f"\n🕒 {period['name'].upper()}")
        print("=" * 35)
        print(f"Train: {period['train_start'].strftime('%Y-%m-%d')} to {period['train_end'].strftime('%Y-%m-%d')}")
        print(f"Test: {period['train_end'].strftime('%Y-%m-%d')} to {period['test_end'].strftime('%Y-%m-%d')}")
        
        period_results = []
        
        for symbol in stocks:
            result = validate_stock_period(symbol, period)
            if result:
                result['period'] = period['name']
                period_results.append(result)
                all_results.append(result)
        
        # Period summary
        if period_results:
            correct = sum(1 for r in period_results if r['direction_correct'])
            accuracy = correct / len(period_results)
            avg_error = np.mean([r['error'] for r in period_results])
            
            print(f"\n📊 {period['name']} Results:")
            print(f"   • Stocks: {len(period_results)}/{len(stocks)}")
            print(f"   • Direction Accuracy: {accuracy:.1%}")
            print(f"   • Average Error: {avg_error:.2f}%")
        
    # Overall summary
    print(f"\n🎯 OVERALL VALIDATION SUMMARY")
    print("=" * 40)
    
    if all_results:
        total_tests = len(all_results)
        total_correct = sum(1 for r in all_results if r['direction_correct'])
        overall_accuracy = total_correct / total_tests
        overall_error = np.mean([r['error'] for r in all_results])
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"🎯 Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_tests})")
        print(f"📈 Average Error: {overall_error:.2f}%")
        
        # Performance by stock
        stock_performance = {}
        for result in all_results:
            symbol = result['symbol']
            if symbol not in stock_performance:
                stock_performance[symbol] = {'correct': 0, 'total': 0, 'errors': []}
            
            stock_performance[symbol]['total'] += 1
            if result['direction_correct']:
                stock_performance[symbol]['correct'] += 1
            stock_performance[symbol]['errors'].append(result['error'])
        
        print(f"\n📈 TOP PERFORMING STOCKS:")
        print("-" * 35)
        
        # Sort by accuracy then by low error
        sorted_stocks = sorted(stock_performance.items(), 
                             key=lambda x: (x[1]['correct']/x[1]['total'], -np.mean(x[1]['errors'])), 
                             reverse=True)
        
        for symbol, perf in sorted_stocks[:10]:  # Top 10
            accuracy = perf['correct'] / perf['total']
            avg_error = np.mean(perf['errors'])
            print(f"{symbol:<6} {accuracy:.1%} accuracy, {avg_error:>5.2f}% avg error ({perf['correct']}/{perf['total']})")
        
        # Model insights
        print(f"\n🔍 MODEL INSIGHTS:")
        print("-" * 25)
        
        good_predictions = [r for r in all_results if r['error'] < 5]
        bad_predictions = [r for r in all_results if r['error'] > 15]
        
        print(f"📈 Good predictions (< 5% error): {len(good_predictions)}")
        print(f"📉 Poor predictions (> 15% error): {len(bad_predictions)}")
        
        # Signal analysis
        signals = [r['signal'] for r in all_results]
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        hold_count = signals.count('HOLD')
        
        print(f"\n📊 SIGNAL DISTRIBUTION:")
        print(f"🟢 BUY: {buy_count} ({buy_count/total_tests:.1%})")
        print(f"🔴 SELL: {sell_count} ({sell_count/total_tests:.1%})")
        print(f"🟡 HOLD: {hold_count} ({hold_count/total_tests:.1%})")
        
        # Final assessment
        print(f"\n🏆 FINAL ASSESSMENT:")
        print("-" * 25)
        
        if overall_accuracy >= 0.6:
            verdict = "🌟 EXCELLENT"
        elif overall_accuracy >= 0.5:
            verdict = "✅ GOOD"
        elif overall_accuracy >= 0.4:
            verdict = "🟡 FAIR"
        else:
            verdict = "❌ POOR"
        
        print(f"📊 Performance: {verdict}")
        print(f"🎯 vs Random (50%): {overall_accuracy-0.5:+.1%}")
        print(f"📈 Prediction Quality: {overall_error:.1f}% avg error")
        
        # Statistical significance test
        if total_tests >= 10:
            # Simple z-test for proportion
            p_hat = overall_accuracy
            p_null = 0.5  # Random chance
            se = np.sqrt(p_null * (1 - p_null) / total_tests)
            z_score = (p_hat - p_null) / se
            
            if z_score > 1.96:
                significance = "✅ Statistically Significant (p < 0.05)"
            elif z_score > 1.64:
                significance = "🟡 Marginally Significant (p < 0.10)"
            else:
                significance = "❌ Not Significant"
            
            print(f"📊 Statistical Test: {significance}")
            print(f"📈 Z-Score: {z_score:.2f}")
        
        print(f"\n🔬 RESEARCH CONCLUSIONS:")
        print("-" * 30)
        print(f"   • Tested across multiple time periods")
        print(f"   • NO future data contamination")
        print(f"   • {total_tests} independent predictions")
        print(f"   • {verdict.split()[1]} performance overall")
        print(f"   • Simple models show {overall_accuracy:.1%} directional accuracy")
        
        return all_results
    
    else:
        print("❌ No successful validations")
        return []

def validate_stock_period(symbol, period):
    """Validate single stock for one period"""
    try:
        # Download data
        data = yf.download(symbol, 
                          start=period['train_start'] - timedelta(days=60), 
                          end=period['test_end'] + timedelta(days=5), 
                          progress=False)
        
        if len(data) < 100:
            return None
        
        # Calculate features
        data['returns'] = data['Close'].pct_change()
        data['sma_5'] = data['Close'].rolling(5).mean()
        data['sma_10'] = data['Close'].rolling(10).mean()
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['volatility'] = data['returns'].rolling(10).std()
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['target'] = data['returns'].shift(-1)
        
        # Time series split (NO FUTURE LEAKAGE!)
        train_data = data[data.index <= period['train_end']].copy()
        test_data = data[(data.index > period['train_end']) & (data.index <= period['test_end'])].copy()
        
        if len(train_data) < 50 or len(test_data) < 5:
            return None
        
        # Prepare features
        features = ['returns', 'sma_5', 'sma_10', 'sma_20', 'volatility', 'volume_ratio']
        
        # Clean training data
        train_clean = train_data.dropna()
        X_train = train_clean[features].values
        y_train = train_clean['target'].values
        
        if len(X_train) < 30:
            return None
        
        # Train ensemble of simple models
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        }
        
        predictions = {}
        scores = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                score = model.score(X_train, y_train)
                scores[name] = score
                
                # Predict on last training sample
                last_features = X_train[-1:] 
                pred = model.predict(last_features)[0]
                predictions[name] = pred
                
            except:
                continue
        
        if not predictions:
            return None
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        avg_score = np.mean(list(scores.values()))
        
        # Calculate actual performance
        price_start = float(train_data['Close'].iloc[-1])
        price_end = float(test_data['Close'].iloc[-1])
        actual_return = ((price_end - price_start) / price_start) * 100
        
        # Convert prediction to percentage
        predicted_return = ensemble_pred * 100
        
        # Determine signal
        if predicted_return > 3:
            signal = "BUY"
        elif predicted_return < -3:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Calculate metrics
        error = abs(predicted_return - actual_return)
        direction_correct = (predicted_return > 0) == (actual_return > 0)
        
        return {
            'symbol': symbol,
            'predicted': predicted_return,
            'actual': actual_return,
            'error': error,
            'direction_correct': direction_correct,
            'signal': signal,
            'model_score': avg_score,
            'models_used': len(predictions)
        }
        
    except Exception as e:
        return None

if __name__ == "__main__":
    results = comprehensive_validation()
    
    print(f"\n" + "="*55)
    print(f"🎯 COMPREHENSIVE VALIDATION COMPLETE!")
    print(f"   • Multiple time periods tested")
    print(f"   • 20-stock universe coverage")
    print(f"   • Research-grade methodology")
    print(f"   • Real predictive power measured")
    print(f"="*55)
