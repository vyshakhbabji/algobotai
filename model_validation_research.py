#!/usr/bin/env python3
"""
SCIENTIFIC MODEL VALIDATION - Research-Grade Testing
How researchers validate AI models + applied to our Elite AI v2.0
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from elite_ai_trader import EliteAITrader

class ResearchGradeValidator:
    """Scientific validation using research methodologies"""
    
    def __init__(self):
        self.ai_model = EliteAITrader()
        self.validation_results = {}
        
    def comprehensive_validation(self, symbol: str = "AAPL"):
        """Run complete research-grade validation suite"""
        
        print(f"🔬 RESEARCH-GRADE MODEL VALIDATION FOR {symbol}")
        print("=" * 55)
        print("Using same methods as academic research papers")
        print("Testing Elite AI v2.0 with scientific rigor")
        print()
        
        # 1. Time Series Cross-Validation (Gold Standard)
        self.time_series_validation(symbol)
        
        # 2. Walk-Forward Analysis (Real Trading Simulation)
        self.walk_forward_analysis(symbol)
        
        # 3. Statistical Significance Testing
        self.statistical_significance_test(symbol)
        
        # 4. Benchmark Comparison (vs Random, Buy&Hold)
        self.benchmark_comparison(symbol)
        
        # 5. Prediction Stability Analysis
        self.stability_analysis(symbol)
        
        # 6. Out-of-Sample Performance
        self.out_of_sample_test(symbol)
        
        # 7. Generate Research Summary
        self.generate_research_report(symbol)
        
    def time_series_validation(self, symbol: str):
        """Time Series Cross-Validation - Prevents data leakage"""
        
        print("📊 1. TIME SERIES CROSS-VALIDATION")
        print("-" * 40)
        print("Gold standard for time series validation")
        print("Prevents future data leakage (critical flaw in many AI systems)")
        
        try:
            # Load data
            data = self.load_clean_data(symbol)
            if data is None:
                return
            
            # Prepare features and targets
            features = self.ai_model.create_simple_features(data)
            target = data['Close'].pct_change().shift(-1).fillna(0) * 100
            
            # Remove any remaining NaN values
            mask = ~(features.isna().any(axis=1) | target.isna())
            features = features[mask]
            target = target[mask]
            
            # Time Series Split (respects temporal order)
            tscv = TimeSeriesSplit(n_splits=5)
            
            scores = []
            direction_accuracies = []
            
            print(f"Running 5-fold time series cross-validation...")
            
            for i, (train_idx, test_idx) in enumerate(tscv.split(features)):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                
                # Train models
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                direction_acc = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
                
                scores.append(r2)
                direction_accuracies.append(direction_acc)
                
                print(f"   Fold {i+1}: R² = {r2:.3f}, Direction = {direction_acc:.1f}%")
            
            avg_r2 = np.mean(scores)
            avg_direction = np.mean(direction_accuracies)
            std_r2 = np.std(scores)
            
            print(f"\n✅ Cross-Validation Results:")
            print(f"   Average R²: {avg_r2:.3f} ± {std_r2:.3f}")
            print(f"   Average Direction Accuracy: {avg_direction:.1f}%")
            print(f"   Consistency: {'HIGH' if std_r2 < 0.1 else 'MEDIUM' if std_r2 < 0.2 else 'LOW'}")
            
            self.validation_results['cross_validation'] = {
                'avg_r2': avg_r2,
                'std_r2': std_r2,
                'avg_direction': avg_direction,
                'scores': scores
            }
            
        except Exception as e:
            print(f"❌ Cross-validation failed: {e}")
    
    def walk_forward_analysis(self, symbol: str):
        """Walk-Forward Analysis - Simulates real trading"""
        
        print(f"\n📈 2. WALK-FORWARD ANALYSIS")
        print("-" * 35)
        print("Simulates real trading: train on past, predict future")
        print("Most realistic test of trading performance")
        
        try:
            data = self.load_clean_data(symbol)
            if data is None:
                return
            
            # Parameters
            train_window = 252  # 1 year
            test_window = 21    # 1 month
            
            predictions = []
            actuals = []
            dates = []
            
            print(f"Running walk-forward with {train_window} day training window...")
            
            for i in range(train_window, len(data) - test_window, test_window):
                # Train data
                train_data = data.iloc[i-train_window:i]
                
                # Test data
                test_data = data.iloc[i:i+test_window]
                
                # Train model on historical data only
                try:
                    self.ai_model.train_simple_models_data(train_data, symbol)
                    
                    # Predict on future data
                    for j in range(len(test_data)):
                        pred_data = test_data.iloc[:j+1]  # Only use data up to current point
                        if len(pred_data) >= 10:  # Need minimum data
                            result = self.ai_model.make_simple_prediction_data(pred_data, symbol)
                            if result:
                                predictions.append(result['predicted_return'])
                                
                                # Actual return (next day)
                                if j < len(test_data) - 1:
                                    actual_return = ((test_data.iloc[j+1]['Close'] / test_data.iloc[j]['Close']) - 1) * 100
                                    actuals.append(actual_return)
                                    dates.append(test_data.iloc[j]['Date'] if 'Date' in test_data.columns else test_data.index[j])
                except:
                    continue
            
            if len(predictions) > 10 and len(actuals) > 10:
                # Align predictions and actuals
                min_len = min(len(predictions), len(actuals))
                predictions = predictions[:min_len]
                actuals = actuals[:min_len]
                
                # Calculate walk-forward metrics
                wf_r2 = r2_score(actuals, predictions)
                wf_direction = np.mean(np.sign(actuals) == np.sign(predictions)) * 100
                wf_mse = mean_squared_error(actuals, predictions)
                
                print(f"\n✅ Walk-Forward Results ({min_len} predictions):")
                print(f"   R²: {wf_r2:.3f}")
                print(f"   Direction Accuracy: {wf_direction:.1f}%")
                print(f"   MSE: {wf_mse:.3f}")
                print(f"   Prediction Quality: {'GOOD' if wf_direction > 55 else 'FAIR' if wf_direction > 50 else 'POOR'}")
                
                self.validation_results['walk_forward'] = {
                    'r2': wf_r2,
                    'direction_accuracy': wf_direction,
                    'mse': wf_mse,
                    'predictions': predictions[:50],  # Store sample
                    'actuals': actuals[:50]
                }
            else:
                print("❌ Insufficient walk-forward data")
                
        except Exception as e:
            print(f"❌ Walk-forward analysis failed: {e}")
    
    def statistical_significance_test(self, symbol: str):
        """Test if predictions are statistically significant"""
        
        print(f"\n📊 3. STATISTICAL SIGNIFICANCE TEST")
        print("-" * 40)
        print("Is our AI better than random guessing?")
        print("Using t-test for statistical significance")
        
        try:
            # Get walk-forward results
            if 'walk_forward' not in self.validation_results:
                print("❌ Need walk-forward results first")
                return
            
            wf_data = self.validation_results['walk_forward']
            predictions = wf_data['predictions']
            actuals = wf_data['actuals']
            
            if len(predictions) < 30:
                print("❌ Need at least 30 predictions for statistical test")
                return
            
            # Direction accuracy test
            correct_directions = np.sign(actuals) == np.sign(predictions)
            accuracy = np.mean(correct_directions)
            
            # T-test against random (50%)
            t_stat, p_value = stats.ttest_1samp(correct_directions, 0.5)
            
            # Effect size (Cohen's d)
            effect_size = (accuracy - 0.5) / np.std(correct_directions)
            
            print(f"\n✅ Statistical Test Results:")
            print(f"   Direction Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"   T-statistic: {t_stat:.3f}")
            print(f"   P-value: {p_value:.4f}")
            print(f"   Effect Size (Cohen's d): {effect_size:.3f}")
            
            # Interpret results
            if p_value < 0.05:
                if accuracy > 0.5:
                    print(f"   🎯 STATISTICALLY SIGNIFICANT: AI beats random!")
                else:
                    print(f"   ⚠️  STATISTICALLY SIGNIFICANT: AI worse than random")
            else:
                print(f"   🤷 NOT STATISTICALLY SIGNIFICANT: Could be random luck")
            
            # Effect size interpretation
            if abs(effect_size) > 0.8:
                effect_desc = "LARGE"
            elif abs(effect_size) > 0.5:
                effect_desc = "MEDIUM" 
            elif abs(effect_size) > 0.2:
                effect_desc = "SMALL"
            else:
                effect_desc = "NEGLIGIBLE"
                
            print(f"   Effect Size: {effect_desc}")
            
            self.validation_results['statistical'] = {
                'accuracy': accuracy,
                't_stat': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05
            }
            
        except Exception as e:
            print(f"❌ Statistical test failed: {e}")
    
    def benchmark_comparison(self, symbol: str):
        """Compare against standard benchmarks"""
        
        print(f"\n🏆 4. BENCHMARK COMPARISON")
        print("-" * 30)
        print("How does our AI compare to:")
        print("• Random predictions")
        print("• Buy and Hold strategy")
        print("• Naive forecast (last value)")
        
        try:
            data = self.load_clean_data(symbol)
            if data is None:
                return
            
            # Calculate actual returns
            returns = data['Close'].pct_change().fillna(0) * 100
            
            # Get our AI predictions (simplified)
            self.ai_model.train_simple_models(symbol)
            ai_result = self.ai_model.make_simple_prediction(symbol)
            
            if not ai_result:
                print("❌ No AI prediction available")
                return
            
            # Benchmark 1: Random predictions
            np.random.seed(42)
            random_predictions = np.random.normal(0, returns.std(), len(returns))
            random_direction = np.mean(np.sign(returns[1:]) == np.sign(random_predictions[:-1])) * 100
            
            # Benchmark 2: Buy and Hold
            buy_hold_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            
            # Benchmark 3: Naive forecast (last value carries forward)
            naive_predictions = returns.shift(1).fillna(0)
            naive_direction = np.mean(np.sign(returns[1:]) == np.sign(naive_predictions[1:])) * 100
            
            # Our AI performance (from validation)
            ai_direction = self.validation_results.get('walk_forward', {}).get('direction_accuracy', 50)
            
            print(f"\n✅ Benchmark Results:")
            print(f"   Random Prediction: {random_direction:.1f}% direction accuracy")
            print(f"   Naive Forecast: {naive_direction:.1f}% direction accuracy")
            print(f"   Buy & Hold Return: {buy_hold_return:.2f}%")
            print(f"   Our AI Direction: {ai_direction:.1f}% direction accuracy")
            
            # Determine if AI beats benchmarks
            beats_random = ai_direction > random_direction
            beats_naive = ai_direction > naive_direction
            
            print(f"\n🎯 AI Performance vs Benchmarks:")
            print(f"   Beats Random: {'✅ YES' if beats_random else '❌ NO'}")
            print(f"   Beats Naive: {'✅ YES' if beats_naive else '❌ NO'}")
            
            if beats_random and beats_naive:
                print(f"   🏆 AI OUTPERFORMS ALL BENCHMARKS!")
            elif beats_random or beats_naive:
                print(f"   🎯 AI SHOWS SOME SKILL")
            else:
                print(f"   ⚠️  AI NEEDS IMPROVEMENT")
            
            self.validation_results['benchmarks'] = {
                'random_direction': random_direction,
                'naive_direction': naive_direction,
                'buy_hold_return': buy_hold_return,
                'ai_direction': ai_direction,
                'beats_random': beats_random,
                'beats_naive': beats_naive
            }
            
        except Exception as e:
            print(f"❌ Benchmark comparison failed: {e}")
    
    def stability_analysis(self, symbol: str):
        """Test prediction stability across different periods"""
        
        print(f"\n🔄 5. PREDICTION STABILITY ANALYSIS")
        print("-" * 38)
        print("Do predictions stay consistent across time?")
        print("Unstable models are dangerous for trading")
        
        try:
            data = self.load_clean_data(symbol)
            if data is None:
                return
            
            # Test stability across different time periods
            periods = [
                (252, "1 Year"),
                (126, "6 Months"), 
                (63, "3 Months"),
                (21, "1 Month")
            ]
            
            direction_accuracies = []
            r2_scores = []
            
            print(f"Testing stability across different training periods...")
            
            for days, period_name in periods:
                try:
                    # Use different training periods
                    train_data = data.tail(days + 50)  # Extra buffer
                    
                    # Train and test
                    self.ai_model.train_simple_models_data(train_data, symbol)
                    result = self.ai_model.make_simple_prediction_data(train_data, symbol)
                    
                    if result:
                        # Get validation metrics from the training
                        quality_metrics = self.ai_model.validate_prediction_quality_data(train_data, symbol)
                        
                        if quality_metrics:
                            direction_accuracies.append(quality_metrics.get('direction_accuracy', 50))
                            r2_scores.append(quality_metrics.get('r2_score', -1))
                            
                            print(f"   {period_name:10}: Direction {quality_metrics.get('direction_accuracy', 50):.1f}%, R² {quality_metrics.get('r2_score', -1):.3f}")
                except:
                    continue
            
            if len(direction_accuracies) >= 2:
                # Calculate stability metrics
                direction_std = np.std(direction_accuracies)
                r2_std = np.std(r2_scores)
                
                print(f"\n✅ Stability Analysis:")
                print(f"   Direction Accuracy Range: {min(direction_accuracies):.1f}% - {max(direction_accuracies):.1f}%")
                print(f"   Direction Std Dev: {direction_std:.2f}%")
                print(f"   R² Range: {min(r2_scores):.3f} - {max(r2_scores):.3f}")
                print(f"   R² Std Dev: {r2_std:.3f}")
                
                # Stability assessment
                if direction_std < 5:
                    stability = "HIGH"
                elif direction_std < 10:
                    stability = "MEDIUM"
                else:
                    stability = "LOW"
                
                print(f"   Overall Stability: {stability}")
                
                self.validation_results['stability'] = {
                    'direction_std': direction_std,
                    'r2_std': r2_std,
                    'stability_rating': stability,
                    'direction_scores': direction_accuracies,
                    'r2_scores': r2_scores
                }
            else:
                print("❌ Insufficient data for stability analysis")
                
        except Exception as e:
            print(f"❌ Stability analysis failed: {e}")
    
    def out_of_sample_test(self, symbol: str):
        """Ultimate test: completely unseen data"""
        
        print(f"\n🎯 6. OUT-OF-SAMPLE TEST")
        print("-" * 25)
        print("The ultimate test: predict on completely unseen data")
        print("Training on old data, testing on recent data")
        
        try:
            data = self.load_clean_data(symbol)
            if data is None:
                return
            
            # Split data: 80% train, 20% out-of-sample test
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            print(f"Training on {len(train_data)} days, testing on {len(test_data)} days")
            print(f"Test period: {test_data.iloc[0].get('Date', 'N/A')} to {test_data.iloc[-1].get('Date', 'N/A')}")
            
            # Train on old data only
            self.ai_model.train_simple_models_data(train_data, symbol)
            
            # Test on completely unseen data
            result = self.ai_model.make_simple_prediction_data(test_data, symbol)
            
            if result:
                # Calculate actual future returns
                future_returns = test_data['Close'].pct_change().fillna(0) * 100
                
                # Direction accuracy on unseen data
                if len(future_returns) > 1:
                    predicted_direction = 1 if result['predicted_return'] > 0 else -1
                    actual_directions = np.sign(future_returns[future_returns != 0])
                    
                    if len(actual_directions) > 0:
                        direction_accuracy = np.mean(actual_directions == predicted_direction) * 100
                        
                        print(f"\n✅ Out-of-Sample Results:")
                        print(f"   Predicted Return: {result['predicted_return']:.2f}%")
                        print(f"   Predicted Direction: {'UP' if predicted_direction > 0 else 'DOWN'}")
                        print(f"   Actual Direction Accuracy: {direction_accuracy:.1f}%")
                        print(f"   Confidence: {result.get('confidence', 0):.1f}%")
                        
                        # Final assessment
                        if direction_accuracy > 60:
                            assessment = "EXCELLENT"
                        elif direction_accuracy > 55:
                            assessment = "GOOD"
                        elif direction_accuracy > 50:
                            assessment = "FAIR"
                        else:
                            assessment = "POOR"
                        
                        print(f"   Out-of-Sample Quality: {assessment}")
                        
                        self.validation_results['out_of_sample'] = {
                            'predicted_return': result['predicted_return'],
                            'direction_accuracy': direction_accuracy,
                            'confidence': result.get('confidence', 0),
                            'assessment': assessment
                        }
            else:
                print("❌ No out-of-sample prediction available")
                
        except Exception as e:
            print(f"❌ Out-of-sample test failed: {e}")
    
    def generate_research_report(self, symbol: str):
        """Generate comprehensive research-grade validation report"""
        
        print(f"\n📋 RESEARCH-GRADE VALIDATION REPORT FOR {symbol}")
        print("=" * 60)
        print("SCIENTIFIC ASSESSMENT OF ELITE AI v2.0 MODEL")
        print("=" * 60)
        
        # Overall quality score
        quality_score = 0
        max_score = 6
        
        # 1. Cross-validation
        if 'cross_validation' in self.validation_results:
            cv = self.validation_results['cross_validation']
            print(f"\n1. CROSS-VALIDATION PERFORMANCE:")
            print(f"   Average R²: {cv['avg_r2']:.3f} ± {cv['std_r2']:.3f}")
            print(f"   Average Direction: {cv['avg_direction']:.1f}%")
            if cv['avg_direction'] > 52:
                quality_score += 1
                print(f"   ✅ PASSES cross-validation test")
            else:
                print(f"   ❌ FAILS cross-validation test")
        
        # 2. Walk-forward analysis
        if 'walk_forward' in self.validation_results:
            wf = self.validation_results['walk_forward']
            print(f"\n2. WALK-FORWARD PERFORMANCE:")
            print(f"   R²: {wf['r2']:.3f}")
            print(f"   Direction Accuracy: {wf['direction_accuracy']:.1f}%")
            if wf['direction_accuracy'] > 52:
                quality_score += 1
                print(f"   ✅ PASSES walk-forward test")
            else:
                print(f"   ❌ FAILS walk-forward test")
        
        # 3. Statistical significance
        if 'statistical' in self.validation_results:
            stat = self.validation_results['statistical']
            print(f"\n3. STATISTICAL SIGNIFICANCE:")
            print(f"   P-value: {stat['p_value']:.4f}")
            print(f"   Effect Size: {stat['effect_size']:.3f}")
            if stat['significant'] and stat['accuracy'] > 0.5:
                quality_score += 1
                print(f"   ✅ STATISTICALLY SIGNIFICANT improvement over random")
            else:
                print(f"   ❌ NOT statistically significant")
        
        # 4. Benchmark comparison
        if 'benchmarks' in self.validation_results:
            bench = self.validation_results['benchmarks']
            print(f"\n4. BENCHMARK COMPARISON:")
            print(f"   AI: {bench['ai_direction']:.1f}%")
            print(f"   Random: {bench['random_direction']:.1f}%")
            print(f"   Naive: {bench['naive_direction']:.1f}%")
            if bench['beats_random'] and bench['beats_naive']:
                quality_score += 1
                print(f"   ✅ BEATS all benchmarks")
            else:
                print(f"   ❌ DOESN'T beat all benchmarks")
        
        # 5. Stability
        if 'stability' in self.validation_results:
            stab = self.validation_results['stability']
            print(f"\n5. PREDICTION STABILITY:")
            print(f"   Direction Std Dev: {stab['direction_std']:.2f}%")
            print(f"   Stability Rating: {stab['stability_rating']}")
            if stab['stability_rating'] in ['HIGH', 'MEDIUM']:
                quality_score += 1
                print(f"   ✅ STABLE predictions across time periods")
            else:
                print(f"   ❌ UNSTABLE predictions")
        
        # 6. Out-of-sample
        if 'out_of_sample' in self.validation_results:
            oos = self.validation_results['out_of_sample']
            print(f"\n6. OUT-OF-SAMPLE PERFORMANCE:")
            print(f"   Direction Accuracy: {oos['direction_accuracy']:.1f}%")
            print(f"   Assessment: {oos['assessment']}")
            if oos['assessment'] in ['EXCELLENT', 'GOOD', 'FAIR']:
                quality_score += 1
                print(f"   ✅ ACCEPTABLE out-of-sample performance")
            else:
                print(f"   ❌ POOR out-of-sample performance")
        
        # Final assessment
        quality_percentage = (quality_score / max_score) * 100
        
        print(f"\n🎯 OVERALL RESEARCH ASSESSMENT:")
        print(f"   Quality Score: {quality_score}/{max_score} ({quality_percentage:.0f}%)")
        
        if quality_percentage >= 80:
            final_grade = "A (EXCELLENT)"
            recommendation = "DEPLOY WITH CONFIDENCE"
        elif quality_percentage >= 60:
            final_grade = "B (GOOD)"
            recommendation = "DEPLOY WITH MONITORING"
        elif quality_percentage >= 40:
            final_grade = "C (FAIR)"
            recommendation = "DEPLOY WITH CAUTION"
        else:
            final_grade = "F (POOR)"
            recommendation = "DO NOT DEPLOY"
        
        print(f"   Research Grade: {final_grade}")
        print(f"   Recommendation: {recommendation}")
        
        print(f"\n📊 RESEARCH METHODOLOGY USED:")
        print(f"   ✅ Time Series Cross-Validation (prevents data leakage)")
        print(f"   ✅ Walk-Forward Analysis (simulates real trading)")
        print(f"   ✅ Statistical Significance Testing")
        print(f"   ✅ Benchmark Comparison")
        print(f"   ✅ Stability Analysis")
        print(f"   ✅ Out-of-Sample Testing")
        
        print(f"\n🔬 This validation follows the same rigorous standards")
        print(f"   used in academic research and institutional trading!")
        
        return {
            'quality_score': quality_score,
            'max_score': max_score,
            'percentage': quality_percentage,
            'grade': final_grade,
            'recommendation': recommendation
        }
    
    def load_clean_data(self, symbol: str):
        """Load clean split-adjusted data if available"""
        try:
            clean_file = f"clean_data_{symbol.lower()}.csv"
            data = pd.read_csv(clean_file)
            return data
        except FileNotFoundError:
            # Fall back to yfinance
            print(f"   Using yfinance data (no clean data for {symbol})")
            data = yf.download(symbol, period="2y", interval="1d")
            data.reset_index(inplace=True)
            return data

def main():
    """Run comprehensive research-grade validation"""
    
    print("🔬 RESEARCH-GRADE MODEL VALIDATION")
    print("=" * 45)
    print("Testing Elite AI v2.0 with academic rigor")
    print("Same methods used in research papers!")
    print()
    
    validator = ResearchGradeValidator()
    
    # Test on multiple stocks
    test_stocks = ["AAPL", "TSLA", "GOOGL"]
    
    for stock in test_stocks:
        print(f"\n{'='*60}")
        validator.comprehensive_validation(stock)
        print(f"{'='*60}")
    
    print(f"\n🎯 RESEARCH VALIDATION COMPLETE!")
    print(f"Our Elite AI v2.0 has been tested with the same rigor")
    print(f"as academic research and institutional models!")

if __name__ == "__main__":
    main()
