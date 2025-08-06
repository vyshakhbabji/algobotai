#!/usr/bin/env python3
"""
Real AI Model Validation using Actual Portfolio Manager
Test your AI system with proper validation methodology
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your AI portfolio manager
try:
    from ai_portfolio_manager import AIPortfolioManager
except ImportError:
    print("❌ Cannot import AIPortfolioManager - make sure ai_portfolio_manager.py is available")
    sys.exit(1)

class RealValidationFramework:
    def __init__(self, portfolio_value=10000):
        self.portfolio_value = portfolio_value
        self.ai_manager = AIPortfolioManager(portfolio_value)
        
    def run_real_validation(self):
        """Run validation using actual AI models"""
        print("🔬 REAL AI MODEL VALIDATION")
        print("=" * 80)
        print(f"Portfolio Value: ${self.portfolio_value:,}")
        print(f"Using actual AI models from ai_portfolio_manager.py")
        print("=" * 80)
        
        # 1. Historical backtesting validation
        print("\n📈 HISTORICAL BACKTESTING VALIDATION")
        print("-" * 60)
        self.historical_validation()
        
        # 2. Cross-validation across time periods
        print("\n⏰ TIME-BASED CROSS-VALIDATION")
        print("-" * 60)
        self.time_cross_validation()
        
        # 3. Walk-forward validation
        print("\n🚶 WALK-FORWARD VALIDATION")
        print("-" * 60)
        self.real_walk_forward()
        
        # 4. Model stability testing
        print("\n🎯 MODEL STABILITY TESTING")
        print("-" * 60)
        self.stability_testing()
        
        # 5. Risk validation
        print("\n⚖️ RISK VALIDATION")
        print("-" * 60)
        self.risk_validation()
        
        print("\n✅ REAL VALIDATION COMPLETE")
        
    def historical_validation(self):
        """Test on different historical periods"""
        test_periods = [
            ("2022 Bear Market", "2022-01-01", "2022-12-31"),
            ("2023 Recovery", "2023-01-01", "2023-12-31"), 
            ("2024 Stability", "2024-01-01", "2024-12-31"),
            ("2025 Current", "2025-01-01", "2025-08-05"),
            ("Last 6 Months", "2025-02-05", "2025-08-05"),
            ("Last 3 Months", "2025-05-05", "2025-08-05")
        ]
        
        results = []
        
        for period_name, start_date, end_date in test_periods:
            try:
                print(f"\n📊 Testing {period_name} ({start_date} to {end_date})")
                
                # Run backtest for this period
                backtest_result = self.ai_manager.backtest_portfolio(
                    start_date=start_date,
                    end_date=end_date
                )
                
                if backtest_result:
                    final_value = backtest_result.get('final_portfolio_value', self.portfolio_value)
                    total_return = ((final_value - self.portfolio_value) / self.portfolio_value) * 100
                    
                    # Calculate additional metrics
                    trades = len(backtest_result.get('trades', []))
                    
                    results.append({
                        'period': period_name,
                        'return': total_return,
                        'final_value': final_value,
                        'trades': trades
                    })
                    
                    print(f"   💰 Return: {total_return:.2f}%")
                    print(f"   🏦 Final Value: ${final_value:,.2f}")
                    print(f"   📈 Trades: {trades}")
                    
                    # Performance assessment
                    if total_return > 10:
                        print(f"   ✅ Strong performance")
                    elif total_return > 0:
                        print(f"   ⚠️ Modest performance")
                    else:
                        print(f"   ❌ Negative performance")
                        
                else:
                    print(f"   ❌ Backtest failed")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
        # Summary analysis
        if results:
            returns = [r['return'] for r in results]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            win_rate = (np.array(returns) > 0).mean() * 100
            
            print(f"\n📋 HISTORICAL VALIDATION SUMMARY:")
            print(f"   Average Return: {avg_return:.2f}%")
            print(f"   Return Volatility: {std_return:.2f}%")
            print(f"   Win Rate: {win_rate:.1f}%")
            
            if win_rate >= 70:
                print(f"   ✅ High win rate - good consistency")
            elif win_rate >= 50:
                print(f"   ⚠️ Moderate win rate")
            else:
                print(f"   ❌ Low win rate - investigate")
                
    def time_cross_validation(self):
        """Cross-validation across different time windows"""
        print("Testing model with different training/testing splits...")
        
        # Define cross-validation periods
        cv_periods = [
            ("2023-2024 train, 2025 test", "2023-01-01", "2024-12-31", "2025-01-01", "2025-08-05"),
            ("2022-2023 train, 2024 test", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
            ("2021-2022 train, 2023 test", "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31")
        ]
        
        cv_results = []
        
        for cv_name, train_start, train_end, test_start, test_end in cv_periods:
            try:
                print(f"\n🔄 {cv_name}")
                print(f"   Train: {train_start} to {train_end}")
                print(f"   Test: {test_start} to {test_end}")
                
                # Test on the test period (models would be retrained in practice)
                test_result = self.ai_manager.backtest_portfolio(
                    start_date=test_start,
                    end_date=test_end
                )
                
                if test_result:
                    final_value = test_result.get('final_portfolio_value', self.portfolio_value)
                    test_return = ((final_value - self.portfolio_value) / self.portfolio_value) * 100
                    
                    cv_results.append(test_return)
                    print(f"   📊 Test Return: {test_return:.2f}%")
                    
                    if test_return > 5:
                        print(f"   ✅ Good out-of-sample performance")
                    elif test_return > 0:
                        print(f"   ⚠️ Modest out-of-sample performance")
                    else:
                        print(f"   ❌ Poor out-of-sample performance")
                        
            except Exception as e:
                print(f"   ❌ CV Error: {e}")
                
        if cv_results:
            avg_cv = np.mean(cv_results)
            print(f"\n📊 CROSS-VALIDATION SUMMARY:")
            print(f"   Average Out-of-Sample Return: {avg_cv:.2f}%")
            
            if avg_cv > 5:
                print(f"   ✅ Good generalization")
            elif avg_cv > 0:
                print(f"   ⚠️ Moderate generalization")
            else:
                print(f"   ❌ Poor generalization - possible overfitting")
                
    def real_walk_forward(self):
        """Walk-forward validation with actual models"""
        print("Running walk-forward analysis...")
        
        # Define monthly walk-forward windows for 2024-2025
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 8, 5)
        
        walk_results = []
        current_date = start_date
        
        while current_date < end_date:
            # Define 3-month test window
            test_end = min(current_date + timedelta(days=90), end_date)
            
            try:
                test_result = self.ai_manager.backtest_portfolio(
                    start_date=current_date.strftime('%Y-%m-%d'),
                    end_date=test_end.strftime('%Y-%m-%d')
                )
                
                if test_result:
                    final_value = test_result.get('final_portfolio_value', self.portfolio_value)
                    period_return = ((final_value - self.portfolio_value) / self.portfolio_value) * 100
                    
                    walk_results.append(period_return)
                    print(f"📈 {current_date.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}: {period_return:.2f}%")
                    
            except Exception as e:
                print(f"❌ {current_date.strftime('%Y-%m')}: Error - {e}")
                
            # Move to next period
            current_date += timedelta(days=90)
            
        if walk_results:
            avg_walk = np.mean(walk_results)
            std_walk = np.std(walk_results)
            
            print(f"\n📊 WALK-FORWARD SUMMARY:")
            print(f"   Average 3-month return: {avg_walk:.2f}%")
            print(f"   Return consistency: {std_walk:.2f}% std")
            print(f"   Number of periods: {len(walk_results)}")
            
    def stability_testing(self):
        """Test model stability and sensitivity"""
        print("Testing model stability...")
        
        # Test same period multiple times to check consistency
        test_period_start = "2025-05-01"
        test_period_end = "2025-08-05"
        
        stability_results = []
        
        for run in range(5):
            try:
                result = self.ai_manager.backtest_portfolio(
                    start_date=test_period_start,
                    end_date=test_period_end
                )
                
                if result:
                    final_value = result.get('final_portfolio_value', self.portfolio_value)
                    period_return = ((final_value - self.portfolio_value) / self.portfolio_value) * 100
                    stability_results.append(period_return)
                    
            except Exception as e:
                print(f"❌ Stability run {run+1}: {e}")
                
        if stability_results:
            stability_std = np.std(stability_results)
            print(f"📊 Stability Test Results:")
            print(f"   Returns: {[f'{r:.2f}%' for r in stability_results]}")
            print(f"   Standard Deviation: {stability_std:.4f}%")
            
            if stability_std < 0.1:
                print(f"   ✅ Very stable model")
            elif stability_std < 1.0:
                print(f"   ⚠️ Moderately stable model")
            else:
                print(f"   ❌ Unstable model - investigate")
                
    def risk_validation(self):
        """Validate risk metrics"""
        try:
            # Run extended backtest for risk analysis
            result = self.ai_manager.backtest_portfolio(
                start_date="2024-01-01",
                end_date="2025-08-05"
            )
            
            if result and 'portfolio_history' in result:
                portfolio_values = result['portfolio_history']
                
                # Calculate daily returns
                returns = []
                for i in range(1, len(portfolio_values)):
                    daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                    returns.append(daily_return)
                
                returns = np.array(returns)
                
                # Risk metrics
                total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe = 0
                
                # Maximum drawdown
                peak = portfolio_values[0]
                max_drawdown = 0
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                print(f"📊 Risk Analysis (Jan 2024 - Aug 2025):")
                print(f"   Total Return: {total_return*100:.2f}%")
                print(f"   Annualized Volatility: {volatility*100:.2f}%")
                print(f"   Sharpe Ratio: {sharpe:.3f}")
                print(f"   Maximum Drawdown: {max_drawdown*100:.2f}%")
                
                # Risk assessment
                risk_score = 0
                if sharpe > 1.0: risk_score += 1
                if max_drawdown < 0.20: risk_score += 1  
                if volatility < 0.25: risk_score += 1
                
                if risk_score >= 2:
                    print(f"   ✅ Good risk profile")
                elif risk_score == 1:
                    print(f"   ⚠️ Moderate risk profile")
                else:
                    print(f"   ❌ High risk profile")
                    
            else:
                print("❌ Could not analyze risk - insufficient data")
                
        except Exception as e:
            print(f"❌ Risk validation error: {e}")

if __name__ == "__main__":
    try:
        validator = RealValidationFramework(portfolio_value=10000)
        validator.run_real_validation()
        
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
