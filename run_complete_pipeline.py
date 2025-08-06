#!/usr/bin/env python3
"""
Complete Fixed Algorithmic Trading Bot
Run the entire pipeline from data fetching to strategy backtesting
"""

import sys
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n{'='*20} STEP {step_num}: {title} {'='*20}")

def main():
    """
    Run the complete algorithmic trading pipeline
    """
    start_time = time.time()
    
    print_header("FIXED ALGORITHMIC TRADING BOT")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Data Fetching
        print_step(1, "DATA FETCHING")
        print("Fetching 6 months of NVDA data with technical indicators...")
        
        from fixed_data_fetcher import main as fetch_data
        data, features = fetch_data()
        
        print(f"✅ Data fetching completed successfully!")
        print(f"   - Samples: {len(data)}")
        print(f"   - Features: {len(features)}")
        print(f"   - Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Step 2: Data Preprocessing
        print_step(2, "DATA PREPROCESSING")
        print("Cleaning, scaling, and preparing data for modeling...")
        
        from fixed_preprocessor import main as preprocess_data
        preprocessor, model_data = preprocess_data()
        
        print(f"✅ Data preprocessing completed successfully!")
        print(f"   - Train samples: {len(model_data['X_train_seq'])}")
        print(f"   - Test samples: {len(model_data['X_test_seq'])}")
        print(f"   - Input shape: {model_data['X_train_seq'].shape}")
        
        # Step 3: Model Training
        print_step(3, "MODEL TRAINING")
        print("Training multiple ML models and creating ensemble...")
        
        from fixed_model_trainer import main as train_models
        trainer, models, scores = train_models()
        
        print(f"✅ Model training completed successfully!")
        print(f"   - Models trained: {len(models)}")
        print("   - Model performance (MAE):")
        for model_name, metrics in scores.items():
            print(f"     • {model_name}: {metrics['MAE']:.4f}")
        
        # Find best model
        best_model = min(scores.items(), key=lambda x: x[1]['MAE'])
        print(f"   - Best model: {best_model[0]} (MAE: {best_model[1]['MAE']:.4f})")
        
        # Step 4: Trading Strategy
        print_step(4, "TRADING STRATEGY & BACKTESTING")
        print("Generating trading signals and backtesting strategy...")
        
        from fixed_trading_strategy import main as run_strategy
        strategy, portfolio, metrics, all_metrics = run_strategy()
        
        print(f"✅ Trading strategy completed successfully!")
        print("   - Strategy performance:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"     • {key}: {value:.2f}")
            else:
                print(f"     • {key}: {value}")
        
        # Step 5: Results Summary
        print_step(5, "RESULTS SUMMARY")
        
        # Model comparison
        print("\nModel Comparison (Total Return %):")
        for model_name, model_metrics in all_metrics.items():
            total_return = model_metrics.get('Total Return (%)', 0)
            print(f"   • {model_name}: {total_return:.2f}%")
        
        # Best trading model
        best_trading_model = max(all_metrics.items(), key=lambda x: x[1].get('Total Return (%)', 0))
        print(f"\nBest Trading Model: {best_trading_model[0]}")
        print(f"Total Return: {best_trading_model[1]['Total Return (%)']:.2f}%")
        print(f"Sharpe Ratio: {best_trading_model[1]['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {best_trading_model[1]['Max Drawdown (%)']:.2f}%")
        
        # Files created
        print(f"\n📁 Files and results saved in:")
        print(f"   • Raw data: fixed_data/nvda_data_processed.csv")
        print(f"   • Preprocessed data: fixed_data/preprocessed/")
        print(f"   • Trained models: fixed_data/models/")
        print(f"   • Trading results: fixed_data/results/")
        print(f"   • Visualizations: fixed_data/models/ and fixed_data/results/")
        
        # Execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n⏱️  Total execution time: {execution_time/60:.2f} minutes")
        
        print_header("ALGORITHMIC TRADING BOT COMPLETED SUCCESSFULLY! 🎉")
        
        # Final recommendations
        print("\n🎯 TRADING RECOMMENDATIONS:")
        print(f"   • Use {best_trading_model[0]} model for live trading")
        print(f"   • Expected return: {best_trading_model[1]['Total Return (%)']:.2f}%")
        print(f"   • Risk level: {best_trading_model[1]['Max Drawdown (%)']:.2f}% max drawdown")
        print(f"   • Win rate: {best_trading_model[1]['Win Rate (%)']:.2f}%")
        
        print("\n📊 NEXT STEPS:")
        print("   1. Review backtesting results in fixed_data/results/")
        print("   2. Analyze model predictions vs actual prices")
        print("   3. Consider paper trading before live implementation")
        print("   4. Monitor model performance and retrain periodically")
        print("   5. Implement risk management and position sizing")
        
        return {
            'data': data,
            'preprocessor': preprocessor,
            'models': models,
            'strategy': strategy,
            'portfolio': portfolio,
            'metrics': metrics,
            'all_metrics': all_metrics
        }
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        print("\nDebugging information:")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    results = main()
