#!/usr/bin/env python3
"""
Quick Demo Runner for Unified ML Trading System
Demonstrates consolidated sophisticated components

This script shows how your existing sophisticated infrastructure
(Elite Stock Selector, AI Portfolio Manager, Options Trader, Live Trading)
is now unified into a single system targeting 30%+ YoY returns.
"""

import os
import sys
import json
from datetime import datetime
import traceback

def run_demo():
    """Run demonstration of unified trading system"""
    
    print("🚀 UNIFIED ML TRADING SYSTEM DEMO")
    print("Consolidating Your Sophisticated Trading Infrastructure")
    print("=" * 80)
    
    try:
        # Import the unified system
        from unified_ml_trading_system import UnifiedMLTradingSystem
        
        print("✅ Unified ML Trading System imported successfully")
        
        # Initialize the system
        print("\\n🔧 Initializing system...")
        trading_system = UnifiedMLTradingSystem()
        
        target_return = trading_system.config.get('performance', {}).get('target_annual_return', 30)
        print(f"✅ System initialized with target: {target_return}% YoY")
        
        # Demonstrate elite stock selection
        print("\\n🔍 Elite Stock Selection (AI-Powered)...")
        universe = trading_system.get_elite_stock_universe()
        print(f"✅ Selected {len(universe)} elite stocks: {universe[:10]}...")
        
        # Demonstrate ML prediction capability
        print("\\n🤖 ML Prediction Demo...")
        if universe:
            test_symbol = universe[0]
            print(f"   Testing ML prediction for {test_symbol}...")
            
            import yfinance as yf
            ticker = yf.Ticker(test_symbol)
            data = ticker.history(period='6m')
            
            if len(data) > 100:
                # Train model
                success = trading_system.train_ensemble_models(test_symbol, data)
                if success:
                    print(f"   ✅ Trained ensemble models for {test_symbol}")
                    
                    # Get prediction
                    prediction = trading_system.get_ml_prediction(test_symbol, data)
                    print(f"   📊 ML Prediction: {prediction:.1f}/100")
                else:
                    print(f"   ⚠️ Model training failed for {test_symbol}")
            else:
                print(f"   ⚠️ Insufficient data for {test_symbol}")
        
        # Demonstrate technical analysis
        print("\\n📈 Technical Analysis Demo...")
        if universe and len(data) > 50:
            tech_signals = trading_system.calculate_technical_signals(test_symbol, data)
            print(f"   Signal: {tech_signals.get('signal', 'HOLD')}")
            print(f"   Strength: {tech_signals.get('strength', 0):.2f}")
            print(f"   RSI: {tech_signals.get('rsi', 0):.1f}")
            print(f"   Momentum: {tech_signals.get('momentum', 0):.3f}")
        
        # Demonstrate composite signal generation
        print("\\n🎯 Composite Signal Generation...")
        if universe and len(data) > 50:
            current_price = float(data['Close'].iloc[-1])
            
            # Get options recommendations
            options_rec = trading_system.get_options_recommendations(
                test_symbol, current_price, prediction, tech_signals
            )
            
            # Generate composite signal
            composite = trading_system.calculate_composite_signal(
                test_symbol, prediction, tech_signals, options_rec
            )
            
            print(f"   Composite Signal: {composite.get('signal', 'HOLD')}")
            print(f"   Strength: {composite.get('strength', 0):.2f}")
            print(f"   Conviction: {composite.get('conviction', 0):.2f}")
            
            if options_rec:
                print(f"   Options Strategy: {options_rec.get('strategy', 'none')}")
        
        # Test Alpaca connection (if available)
        print("\\n🔗 Alpaca Connection Test...")
        if trading_system.alpaca_connected:
            account = trading_system.trading_client.get_account()
            print(f"   ✅ Connected to Alpaca Paper Trading")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        else:
            print("   ⚠️ Alpaca not connected (set ALPACA_API_KEY and ALPACA_SECRET_KEY)")
        
        # Demonstrate risk management
        print("\\n🛡️ Risk Management Configuration...")
        risk_config = trading_system.config['risk_management']
        print(f"   Max Daily Loss: {risk_config['max_daily_loss']*100:.1f}%")
        print(f"   Max Drawdown: {risk_config['max_drawdown']*100:.1f}%")
        print(f"   Stop Loss: {risk_config['stop_loss_atr']:.1f}x ATR")
        print(f"   Max Position Size: {trading_system.config['portfolio']['max_position_size']*100:.0f}%")
        
        print("\\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("\\n📋 NEXT STEPS:")
        print("1. Set up Alpaca API keys for live trading")
        print("2. Run backtest validation: python comprehensive_backtester.py")
        print("3. Execute live trading: python unified_live_runner.py --mode single")
        print("4. For automated daily trading: python unified_live_runner.py --mode live")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\\nMissing dependencies detected. Install with:")
        print("pip install yfinance pandas numpy scikit-learn lightgbm alpaca-py")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\\nFull error traceback:")
        traceback.print_exc()
        return False

def show_architecture():
    """Show the unified system architecture"""
    
    print("\\n🏗️ UNIFIED SYSTEM ARCHITECTURE")
    print("=" * 60)
    
    architecture = """
    📊 DATA LAYER
    ├── Elite Stock Selector (AI-powered screening)
    ├── Market Data (yfinance, Alpaca)
    └── Technical Indicators (RSI, MA, momentum, volatility)
    
    🤖 ML LAYER  
    ├── Ensemble Models (RandomForest, GradientBoosting, LightGBM)
    ├── Feature Engineering (17+ technical features)
    ├── Cross-validation (Time series, purged, embargoed)
    └── Model Calibration & Drift Detection
    
    📈 SIGNAL LAYER
    ├── ML Predictions (40% weight)
    ├── Technical Signals (30% weight)  
    ├── Options Strategies (30% weight)
    └── Composite Signal Generation
    
    💼 PORTFOLIO LAYER
    ├── Position Sizing (Kelly-capped, volatility-scaled)
    ├── Risk Management (Stop loss, drawdown limits)
    ├── Sector/Concentration Limits
    └── Transaction Cost Modeling
    
    🎯 EXECUTION LAYER
    ├── Alpaca Paper Trading Integration
    ├── Smart Order Routing
    ├── Real-time Monitoring
    └── Performance Attribution
    
    📊 TARGETS
    ├── 30%+ Annual Returns
    ├── Sharpe Ratio > 1.0
    ├── Max Drawdown < 10%
    └── Win Rate > 55%
    """
    
    print(architecture)

def main():
    """Main demo execution"""
    
    print("🌟 Welcome to the Unified ML Trading System!")
    print("This consolidates all your sophisticated trading components.")
    print()
    
    # Show architecture
    show_architecture()
    
    # Run demo
    success = run_demo()
    
    if success:
        print("\\n✨ Your sophisticated trading infrastructure is now unified!")
        print("🎯 Ready to target 30%+ annual returns with institutional-grade risk management.")
    else:
        print("\\n🔧 Please address the issues above and try again.")

if __name__ == "__main__":
    main()
