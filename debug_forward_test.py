#!/usr/bin/env python3
"""
Debug version of the forward test to see why no trades are being made
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from algobot.features.advanced import build_advanced_features, ADV_FEATURE_COLUMNS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def debug_forward_test():
    """Debug version with verbose output"""
    
    print("Debug Forward Test for NVDA")
    print("=" * 50)
    
    # Get data
    start_date = "2024-05-13"
    end_date = "2024-08-13"
    lookback_days = 252
    
    print(f"Test period: {start_date} to {end_date}")
    
    # Download full data (training + test)
    full_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days * 1.5)
    full_data = yf.download('NVDA', start=full_start.strftime('%Y-%m-%d'), end=end_date, progress=False)
    test_data = yf.download('NVDA', start=start_date, end=end_date, progress=False)
    
    print(f"Full data: {len(full_data)} days")
    print(f"Test data: {len(test_data)} days")
    
    # Build features
    print("\nBuilding features...")
    feat_df = build_advanced_features(full_data)
    print(f"Features built: {feat_df.shape}")
    
    # Get test dates
    test_dates = test_data.index
    print(f"Test dates: {len(test_dates)} days")
    
    # Simulate the forward test process
    cash = 100000.0
    holdings = 0.0
    trades = []
    
    print(f"\nStarting simulation with ${cash:,.2f}")
    print(f"Parameters:")
    print(f"  - Buy threshold: 0.60")
    print(f"  - Exit threshold: 0.50") 
    print(f"  - Hard exit threshold: 0.45")
    print(f"  - Lookback days: {lookback_days}")
    
    model = None
    last_train_month = None
    
    for i, current_date in enumerate(test_dates[:10]):  # Test first 10 days
        print(f"\n--- {current_date.strftime('%Y-%m-%d')} (Day {i+1}) ---")
        
        # Check if we have enough historical data
        hist_data = feat_df.loc[:current_date]
        if len(hist_data) < lookback_days:
            print(f"  Not enough historical data: {len(hist_data)} < {lookback_days}")
            continue
            
        # Get current price
        if current_date in full_data.index:
            current_price = float(full_data.loc[current_date, 'Close'])
            print(f"  Current price: ${current_price:.2f}")
        else:
            print(f"  No price data for {current_date}")
            continue
        
        # Check if we need to retrain (monthly)
        current_month = current_date.to_period('M')
        if last_train_month is None or current_month != last_train_month:
            print(f"  Retraining model for month {current_month}")
            
            # Get training window
            train_window = hist_data.iloc[-lookback_days:]
            X_train = train_window[ADV_FEATURE_COLUMNS]
            y_train = train_window['target_up']
            
            print(f"    Training samples: {len(X_train)}")
            print(f"    Up days in training: {y_train.mean():.2%}")
            
            # Train models
            gb = GradientBoostingClassifier(random_state=42)
            lr = LogisticRegression(max_iter=500)
            
            gb.fit(X_train, y_train)
            lr.fit(X_train, y_train)
            
            model = {'gb': gb, 'lr': lr}
            last_train_month = current_month
            print(f"    ✓ Model retrained")
        
        # Make prediction
        if model and current_date in feat_df.index:
            current_features = feat_df.loc[[current_date]][ADV_FEATURE_COLUMNS]
            
            p_gb = model['gb'].predict_proba(current_features)[:, 1]
            p_lr = model['lr'].predict_proba(current_features)[:, 1]
            prob = float(0.5 * p_gb[0] + 0.5 * p_lr[0])
            
            print(f"  Prediction: {prob:.4f} (GB: {p_gb[0]:.4f}, LR: {p_lr[0]:.4f})")
            
            # Current position value
            position_value = holdings * current_price
            total_equity = cash + position_value
            current_weight = position_value / total_equity if total_equity > 0 else 0
            
            print(f"  Current position: {holdings:.4f} shares (${position_value:,.2f}, {current_weight:.2%})")
            print(f"  Total equity: ${total_equity:,.2f}")
            
            # Trading logic
            if holdings == 0:  # No position
                if prob >= 0.60:  # Buy signal
                    # Calculate position size (for simplicity, use all cash)
                    shares_to_buy = cash / current_price
                    cost = shares_to_buy * current_price
                    
                    if cost <= cash:
                        holdings += shares_to_buy
                        cash -= cost
                        
                        trade = {
                            'date': current_date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost,
                            'prob': prob
                        }
                        trades.append(trade)
                        
                        print(f"  🟢 BUY: {shares_to_buy:.4f} shares at ${current_price:.2f}")
                        print(f"     Cost: ${cost:,.2f}, Remaining cash: ${cash:,.2f}")
                    else:
                        print(f"  ❌ Buy signal but insufficient cash")
                else:
                    print(f"  ⚪ No buy signal (prob {prob:.4f} < 0.60)")
            
            else:  # Have position
                if prob <= 0.45:  # Hard exit
                    proceeds = holdings * current_price
                    cash += proceeds
                    
                    trade = {
                        'date': current_date,
                        'action': 'SELL',
                        'shares': holdings,
                        'price': current_price,
                        'proceeds': proceeds,
                        'prob': prob
                    }
                    trades.append(trade)
                    
                    print(f"  🔴 HARD EXIT: {holdings:.4f} shares at ${current_price:.2f}")
                    print(f"     Proceeds: ${proceeds:,.2f}, Total cash: ${cash:,.2f}")
                    
                    holdings = 0
                    
                elif prob <= 0.50:  # Regular exit
                    proceeds = holdings * current_price
                    cash += proceeds
                    
                    trade = {
                        'date': current_date,
                        'action': 'SELL',
                        'shares': holdings,
                        'price': current_price,
                        'proceeds': proceeds,
                        'prob': prob
                    }
                    trades.append(trade)
                    
                    print(f"  🔴 EXIT: {holdings:.4f} shares at ${current_price:.2f}")
                    print(f"     Proceeds: ${proceeds:,.2f}, Total cash: ${cash:,.2f}")
                    
                    holdings = 0
                    
                else:
                    print(f"  🟡 HOLD position (prob {prob:.4f} > 0.50)")
        
        else:
            print(f"  ❌ No model or features available")
    
    # Final summary
    print(f"\n" + "=" * 50)
    print(f"SUMMARY")
    print(f"=" * 50)
    
    final_position_value = holdings * current_price if holdings > 0 else 0
    final_total = cash + final_position_value
    total_return = (final_total - 100000) / 100000
    
    print(f"Final cash: ${cash:,.2f}")
    print(f"Final position: {holdings:.4f} shares (${final_position_value:,.2f})")
    print(f"Final total: ${final_total:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Number of trades: {len(trades)}")
    
    if trades:
        print(f"\nTrades:")
        for trade in trades:
            print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} {trade.get('shares', 0):.4f} shares at ${trade.get('price', 0):.2f} (prob: {trade['prob']:.4f})")


if __name__ == '__main__':
    debug_forward_test()
