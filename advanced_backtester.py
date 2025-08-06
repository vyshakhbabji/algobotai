#!/usr/bin/env python3
"""
Advanced Backtesting System
Test prediction accuracy using walk-forward analysis
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktester:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_models(self):
        """Load trained models"""
        print("🤖 Loading trained models...")
        
        try:
            self.scalers['feature'] = joblib.load('fixed_data/preprocessed/feature_scaler.pkl')
            self.scalers['target'] = joblib.load('fixed_data/preprocessed/target_scaler.pkl')
            
            model_files = {
                'rf': 'fixed_data/models/random_forest_model.pkl',
                'gb': 'fixed_data/models/gradient_boosting_model.pkl',
                'linear': 'fixed_data/models/linear_regression_model.pkl',
                'ridge': 'fixed_data/models/ridge_model.pkl'
            }
            
            for name, path in model_files.items():
                try:
                    self.models[name] = joblib.load(path)
                    print(f"  ✅ {name.upper()} loaded")
                except Exception as e:
                    print(f"  ❌ {name.upper()}: {e}")
            
            print(f"📊 Successfully loaded {len(self.models)} models")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def fetch_extended_data(self, symbol='NVDA', period='1y'):
        """Fetch extended historical data for backtesting"""
        print(f"📡 Fetching extended data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) == 0:
                print("❌ No data retrieved")
                return None
            
            print(f"📈 Retrieved {len(hist)} days of data")
            print(f"📅 Date range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
            
            return hist
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators like in training"""
        print("🔧 Calculating technical indicators...")
        
        df = data.copy()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # EMAs with specific spans to match training
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Also create the ones we were calculating before
        for window in [5, 10, 20, 50]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # On Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # Price-based features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Returns
        for period in [1, 2, 3, 5, 10]:
            df[f'Return_{period}d'] = df['Close'].pct_change(period) * 100
        
        # Volatility
        for window in [5, 10, 20]:
            df[f'Volatility_{window}d'] = df['Close'].pct_change().rolling(window=window).std() * np.sqrt(252) * 100
        
        # Target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        print(f"✅ Calculated technical indicators, dataset now has {df.shape[1]} columns")
        return df
    
    def prepare_features(self, df):
        """Prepare features exactly like in training"""
        print("🔧 Preparing features for prediction...")
        
        # Load the exact feature columns from training
        try:
            with open('fixed_data/preprocessed/metadata.json', 'r') as f:
                metadata = json.load(f)
                feature_columns = metadata['feature_columns']
                print(f"📋 Loaded {len(feature_columns)} features from training metadata")
        except:
            print("⚠️ Could not load metadata, using manual feature list")
            # Fallback to manual feature list matching training
            feature_columns = [
                "Open","High","Low","Volume","Returns","HL_PCT","PC_PCT","SMA_10","SMA_20","SMA_50",
                "EMA_12","EMA_26","RSI_14","MACD","MACD_signal","MACD_hist","BB_upper","BB_middle",
                "BB_lower","BB_width","BB_position","Stoch_k","Stoch_d","ATR","OBV","VWAP",
                "Volume_SMA","Volume_ratio","Price_above_SMA20","Price_above_SMA50","RSI_oversold",
                "RSI_overbought","Price_momentum_5","Price_momentum_10","High_20","Low_20",
                "Distance_to_high","Distance_to_low"
            ]
        
        # Map our calculated features to training feature names
        feature_mapping = {
            'Returns': 'Return_1d',
            'HL_PCT': 'High_Low_Pct', 
            'PC_PCT': 'Open_Close_Pct',
            'SMA_10': 'SMA_10',
            'SMA_20': 'SMA_20', 
            'SMA_50': 'SMA_50',
            'EMA_12': 'EMA_12',
            'EMA_26': 'EMA_26',
            'RSI_14': 'RSI',
            'MACD': 'MACD',
            'MACD_signal': 'MACD_Signal',
            'MACD_hist': 'MACD_Histogram',
            'BB_upper': 'BB_Upper',
            'BB_middle': 'BB_Middle', 
            'BB_lower': 'BB_Lower',
            'BB_width': 'BB_Width',
            'BB_position': 'BB_Position',
            'Stoch_k': 'Stoch_K',
            'Stoch_d': 'Stoch_D',
            'Volume_ratio': 'Volume_Ratio'
        }
        
        # Create feature matrix with proper names
        df_features = df.copy()
        
        # Add missing features that training expects
        df_features['Returns'] = df_features['Close'].pct_change()
        df_features['Price_above_SMA20'] = (df_features['Close'] > df_features['SMA_20']).astype(int)
        df_features['Price_above_SMA50'] = (df_features['Close'] > df_features['SMA_50']).astype(int)
        df_features['RSI_oversold'] = (df_features['RSI'] < 30).astype(int)
        df_features['RSI_overbought'] = (df_features['RSI'] > 70).astype(int)
        df_features['Price_momentum_5'] = df_features['Return_5d']
        df_features['Price_momentum_10'] = df_features['Return_10d']
        df_features['High_20'] = df_features['High'].rolling(window=20).max()
        df_features['Low_20'] = df_features['Low'].rolling(window=20).min()
        df_features['Distance_to_high'] = (df_features['High_20'] - df_features['Close']) / df_features['Close']
        df_features['Distance_to_low'] = (df_features['Close'] - df_features['Low_20']) / df_features['Close']
        
        # Rename features to match training
        for new_name, old_name in feature_mapping.items():
            if old_name in df_features.columns:
                df_features[new_name] = df_features[old_name]
        
        # Select only the features used in training
        try:
            df_clean = df_features[feature_columns].dropna()
            print(f"✅ Prepared {len(df_clean)} samples with {len(feature_columns)} features")
            return df_clean, df.loc[df_clean.index, 'Target'].dropna()
        except KeyError as e:
            print(f"❌ Missing feature: {e}")
            available_features = [f for f in feature_columns if f in df_features.columns]
            print(f"📋 Available features: {len(available_features)} of {len(feature_columns)}")
            df_clean = df_features[available_features].dropna()
            return df_clean, df.loc[df_clean.index, 'Target'].dropna()
    
    def walk_forward_backtest(self, data):
        """
        Simulate the scenario: Train until June, predict July-August
        """
        print(f"\n🚀 Starting Walk-Forward Backtest: Train until June, Test July-August")
        
        # Prepare data
        df_with_indicators = self.calculate_technical_indicators(data)
        X, y = self.prepare_features(df_with_indicators)
        
        if len(X) == 0:
            print("❌ No valid data for backtesting")
            return None
        
        # Ensure X and y have same index
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Convert dates to datetime for filtering
        dates = pd.to_datetime(X.index)
        
        # Find cutoff date (end of June 2025) - make timezone aware to match data
        cutoff_date = pd.Timestamp('2025-06-30', tz=dates.tz)
        
        # Split data
        train_mask = dates <= cutoff_date
        test_mask = dates > cutoff_date
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        test_dates = dates[test_mask]
        
        print(f"📅 Training period: {dates[train_mask].min().strftime('%Y-%m-%d')} to {dates[train_mask].max().strftime('%Y-%m-%d')}")
        print(f"📅 Testing period: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}")
        print(f"🎯 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
        
        if len(X_test) == 0:
            print("❌ No test data available for July-August period")
            return None
        
        if len(X_test) < 10:
            print(f"❌ Insufficient test data ({len(X_test)} samples) - need at least 10 for sequences")
            return None
        
        # Always use sequence data since models expect sequence input
        print("🔧 Creating sequences for model compatibility...")
        
        sequence_length = 10
        
        # Simple approach: create sequences from the full dataset
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        dates_full = pd.concat([pd.Series(dates[train_mask]), pd.Series(test_dates)])
        
        # Create all sequences
        X_sequences = []
        y_sequences = []
        seq_dates = []
        seq_is_test = []
        
        for i in range(sequence_length, len(X_full)):
            X_sequences.append(X_full.iloc[i-sequence_length:i].values)
            y_sequences.append(y_full.iloc[i])
            seq_dates.append(dates_full.iloc[i])
            seq_is_test.append(dates_full.iloc[i] in test_dates.values)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        seq_dates = pd.to_datetime(seq_dates)
        seq_is_test = np.array(seq_is_test)
        
        # Split into train and test
        train_mask_seq = ~seq_is_test
        test_mask_seq = seq_is_test
        
        X_train_flat = X_sequences[train_mask_seq].reshape(np.sum(train_mask_seq), -1)
        y_train_seq = y_sequences[train_mask_seq]
        X_test_flat = X_sequences[test_mask_seq].reshape(np.sum(test_mask_seq), -1)
        y_test_seq = y_sequences[test_mask_seq]
        test_dates_final = seq_dates[test_mask_seq]
        
        if len(X_test_flat) == 0:
            print("❌ No test sequences created")
            return None
        
        # Create sequences for training
        sequence_length = 10
        X_train_seq = []
        y_train_seq = []
        for i in range(sequence_length, len(X_train)):
            X_train_seq.append(X_train.iloc[i-sequence_length:i].values)
            y_train_seq.append(y_train.iloc[i])
        
        if len(X_train_seq) == 0:
            print("❌ Insufficient training data for sequences")
            return None
            
        X_train_seq = np.array(X_train_seq)
        y_train_seq = np.array(y_train_seq)
        
        # Create sequences for testing - simplified approach
        X_test_seq = []
        y_test_seq = []
        test_dates_seq = []
        
        # Use all available data for creating test sequences
        all_X = pd.concat([X_train, X_test])
        all_y = pd.concat([y_train, y_test])
        all_dates = pd.concat([pd.Series(dates[train_mask]), pd.Series(test_dates)])
        
        # Create sequences that end in the test period
        for i in range(sequence_length, len(all_X)):
            current_date = all_dates.iloc[i]
            if current_date in test_dates.values:  # This is a test date
                X_test_seq.append(all_X.iloc[i-sequence_length:i].values)
                y_test_seq.append(all_y.iloc[i])
                test_dates_seq.append(current_date)
        
        if len(X_test_seq) == 0:
            print("❌ Could not create test sequences")
            return None
        
        X_test_seq = np.array(X_test_seq)
        y_test_seq = np.array(y_test_seq)
        test_dates_seq = pd.to_datetime(test_dates_seq)
        
        # Flatten sequences for sklearn models (they expect 2D input)
        X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
        X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], -1)
        
        print(f"📊 Final data: {len(X_train_flat)} train, {len(X_test_flat)} test samples")
        print(f"📐 Feature dimensions: {X_train_flat.shape[1]} (10 days × {len(X_train.columns)} features)")
        
        test_dates_final = test_dates_final
        
        # Test each model
        results = {
            'train_period': (dates[train_mask].min(), dates[train_mask].max()),
            'test_period': (test_dates_final.min(), test_dates_final.max()),
            'actual_prices': y_test_seq,
            'test_dates': test_dates_final,
            'models': {}
        }
        
        for model_name, model in self.models.items():
            try:
                print(f"\n🧠 Testing {model_name.upper()} model...")
                
                # Make predictions
                predictions = model.predict(X_test_flat)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test_seq, predictions)
                rmse = np.sqrt(mean_squared_error(y_test_seq, predictions))
                r2 = r2_score(y_test_seq, predictions)
                
                # Calculate direction accuracy
                if len(y_test_seq) > 1:
                    actual_direction = np.sign(np.diff(y_test_seq))
                    pred_direction = np.sign(np.diff(predictions))
                    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                else:
                    direction_accuracy = 0
                
                # Calculate price movement accuracy
                actual_start = y_test_seq[0]
                actual_end = y_test_seq[-1]
                pred_start = predictions[0]
                pred_end = predictions[-1]
                
                actual_total_change = ((actual_end - actual_start) / actual_start) * 100
                pred_total_change = ((pred_end - pred_start) / pred_start) * 100
                
                direction_match = np.sign(actual_total_change) == np.sign(pred_total_change)
                
                results['models'][model_name] = {
                    'predictions': predictions,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'actual_total_change': actual_total_change,
                    'pred_total_change': pred_total_change,
                    'direction_match': direction_match,
                    'actual_start_price': actual_start,
                    'actual_end_price': actual_end,
                    'pred_start_price': pred_start,
                    'pred_end_price': pred_end
                }
                
                print(f"  📊 MAE: ${mae:.2f}")
                print(f"  📊 Direction Accuracy: {direction_accuracy:.1f}%")
                print(f"  📊 Actual Change: {actual_total_change:+.1f}%")
                print(f"  📊 Predicted Change: {pred_total_change:+.1f}%")
                print(f"  📊 Direction Match: {'✅' if direction_match else '❌'}")
                
            except Exception as e:
                print(f"  ❌ {model_name.upper()}: {e}")
        
        self.results = results
        return results
    
    def analyze_results(self):
        """Analyze backtest results"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print(f"\n📊 DETAILED BACKTESTING ANALYSIS")
        print(f"{'='*80}")
        
        train_start, train_end = self.results['train_period']
        test_start, test_end = self.results['test_period']
        
        print(f"📅 Training Period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"📅 Testing Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        print(f"📅 Testing Duration: {(test_end - test_start).days} days")
        
        actual_prices = self.results['actual_prices']
        actual_start = actual_prices[0]
        actual_end = actual_prices[-1]
        actual_change = ((actual_end - actual_start) / actual_start) * 100
        
        print(f"\n💰 ACTUAL MARKET PERFORMANCE:")
        print(f"   Start Price: ${actual_start:.2f}")
        print(f"   End Price: ${actual_end:.2f}")
        print(f"   Total Change: {actual_change:+.1f}%")
        
        print(f"\n🤖 MODEL PREDICTIONS vs REALITY:")
        print(f"{'Model':<12} {'Pred Change':<12} {'Actual':<12} {'Direction':<10} {'MAE':<10} {'Accuracy':<10}")
        print(f"{'-'*80}")
        
        best_model = None
        best_mae = float('inf')
        
        for model_name, metrics in self.results['models'].items():
            pred_change = metrics['pred_total_change']
            direction_match = "✅ Correct" if metrics['direction_match'] else "❌ Wrong"
            mae = metrics['mae']
            direction_acc = metrics['direction_accuracy']
            
            print(f"{model_name.upper():<12} "
                  f"{pred_change:+8.1f}% "
                  f"{actual_change:+8.1f}% "
                  f"{direction_match:<10} "
                  f"${mae:<9.2f} "
                  f"{direction_acc:<9.1f}%")
            
            if mae < best_mae:
                best_mae = mae
                best_model = model_name
        
        # Only analyze if we have successful model results
        if not self.results['models']:
            print("❌ No successful model predictions to analyze")
            return None
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model.upper() if best_model else 'None'}")
        if best_model:
            best_metrics = self.results['models'][best_model]
            print(f"   📊 Average Error: ${best_metrics['mae']:.2f}")
            print(f"   📊 Direction Accuracy: {best_metrics['direction_accuracy']:.1f}%")
            print(f"   📊 Predicted vs Actual: {best_metrics['pred_total_change']:+.1f}% vs {actual_change:+.1f}%")
        
        # Calculate what would have happened with trading
        if best_model:
            best_metrics = self.results['models'][best_model]
            print(f"\n💼 TRADING SIMULATION:")
            print(f"   If you invested $10,000 based on {best_model.upper()} prediction:")
            
            if best_metrics['direction_match']:
                # Correct direction prediction
                if actual_change > 0:
                    profit = 10000 * (actual_change / 100)
                    print(f"   ✅ Predicted UP, Market went UP")
                    print(f"   💰 Result: ${10000 + profit:.2f} (+{profit:.2f})")
                else:
                    # Correctly predicted down, assume we shorted or stayed out
                    print(f"   ✅ Predicted DOWN, Market went DOWN") 
                    print(f"   💰 Result: Avoided loss of ${abs(10000 * actual_change / 100):.2f}")
            else:
                # Wrong direction prediction
                if actual_change > 0:
                    loss = 10000 * (actual_change / 100)
                    print(f"   ❌ Predicted DOWN, Market went UP")
                    print(f"   💸 Result: Missed profit of ${loss:.2f}")
                else:
                    loss = 10000 * (abs(actual_change) / 100)
                    print(f"   ❌ Predicted UP, Market went DOWN")
                    print(f"   💸 Result: Lost ${loss:.2f}")
            
            # Compare to benchmark
            print(f"\n📈 BENCHMARK COMPARISON:")
            print(f"   Buy & Hold Strategy: {actual_change:+.1f}%")
            print(f"   Best Model Prediction: {best_metrics['pred_total_change']:+.1f}%")
            print(f"   Prediction Error: {abs(best_metrics['pred_total_change'] - actual_change):.1f} percentage points")
        
        # Model sophistication assessment
        print(f"\n🎯 SOPHISTICATION ASSESSMENT:")
        direction_accuracy = np.mean([m['direction_accuracy'] for m in self.results['models'].values()])
        avg_mae = np.mean([m['mae'] for m in self.results['models'].values()])
        
        if direction_accuracy > 60:
            sophistication = "HIGH"
        elif direction_accuracy > 50:
            sophistication = "MEDIUM"
        else:
            sophistication = "LOW"
        
        print(f"   📊 Average Direction Accuracy: {direction_accuracy:.1f}%")
        print(f"   📊 Average Price Error: ${avg_mae:.2f}")
        print(f"   📊 Overall Sophistication: {sophistication}")
        
        if direction_accuracy > 50:
            print(f"   ✅ Your models beat random chance!")
        else:
            print(f"   ❌ Models need improvement to beat random chance")
        
        return {
            'best_model': best_model,
            'best_mae': best_mae,
            'direction_accuracy': direction_accuracy,
            'sophistication': sophistication,
            'actual_change': actual_change
        }
    
    def create_visualization(self):
        """Create visualization of results"""
        if not self.results or not self.results['models']:
            print("❌ No model results to visualize")
            return
        
        print(f"\n📊 Creating detailed visualization...")
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        dates = self.results['test_dates']
        actual_prices = self.results['actual_prices']
        
        # Plot 1: Price predictions vs actual
        ax1 = axes[0, 0]
        ax1.plot(dates, actual_prices, label='Actual Price', linewidth=3, color='black')
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, metrics) in enumerate(self.results['models'].items()):
            predictions = metrics['predictions']
            ax1.plot(dates, predictions, label=f'{model_name.upper()} Prediction', 
                    alpha=0.8, linewidth=2, color=colors[i % len(colors)])
        
        ax1.set_title('Price Predictions vs Actual (Test Period)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add annotations
        start_price = actual_prices[0]
        end_price = actual_prices[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        ax1.annotate(f'Actual Change: {change_pct:+.1f}%', 
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10, fontweight='bold', verticalalignment='top')
        
        # Plot 2: Prediction errors
        ax2 = axes[0, 1]
        for i, (model_name, metrics) in enumerate(self.results['models'].items()):
            predictions = metrics['predictions']
            errors = np.array(predictions) - np.array(actual_prices)
            ax2.plot(dates, errors, label=f'{model_name.upper()}', 
                    alpha=0.8, linewidth=2, color=colors[i % len(colors)])
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model performance comparison
        ax3 = axes[1, 0]
        model_names = list(self.results['models'].keys())
        maes = [self.results['models'][m]['mae'] for m in model_names]
        direction_accs = [self.results['models'][m]['direction_accuracy'] for m in model_names]
        
        x_pos = np.arange(len(model_names))
        ax3_twin = ax3.twinx()
        
        bars1 = ax3.bar(x_pos - 0.2, maes, 0.4, label='MAE ($)', alpha=0.7, color='skyblue')
        bars2 = ax3_twin.bar(x_pos + 0.2, direction_accs, 0.4, label='Direction Accuracy (%)', alpha=0.7, color='lightcoral')
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('MAE ($)', color='skyblue')
        ax3_twin.set_ylabel('Direction Accuracy (%)', color='lightcoral')
        ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([name.upper() for name in model_names])
        
        # Add value labels
        for bar, value in zip(bars1, maes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${value:.1f}', ha='center', va='bottom', fontsize=9)
        
        for bar, value in zip(bars2, direction_accs):
            height = bar.get_height()
            ax3_twin.text(bar.get_x() + bar.get_width()/2., height + 2,
                         f'{value:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Cumulative returns comparison
        ax4 = axes[1, 1]
        
        # Calculate daily returns
        if len(actual_prices) > 1:
            actual_returns = np.diff(actual_prices) / actual_prices[:-1] * 100
            actual_cumulative = np.cumsum(actual_returns)
            
            ax4.plot(dates[1:], actual_cumulative, label='Actual Returns', 
                    linewidth=3, color='black')
            
            for i, (model_name, metrics) in enumerate(self.results['models'].items()):
                predictions = metrics['predictions']
                if len(predictions) > 1:
                    pred_returns = np.diff(predictions) / predictions[:-1] * 100
                    pred_cumulative = np.cumsum(pred_returns)
                    
                    ax4.plot(dates[1:], pred_cumulative, label=f'{model_name.upper()} Predicted', 
                            alpha=0.8, linewidth=2, color=colors[i % len(colors)])
        
        ax4.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('fixed_data/results/backtest_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved to 'fixed_data/results/backtest_visualization.png'")

def main():
    """Run the June-to-August backtest"""
    print(f"🚀 JUNE-TO-AUGUST BACKTEST ANALYSIS")
    print(f"{'='*80}")
    print(f"📅 Scenario: Train models until June 30, predict July-August")
    print(f"🎯 Question: How accurate were predictions for the next 1-2 months?")
    
    backtester = AdvancedBacktester()
    
    # Load models
    if not backtester.load_models():
        return
    
    # Fetch extended data
    data = backtester.fetch_extended_data('NVDA', '1y')
    if data is None:
        return
    
    # Run the specific backtest
    results = backtester.walk_forward_backtest(data)
    
    if results:
        # Analyze results
        performance = backtester.analyze_results()
        
        # Create visualization
        backtester.create_visualization()
        
        print(f"\n✅ BACKTEST COMPLETE!")
        print(f"📊 Check 'fixed_data/results/june_to_august_backtest.png' for charts")
        
        return backtester, results, performance
    
    return None

if __name__ == "__main__":
    backtester, results, performance = main()
