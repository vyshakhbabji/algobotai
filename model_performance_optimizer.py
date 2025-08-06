#!/usr/bin/env python3
"""
Model Performance Optimizer
Advanced fine-tuning and optimization for AI trading models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceOptimizer:
    """Advanced model optimization with hyperparameter tuning"""
    
    def __init__(self, optimize_level='comprehensive'):
        """
        Initialize optimizer
        
        Args:
            optimize_level: 'basic', 'advanced', 'comprehensive'
        """
        self.optimize_level = optimize_level
        self.best_models = {}
        self.optimization_results = {}
        
    def optimize_random_forest(self, X_train, y_train, X_val, y_val, n_trials=100):
        """Optimize Random Forest using Optuna"""
        print("🌳 Optimizing Random Forest...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            return r2_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', study_name='rf_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        return best_model, best_params, study.best_value
    
    def optimize_gradient_boosting(self, X_train, y_train, X_val, y_val, n_trials=100):
        """Optimize Gradient Boosting using Optuna"""
        print("🚀 Optimizing Gradient Boosting...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'random_state': 42
            }
            
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            return r2_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', study_name='gb_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_model = GradientBoostingRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        return best_model, best_params, study.best_value
    
    def optimize_ensemble_weights(self, models, X_val, y_val, n_trials=50):
        """Optimize ensemble model weights"""
        print("⚖️ Optimizing ensemble weights...")
        
        def objective(trial):
            weights = []
            for i in range(len(models)):
                weights.append(trial.suggest_float(f'weight_{i}', 0.0, 1.0))
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Ensemble prediction
            predictions = []
            for i, model in enumerate(models):
                pred = model.predict(X_val)
                predictions.append(pred * weights[i])
            
            ensemble_pred = np.sum(predictions, axis=0)
            
            return r2_score(y_val, ensemble_pred)
        
        study = optuna.create_study(direction='maximize', study_name='ensemble_weights')
        study.optimize(objective, n_trials=n_trials)
        
        # Extract best weights
        best_weights = []
        for i in range(len(models)):
            best_weights.append(study.best_params[f'weight_{i}'])
        
        best_weights = np.array(best_weights)
        best_weights = best_weights / np.sum(best_weights)
        
        return best_weights, study.best_value
    
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering for better model performance"""
        print("🔧 Advanced feature engineering...")
        
        # Price-based features
        df['price_momentum_5'] = df['Close'].pct_change(5)
        df['price_momentum_10'] = df['Close'].pct_change(10)
        df['price_momentum_20'] = df['Close'].pct_change(20)
        
        # Volatility features
        df['volatility_5'] = df['Close'].rolling(5).std()
        df['volatility_10'] = df['Close'].rolling(10).std()
        df['volatility_20'] = df['Close'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Volume features
        df['volume_sma_5'] = df['Volume'].rolling(5).mean()
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['price_volume'] = df['Close'] * df['Volume']
        
        # Technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Support and Resistance levels
        df['support'] = df['Low'].rolling(20).min()
        df['resistance'] = df['High'].rolling(20).max()
        df['support_distance'] = (df['Close'] - df['support']) / df['Close']
        df['resistance_distance'] = (df['resistance'] - df['Close']) / df['Close']
        
        # Trend features
        df['trend_5'] = np.where(df['Close'] > df['Close'].shift(5), 1, -1)
        df['trend_10'] = np.where(df['Close'] > df['Close'].shift(10), 1, -1)
        df['trend_20'] = np.where(df['Close'] > df['Close'].shift(20), 1, -1)
        
        # Price position in range
        df['price_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                              (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        return df
    
    def optimize_preprocessing(self, X_train, y_train, X_val, y_val):
        """Optimize preprocessing techniques"""
        print("📊 Optimizing preprocessing...")
        
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'none': None
        }
        
        best_scaler = None
        best_score = -np.inf
        
        for scaler_name, scaler in scalers.items():
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Quick model test
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            score = r2_score(y_val, y_pred)
            
            print(f"  {scaler_name}: R² = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_scaler = scaler
        
        print(f"  ✅ Best scaler: {type(best_scaler).__name__ if best_scaler else 'None'}")
        return best_scaler
    
    def comprehensive_optimization(self, X_train, y_train, X_val, y_val):
        """Run comprehensive model optimization"""
        print("🎯 COMPREHENSIVE MODEL OPTIMIZATION")
        print("=" * 60)
        
        results = {}
        
        # 1. Optimize preprocessing
        best_scaler = self.optimize_preprocessing(X_train, y_train, X_val, y_val)
        
        if best_scaler:
            X_train_scaled = best_scaler.fit_transform(X_train)
            X_val_scaled = best_scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # 2. Optimize individual models
        if self.optimize_level == 'comprehensive':
            n_trials = 100
        elif self.optimize_level == 'advanced':
            n_trials = 50
        else:
            n_trials = 20
        
        # Random Forest
        rf_model, rf_params, rf_score = self.optimize_random_forest(
            X_train_scaled, y_train, X_val_scaled, y_val, n_trials
        )
        
        # Gradient Boosting
        gb_model, gb_params, gb_score = self.optimize_gradient_boosting(
            X_train_scaled, y_train, X_val_scaled, y_val, n_trials
        )
        
        # 3. Optimize ensemble
        models = [rf_model, gb_model]
        ensemble_weights, ensemble_score = self.optimize_ensemble_weights(
            models, X_val_scaled, y_val
        )
        
        # 4. Results
        results = {
            'scaler': best_scaler,
            'random_forest': {
                'model': rf_model,
                'params': rf_params,
                'score': rf_score
            },
            'gradient_boosting': {
                'model': gb_model,
                'params': gb_params,
                'score': gb_score
            },
            'ensemble': {
                'weights': ensemble_weights,
                'score': ensemble_score,
                'models': models
            }
        }
        
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        print(f"  Random Forest R²: {rf_score:.4f}")
        print(f"  Gradient Boosting R²: {gb_score:.4f}")
        print(f"  Ensemble R²: {ensemble_score:.4f}")
        print(f"  Best approach: {'Ensemble' if ensemble_score > max(rf_score, gb_score) else 'RF' if rf_score > gb_score else 'GB'}")
        
        return results
    
    def save_optimized_models(self, results, symbol, save_dir='optimized_models'):
        """Save optimized models and parameters"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        joblib.dump(results['random_forest']['model'], f"{save_dir}/{symbol}_rf_optimized.pkl")
        joblib.dump(results['gradient_boosting']['model'], f"{save_dir}/{symbol}_gb_optimized.pkl")
        
        if results['scaler']:
            joblib.dump(results['scaler'], f"{save_dir}/{symbol}_scaler_optimized.pkl")
        
        # Save parameters
        import json
        params = {
            'random_forest': results['random_forest']['params'],
            'gradient_boosting': results['gradient_boosting']['params'],
            'ensemble_weights': results['ensemble']['weights'].tolist(),
            'scores': {
                'rf': results['random_forest']['score'],
                'gb': results['gradient_boosting']['score'],
                'ensemble': results['ensemble']['score']
            }
        }
        
        with open(f"{save_dir}/{symbol}_optimized_params.json", 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"✅ Saved optimized models for {symbol}")

def demonstrate_optimization():
    """Demonstrate the optimization process"""
    print("🚀 DEMONSTRATING MODEL OPTIMIZATION")
    print("=" * 60)
    
    # Create sample data (in real use, this would be your actual stock data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.02  # Stock returns
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Initialize optimizer
    optimizer = ModelPerformanceOptimizer(optimize_level='basic')  # Use 'basic' for demo
    
    # Run optimization
    results = optimizer.comprehensive_optimization(X_train, y_train, X_val, y_val)
    
    # Save results
    optimizer.save_optimized_models(results, 'DEMO')
    
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    best_score = max(
        results['random_forest']['score'],
        results['gradient_boosting']['score'],
        results['ensemble']['score']
    )
    
    if results['ensemble']['score'] == best_score:
        print("  ✅ Use ensemble model for best performance")
        print(f"  📊 Ensemble weights: RF={results['ensemble']['weights'][0]:.3f}, GB={results['ensemble']['weights'][1]:.3f}")
    elif results['random_forest']['score'] == best_score:
        print("  ✅ Use Random Forest model")
    else:
        print("  ✅ Use Gradient Boosting model")
    
    print(f"  🎯 Expected R² Score: {best_score:.4f}")

if __name__ == "__main__":
    demonstrate_optimization()
