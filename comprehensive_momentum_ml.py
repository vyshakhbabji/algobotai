#!/usr/bin/env python3
"""
COMPREHENSIVE MOMENTUM ML MODEL
Full institutional momentum features integrated with advanced machine learning
Uses complete feature set from our deployed momentum portfolio
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json
import joblib

class ComprehensiveMomentumMLModel:
    def __init__(self):
        # Our deployed momentum portfolio stocks (WINNING PORTFOLIO)
        self.momentum_stocks = ['AMD', 'GE', 'PLTR', 'MSFT', 'NVDA', 'JNJ', 'CAT', 'GOOGL']
        
        # Institutional momentum parameters (from deployed portfolio)
        self.momentum_periods = {
            'long_momentum': 126,    # 6 months (institutional standard)
            'medium_momentum': 63,   # 3 months (institutional standard)
            'short_momentum': 21,    # 1 month (institutional standard)
            'micro_momentum': 5,     # 1 week (tactical adjustment)
            'trend_momentum': 10     # 2 weeks (trend confirmation)
        }
        
        # Advanced ML ensemble
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150, 
                learning_rate=0.1, 
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        self.scaler = StandardScaler()
        self.trained_models = {}
        self.feature_names = []
        
    def calculate_comprehensive_momentum_features(self, data):
        """Calculate ALL institutional momentum features (no simplification)"""
        
        if len(data) < 150:
            return None
            
        close_prices = data['Close'].values
        volume_data = data['Volume'].values if 'Volume' in data.columns else None
        high_prices = data['High'].values if 'High' in data.columns else close_prices
        low_prices = data['Low'].values if 'Low' in data.columns else close_prices
        
        features = {}
        
        # ===== CORE INSTITUTIONAL MOMENTUM METRICS =====
        for period_name, period_days in self.momentum_periods.items():
            if len(close_prices) > period_days + 10:
                # Raw momentum (price change)
                start_price = close_prices[-period_days-1]
                end_price = close_prices[-1]
                momentum = (end_price - start_price) / start_price
                features[f'{period_name}_momentum'] = momentum
                
                # Risk-adjusted momentum (Sharpe-like ratio)
                period_returns = np.diff(close_prices[-period_days:]) / close_prices[-period_days:-1]
                if len(period_returns) > 0 and np.std(period_returns) > 1e-8:
                    risk_adj_momentum = np.mean(period_returns) / np.std(period_returns) * np.sqrt(252)
                    features[f'{period_name}_risk_adj'] = risk_adj_momentum
                else:
                    features[f'{period_name}_risk_adj'] = 0
                
                # Momentum consistency (percentage of positive days)
                positive_days = np.sum(period_returns > 0) / len(period_returns) if len(period_returns) > 0 else 0.5
                features[f'{period_name}_consistency'] = positive_days
                
                # Momentum acceleration (current vs previous period)
                if len(close_prices) > period_days * 2:
                    prev_start = close_prices[-period_days*2-1]
                    prev_end = close_prices[-period_days-1]
                    prev_momentum = (prev_end - prev_start) / prev_start
                    features[f'{period_name}_acceleration'] = momentum - prev_momentum
                else:
                    features[f'{period_name}_acceleration'] = 0
                
                # Maximum favorable excursion in period
                period_highs = high_prices[-period_days:]
                period_entry = close_prices[-period_days-1]
                max_gain = (np.max(period_highs) - period_entry) / period_entry
                features[f'{period_name}_max_gain'] = max_gain
                
                # Maximum adverse excursion in period
                period_lows = low_prices[-period_days:]
                max_loss = (np.min(period_lows) - period_entry) / period_entry
                features[f'{period_name}_max_loss'] = max_loss
        
        # ===== VOLATILITY AND RISK METRICS =====
        for vol_period in [10, 20, 60]:
            if len(close_prices) > vol_period:
                returns = np.diff(close_prices[-vol_period:]) / close_prices[-vol_period:-1]
                
                # Standard volatility
                vol = np.std(returns) * np.sqrt(252)
                features[f'volatility_{vol_period}d'] = vol
                
                # Downside deviation
                negative_returns = returns[returns < 0]
                downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
                features[f'downside_vol_{vol_period}d'] = downside_vol
                
                # Skewness and kurtosis
                if len(returns) > 3:
                    features[f'skewness_{vol_period}d'] = pd.Series(returns).skew()
                    features[f'kurtosis_{vol_period}d'] = pd.Series(returns).kurtosis()
                
                # VaR (Value at Risk) 5%
                var_5 = np.percentile(returns, 5)
                features[f'var_5_{vol_period}d'] = var_5
        
        # Volatility regime (current vs historical)
        if len(close_prices) > 60:
            current_vol = features.get('volatility_20d', 0)
            historical_vol = features.get('volatility_60d', current_vol)
            features['vol_regime'] = current_vol / historical_vol if historical_vol > 0 else 1
        
        # ===== VOLUME ANALYSIS =====
        if volume_data is not None and len(volume_data) > 20:
            # Volume momentum
            recent_vol = np.mean(volume_data[-5:])
            historical_vol = np.mean(volume_data[-21:-1])
            features['volume_momentum'] = (recent_vol - historical_vol) / historical_vol if historical_vol > 0 else 0
            
            # Volume trend (5-day vs 20-day)
            vol_ma_5 = np.mean(volume_data[-5:])
            vol_ma_20 = np.mean(volume_data[-20:])
            features['volume_trend'] = vol_ma_5 / vol_ma_20 if vol_ma_20 > 0 else 1
            
            # Price-volume relationship
            price_changes = np.diff(close_prices[-20:])
            volume_changes = np.diff(volume_data[-20:])
            if len(price_changes) > 0 and len(volume_changes) > 0:
                correlation = np.corrcoef(price_changes, volume_changes)[0,1]
                features['price_volume_corr'] = correlation if not np.isnan(correlation) else 0
            
            # On-balance volume momentum
            obv = np.cumsum(np.where(np.diff(close_prices[-21:]) > 0, volume_data[-20:], 
                                   np.where(np.diff(close_prices[-21:]) < 0, -volume_data[-20:], 0)))
            if len(obv) > 10:
                obv_momentum = (obv[-1] - obv[-11]) / abs(obv[-11]) if obv[-11] != 0 else 0
                features['obv_momentum'] = obv_momentum
        
        # ===== TECHNICAL MOMENTUM FEATURES =====
        # Moving average relationships
        ma_periods = [5, 10, 20, 50, 100]
        current_price = close_prices[-1]
        
        moving_averages = {}
        for period in ma_periods:
            if len(close_prices) > period:
                ma = np.mean(close_prices[-period:])
                moving_averages[period] = ma
                features[f'price_vs_ma{period}'] = (current_price - ma) / ma
        
        # MA crossover signals
        if 5 in moving_averages and 20 in moving_averages:
            features['ma5_vs_ma20'] = (moving_averages[5] - moving_averages[20]) / moving_averages[20]
        if 10 in moving_averages and 50 in moving_averages:
            features['ma10_vs_ma50'] = (moving_averages[10] - moving_averages[50]) / moving_averages[50]
        if 20 in moving_averages and 100 in moving_averages:
            features['ma20_vs_ma100'] = (moving_averages[20] - moving_averages[100]) / moving_averages[100]
        
        # Price position in recent range
        if len(close_prices) > 20:
            recent_high = np.max(high_prices[-20:])
            recent_low = np.min(low_prices[-20:])
            range_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            features['range_position_20d'] = range_position
        
        # ===== DRAWDOWN ANALYSIS =====
        if len(close_prices) > 60:
            # Maximum drawdown analysis for multiple periods
            for dd_period in [20, 60, 126]:
                if len(close_prices) > dd_period:
                    period_prices = close_prices[-dd_period:]
                    peak = np.maximum.accumulate(period_prices)
                    drawdown = (period_prices - peak) / peak
                    
                    features[f'max_drawdown_{dd_period}d'] = np.min(drawdown)
                    features[f'current_drawdown_{dd_period}d'] = drawdown[-1]
                    
                    # Drawdown recovery
                    current_peak_idx = np.argmax(peak == peak[-1])
                    if current_peak_idx > 0:
                        days_from_peak = len(peak) - current_peak_idx - 1
                        features[f'days_from_peak_{dd_period}d'] = days_from_peak / dd_period
                    else:
                        features[f'days_from_peak_{dd_period}d'] = 0
        
        # ===== MOMENTUM PATTERNS =====
        if len(close_prices) > 50:
            # Consecutive up/down days
            daily_changes = np.diff(close_prices[-20:])
            
            # Current streak
            current_streak = 0
            for i in range(len(daily_changes)-1, -1, -1):
                if (daily_changes[i] > 0) == (daily_changes[-1] > 0):
                    current_streak += 1
                else:
                    break
            features['current_streak'] = current_streak
            features['streak_direction'] = 1 if daily_changes[-1] > 0 else -1
            
            # Momentum divergence (price vs momentum)
            recent_momentum = features.get('short_momentum_momentum', 0)
            price_change_5d = (close_prices[-1] - close_prices[-6]) / close_prices[-6]
            features['momentum_divergence'] = recent_momentum - price_change_5d
        
        # ===== RELATIVE STRENGTH =====
        try:
            # Download SPY for market comparison
            spy_start = data.index[0] - timedelta(days=5)
            spy_end = data.index[-1] + timedelta(days=1)
            spy_data = yf.download('SPY', start=spy_start, end=spy_end, progress=False, auto_adjust=True)
            
            if not spy_data.empty and len(spy_data) > 50:
                spy_closes = spy_data['Close'].values
                
                # Align data (take last N days matching stock data)
                min_length = min(len(close_prices), len(spy_closes))
                stock_prices = close_prices[-min_length:]
                market_prices = spy_closes[-min_length:]
                
                # Relative strength for multiple periods
                for rs_period in [21, 63, 126]:
                    if min_length > rs_period:
                        stock_return = (stock_prices[-1] - stock_prices[-rs_period-1]) / stock_prices[-rs_period-1]
                        market_return = (market_prices[-1] - market_prices[-rs_period-1]) / market_prices[-rs_period-1]
                        features[f'relative_strength_{rs_period}d'] = stock_return - market_return
                
                # Beta calculation (60-day)
                if min_length > 60:
                    stock_returns = np.diff(stock_prices[-60:]) / stock_prices[-60:-1]
                    market_returns = np.diff(market_prices[-60:]) / market_prices[-60:-1]
                    
                    if np.var(market_returns) > 1e-8:
                        beta = np.cov(stock_returns, market_returns)[0,1] / np.var(market_returns)
                        features['beta_60d'] = beta
                    else:
                        features['beta_60d'] = 1
        except:
            # Default values if SPY data unavailable
            for rs_period in [21, 63, 126]:
                features[f'relative_strength_{rs_period}d'] = 0
            features['beta_60d'] = 1
        
        # ===== SEASONAL AND CYCLICAL FACTORS =====
        # Month and quarter effects
        current_date = data.index[-1]
        features['month'] = current_date.month
        features['quarter'] = (current_date.month - 1) // 3 + 1
        features['day_of_week'] = current_date.weekday()
        
        # Handle any remaining NaN or infinite values
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0
        
        return features
    
    def create_sophisticated_signals(self, features, future_returns=None):
        """Create sophisticated trading signals based on comprehensive analysis"""
        if not features:
            return 0
        
        # If we have future returns (for training), use them
        if future_returns is not None:
            # Multi-class signal based on future performance
            if future_returns > 0.05:  # Strong outperformance
                return 3  # STRONG BUY
            elif future_returns > 0.02:  # Good performance
                return 2  # BUY
            elif future_returns > -0.02:  # Neutral performance
                return 1  # HOLD
            else:  # Poor performance
                return 0  # SELL
        
        # For live predictions, use feature-based logic
        momentum_score = 0
        risk_score = 0
        technical_score = 0
        volume_score = 0
        
        # MOMENTUM SCORING
        long_momentum = features.get('long_momentum_momentum', 0)
        medium_momentum = features.get('medium_momentum_momentum', 0)
        short_momentum = features.get('short_momentum_momentum', 0)
        
        # Weight longer-term momentum more heavily (institutional approach)
        momentum_score = (long_momentum * 0.5 + medium_momentum * 0.3 + short_momentum * 0.2)
        
        # Risk-adjusted momentum bonus
        long_risk_adj = features.get('long_momentum_risk_adj', 0)
        if long_risk_adj > 1.0:
            momentum_score += 0.02
        elif long_risk_adj < -0.5:
            momentum_score -= 0.02
        
        # RISK SCORING
        vol_regime = features.get('vol_regime', 1)
        max_drawdown = features.get('max_drawdown_60d', 0)
        current_drawdown = features.get('current_drawdown_60d', 0)
        
        if vol_regime < 1.2 and current_drawdown > -0.05:  # Low vol, small drawdown
            risk_score = 0.01
        elif vol_regime > 1.5 or current_drawdown < -0.15:  # High vol or large drawdown
            risk_score = -0.02
        
        # TECHNICAL SCORING
        ma_alignment = (
            features.get('price_vs_ma5', 0) + 
            features.get('price_vs_ma20', 0) + 
            features.get('price_vs_ma50', 0)
        ) / 3
        
        range_position = features.get('range_position_20d', 0.5)
        
        if ma_alignment > 0.02 and range_position > 0.7:  # Above MAs and near highs
            technical_score = 0.01
        elif ma_alignment < -0.02 and range_position < 0.3:  # Below MAs and near lows
            technical_score = -0.01
        
        # VOLUME SCORING
        volume_momentum = features.get('volume_momentum', 0)
        price_volume_corr = features.get('price_volume_corr', 0)
        
        if volume_momentum > 0.2 and price_volume_corr > 0.3:  # Strong volume with price
            volume_score = 0.005
        elif volume_momentum < -0.2:  # Declining volume
            volume_score = -0.005
        
        # RELATIVE STRENGTH BONUS
        relative_strength = features.get('relative_strength_63d', 0)
        if relative_strength > 0.05:  # Outperforming market
            momentum_score += 0.01
        elif relative_strength < -0.05:  # Underperforming market
            momentum_score -= 0.01
        
        # COMBINED SCORE
        total_score = momentum_score + risk_score + technical_score + volume_score
        
        # Convert to signal
        if total_score > 0.04:
            return 3  # STRONG BUY
        elif total_score > 0.02:
            return 2  # BUY
        elif total_score > -0.02:
            return 1  # HOLD
        else:
            return 0  # SELL
    
    def prepare_comprehensive_training_data(self, symbol, period_days=1095):  # 3 years
        """Prepare comprehensive training dataset"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 300)  # Extra for features
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty or len(data) < 300:
                return None, None
            
            features_list = []
            signals_list = []
            
            # Create training samples with 30-day rolling windows
            window_size = 200  # Minimum data for comprehensive features
            
            for i in range(window_size, len(data) - 10):  # Leave 10 days for future return
                window_data = data.iloc[:i+1]
                features = self.calculate_comprehensive_momentum_features(window_data)
                
                if features:
                    # Calculate 10-day future return for signal generation
                    current_price = data['Close'].iloc[i]
                    future_price = data['Close'].iloc[i+10]
                    future_return = (future_price - current_price) / current_price
                    
                    signal = self.create_sophisticated_signals(features, future_return)
                    
                    features_list.append(list(features.values()))
                    signals_list.append(signal)
                    
                    # Store feature names (first iteration only)
                    if not self.feature_names:
                        self.feature_names = list(features.keys())
            
            if len(features_list) == 0:
                return None, None
            
            X = np.array(features_list)
            y = np.array(signals_list)
            
            # Clean data
            X = np.nan_to_num(X, nan=0, posinf=1, neginf=-1)
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error preparing training data for {symbol}: {str(e)}")
            return None, None
    
    def train_ensemble_models(self):
        """Train comprehensive ensemble of ML models"""
        print("🤖 TRAINING COMPREHENSIVE MOMENTUM ML ENSEMBLE")
        print("=" * 60)
        print("🏛️ Using full institutional momentum feature set")
        print("📊 Training on deployed momentum portfolio stocks")
        print("🎯 Multi-class classification: SELL/HOLD/BUY/STRONG_BUY")
        print("=" * 60)
        
        all_X = []
        all_y = []
        
        for symbol in self.momentum_stocks:
            print(f"📈 Preparing comprehensive data for {symbol}...")
            X, y = self.prepare_comprehensive_training_data(symbol)
            
            if X is not None and y is not None:
                all_X.append(X)
                all_y.append(y)
                print(f"   ✅ {symbol}: {len(X)} samples, {X.shape[1]} features")
                print(f"   📊 Signal distribution: {np.bincount(y)}")
            else:
                print(f"   ❌ {symbol}: Failed to prepare data")
        
        if not all_X:
            print("❌ No training data available!")
            return False
        
        # Combine all training data
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        print(f"\n📈 COMBINED TRAINING DATASET:")
        print(f"   Total samples: {len(X_combined):,}")
        print(f"   Features: {X_combined.shape[1]}")
        print(f"   Signal distribution: {dict(zip(['SELL', 'HOLD', 'BUY', 'STRONG_BUY'], np.bincount(y_combined)))}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n🎯 TRAINING ENSEMBLE MODELS:")
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\n   🔄 Training {model_name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            print(f"   📊 CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train final model
            model.fit(X_train_scaled, y_train)
            
            # Test performance
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"   ✅ Test Accuracy: {accuracy:.3f}")
            
            # Classification report
            print(f"   📋 Classification Report:")
            report = classification_report(y_test, y_pred, 
                                         target_names=['SELL', 'HOLD', 'BUY', 'STRONG_BUY'],
                                         output_dict=True)
            
            for signal in ['SELL', 'HOLD', 'BUY', 'STRONG_BUY']:
                if signal in report:
                    precision = report[signal]['precision']
                    recall = report[signal]['recall']
                    print(f"      {signal}: Precision={precision:.3f}, Recall={recall:.3f}")
            
            self.trained_models[model_name] = model
        
        # Feature importance analysis
        if 'random_forest' in self.trained_models:
            print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
            feature_importance = self.trained_models['random_forest'].feature_importances_
            
            importance_pairs = list(zip(self.feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, importance) in enumerate(importance_pairs[:10], 1):
                print(f"   {i:2d}. {name}: {importance:.4f}")
        
        # Save models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"momentum_ml_models_{timestamp}.joblib"
        
        model_data = {
            'models': self.trained_models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_stats': {
                'total_samples': len(X_combined),
                'features': X_combined.shape[1],
                'signal_distribution': dict(zip(['SELL', 'HOLD', 'BUY', 'STRONG_BUY'], np.bincount(y_combined)))
            }
        }
        
        joblib.dump(model_data, model_filename)
        print(f"\n💾 Models saved to {model_filename}")
        
        return True
    
    def predict_comprehensive_signals(self, symbol):
        """Generate comprehensive ML predictions for a stock"""
        if not self.trained_models:
            print("❌ Models not trained yet!")
            return None
        
        try:
            # Get comprehensive recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=300)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty or len(data) < 200:
                return None
            
            # Calculate comprehensive features
            features = self.calculate_comprehensive_momentum_features(data)
            if not features:
                return None
            
            # Prepare feature array
            X = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            X = np.nan_to_num(X, nan=0, posinf=1, neginf=-1)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.trained_models.items():
                pred = model.predict(X_scaled)[0]
                predictions[model_name] = pred
                
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    probabilities[model_name] = {
                        'SELL': proba[0],
                        'HOLD': proba[1] if len(proba) > 1 else 0,
                        'BUY': proba[2] if len(proba) > 2 else 0,
                        'STRONG_BUY': proba[3] if len(proba) > 3 else 0
                    }
            
            # Ensemble prediction (weighted average)
            # Weight Random Forest more heavily as it typically performs best
            weights = {'random_forest': 0.4, 'gradient_boost': 0.35, 'logistic_regression': 0.25}
            
            ensemble_score = sum(predictions[model] * weights.get(model, 1/len(predictions)) 
                               for model in predictions)
            
            # Convert ensemble score to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY', 3: 'STRONG_BUY'}
            ensemble_signal = signal_map[round(ensemble_score)]
            
            # Calculate confidence (agreement between models)
            prediction_values = list(predictions.values())
            confidence = 1.0 - (np.std(prediction_values) / 1.5)  # Normalize by max possible std
            
            # Get top contributing features
            if 'random_forest' in self.trained_models:
                feature_importance = self.trained_models['random_forest'].feature_importances_
                feature_contributions = []
                
                for i, (feature_name, importance) in enumerate(zip(self.feature_names, feature_importance)):
                    contribution = X[0, i] * importance
                    feature_contributions.append((feature_name, contribution, features[feature_name]))
                
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_features = feature_contributions[:5]
            else:
                top_features = []
            
            return {
                'symbol': symbol,
                'signal': ensemble_signal,
                'confidence': confidence,
                'ensemble_score': ensemble_score,
                'model_predictions': predictions,
                'probabilities': probabilities,
                'top_features': top_features,
                'current_price': data['Close'].iloc[-1],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error predicting for {symbol}: {str(e)}")
            return None
    
    def generate_portfolio_analysis(self):
        """Generate comprehensive ML analysis for momentum portfolio"""
        print("\n🚀 COMPREHENSIVE ML MOMENTUM PORTFOLIO ANALYSIS")
        print("=" * 65)
        
        if not self.trained_models:
            print("⚠️  Training models first...")
            if not self.train_ensemble_models():
                return None
        
        portfolio_analysis = []
        
        for symbol in self.momentum_stocks:
            print(f"\n🔍 Comprehensive analysis for {symbol}...")
            result = self.predict_comprehensive_signals(symbol)
            
            if result:
                portfolio_analysis.append(result)
                
                signal_emoji = {"STRONG_BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "❌"}
                emoji = signal_emoji.get(result['signal'], "❓")
                
                print(f"   {emoji} {result['signal']} (Confidence: {result['confidence']:.2f})")
                print(f"   💰 Price: ${result['current_price']:.2f}")
                print(f"   🎯 Ensemble Score: {result['ensemble_score']:.2f}")
                
                # Show model agreement
                predictions = result['model_predictions']
                print(f"   🤖 Model Predictions: RF={predictions.get('random_forest', 'N/A')}, "
                      f"GB={predictions.get('gradient_boost', 'N/A')}, "
                      f"LR={predictions.get('logistic_regression', 'N/A')}")
                
                # Show top features
                if result['top_features']:
                    print(f"   📊 Key Factors:")
                    for feature_name, contribution, value in result['top_features'][:3]:
                        print(f"      • {feature_name}: {value:.4f} (impact: {contribution:.4f})")
                
            else:
                print(f"   ❌ Could not analyze {symbol}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comprehensive_ml_momentum_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(portfolio_analysis, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive analysis saved to {filename}")
        
        # Portfolio summary
        if portfolio_analysis:
            signal_counts = {}
            total_confidence = 0
            
            for result in portfolio_analysis:
                signal = result['signal']
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
                total_confidence += result['confidence']
            
            avg_confidence = total_confidence / len(portfolio_analysis)
            
            print(f"\n📊 COMPREHENSIVE ML MOMENTUM PORTFOLIO SUMMARY:")
            print(f"   🔬 Analysis Method: Institutional Momentum + Advanced ML")
            print(f"   📈 Stocks Analyzed: {len(portfolio_analysis)}")
            print(f"   🎯 Average Confidence: {avg_confidence:.2f}")
            print(f"   🤖 Model Ensemble: Random Forest + Gradient Boost + Logistic Regression")
            
            for signal, count in signal_counts.items():
                emoji = {"STRONG_BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "❌"}[signal]
                print(f"   {emoji} {signal}: {count}")
            
            # High confidence recommendations
            high_conf_signals = [r for r in portfolio_analysis if r['confidence'] > 0.8]
            if high_conf_signals:
                print(f"\n🎯 HIGH CONFIDENCE RECOMMENDATIONS (>80%):")
                for result in high_conf_signals:
                    emoji = {"STRONG_BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "SELL": "❌"}[result['signal']]
                    print(f"   {emoji} {result['symbol']}: {result['signal']} ({result['confidence']:.1%})")
        
        return portfolio_analysis

def main():
    """Main execution function"""
    print("🤖 COMPREHENSIVE MOMENTUM ML MODEL")
    print("🏛️ Full Institutional Feature Set + Advanced Machine Learning")
    print("=" * 70)
    
    # Initialize comprehensive model
    ml_model = ComprehensiveMomentumMLModel()
    
    # Run comprehensive analysis
    results = ml_model.generate_portfolio_analysis()
    
    if results:
        print(f"\n🎯 COMPREHENSIVE ML ANALYSIS COMPLETE!")
        print(f"✅ Advanced ensemble models trained on institutional features")
        print(f"📊 Multi-class predictions with confidence scores")
        print(f"🔬 Feature importance analysis included")
        print(f"🏆 Ready for integration with deployed momentum strategy")
        print(f"\n💡 Next Steps:")
        print(f"   • Compare ML signals with deployed momentum portfolio")
        print(f"   • Use high-confidence signals for position sizing")
        print(f"   • Monitor model performance vs institutional benchmark")
    
    return ml_model

if __name__ == "__main__":
    ml_model = main()
