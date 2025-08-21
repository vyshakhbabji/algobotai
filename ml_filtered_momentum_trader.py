#!/usr/bin/env python3
"""
ML-FILTERED INSTITUTIONAL MOMENTUM TRADER
Combines machine learning stock classification with institutional momentum signals
Uses ML to identify momentum-friendly stocks, then applies proven institutional configs
Full sophisticated implementation without simplifications
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json
import joblib

class MLFilteredMomentumTrader:
    def __init__(self, starting_capital=10000):
        self.starting_capital = starting_capital
        
        # Full stock universe from optimizer
        self.all_stocks = [
            # TECH GIANTS
            "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN",
            # VOLATILE GROWTH  
            "TSLA", "META", "NFLX", "CRM", "UBER",
            # TRADITIONAL VALUE
            "JPM", "WMT", "JNJ", "PG", "KO",
            # EMERGING/SPECULATIVE
            "PLTR", "COIN", "SNOW", "AMD", "INTC",
            # ENERGY/MATERIALS
            "XOM", "CVX", "CAT", "BA", "GE"
        ]
        
        # Institutional momentum parameters (from research)
        self.institutional_config = {
            'trend_5d_buy_threshold': 0.025,    # Jegadeesh & Titman (1993)
            'trend_5d_sell_threshold': -0.02,   # Risk management threshold
            'trend_10d_buy_threshold': 0.025,   # Medium-term confirmation
            'trend_10d_sell_threshold': -0.045, # Strict exit signal
            'rsi_overbought': 65,               # Conservative overbought
            'rsi_oversold': 20,                 # Conservative oversold
            'volatility_threshold': 0.07,       # 7% volatility limit
            'volume_ratio_threshold': 1.6       # 1.6x volume confirmation
        }
        
        # Aggressive momentum config for selected stocks
        self.aggressive_config = {
            'trend_5d_buy_threshold': 0.015,    # More sensitive
            'trend_5d_sell_threshold': -0.025,  # Tighter stops
            'trend_10d_buy_threshold': 0.015,   # Earlier entry
            'trend_10d_sell_threshold': -0.045, # Keep strict exits
            'rsi_overbought': 85,               # Allow momentum runs
            'rsi_oversold': 30,                 # More realistic
            'volatility_threshold': 0.10,       # Higher tolerance
            'volume_ratio_threshold': 1.1       # Easier to trigger
        }
        
        # ML models for stock classification
        self.ml_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150, 
                learning_rate=0.1, 
                max_depth=8,
                min_samples_split=5,
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
        self.momentum_friendly_stocks = []
        self.momentum_hostile_stocks = []
        
        print(f"🧠 ML-FILTERED INSTITUTIONAL MOMENTUM TRADER")
        print(f"💰 Capital per stock: ${starting_capital:,}")
        print(f"📊 Total stock universe: {len(self.all_stocks)} stocks")
        print(f"🤖 ML Models: Random Forest, Gradient Boost, Logistic Regression")
        print(f"🏛️  Using institutional momentum parameters")
        
    def calculate_comprehensive_features(self, data, lookback_periods=[5, 10, 21, 63, 126]):
        """Calculate comprehensive momentum and technical features"""
        features = {}
        close_prices = data['Close']
        volumes = data['Volume']
        
        for period in lookback_periods:
            if len(close_prices) > period:
                # Momentum features
                features[f'momentum_{period}d'] = (close_prices.iloc[-1] - close_prices.iloc[-period]) / close_prices.iloc[-period]
                features[f'momentum_strength_{period}d'] = abs(features[f'momentum_{period}d'])
                
                # Trend consistency (how often price went up in period)
                recent_returns = close_prices.pct_change().iloc[-period:]
                if not recent_returns.empty:
                    features[f'trend_consistency_{period}d'] = float((recent_returns > 0).mean())
                else:
                    features[f'trend_consistency_{period}d'] = 0.5
                
                # Volatility features
                recent_prices = close_prices.iloc[-period:]
                current_vol = float(recent_prices.std())
                features[f'volatility_{period}d'] = current_vol / float(recent_prices.mean()) if recent_prices.mean() != 0 else 0.0
                
                rolling_vol = recent_prices.rolling(max(2, period//2)).std().mean()
                if not pd.isna(rolling_vol) and rolling_vol != 0:
                    features[f'volatility_rank_{period}d'] = float(current_vol > rolling_vol)
                else:
                    features[f'volatility_rank_{period}d'] = 0.0
                
                # Volume features
                if len(volumes) > period:
                    recent_volumes = volumes.iloc[-period:]
                    avg_volume = volumes.iloc[-period*2:-period].mean()
                    features[f'volume_ratio_{period}d'] = float(recent_volumes.mean() / avg_volume) if avg_volume > 0 else 1.0
                    features[f'volume_trend_{period}d'] = float(volumes.iloc[-1] / volumes.iloc[-period]) if volumes.iloc[-period] > 0 else 1.0
        
        # Technical indicators
        if len(close_prices) >= 14:
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            features['rsi_momentum'] = float(rsi.iloc[-1] - rsi.iloc[-5]) if len(rsi) >= 5 and not pd.isna(rsi.iloc[-1]) and not pd.isna(rsi.iloc[-5]) else 0.0
        
        # Moving average features
        for ma_period in [5, 10, 20, 50]:
            if len(close_prices) > ma_period:
                ma = close_prices.rolling(ma_period).mean()
                features[f'ma_{ma_period}_slope'] = float((ma.iloc[-1] - ma.iloc[-5]) / ma.iloc[-5]) if len(ma) >= 5 and not pd.isna(ma.iloc[-1]) and not pd.isna(ma.iloc[-5]) and ma.iloc[-5] != 0 else 0.0
                features[f'price_vs_ma_{ma_period}'] = float((close_prices.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) and ma.iloc[-1] != 0 else 0.0
        
        # Price action features
        features['price_change_1d'] = float(close_prices.pct_change().iloc[-1]) if len(close_prices) >= 2 and not pd.isna(close_prices.pct_change().iloc[-1]) else 0.0
        features['price_change_5d'] = float((close_prices.iloc[-1] - close_prices.iloc[-6]) / close_prices.iloc[-6]) if len(close_prices) >= 6 and close_prices.iloc[-6] != 0 else 0.0
        
        # Momentum persistence (how long has momentum been in same direction)
        if len(close_prices) >= 10:
            returns = close_prices.pct_change().iloc[-10:]
            if not returns.empty and not pd.isna(returns.iloc[-1]):
                current_direction = 1 if returns.iloc[-1] > 0 else -1
                persistence = 0
                for i in range(len(returns)-1, -1, -1):
                    if not pd.isna(returns.iloc[i]):
                        if (returns.iloc[i] > 0 and current_direction > 0) or (returns.iloc[i] < 0 and current_direction < 0):
                            persistence += 1
                        else:
                            break
                features['momentum_persistence'] = float(persistence)
            else:
                features['momentum_persistence'] = 0.0
        else:
            features['momentum_persistence'] = 0.0
        
        # Replace any infinite or NaN values
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                features[key] = 0
                
        return features
    
    def create_training_labels(self, symbol, data, forward_days=21):
        """Create labels based on future momentum performance"""
        close_prices = data['Close']
        
        if len(close_prices) < forward_days + 30:
            return None
        
        # Calculate forward returns for each point
        labels = []
        for i in range(30, len(close_prices) - forward_days):
            current_price = float(close_prices.iloc[i])
            future_price = float(close_prices.iloc[i + forward_days])
            forward_return = (future_price - current_price) / current_price
            
            # Binary classification: momentum-friendly (1) vs momentum-hostile (0)
            # Momentum-friendly = consistent positive returns with low volatility
            recent_returns = close_prices.iloc[i-21:i].pct_change().dropna()
            volatility = float(recent_returns.std()) if len(recent_returns) > 0 else 0.0
            trend_strength = float(recent_returns.mean()) if len(recent_returns) > 0 else 0.0
            
            # Label as momentum-friendly if:
            # 1. Future return is positive AND
            # 2. Recent trend is consistent (low volatility relative to return) AND
            # 3. Has sustained momentum
            if forward_return > 0.02 and trend_strength > 0 and volatility < 0.05:
                labels.append(1)  # Momentum-friendly
            elif forward_return < -0.02 or volatility > 0.08:
                labels.append(0)  # Momentum-hostile
            else:
                labels.append(0)  # Neutral -> treat as hostile for conservative approach
        
        return labels
    
    def prepare_training_data(self):
        """Prepare training data from historical stock performance"""
        print(f"\n🤖 PREPARING ML TRAINING DATA")
        print("=" * 50)
        
        all_features = []
        all_labels = []
        feature_names = None
        
        for symbol in self.all_stocks:
            try:
                print(f"📊 Processing {symbol}...")
                
                # Download 2 years of data for training
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if data.empty or len(data) < 100:
                    continue
                
                # Generate labels
                labels = self.create_training_labels(symbol, data)
                if not labels:
                    continue
                
                # Generate features for each training point
                for i in range(30, len(data) - 21):
                    try:
                        features = self.calculate_comprehensive_features(data.iloc[:i+1])
                        
                        if feature_names is None:
                            feature_names = list(features.keys())
                        
                        # Ensure consistent feature order
                        feature_vector = [features.get(name, 0) for name in feature_names]
                        all_features.append(feature_vector)
                        all_labels.append(labels[i-30])
                    except Exception as fe:
                        print(f"   ⚠️  Feature error for {symbol} at index {i}: {str(fe)}")
                        continue
                
                print(f"   ✅ {symbol}: {len(labels)} training samples")
                
            except Exception as e:
                print(f"   ❌ {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_features:
            print("❌ No training data generated!")
            return None, None, None
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n📊 TRAINING DATA SUMMARY:")
        print(f"   🔢 Total samples: {len(X):,}")
        print(f"   📈 Features per sample: {len(feature_names)}")
        print(f"   ✅ Momentum-friendly samples: {sum(y):,} ({sum(y)/len(y):.1%})")
        print(f"   ❌ Momentum-hostile samples: {len(y)-sum(y):,} ({(len(y)-sum(y))/len(y):.1%})")
        
        return X, y, feature_names
    
    def train_ml_models(self):
        """Train machine learning models for stock classification"""
        print(f"\n🧠 TRAINING ML MODELS")
        print("=" * 50)
        
        # Prepare training data
        X, y, feature_names = self.prepare_training_data()
        if X is None:
            print("❌ Failed to prepare training data!")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train each model
        for model_name, model in self.ml_models.items():
            print(f"\n🔧 Training {model_name}...")
            
            if model_name == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, predictions)
            
            # Store trained model
            self.trained_models[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'feature_names': feature_names
            }
            
            print(f"   ✅ {model_name}: {accuracy:.1%} accuracy")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"   🔝 Top 5 features:")
                for idx, row in importance_df.head().iterrows():
                    print(f"      {row['feature']}: {row['importance']:.3f}")
        
        # Select best model
        best_model_name = max(self.trained_models.keys(), key=lambda x: self.trained_models[x]['accuracy'])
        print(f"\n🏆 Best model: {best_model_name} ({self.trained_models[best_model_name]['accuracy']:.1%} accuracy)")
        
        return True
    
    def classify_stock(self, symbol):
        """Classify a stock as momentum-friendly or momentum-hostile"""
        try:
            # Download recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=200)  # Need enough for features
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 50:
                return 'hostile', 0.0
            
            # Calculate features
            features = self.calculate_comprehensive_features(data)
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for model_name, model_info in self.trained_models.items():
                model = model_info['model']
                feature_names = model_info['feature_names']
                
                # Ensure consistent feature order
                feature_vector = np.array([[features.get(name, 0) for name in feature_names]])
                
                if model_name == 'logistic_regression':
                    feature_vector = self.scaler.transform(feature_vector)
                
                prediction = model.predict(feature_vector)[0]
                
                if hasattr(model, 'predict_proba'):
                    confidence = model.predict_proba(feature_vector)[0].max()
                else:
                    confidence = 0.5
                
                predictions.append(prediction)
                confidences.append(confidence)
            
            # Ensemble prediction (majority vote with confidence weighting)
            weighted_prediction = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
            final_prediction = 'friendly' if weighted_prediction > 0.5 else 'hostile'
            avg_confidence = np.mean(confidences)
            
            return final_prediction, avg_confidence
            
        except Exception as e:
            print(f"❌ Error classifying {symbol}: {str(e)}")
            return 'hostile', 0.0
    
    def classify_all_stocks(self):
        """Classify all stocks using trained ML models"""
        print(f"\n🔍 CLASSIFYING ALL STOCKS")
        print("=" * 50)
        
        self.momentum_friendly_stocks = []
        self.momentum_hostile_stocks = []
        
        for symbol in self.all_stocks:
            classification, confidence = self.classify_stock(symbol)
            
            if classification == 'friendly':
                self.momentum_friendly_stocks.append(symbol)
                status = "✅"
            else:
                self.momentum_hostile_stocks.append(symbol)
                status = "❌"
            
            print(f"{status} {symbol:4s}: {classification:8s} (confidence: {confidence:.1%})")
        
        print(f"\n📊 CLASSIFICATION RESULTS:")
        print(f"   ✅ Momentum-friendly: {len(self.momentum_friendly_stocks)} stocks")
        print(f"   ❌ Momentum-hostile:  {len(self.momentum_hostile_stocks)} stocks")
        print(f"   🎯 Friendly stocks: {self.momentum_friendly_stocks}")
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for trading"""
        data = data.copy()
        
        # Moving averages
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def generate_momentum_signal(self, data, idx, config):
        """Generate momentum trading signal using institutional parameters"""
        try:
            price = float(data['Close'].iloc[idx])
            
            # Trend calculations
            recent_5d = data['Close'].iloc[idx-5:idx]
            recent_10d = data['Close'].iloc[idx-10:idx]
            
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # Technical indicators
            ma5 = float(data['MA5'].iloc[idx]) if not pd.isna(data['MA5'].iloc[idx]) else price
            ma10 = float(data['MA10'].iloc[idx]) if not pd.isna(data['MA10'].iloc[idx]) else price
            rsi = float(data['RSI'].iloc[idx]) if not pd.isna(data['RSI'].iloc[idx]) else 50
            
            # Volume analysis
            recent_vol = float(data['Volume'].iloc[idx-10:idx].mean())
            current_vol = float(data['Volume'].iloc[idx])
            vol_ratio = current_vol / recent_vol if recent_vol > 0 else 1
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # INSTITUTIONAL MOMENTUM SIGNAL LOGIC
            signal = 'HOLD'
            
            # BUY CONDITIONS (multiple criteria must align)
            buy_conditions = []
            
            # Primary: Strong dual-timeframe momentum
            if trend_5d > config['trend_5d_buy_threshold'] and trend_10d > config['trend_10d_buy_threshold']:
                buy_conditions.append('strong_momentum')
            
            # Secondary: Technical confirmation
            if price > ma5 > ma10 and trend_5d > config['trend_5d_buy_threshold']:
                buy_conditions.append('technical_confirmation')
            
            # Tertiary: Volume confirmation
            if vol_ratio > config['volume_ratio_threshold'] and trend_5d > config['trend_5d_buy_threshold']:
                buy_conditions.append('volume_confirmation')
            
            # Quaternary: Oversold bounce with momentum
            if rsi < config['rsi_oversold'] and trend_5d > config['trend_5d_buy_threshold']/2:
                buy_conditions.append('oversold_bounce')
            
            # SELL CONDITIONS (risk management focused)
            sell_conditions = []
            
            # Primary: Dual-timeframe breakdown
            if trend_5d < config['trend_5d_sell_threshold'] and trend_10d < config['trend_10d_sell_threshold']:
                sell_conditions.append('momentum_breakdown')
            
            # Secondary: Technical breakdown
            if price < ma5 < ma10:
                sell_conditions.append('technical_breakdown')
            
            # Tertiary: Overbought with weakening momentum
            if rsi > config['rsi_overbought'] and trend_5d < config['trend_5d_sell_threshold']/2:
                sell_conditions.append('overbought_exhaustion')
            
            # Quaternary: High volatility with weak trend
            if volatility > config['volatility_threshold'] and trend_10d < config['trend_10d_sell_threshold']:
                sell_conditions.append('volatility_spike')
            
            # SIGNAL DETERMINATION
            if sell_conditions:
                signal = 'SELL'
            elif buy_conditions:
                signal = 'BUY'
            
            return {
                'signal': signal,
                'price': price,
                'trend_5d': trend_5d,
                'trend_10d': trend_10d,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': vol_ratio,
                'buy_conditions': buy_conditions,
                'sell_conditions': sell_conditions
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'price': price if 'price' in locals() else 0}
    
    def trade_stock_with_ml_config(self, symbol, use_aggressive=False):
        """Trade a single stock using ML-determined configuration"""
        try:
            classification, confidence = self.classify_stock(symbol)
            
            # Choose config based on ML classification
            if classification == 'friendly' and confidence > 0.7:
                config = self.aggressive_config if use_aggressive else self.institutional_config
                config_type = "aggressive" if use_aggressive else "institutional"
            else:
                # Use conservative config for hostile/uncertain stocks
                config = self.institutional_config
                config_type = "conservative"
            
            print(f"\n📈 Trading {symbol} ({classification}, confidence: {confidence:.1%})")
            print(f"⚙️  Using {config_type} config")
            print("-" * 40)
            
            # Download data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=120)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 20:
                print(f"❌ {symbol}: Insufficient data")
                return None
                
            data = self.calculate_technical_indicators(data)
            
            # Portfolio state
            cash = self.starting_capital
            shares = 0
            position = None
            trades = []
            
            # Trading simulation
            for i in range(15, len(data)):
                date = data.index[i]
                signal_data = self.generate_momentum_signal(data, i, config)
                price = signal_data['price']
                signal = signal_data['signal']
                
                # Execute trades
                if signal == 'BUY' and position != 'LONG' and cash > 0:
                    # Buy with full available cash
                    shares_to_buy = cash / price
                    amount = shares_to_buy * price
                    
                    shares += shares_to_buy
                    cash = 0
                    position = 'LONG'
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'amount': amount,
                        'conditions': signal_data['buy_conditions']
                    })
                    
                    conditions_str = ', '.join(signal_data['buy_conditions'])
                    print(f"📈 BUY:  {date.strftime('%m/%d')} | ${price:6.2f} | {shares_to_buy:5.1f} shares | {conditions_str}")
                    
                elif signal == 'SELL' and position == 'LONG' and shares > 0:
                    # Sell all shares
                    amount = shares * price
                    sold_shares = shares
                    
                    # Calculate trade profit
                    last_buy = None
                    for t in reversed(trades):
                        if t['action'] == 'BUY':
                            last_buy = t
                            break
                    
                    profit = amount - last_buy['amount'] if last_buy else 0
                    profit_pct = (profit / last_buy['amount']) * 100 if last_buy else 0
                    
                    cash = amount
                    shares = 0
                    position = None
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': sold_shares,
                        'amount': amount,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'conditions': signal_data['sell_conditions']
                    })
                    
                    conditions_str = ', '.join(signal_data['sell_conditions'])
                    print(f"📉 SELL: {date.strftime('%m/%d')} | ${price:6.2f} | Profit: {profit_pct:+5.1f}% | {conditions_str}")
            
            # Final calculations
            final_price = float(data['Close'].iloc[-1])
            final_value = cash + (shares * final_price)
            
            start_price = float(data['Close'].iloc[15])
            buy_hold_value = self.starting_capital * (final_price / start_price)
            
            strategy_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
            buy_hold_return = ((buy_hold_value - self.starting_capital) / self.starting_capital) * 100
            outperformance = strategy_return - buy_hold_return
            
            result = {
                'symbol': symbol,
                'ml_classification': classification,
                'ml_confidence': confidence,
                'config_used': config_type,
                'final_value': final_value,
                'strategy_return': strategy_return,
                'buy_hold_value': buy_hold_value,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'trades': trades,
                'final_cash': cash,
                'final_shares': shares,
                'price_change': ((final_price - start_price) / start_price) * 100
            }
            
            # Display results
            status = "✅" if outperformance > 0 else "❌"
            print(f"{status} Final: ${final_value:7,.0f} ({strategy_return:+5.1f}%) vs BH: {buy_hold_return:+5.1f}% | Diff: {outperformance:+5.1f}% | Trades: {len(trades)}")
            
            return result
            
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")
            return None
    
    def run_ml_filtered_momentum_strategy(self):
        """Run the complete ML-filtered momentum strategy"""
        print(f"\n🚀 RUNNING ML-FILTERED MOMENTUM STRATEGY")
        print("=" * 70)
        
        # Step 1: Train ML models
        if not self.train_ml_models():
            print("❌ Failed to train ML models!")
            return None
        
        # Step 2: Classify all stocks
        self.classify_all_stocks()
        
        # Step 3: Trade all stocks with appropriate configs
        all_results = []
        total_deployed = 0
        total_final_value = 0
        
        for symbol in self.all_stocks:
            result = self.trade_stock_with_ml_config(symbol, use_aggressive=True)
            if result:
                all_results.append(result)
                total_deployed += self.starting_capital
                total_final_value += result['final_value']
        
        # Portfolio analysis
        print(f"\n" + "="*70)
        print(f"🏆 ML-FILTERED MOMENTUM PORTFOLIO RESULTS")
        print("="*70)
        
        if all_results:
            # Sort by performance
            all_results.sort(key=lambda x: x['outperformance'], reverse=True)
            
            # Portfolio totals
            total_strategy_return = ((total_final_value - total_deployed) / total_deployed) * 100
            total_buy_hold_value = sum(r['buy_hold_value'] for r in all_results)
            total_buy_hold_return = ((total_buy_hold_value - total_deployed) / total_deployed) * 100
            total_outperformance = total_strategy_return - total_buy_hold_return
            
            print(f"💰 PORTFOLIO TOTALS:")
            print(f"   🏦 Capital Deployed:  ${total_deployed:10,.0f}")
            print(f"   📈 Strategy Value:    ${total_final_value:10,.0f} ({total_strategy_return:+6.1f}%)")
            print(f"   🎯 Buy-Hold Value:    ${total_buy_hold_value:10,.0f} ({total_buy_hold_return:+6.1f}%)")
            print(f"   🏆 Total Outperform:  ${total_final_value - total_buy_hold_value:10,.0f} ({total_outperformance:+6.1f}%)")
            
            # ML Classification analysis
            friendly_results = [r for r in all_results if r['ml_classification'] == 'friendly']
            hostile_results = [r for r in all_results if r['ml_classification'] == 'hostile']
            
            print(f"\n🤖 ML CLASSIFICATION PERFORMANCE:")
            if friendly_results:
                friendly_outperform = sum(r['outperformance'] for r in friendly_results) / len(friendly_results)
                friendly_winners = len([r for r in friendly_results if r['outperformance'] > 0])
                print(f"   ✅ Momentum-friendly stocks: {len(friendly_results)}")
                print(f"      📈 Avg outperformance: {friendly_outperform:+.1f}%")
                print(f"      🎯 Win rate: {friendly_winners}/{len(friendly_results)} ({friendly_winners/len(friendly_results):.1%})")
            
            if hostile_results:
                hostile_outperform = sum(r['outperformance'] for r in hostile_results) / len(hostile_results)
                hostile_winners = len([r for r in hostile_results if r['outperformance'] > 0])
                print(f"   ❌ Momentum-hostile stocks: {len(hostile_results)}")
                print(f"      📈 Avg outperformance: {hostile_outperform:+.1f}%")
                print(f"      🎯 Win rate: {hostile_winners}/{len(hostile_results)} ({hostile_winners/len(hostile_results):.1%})")
            
            # Top performers
            print(f"\n🏅 TOP 10 PERFORMERS:")
            for i, result in enumerate(all_results[:10], 1):
                status = "✅" if result['outperformance'] > 0 else "❌"
                ml_status = "🧠" if result['ml_classification'] == 'friendly' else "🔴"
                print(f"   #{i:2d} {status} {ml_status} {result['symbol']:4s}: ${result['final_value']:7,.0f} ({result['strategy_return']:+6.1f}%) | "
                      f"vs BH: {result['outperformance']:+6.1f}% | {result['config_used']} | Trades: {len(result['trades'])}")
            
            # Performance breakdown
            winners = [r for r in all_results if r['outperformance'] > 0]
            losers = [r for r in all_results if r['outperformance'] <= 0]
            
            print(f"\n📊 OVERALL PERFORMANCE BREAKDOWN:")
            print(f"   ✅ Winners: {len(winners)}/{len(all_results)} ({len(winners)/len(all_results):.1%})")
            print(f"   ❌ Losers:  {len(losers)}/{len(all_results)} ({len(losers)/len(all_results):.1%})")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ml_filtered_momentum_results_{timestamp}.json"
            
            portfolio_summary = {
                'total_deployed': total_deployed,
                'total_final_value': total_final_value,
                'total_strategy_return': total_strategy_return,
                'total_buy_hold_return': total_buy_hold_return,
                'total_outperformance': total_outperformance,
                'ml_friendly_count': len(friendly_results),
                'ml_hostile_count': len(hostile_results),
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': len(winners)/len(all_results),
                'individual_results': all_results,
                'ml_model_accuracies': {name: info['accuracy'] for name, info in self.trained_models.items()},
                'institutional_config': self.institutional_config,
                'aggressive_config': self.aggressive_config
            }
            
            with open(filename, 'w') as f:
                json.dump(portfolio_summary, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to {filename}")
            
            return portfolio_summary
        
        else:
            print("❌ No successful trades!")
            return None

def main():
    """Run ML-filtered institutional momentum trader"""
    trader = MLFilteredMomentumTrader(starting_capital=10000)
    results = trader.run_ml_filtered_momentum_strategy()
    
    if results:
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"   💰 Deployed ${results['total_deployed']:,} across {len(trader.all_stocks)} stocks")
        print(f"   📈 Returned ${results['total_final_value']:,.0f} ({results['total_strategy_return']:+.1f}%)")
        print(f"   🏆 Beat buy-and-hold by {results['total_outperformance']:+.1f}%")
        print(f"   🎯 Win rate: {results['win_rate']:.1%} ({results['winners']}/{results['winners'] + results['losers']})")
        print(f"   🧠 ML-friendly stocks: {results['ml_friendly_count']}")
        print(f"   🔴 ML-hostile stocks: {results['ml_hostile_count']}")
        
        # Key insights
        if results['total_outperformance'] > 0:
            print(f"   ✅ ML-FILTERED MOMENTUM STRATEGY SUCCESSFUL!")
        else:
            print(f"   ⚠️  Strategy needs refinement")
    
    return results

if __name__ == "__main__":
    results = main()
