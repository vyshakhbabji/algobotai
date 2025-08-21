#!/usr/bin/env python3
"""
🧠 SIMPLIFIED ML-FILTERED INSTITUTIONAL MOMENTUM TRADER
Combines machine learning with institutional momentum research to improve stock selection.
Uses simplified feature engineering to avoid pandas Series ambiguity errors.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class SimplifiedMLMomentumTrader:
    def __init__(self, capital_per_stock=10000):
        self.capital_per_stock = capital_per_stock
        
        # All 25 stocks in our universe
        self.all_stocks = [
            'NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'CRM', 'UBER',
            'JPM', 'WMT', 'JNJ', 'PG', 'KO', 'PLTR', 'COIN', 'SNOW', 'AMD', 'INTC',
            'XOM', 'CVX', 'CAT', 'BA', 'GE'
        ]
        
        # Institutional momentum parameters (Jegadeesh & Titman, 1993)
        self.conservative_config = {
            'momentum_lookback': 126,  # 6 months
            'formation_period': 63,   # 3 months
            'holding_period': 21,     # 1 month
            'volatility_threshold': 0.4,
            'min_volume': 1000000
        }
        
        # Aggressive configuration for ML-identified momentum-friendly stocks
        self.aggressive_config = {
            'momentum_lookback': 63,   # 3 months
            'formation_period': 21,   # 1 month
            'holding_period': 10,     # 2 weeks
            'volatility_threshold': 0.6,
            'min_volume': 500000
        }
        
        # ML models for stock classification
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
        self.models_trained = False
        
    def calculate_simple_features(self, data):
        """Calculate simplified momentum and technical features using safe scalar operations"""
        features = {}
        
        # Flatten multi-index columns if present
        if hasattr(data.columns, 'levels'):
            data.columns = data.columns.droplevel(1)
        
        close_prices = data['Close']
        volumes = data['Volume']
        
        # Basic momentum features (using safe iloc indexing and .item() for scalar extraction)
        if len(close_prices) >= 21:
            momentum_21 = (close_prices.iloc[-1] - close_prices.iloc[-21]) / close_prices.iloc[-21]
            features['momentum_21d'] = float(momentum_21.item() if hasattr(momentum_21, 'item') else momentum_21)
            
            vol_21 = close_prices.iloc[-21:].std() / close_prices.iloc[-21:].mean()
            features['volatility_21d'] = float(vol_21.item() if hasattr(vol_21, 'item') else vol_21)
            
        if len(close_prices) >= 63:
            momentum_63 = (close_prices.iloc[-1] - close_prices.iloc[-63]) / close_prices.iloc[-63]
            features['momentum_63d'] = float(momentum_63.item() if hasattr(momentum_63, 'item') else momentum_63)
            
            vol_63 = close_prices.iloc[-63:].std() / close_prices.iloc[-63:].mean()
            features['volatility_63d'] = float(vol_63.item() if hasattr(vol_63, 'item') else vol_63)
            
        if len(close_prices) >= 126:
            momentum_126 = (close_prices.iloc[-1] - close_prices.iloc[-126]) / close_prices.iloc[-126]
            features['momentum_126d'] = float(momentum_126.item() if hasattr(momentum_126, 'item') else momentum_126)
            
            vol_126 = close_prices.iloc[-126:].std() / close_prices.iloc[-126:].mean()
            features['volatility_126d'] = float(vol_126.item() if hasattr(vol_126, 'item') else vol_126)
            
        # Volume features
        if len(volumes) >= 42:
            vol_ratio = volumes.iloc[-21:].mean() / volumes.iloc[-42:-21].mean()
            features['volume_ratio_21d'] = float(vol_ratio.item() if hasattr(vol_ratio, 'item') else vol_ratio)
        elif len(volumes) >= 21:
            features['volume_ratio_21d'] = 1.0
            
        # Price action features
        if len(close_prices) >= 5:
            price_change = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
            features['price_change_5d'] = float(price_change.item() if hasattr(price_change, 'item') else price_change)
            
        # Simple moving average features
        if len(close_prices) >= 20:
            ma_20 = close_prices.iloc[-20:].mean()
            price_vs_ma = (close_prices.iloc[-1] - ma_20) / ma_20
            features['price_vs_ma20'] = float(price_vs_ma.item() if hasattr(price_vs_ma, 'item') else price_vs_ma)
            
        if len(close_prices) >= 50:
            ma_50 = close_prices.iloc[-50:].mean()
            price_vs_ma = (close_prices.iloc[-1] - ma_50) / ma_50
            features['price_vs_ma50'] = float(price_vs_ma.item() if hasattr(price_vs_ma, 'item') else price_vs_ma)
            
        # Trend consistency (simplified)
        if len(close_prices) >= 10:
            recent_returns = close_prices.iloc[-10:].pct_change().dropna()
            if len(recent_returns) > 0:
                positive_days = (recent_returns > 0).sum()
                positive_count = float(positive_days.item() if hasattr(positive_days, 'item') else positive_days)
                features['trend_consistency'] = positive_count / len(recent_returns)
            else:
                features['trend_consistency'] = 0.5
            
        # Replace any missing values with defaults
        default_features = {
            'momentum_21d': 0.0, 'momentum_63d': 0.0, 'momentum_126d': 0.0,
            'volatility_21d': 0.0, 'volatility_63d': 0.0, 'volatility_126d': 0.0,
            'volume_ratio_21d': 1.0, 'price_change_5d': 0.0,
            'price_vs_ma20': 0.0, 'price_vs_ma50': 0.0, 'trend_consistency': 0.5
        }
        
        for key, default_value in default_features.items():
            if key not in features:
                features[key] = default_value
                
        # Final safety check - convert all features to float
        for key in features:
            try:
                features[key] = float(features[key])
            except:
                features[key] = 0.0
                
        return features
    
    def create_simple_labels(self, symbol, data, forward_days=21):
        """Create labels based on future momentum performance using simple logic"""
        # Flatten multi-index columns if present
        if hasattr(data.columns, 'levels'):
            data.columns = data.columns.droplevel(1)
            
        close_prices = data['Close']
        
        if len(close_prices) < forward_days + 30:
            return None
        
        labels = []
        for i in range(30, len(close_prices) - forward_days):
            current_price = close_prices.iloc[i]
            future_price = close_prices.iloc[i + forward_days]
            
            # Extract scalar values safely
            current_val = float(current_price.item() if hasattr(current_price, 'item') else current_price)
            future_val = float(future_price.item() if hasattr(future_price, 'item') else future_price)
            
            forward_return = (future_val - current_val) / current_val
            
            # Simple classification: good momentum (1) vs poor momentum (0)
            if forward_return > 0.03:  # >3% gain
                labels.append(1)
            else:
                labels.append(0)
        
        return labels
    
    def prepare_training_data(self):
        """Prepare training data from historical stock performance"""
        print(f"\n🤖 PREPARING SIMPLIFIED ML TRAINING DATA")
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
                labels = self.create_simple_labels(symbol, data)
                if not labels:
                    continue
                
                # Generate features for each training point
                for i in range(30, len(data) - 21):
                    try:
                        features = self.calculate_simple_features(data.iloc[:i+1])
                        
                        if feature_names is None:
                            feature_names = list(features.keys())
                        
                        # Ensure consistent feature order
                        feature_vector = [features.get(name, 0.0) for name in feature_names]
                        all_features.append(feature_vector)
                        all_labels.append(labels[i-30])
                    except Exception as fe:
                        print(f"   ⚠️  Feature error for {symbol} at index {i}: {str(fe)}")
                        continue
                
                print(f"   ✅ {symbol}: {len(labels)} training samples")
                
            except Exception as e:
                print(f"   ❌ {symbol}: {str(e)}")
                continue
        
        if len(all_features) == 0:
            print("❌ No training data generated!")
            return None
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Positive momentum samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        
        return X, y, feature_names
    
    def train_ml_models(self):
        """Train ML models to classify momentum-friendly vs momentum-hostile stocks"""
        print(f"\n🧠 TRAINING SIMPLIFIED ML MODELS")
        print("=" * 50)
        
        training_data = self.prepare_training_data()
        if training_data is None:
            print("❌ Failed to prepare training data!")
            return False
        
        X, y, feature_names = training_data
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("🌲 Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train)
        rf_score = self.rf_model.score(X_test_scaled, y_test)
        
        print("⚡ Training Gradient Boosting...")
        self.gb_model.fit(X_train_scaled, y_train)
        gb_score = self.gb_model.score(X_test_scaled, y_test)
        
        print("📈 Training Logistic Regression...")
        self.lr_model.fit(X_train_scaled, y_train)
        lr_score = self.lr_model.score(X_test_scaled, y_test)
        
        print(f"\n✅ MODEL PERFORMANCE:")
        print(f"   🌲 Random Forest: {rf_score:.3f}")
        print(f"   ⚡ Gradient Boosting: {gb_score:.3f}")
        print(f"   📈 Logistic Regression: {lr_score:.3f}")
        
        self.models_trained = True
        self.feature_names = feature_names
        return True
    
    def classify_stock(self, symbol, data):
        """Classify if a stock is momentum-friendly using ensemble ML models"""
        if not self.models_trained:
            return 'conservative'  # Default to conservative if models not trained
        
        try:
            features = self.calculate_simple_features(data)
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get predictions from all models
            rf_pred = self.rf_model.predict_proba(feature_vector_scaled)[0][1]
            gb_pred = self.gb_model.predict_proba(feature_vector_scaled)[0][1]
            lr_pred = self.lr_model.predict_proba(feature_vector_scaled)[0][1]
            
            # Ensemble prediction (average)
            ensemble_score = (rf_pred + gb_pred + lr_pred) / 3
            
            # Classify based on ensemble score
            if ensemble_score > 0.6:
                return 'aggressive'  # High confidence momentum-friendly
            else:
                return 'conservative'  # Default to conservative approach
                
        except Exception as e:
            print(f"   ⚠️  Classification error for {symbol}: {str(e)}")
            return 'conservative'
    
    def calculate_momentum_signal(self, data, config):
        """Calculate institutional momentum signal using given configuration"""
        # Flatten multi-index columns if present
        if hasattr(data.columns, 'levels'):
            data.columns = data.columns.droplevel(1)
            
        close_prices = data['Close']
        volumes = data['Volume']
        
        if len(close_prices) < config['momentum_lookback']:
            return 0
        
        # Formation period return
        current_price = close_prices.iloc[-1]
        formation_price = close_prices.iloc[-config['formation_period']]
        
        # Extract scalar values safely
        current_val = float(current_price.item() if hasattr(current_price, 'item') else current_price)
        formation_val = float(formation_price.item() if hasattr(formation_price, 'item') else formation_price)
        
        formation_return = (current_val - formation_val) / formation_val
        
        # Volatility check
        volatility_series = close_prices.iloc[-config['momentum_lookback']:].std() / close_prices.iloc[-config['momentum_lookback']:].mean()
        volatility = float(volatility_series.item() if hasattr(volatility_series, 'item') else volatility_series)
        
        # Volume check
        avg_volume_series = volumes.iloc[-21:].mean()
        avg_volume = float(avg_volume_series.item() if hasattr(avg_volume_series, 'item') else avg_volume_series)
        
        # Risk-adjusted momentum signal
        if volatility < config['volatility_threshold'] and avg_volume > config['min_volume']:
            return formation_return / volatility  # Risk-adjusted return
        else:
            return 0
    
    def run_ml_filtered_strategy(self):
        """Run the ML-filtered institutional momentum strategy"""
        print(f"\n🚀 RUNNING SIMPLIFIED ML-FILTERED MOMENTUM STRATEGY")
        print("=" * 70)
        
        # Train ML models first
        if not self.train_ml_models():
            print("❌ Failed to train ML models!")
            return
        
        print(f"\n📊 TESTING ML-FILTERED STRATEGY")
        print("=" * 50)
        
        total_invested = 0
        total_returns = 0
        total_buy_hold_returns = 0
        successful_trades = 0
        total_trades = 0
        
        results = []
        
        for symbol in self.all_stocks:
            try:
                print(f"\n🔍 Analyzing {symbol}...")
                
                # Download recent data for testing
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)  # 6 months for testing
                
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if data.empty or len(data) < 63:
                    print(f"   ❌ Insufficient data")
                    continue
                
                # Flatten multi-index columns if present
                if hasattr(data.columns, 'levels'):
                    data.columns = data.columns.droplevel(1)
                
                # Classify stock using ML
                classification = self.classify_stock(symbol, data)
                config = self.aggressive_config if classification == 'aggressive' else self.conservative_config
                
                print(f"   🤖 ML Classification: {classification.upper()}")
                
                # Calculate momentum signal
                momentum_signal = self.calculate_momentum_signal(data, config)
                
                if momentum_signal > 0:
                    # Simulate trade - extract scalar values safely
                    entry_price_series = data['Close'].iloc[-config['holding_period']]
                    exit_price_series = data['Close'].iloc[-1]
                    
                    entry_price = float(entry_price_series.item() if hasattr(entry_price_series, 'item') else entry_price_series)
                    exit_price = float(exit_price_series.item() if hasattr(exit_price_series, 'item') else exit_price_series)
                    
                    position_return = (exit_price - entry_price) / entry_price
                    
                    # Buy and hold return for comparison
                    start_price_series = data['Close'].iloc[0]
                    end_price_series = data['Close'].iloc[-1]
                    
                    start_price = float(start_price_series.item() if hasattr(start_price_series, 'item') else start_price_series)
                    end_price = float(end_price_series.item() if hasattr(end_price_series, 'item') else end_price_series)
                    
                    buy_hold_return = (end_price - start_price) / start_price
                    
                    total_invested += self.capital_per_stock
                    total_returns += self.capital_per_stock * position_return
                    total_buy_hold_returns += self.capital_per_stock * buy_hold_return
                    
                    if position_return > 0:
                        successful_trades += 1
                    total_trades += 1
                    
                    results.append({
                        'symbol': symbol,
                        'classification': classification,
                        'momentum_signal': momentum_signal,
                        'position_return': position_return,
                        'buy_hold_return': buy_hold_return,
                        'outperformed': position_return > buy_hold_return
                    })
                    
                    status = "✅" if position_return > 0 else "❌"
                    outperform = "🎯" if position_return > buy_hold_return else "📉"
                    print(f"   {status} Position Return: {position_return*100:.1f}% {outperform}")
                    print(f"   📊 Buy & Hold: {buy_hold_return*100:.1f}%")
                else:
                    print(f"   ⏸️  No momentum signal - SKIP")
                    
            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {str(e)}")
                continue
        
        # Calculate final results
        if total_invested > 0:
            strategy_return = total_returns / total_invested
            buy_hold_return = total_buy_hold_returns / total_invested
            win_rate = successful_trades / total_trades if total_trades > 0 else 0
            
            print(f"\n🎯 FINAL RESULTS")
            print("=" * 50)
            print(f"💰 Total Capital Deployed: ${total_invested:,.0f}")
            print(f"📈 ML-Filtered Strategy Return: {strategy_return*100:.1f}%")
            print(f"📊 Buy & Hold Return: {buy_hold_return*100:.1f}%")
            print(f"🎲 Win Rate: {win_rate*100:.1f}% ({successful_trades}/{total_trades})")
            print(f"⚡ Alpha: {(strategy_return - buy_hold_return)*100:.1f}%")
            
            # Show outperformance stats
            outperformed = sum(1 for r in results if r['outperformed'])
            outperform_rate = outperformed / len(results) if results else 0
            print(f"🎯 Outperformed Buy & Hold: {outperform_rate*100:.1f}% ({outperformed}/{len(results)})")
            
            print(f"\n📋 INDIVIDUAL RESULTS:")
            print("-" * 60)
            for result in results:
                symbol = result['symbol']
                classification = result['classification'][:3].upper()
                pos_ret = result['position_return']
                bh_ret = result['buy_hold_return']
                alpha = pos_ret - bh_ret
                
                status = "✅" if pos_ret > 0 else "❌"
                outperform = "🎯" if result['outperformed'] else "📉"
                
                print(f"{symbol:5} [{classification}]: {status} {pos_ret*100:+6.1f}% vs {bh_ret*100:+6.1f}% {outperform} α={alpha*100:+5.1f}%")
        else:
            print("\n❌ No trades executed!")

if __name__ == "__main__":
    trader = SimplifiedMLMomentumTrader(capital_per_stock=10000)
    trader.run_ml_filtered_strategy()
