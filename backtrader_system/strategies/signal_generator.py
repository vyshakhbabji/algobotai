"""
Signal Generator for Backtrader
Extracted from RealisticLiveTradingSystem for clean separation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, r2_score
import yfinance as yf
import logging


class MLSignalGenerator:
    """
    ML Signal Generator that mirrors the logic from RealisticLiveTradingSystem
    but works within Backtrader framework
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model storage (mirrors RealisticLiveTradingSystem)
        self.daily_models = {}  # {date: {symbol: {model_type: model}}}
        self.daily_scalers = {}  # {date: {symbol: scaler}}
        self.model_evolution = {}  # Track model performance over time
        
        # Feature columns from config
        self.feature_cols = config.get('feature_columns', [
            'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50',
            'rsi_normalized', 'volatility_10d', 'volatility_20d',
            'volume_ratio', 'bb_position', 'macd_histogram',
            'trend_consistency'
        ])
        
        self.min_training_days = config.get('min_training_days', 60)
        
    def get_elite_stocks(self) -> List[str]:
        """Top 20 elite mega-cap stocks - from RealisticLiveTradingSystem"""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ORCL",
            "JPM", "BAC", "V", "MA",
            "UNH", "JNJ", "PG", "KO", 
            "XOM", "HD", "DIS", "CRM"
        ]
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators - extracted from RealisticLiveTradingSystem
        """
        try:
            df = df.copy()
            
            # Ensure we have returns
            df['returns'] = df['Close'].pct_change()
            
            # Moving averages
            df['ma_5'] = df['Close'].rolling(5).mean()
            df['ma_20'] = df['Close'].rolling(20).mean()
            df['ma_50'] = df['Close'].rolling(50).mean()
            
            # Price relative positions
            df['price_vs_ma5'] = (df['Close'] - df['ma_5']) / df['ma_5']
            df['price_vs_ma20'] = (df['Close'] - df['ma_20']) / df['ma_20']
            df['price_vs_ma50'] = (df['Close'] - df['ma_50']) / df['ma_50']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
            
            # Volatility
            df['volatility_10d'] = df['returns'].rolling(10).std()
            df['volatility_20d'] = df['returns'].rolling(20).std()
            
            # Volume ratio (handle missing volume data)
            if 'Volume' in df.columns:
                df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            else:
                df['volume_ratio'] = 1.0  # Default if no volume data
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Fix divide-by-zero for Bollinger Bands
            bb_width = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / bb_width
            df['bb_position'] = df['bb_position'].fillna(0.5)  # Default to middle
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Trend features
            df['trend_5d'] = np.where(df['Close'] > df['Close'].shift(5), 1, -1)
            df['trend_10d'] = np.where(df['Close'] > df['Close'].shift(10), 1, -1)
            df['trend_consistency'] = (df['trend_5d'] + df['trend_10d']) / 2

            # Additional features from RealisticLiveTradingSystem
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()

            df['momentum_14'] = df['Close'].pct_change(14)
            df['momentum_30'] = df['Close'].pct_change(30)

            # Normalize indicators with rolling z-score
            indicator_cols = [
                'price_vs_ma5', 'price_vs_ma20', 'price_vs_ma50',
                'rsi_normalized', 'volatility_10d', 'volatility_20d',
                'volume_ratio', 'bb_position', 'macd_histogram',
                'trend_consistency', 'atr_14', 'momentum_14', 'momentum_30'
            ]
            
            for col in indicator_cols:
                if col in df.columns:
                    # Rolling z-score normalization
                    rolling_mean = df[col].rolling(50, min_periods=10).mean()
                    rolling_std = df[col].rolling(50, min_periods=10).std()
                    df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df
    
    def create_ml_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML targets for prediction - extracted from RealisticLiveTradingSystem
        """
        try:
            df = df.copy()
            
            # Market regime target (trending vs ranging)
            price_momentum = df['Close'].pct_change(10).abs()
            volatility = df['volatility_10d']
            
            # Regime score: high momentum + low volatility = trending
            regime_score = price_momentum / (volatility + 0.001)
            regime_threshold = regime_score.rolling(50).quantile(0.6)
            df['regime_target'] = (regime_score > regime_threshold).astype(int)
            
            # Signal strength target (0.3-1.0) - ENHANCED VERSION
            future_returns_3d = df['Close'].shift(-3) / df['Close'] - 1
            vol_adj_move = np.abs(future_returns_3d) / (df['volatility_10d'] + 0.001)
            
            # Normalize to 0.3-1.0 range
            signal_strength_raw = np.clip(vol_adj_move * 1.5, 0, 1)
            df['ml_signal_strength'] = 0.3 + (signal_strength_raw * 0.7)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating ML targets: {e}")
            return df
    
    def train_daily_models(self, symbol: str, data: pd.DataFrame, current_date: pd.Timestamp) -> bool:
        """
        Train ML models using only data up to current date - from RealisticLiveTradingSystem
        """
        try:
            if len(data) < self.min_training_days:
                return False
            
            # Calculate features and targets
            df = self.calculate_technical_indicators(data.copy())
            df = self.create_ml_targets(df)
            
            # Clean data (remove future-looking NaN values)
            clean_data = df[self.feature_cols + ['regime_target', 'ml_signal_strength']].dropna()
            
            # Fix target leakage: only train on data BEFORE current_date
            train_mask = clean_data.index < current_date
            clean_data = clean_data[train_mask]
            
            if len(clean_data) < self.min_training_days:
                return False
            
            X = clean_data[self.feature_cols].values
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store scaler for this date
            if current_date not in self.daily_scalers:
                self.daily_scalers[current_date] = {}
            self.daily_scalers[current_date][symbol] = scaler
            
            # Initialize model storage for this date
            if current_date not in self.daily_models:
                self.daily_models[current_date] = {}
            if symbol not in self.daily_models[current_date]:
                self.daily_models[current_date][symbol] = {}
            
            models_trained = 0
            
            # 1. Train Signal Strength Model
            y_strength = clean_data['ml_signal_strength'].values
            try:
                strength_model = RandomForestRegressor(
                    n_estimators=50, max_depth=8, random_state=42
                )
                strength_model.fit(X_scaled, y_strength)
                self.daily_models[current_date][symbol]['strength'] = strength_model
                
                # Track performance
                strength_pred = strength_model.predict(X_scaled)
                strength_r2 = r2_score(y_strength, strength_pred)
                models_trained += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to train strength model for {symbol}: {e}")
                strength_r2 = 0.0
            
            # 2. Train Regime Model
            y_regime = clean_data['regime_target'].values
            try:
                regime_model = RandomForestClassifier(
                    n_estimators=50, max_depth=8, random_state=42
                )
                regime_model.fit(X_scaled, y_regime)
                self.daily_models[current_date][symbol]['regime'] = regime_model
                
                # Track performance
                regime_pred = regime_model.predict(X_scaled)
                regime_accuracy = accuracy_score(y_regime, regime_pred)
                models_trained += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to train regime model for {symbol}: {e}")
                regime_accuracy = 0.5
            
            # Track model evolution
            if symbol not in self.model_evolution:
                self.model_evolution[symbol] = []
            
            self.model_evolution[symbol].append({
                'date': current_date,
                'regime_accuracy': regime_accuracy,
                'strength_r2': strength_r2,
                'models_trained': models_trained,
                'training_samples': len(clean_data)
            })
            
            return models_trained > 0
            
        except Exception as e:
            self.logger.error(f"Error training daily models for {symbol}: {e}")
            return False
    
    def generate_base_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate base technical signal - from RealisticLiveTradingSystem
        """
        try:
            if len(df) < 50:
                return {'signal': 'HOLD', 'strength': 0.0, 'price': df['Close'].iloc[-1]}
            
            latest = df.iloc[-1]
            
            # Technical conditions
            rsi = latest['rsi']
            price = latest['Close']
            price_vs_ma5 = latest['price_vs_ma5']
            price_vs_ma20 = latest['price_vs_ma20']
            volume_ratio = latest.get('volume_ratio', 1.0)
            macd_hist = latest.get('macd_histogram', 0)
            
            # Strong buy conditions
            strong_buy = (
                (rsi < 35 and price_vs_ma5 > -0.03) or
                (price_vs_ma5 > 0.02 and price_vs_ma20 > 0.01 and volume_ratio > 1.2 and macd_hist > 0)
            )
            
            # Strong sell conditions
            strong_sell = (
                (rsi > 75 and price_vs_ma5 < 0.02) or
                (price_vs_ma5 < -0.02 and price_vs_ma20 < -0.01 and macd_hist < 0)
            )
            
            # Calculate signal strength
            if strong_buy:
                # Strength based on multiple confirmations
                strength_factors = []
                if rsi < 35 and price_vs_ma5 > -0.03:
                    strength_factors.append(0.7)
                if price_vs_ma5 > 0.02 and price_vs_ma20 > 0.01:
                    strength_factors.append(0.6)
                if volume_ratio > 1.2:
                    strength_factors.append(0.3)
                if macd_hist > 0:
                    strength_factors.append(0.2)
                
                signal_strength = min(0.9, sum(strength_factors))
                return {'signal': 'BUY', 'strength': signal_strength, 'price': price}
                
            elif strong_sell:
                # Strength based on sell confirmations
                strength_factors = []
                if rsi > 75 and price_vs_ma5 < 0.02:
                    strength_factors.append(0.7)
                if price_vs_ma5 < -0.02 and price_vs_ma20 < -0.01:
                    strength_factors.append(0.6)
                if macd_hist < 0:
                    strength_factors.append(0.2)
                
                signal_strength = min(0.9, sum(strength_factors))
                return {'signal': 'SELL', 'strength': signal_strength, 'price': price}
            else:
                return {'signal': 'HOLD', 'strength': 0.0, 'price': price}
                
        except Exception as e:
            self.logger.error(f"Error generating base signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'price': 0}
    
    def enhance_signal_with_ml(self, symbol: str, base_signal: Dict, df: pd.DataFrame, 
                              current_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Enhance signal using ML models - from RealisticLiveTradingSystem
        """
        try:
            if base_signal['signal'] == 'HOLD':
                return base_signal
            
            # Check if we have models for this date and symbol
            if (current_date not in self.daily_models or 
                symbol not in self.daily_models[current_date] or
                current_date not in self.daily_scalers or
                symbol not in self.daily_scalers[current_date]):
                return base_signal
            
            # Prepare features
            latest_features = df[self.feature_cols].iloc[-1:].values
            scaler = self.daily_scalers[current_date][symbol]
            X_scaled = scaler.transform(latest_features)
            
            # Default enhancements
            ml_multiplier = 1.0
            regime_boost = 1.0
            
            # 1. Predict Signal Strength Enhancement
            if 'strength' in self.daily_models[current_date][symbol]:
                try:
                    strength_model = self.daily_models[current_date][symbol]['strength']
                    predicted_strength = strength_model.predict(X_scaled)[0]
                    predicted_strength = np.clip(predicted_strength, 0.3, 1.0)
                    
                    # ML multiplier based on predicted vs base strength
                    ml_multiplier = predicted_strength / max(base_signal['strength'], 0.1)
                    ml_multiplier = np.clip(ml_multiplier, 0.5, 1.5)
                    
                except Exception as e:
                    self.logger.warning(f"Strength prediction failed for {symbol}: {e}")
            
            # 2. Predict Market Regime Enhancement
            if 'regime' in self.daily_models[current_date][symbol]:
                try:
                    regime_model = self.daily_models[current_date][symbol]['regime']
                    regime_pred = regime_model.predict_proba(X_scaled)[0]
                    
                    # If model predicts trending regime (class 1), boost signal
                    if len(regime_pred) > 1:
                        trending_prob = regime_pred[1]
                        regime_boost = 0.8 + (trending_prob * 0.4)  # 0.8 to 1.2 range
                    
                except Exception as e:
                    self.logger.warning(f"Regime prediction failed for {symbol}: {e}")
            
            # 3. Combine Enhancements
            total_multiplier = ml_multiplier * regime_boost
            total_multiplier = np.clip(total_multiplier, 0.5, 1.8)
            
            # Apply enhancement
            enhanced_strength = base_signal['strength'] * total_multiplier
            enhanced_strength = np.clip(enhanced_strength, 0.0, 1.0)
            
            return {
                'signal': base_signal['signal'],
                'strength': enhanced_strength,
                'price': base_signal['price'],
                'base_strength': base_signal['strength'],
                'ml_multiplier': ml_multiplier,
                'regime_boost': regime_boost,
                'total_enhancement': total_multiplier,
                'ml_enhanced': True
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing signal for {symbol}: {e}")
            return base_signal
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Main signal generation method that combines base + ML enhancement
        """
        # Calculate indicators
        df_with_indicators = self.calculate_technical_indicators(data)
        
        # Generate base signal
        base_signal = self.generate_base_signal(df_with_indicators)
        
        # Enhance with ML if we have trained models
        enhanced_signal = self.enhance_signal_with_ml(
            symbol, base_signal, df_with_indicators, current_date
        )
        
        return enhanced_signal
