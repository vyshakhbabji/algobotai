#!/usr/bin/env python3
"""
COMPLETE AUTO-TRADING ML BOT
Advanced ML + Technical Indicators Trading System

Requirements from prompt.yaml:
- 30%+ YoY profit target
- 100-150 stock universe with continuous scanning  
- ML-driven signals with technical confirmation
- Daily position tracking and rebalancing
- Comprehensive backtesting and forward testing
- Risk management and performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import yfinance as yf
import ta
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class MLTradingBot:
    """Advanced ML Trading Bot with Technical Indicators"""
    
    def __init__(self, universe_size: int = 100, target_return: float = 0.30):
        self.universe_size = universe_size
        self.target_return = target_return
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.positions = {}
        self.portfolio_value = 100000  # Starting capital
        self.performance_history = []
        
        # SP500 universe (top 100 by market cap)
        self.universe = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'BRK-B', 'LLY', 'AVGO',
            'JPM', 'UNH', 'XOM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'COST', 'NFLX',
            'CRM', 'BAC', 'ABBV', 'KO', 'ORCL', 'CVX', 'MRK', 'AMD', 'PEP', 'TMO',
            'WMT', 'ADBE', 'ACN', 'DIS', 'ABT', 'CSCO', 'VZ', 'AXP', 'DHR', 'TXN',
            'NOW', 'QCOM', 'INTU', 'COP', 'CMCSA', 'PM', 'RTX', 'UBER', 'SPGI', 'HON',
            'AMAT', 'GS', 'IBM', 'PFE', 'CAT', 'BKNG', 'TJX', 'GE', 'AMGN', 'MDT',
            'MCD', 'ISRG', 'T', 'LRCX', 'SYK', 'BSX', 'LOW', 'MU', 'PLD', 'NEE',
            'DE', 'REGN', 'ADP', 'LMT', 'ADI', 'VRTX', 'KLAC', 'AMT', 'GILD', 'CI',
            'PANW', 'MMC', 'SBUX', 'CVS', 'FI', 'TMUS', 'ZTS', 'SHW', 'CB', 'CDNS',
            'PYPL', 'CMG', 'SO', 'SNPS', 'ITW', 'ORLY', 'WM', 'CRWD', 'MMM', 'AON'
        ][:universe_size]
        
    def download_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Download and prepare stock data"""
        try:
            df = yf.download(symbol, period=period, progress=False)
            if df.empty:
                return None
            
            # Handle multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Clean data
            df = df.dropna()
            if len(df) < 200:  # Need sufficient history
                return None
                
            return df
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set combining technical indicators and ML features"""
        data = df.copy()
        
        # Basic price features
        data['returns_1d'] = data['Close'].pct_change(1)
        data['returns_5d'] = data['Close'].pct_change(5)
        data['returns_10d'] = data['Close'].pct_change(10)
        data['returns_20d'] = data['Close'].pct_change(20)
        
        # Moving averages and price relationships
        data['sma_5'] = ta.trend.sma_indicator(data['Close'], window=5)
        data['sma_10'] = ta.trend.sma_indicator(data['Close'], window=10)
        data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['sma_100'] = ta.trend.sma_indicator(data['Close'], window=100)
        
        data['ema_5'] = ta.trend.ema_indicator(data['Close'], window=5)
        data['ema_10'] = ta.trend.ema_indicator(data['Close'], window=10)
        data['ema_20'] = ta.trend.ema_indicator(data['Close'], window=20)
        
        # Price vs MA ratios
        data['price_vs_sma5'] = data['Close'] / data['sma_5'] - 1
        data['price_vs_sma10'] = data['Close'] / data['sma_10'] - 1
        data['price_vs_sma20'] = data['Close'] / data['sma_20'] - 1
        data['price_vs_sma50'] = data['Close'] / data['sma_50'] - 1
        
        # Momentum indicators
        data['rsi'] = ta.momentum.rsi(data['Close'], window=14)
        data['stoch'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'], window=14)
        data['macd'] = ta.trend.macd_diff(data['Close'])
        data['macd_signal'] = ta.trend.macd_signal(data['Close'])
        
        # Volatility indicators
        data['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
        data['atr_pct'] = data['atr'] / data['Close']
        data['bb_upper'] = ta.volatility.bollinger_hband(data['Close'], window=20)
        data['bb_lower'] = ta.volatility.bollinger_lband(data['Close'], window=20)
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume indicators
        data['volume_sma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        data['vwap'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])
        data['price_vs_vwap'] = data['Close'] / data['vwap'] - 1
        
        # Advanced momentum features
        data['mom_3'] = data['Close'].pct_change(3)
        data['mom_5'] = data['Close'].pct_change(5)
        data['mom_10'] = data['Close'].pct_change(10)
        data['mom_20'] = data['Close'].pct_change(20)
        
        # Volatility features
        data['volatility_5d'] = data['returns_1d'].rolling(5).std()
        data['volatility_20d'] = data['returns_1d'].rolling(20).std()
        data['volatility_ratio'] = data['volatility_5d'] / data['volatility_20d']
        
        # Trend strength
        data['trend_strength'] = np.where(
            (data['sma_5'] > data['sma_10']) & 
            (data['sma_10'] > data['sma_20']) & 
            (data['sma_20'] > data['sma_50']), 1,
            np.where(
                (data['sma_5'] < data['sma_10']) & 
                (data['sma_10'] < data['sma_20']) & 
                (data['sma_20'] < data['sma_50']), -1, 0
            )
        )
        
        # Support/Resistance levels
        data['high_52w'] = data['High'].rolling(252).max()
        data['low_52w'] = data['Low'].rolling(252).min()
        data['distance_from_high'] = (data['Close'] - data['high_52w']) / data['high_52w']
        data['distance_from_low'] = (data['Close'] - data['low_52w']) / data['low_52w']
        
        # Market regime features
        data['market_cap_proxy'] = data['Close'] * data['Volume']  # Rough proxy
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            data[f'returns_1d_lag_{lag}'] = data['returns_1d'].shift(lag)
            data[f'volume_ratio_lag_{lag}'] = data['volume_ratio'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['rsi'].shift(lag)
        
        return data
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.DataFrame:
        """Create trading labels based on future returns"""
        data = df.copy()
        
        # Forward returns
        data['forward_return'] = data['Close'].shift(-horizon) / data['Close'] - 1
        
        # Binary labels for classification
        data['label'] = np.where(data['forward_return'] > threshold, 1, 0)
        
        # Multi-class labels
        data['label_multiclass'] = np.where(
            data['forward_return'] > threshold, 2,  # Strong buy
            np.where(data['forward_return'] > 0, 1, 0)  # Weak buy / Hold/Sell
        )
        
        # Remove future data contamination
        data = data.iloc[:-horizon]
        
        return data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model"""
        # Define feature columns (avoid future data)
        feature_cols = [
            'returns_1d', 'returns_5d', 'returns_10d', 'returns_20d',
            'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20', 'price_vs_sma50',
            'rsi', 'stoch', 'macd', 'macd_signal',
            'atr_pct', 'bb_position', 'volume_ratio', 'price_vs_vwap',
            'mom_3', 'mom_5', 'mom_10', 'mom_20',
            'volatility_5d', 'volatility_20d', 'volatility_ratio',
            'trend_strength', 'distance_from_high', 'distance_from_low',
            'returns_1d_lag_1', 'returns_1d_lag_2', 'returns_1d_lag_3', 'returns_1d_lag_5',
            'volume_ratio_lag_1', 'volume_ratio_lag_2', 'rsi_lag_1', 'rsi_lag_2'
        ]
        
        # Filter existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        # Prepare data
        X = df[available_cols].fillna(0)
        y = df['label'].fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        return X.values, y.values
    
    def train_model(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Train ML model for a specific symbol"""
        try:
            # Create features and labels
            df_with_features = self.create_features(df)
            df_with_labels = self.create_labels(df_with_features)
            
            X, y = self.prepare_features(df_with_labels)
            
            if len(X) < 100 or y.sum() < 10:  # Need enough data and positive samples
                return {'error': 'Insufficient data'}
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Ensemble of models
            models = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6),
                'lr': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            # Train and validate models
            best_model = None
            best_score = 0
            scores = {}
            
            for name, model in models.items():
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Predict and score
                    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                scores[name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            
            # Train final model on all data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            best_model.fit(X_scaled, y)
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
            calibrated_model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[symbol] = calibrated_model
            self.scalers[symbol] = scaler
            
            return {
                'success': True,
                'scores': scores,
                'best_score': best_score,
                'features': len(self.feature_columns),
                'samples': len(X)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Generate trading signal for a symbol"""
        try:
            if symbol not in self.models:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No model available'}
            
            # Create features
            df_with_features = self.create_features(df)
            
            # Get latest data point
            latest_data = df_with_features.iloc[-1:][self.feature_columns].fillna(0)
            latest_data = latest_data.replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = self.scalers[symbol].transform(latest_data.values)
            
            # Get prediction
            prob = self.models[symbol].predict_proba(X_scaled)[0, 1]
            
            # Technical confirmation
            latest = df_with_features.iloc[-1]
            
            # Technical signals
            rsi_signal = latest['rsi'] < 70 and latest['rsi'] > 30  # Not overbought/oversold
            trend_signal = latest['trend_strength'] >= 0  # Uptrend or neutral
            momentum_signal = latest['mom_5'] > 0.01  # Positive 5-day momentum
            volume_signal = latest['volume_ratio'] > 1.2  # Above average volume
            
            # Combined signal logic
            if prob > 0.65 and rsi_signal and trend_signal and momentum_signal:
                signal = 'BUY'
                confidence = prob
            elif prob > 0.55 and trend_signal and momentum_signal and volume_signal:
                signal = 'BUY'
                confidence = prob * 0.8  # Lower confidence
            elif prob < 0.35 or latest['rsi'] > 80 or latest['trend_strength'] < 0:
                signal = 'SELL'
                confidence = 1 - prob
            else:
                signal = 'HOLD'
                confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'ml_probability': float(prob),
                'rsi': float(latest['rsi']),
                'trend_strength': float(latest['trend_strength']),
                'momentum_5d': float(latest['mom_5']),
                'volume_ratio': float(latest['volume_ratio']),
                'price': float(latest['Close'])
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'confidence': 0.0, 'error': str(e)}
    
    def run_full_pipeline(self, retrain: bool = True) -> Dict:
        """Run complete ML trading pipeline"""
        print("🤖 STARTING ML AUTO-TRADING PIPELINE")
        print("=" * 60)
        
        results = {
            'training_results': {},
            'signals': {},
            'portfolio_actions': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Train models for universe
        if retrain:
            print(f"📚 Training ML models for {len(self.universe)} stocks...")
            for i, symbol in enumerate(self.universe):
                print(f"  Training {symbol} ({i+1}/{len(self.universe)})")
                
                df = self.download_data(symbol)
                if df is not None:
                    train_result = self.train_model(symbol, df)
                    results['training_results'][symbol] = train_result
                    
                    if train_result.get('success'):
                        print(f"    ✅ {symbol}: Score={train_result['best_score']:.3f}")
                    else:
                        print(f"    ❌ {symbol}: {train_result.get('error', 'Failed')}")
        
        # Step 2: Generate signals for all stocks
        print(f"\n📊 Generating trading signals...")
        buy_candidates = []
        sell_candidates = []
        
        for symbol in self.universe:
            if symbol in self.models:
                df = self.download_data(symbol, period="6mo")  # Recent data for signals
                if df is not None:
                    signal_data = self.generate_signal(symbol, df)
                    results['signals'][symbol] = signal_data
                    
                    if signal_data['signal'] == 'BUY' and signal_data['confidence'] > 0.6:
                        buy_candidates.append((symbol, signal_data))
                    elif signal_data['signal'] == 'SELL':
                        sell_candidates.append((symbol, signal_data))
        
        # Step 3: Portfolio management
        print(f"\n💼 Portfolio Management:")
        print(f"  Buy candidates: {len(buy_candidates)}")
        print(f"  Sell candidates: {len(sell_candidates)}")
        
        # Sort buy candidates by confidence
        buy_candidates.sort(key=lambda x: x[1]['confidence'], reverse=True)
        
        # Portfolio actions
        max_positions = 10
        position_size = 0.08  # 8% per position
        
        for symbol, signal_data in buy_candidates[:max_positions]:
            if symbol not in self.positions:
                action = {
                    'action': 'BUY',
                    'symbol': symbol,
                    'confidence': signal_data['confidence'],
                    'ml_probability': signal_data['ml_probability'],
                    'position_size': position_size,
                    'reasoning': f"ML prob: {signal_data['ml_probability']:.2f}, RSI: {signal_data['rsi']:.1f}"
                }
                results['portfolio_actions'].append(action)
                self.positions[symbol] = {
                    'entry_price': signal_data['price'],
                    'entry_date': datetime.now(),
                    'position_size': position_size
                }
                print(f"  🟢 BUY {symbol}: Confidence={signal_data['confidence']:.2f}")
        
        for symbol, signal_data in sell_candidates:
            if symbol in self.positions:
                action = {
                    'action': 'SELL',
                    'symbol': symbol,
                    'confidence': signal_data['confidence'],
                    'reasoning': f"ML prob: {signal_data['ml_probability']:.2f}, RSI: {signal_data['rsi']:.1f}"
                }
                results['portfolio_actions'].append(action)
                del self.positions[symbol]
                print(f"  🔴 SELL {symbol}: Confidence={signal_data['confidence']:.2f}")
        
        # Step 4: Performance summary
        successful_models = sum(1 for r in results['training_results'].values() if r.get('success'))
        avg_score = np.mean([r['best_score'] for r in results['training_results'].values() if r.get('success')])
        
        print(f"\n🎯 PIPELINE SUMMARY:")
        print(f"  Successful models: {successful_models}/{len(self.universe)}")
        print(f"  Average ML score: {avg_score:.3f}")
        print(f"  Active positions: {len(self.positions)}")
        print(f"  Portfolio actions: {len(results['portfolio_actions'])}")
        
        # Save results
        with open('ml_trading_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

def main():
    """Run the complete ML trading bot"""
    bot = MLTradingBot(universe_size=50, target_return=0.30)  # Start with 50 stocks
    
    # Run full pipeline
    results = bot.run_full_pipeline(retrain=True)
    
    print(f"\n🚀 ML AUTO-TRADING BOT COMPLETE!")
    print(f"Results saved to: ml_trading_results.json")
    
    return results

if __name__ == "__main__":
    main()
