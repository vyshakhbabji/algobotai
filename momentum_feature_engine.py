#!/usr/bin/env python3
"""
MOMENTUM-BASED FEATURE ENGINEERING FRAMEWORK
Core feature set for ML models, options trading, and portfolio management
Based on institutional momentum research (Jegadeesh & Titman 1993)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MomentumFeatureEngine:
    def __init__(self):
        """Initialize momentum-based feature engineering system"""
        
        # CORE MOMENTUM PERIODS (Institutional Standard)
        self.momentum_periods = {
            'short_term': [1, 3, 5],      # 1-5 day momentum (micro trends)
            'medium_term': [10, 20, 30],   # 2-6 week momentum (swing trends)
            'long_term': [60, 120, 252],   # 3-12 month momentum (macro trends)
            'institutional': [63, 126, 252] # 3M, 6M, 12M (standard institutional)
        }
        
        # VOLATILITY PERIODS (Risk Adjustment)
        self.volatility_periods = [5, 10, 20, 60, 252]
        
        # VOLUME ANALYSIS PERIODS
        self.volume_periods = [5, 10, 20, 60]

    def calculate_momentum_features(self, data, symbol=None):
        """Calculate comprehensive momentum feature set"""
        features = data.copy()
        
        print(f"🔧 Calculating momentum features for {symbol or 'stock'}...")
        
        # ===========================================
        # 1. CORE MOMENTUM FEATURES (PRIMARY)
        # ===========================================
        
        # Price momentum (returns over different periods)
        for period in [1, 3, 5, 10, 20, 30, 60, 120, 252]:
            features[f'momentum_{period}d'] = features['Close'].pct_change(period)
            
        # Risk-adjusted momentum (Sharpe-style)
        for period in [20, 60, 252]:
            returns = features['Close'].pct_change()
            rolling_mean = returns.rolling(period).mean()
            rolling_std = returns.rolling(period).std()
            features[f'risk_adj_momentum_{period}d'] = rolling_mean / (rolling_std + 1e-8)
        
        # ===========================================
        # 2. INSTITUTIONAL MOMENTUM SCORE
        # ===========================================
        
        # 3-factor momentum score (3M, 6M, 12M)
        mom_3m = features['Close'].pct_change(63)   # 3 months
        mom_6m = features['Close'].pct_change(126)  # 6 months  
        mom_12m = features['Close'].pct_change(252) # 12 months
        
        # Weighted institutional score (6M gets highest weight)
        features['institutional_momentum'] = (
            0.3 * mom_3m + 0.5 * mom_6m + 0.2 * mom_12m
        )
        
        # ===========================================
        # 3. TREND STRENGTH FEATURES
        # ===========================================
        
        # Moving average relationships
        features['MA5'] = features['Close'].rolling(5).mean()
        features['MA10'] = features['Close'].rolling(10).mean()
        features['MA20'] = features['Close'].rolling(20).mean()
        features['MA50'] = features['Close'].rolling(50).mean()
        features['MA200'] = features['Close'].rolling(200).mean()
        
        # Trend strength indicators
        ma20 = features['Close'].rolling(20).mean()
        ma50 = features['Close'].rolling(50).mean()
        ma200 = features['Close'].rolling(200).mean()
        
        features['price_vs_ma20'] = (features['Close'] - ma20) / ma20
        features['price_vs_ma50'] = (features['Close'] - ma50) / ma50
        features['price_vs_ma200'] = (features['Close'] - ma200) / ma200
        
        # MA slope (trend direction)
        for period in [5, 10, 20, 50]:
            ma_col = f'MA{period}'
            features[f'{ma_col}_slope'] = features[ma_col].pct_change(5)
        
        # ===========================================
        # 4. VOLATILITY FEATURES (Risk Metrics)
        # ===========================================
        
        # Rolling volatility (annualized)
        for period in self.volatility_periods:
            returns = features['Close'].pct_change()
            features[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            
        # Volatility momentum (changing volatility)
        features['vol_momentum_20d'] = (
            features['volatility_20d'] / features['volatility_60d']
        ) - 1
        
        # ===========================================
        # 5. VOLUME FEATURES (Momentum Confirmation)
        # ===========================================
        
        # Volume moving averages
        for period in self.volume_periods:
            features[f'volume_ma_{period}d'] = features['Volume'].rolling(period).mean()
            
        # Volume ratios (current vs average)
        for period in [5, 20]:
            vol_ma = features['Volume'].rolling(period).mean()
            features[f'volume_ratio_{period}d'] = features['Volume'] / vol_ma
            
        # Price-volume momentum
        vol_ratio_5d = features['Volume'] / features['Volume'].rolling(5).mean()
        mom_5d = features['Close'].pct_change(5)
        features['pv_momentum_5d'] = mom_5d * vol_ratio_5d
        
        # ===========================================
        # 6. MOMENTUM QUALITY FEATURES
        # ===========================================
        
        # Momentum consistency (what % of days were positive)
        for period in [20, 60]:
            returns = features['Close'].pct_change()
            features[f'momentum_consistency_{period}d'] = (
                returns.rolling(period).apply(lambda x: (x > 0).sum() / len(x))
            )
            
        # Momentum acceleration (momentum of momentum)
        mom_20d = features['Close'].pct_change(20)
        features['momentum_acceleration_20d'] = mom_20d - mom_20d.shift(10)
        
        # ===========================================
        # 7. RELATIVE STRENGTH FEATURES
        # ===========================================
        
        # RSI (traditional)
        delta = features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Stochastic momentum
        for period in [14, 20]:
            low_min = features['Low'].rolling(period).min()
            high_max = features['High'].rolling(period).max()
            features[f'stoch_{period}d'] = (
                (features['Close'] - low_min) / (high_max - low_min) * 100
            )
        
        return features

    def create_ml_features(self, data, symbol=None):
        """Create ML-ready feature matrix"""
        print(f"🤖 Creating ML features for {symbol or 'stock'}...")
        
        # Calculate all momentum features
        features_df = self.calculate_momentum_features(data, symbol)
        
        # Select core ML features (avoid data leakage)
        ml_features = [
            # Core momentum (different timeframes)
            'momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_10d', 
            'momentum_20d', 'momentum_60d', 'momentum_120d',
            
            # Risk-adjusted momentum
            'risk_adj_momentum_20d', 'risk_adj_momentum_60d', 'risk_adj_momentum_252d',
            
            # Institutional momentum
            'institutional_momentum',
            
            # Trend strength
            'price_vs_ma20', 'price_vs_ma50', 'price_vs_ma200',
            'MA5_slope', 'MA10_slope', 'MA20_slope',
            
            # Volatility features
            'volatility_5d', 'volatility_20d', 'volatility_60d',
            'vol_momentum_20d',
            
            # Volume features
            'volume_ratio_5d', 'volume_ratio_20d',
            'pv_momentum_5d',
            
            # Quality features
            'momentum_consistency_20d', 'momentum_acceleration_20d',
            
            # Technical indicators
            'rsi', 'stoch_14d', 'stoch_20d'
        ]
        
        # Create target variable (forward returns)
        features_df['target_1d'] = features_df['Close'].pct_change().shift(-1)  # Next day return
        features_df['target_5d'] = features_df['Close'].pct_change(5).shift(-5)  # 5-day forward return
        features_df['target_20d'] = features_df['Close'].pct_change(20).shift(-20)  # 20-day forward return
        
        # Select features and handle missing values
        ml_data = features_df[ml_features + ['target_1d', 'target_5d', 'target_20d']].dropna()
        
        print(f"✅ Created {len(ml_features)} features, {len(ml_data)} samples")
        
        return ml_data, ml_features

    def create_options_features(self, data, symbol=None):
        """Create options-specific momentum features"""
        print(f"📊 Creating options features for {symbol or 'stock'}...")
        
        # Calculate base momentum features
        features_df = self.calculate_momentum_features(data, symbol)
        
        # ===========================================
        # OPTIONS-SPECIFIC FEATURES
        # ===========================================
        
        # Implied volatility proxies (using realized vol)
        features_df['iv_rank_20d'] = (
            features_df['volatility_20d'].rolling(252).rank(pct=True)
        )
        features_df['iv_rank_60d'] = (
            features_df['volatility_60d'].rolling(252).rank(pct=True)
        )
        
        # Momentum breakout probability
        for period in [5, 10, 20]:
            high_roll = features_df['High'].rolling(period).max()
            low_roll = features_df['Low'].rolling(period).min()
            features_df[f'breakout_prob_{period}d'] = (
                (features_df['Close'] > high_roll.shift(1)).astype(int) +
                (features_df['Close'] < low_roll.shift(1)).astype(int) * -1
            )
        
        # Momentum regime classification
        mom_20d = features_df['momentum_20d']
        mom_60d = features_df['momentum_60d']
        
        conditions = [
            (mom_20d > 0.05) & (mom_60d > 0.05),  # Strong uptrend
            (mom_20d > 0) & (mom_60d > 0),        # Weak uptrend
            (mom_20d < -0.05) & (mom_60d < -0.05), # Strong downtrend
            (mom_20d < 0) & (mom_60d < 0),        # Weak downtrend
        ]
        choices = [3, 1, -3, -1]  # Strong up, weak up, weak down, strong down
        features_df['momentum_regime'] = np.select(conditions, choices, default=0)
        
        # Options strategy signals
        features_df['call_signal'] = (
            (features_df['institutional_momentum'] > 0.1) &
            (features_df['momentum_20d'] > 0.02) &
            (features_df['iv_rank_20d'] < 0.3)  # Low IV
        ).astype(int)
        
        features_df['put_signal'] = (
            (features_df['institutional_momentum'] < -0.1) &
            (features_df['momentum_20d'] < -0.02) &
            (features_df['iv_rank_20d'] < 0.3)  # Low IV
        ).astype(int)
        
        return features_df

    def create_portfolio_signals(self, symbols_data):
        """Create portfolio-level momentum signals"""
        print("📈 Creating portfolio momentum signals...")
        
        portfolio_signals = {}
        
        for symbol, data in symbols_data.items():
            # Calculate momentum features
            features = self.calculate_momentum_features(data, symbol)
            
            # Latest values
            latest = features.iloc[-1]
            
            # Momentum score (institutional formula)
            momentum_score = (
                0.3 * latest['momentum_60d'] +   # Use 60d as proxy for 3M
                0.5 * latest['momentum_120d'] +  # Use 120d as proxy for 6M  
                0.2 * latest['momentum_252d']    # 252d for 12M
            ) if all(pd.notna([latest['momentum_60d'], latest['momentum_120d'], latest['momentum_252d']])) else np.nan
            
            # Risk adjustment
            vol_20d = latest['volatility_20d']
            risk_adjusted_score = momentum_score / (vol_20d + 0.01) if pd.notna(vol_20d) else np.nan
            
            # Signal classification
            if pd.notna(risk_adjusted_score):
                if risk_adjusted_score > 0.5:
                    signal = 'STRONG_BUY'
                elif risk_adjusted_score > 0.2:
                    signal = 'BUY'
                elif risk_adjusted_score < -0.5:
                    signal = 'STRONG_SELL'
                elif risk_adjusted_score < -0.2:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
            else:
                signal = 'HOLD'
            
            portfolio_signals[symbol] = {
                'signal': signal,
                'momentum_score': momentum_score,
                'risk_adjusted_score': risk_adjusted_score,
                'current_price': latest['Close'],
                'vol_20d': vol_20d
            }
        
        return portfolio_signals

def demonstrate_momentum_features():
    """Demonstrate momentum feature engineering"""
    print("🚀 MOMENTUM FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize feature engine
    engine = MomentumFeatureEngine()
    
    # Download sample data
    print("📊 Downloading NVDA data for demonstration...")
    data = yf.download('NVDA', start='2023-01-01', end='2024-08-09', progress=False)
    
    if data.empty:
        print("❌ Could not download data")
        return
    
    # 1. ML Features
    print("\n🤖 MACHINE LEARNING FEATURES:")
    print("-" * 40)
    ml_data, ml_features = engine.create_ml_features(data, 'NVDA')
    print(f"✅ Created {len(ml_features)} ML features")
    print(f"📊 Latest institutional momentum: {ml_data['institutional_momentum'].iloc[-1]:.4f}")
    print(f"📈 Latest 20d momentum: {ml_data['momentum_20d'].iloc[-1]:.4f}")
    print(f"📉 Latest volatility: {ml_data['volatility_20d'].iloc[-1]:.4f}")
    
    # 2. Options Features
    print("\n📊 OPTIONS TRADING FEATURES:")
    print("-" * 40)
    options_data = engine.create_options_features(data, 'NVDA')
    latest_options = options_data.iloc[-1]
    print(f"✅ Options features calculated")
    print(f"📊 Momentum regime: {latest_options['momentum_regime'].iloc[0] if hasattr(latest_options['momentum_regime'], 'iloc') else latest_options['momentum_regime']}")
    print(f"🔄 Call signal: {'YES' if (latest_options['call_signal'].iloc[0] if hasattr(latest_options['call_signal'], 'iloc') else latest_options['call_signal']) else 'NO'}")
    print(f"🔄 Put signal: {'YES' if (latest_options['put_signal'].iloc[0] if hasattr(latest_options['put_signal'], 'iloc') else latest_options['put_signal']) else 'NO'}")
    print(f"📈 IV rank (20d): {latest_options['iv_rank_20d'].iloc[0] if hasattr(latest_options['iv_rank_20d'], 'iloc') else latest_options['iv_rank_20d']:.2f}")
    
    # 3. Portfolio Signals
    print("\n📈 PORTFOLIO MOMENTUM SIGNALS:")
    print("-" * 40)
    symbols_data = {'NVDA': data}
    portfolio_signals = engine.create_portfolio_signals(symbols_data)
    
    nvda_signal = portfolio_signals['NVDA']
    print(f"✅ Portfolio signal: {nvda_signal['signal']}")
    print(f"📊 Momentum score: {nvda_signal['momentum_score']:.4f}")
    print(f"🎯 Risk-adjusted score: {nvda_signal['risk_adjusted_score']:.4f}")
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 30)
    print("✅ Same momentum features work across:")
    print("   🤖 ML models (prediction)")
    print("   📊 Options trading (volatility + direction)")
    print("   📈 Portfolio management (risk-adjusted ranking)")
    print("🏛️ All based on institutional momentum research")
    
    return engine, ml_data, options_data, portfolio_signals

if __name__ == "__main__":
    engine, ml_data, options_data, portfolio_signals = demonstrate_momentum_features()
