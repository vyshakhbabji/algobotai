#!/usr/bin/env python3
"""
Sophisticated AI Trading Advisor
Advanced ML-based trading system with intelligent alerts and confidence scoring
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import accuracy_score
import yfinance as yf
from colorama import Fore, Back, Style, init
init(autoreset=True)

class SophisticatedTradingAdvisor:
    def __init__(self):
        """
        Initialize the sophisticated trading advisor
        """
        self.models = {}
        self.scalers = {}
        self.metadata = None
        self.predictions = {}
        self.signals = {}
        self.confidence_scores = {}
        self.market_regime = None
        self.volatility_regime = None
        
    def load_models_and_data(self):
        """
        Load all trained models and preprocessed data
        """
        print(f"{Fore.CYAN}🤖 Loading AI Models and Market Data...{Style.RESET_ALL}")
        
        # Load metadata
        with open('fixed_data/preprocessed/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load scalers
        self.scalers['feature'] = joblib.load('fixed_data/preprocessed/feature_scaler.pkl')
        self.scalers['target'] = joblib.load('fixed_data/preprocessed/target_scaler.pkl')
        
        # Load test data
        self.X_test_seq = np.load('fixed_data/preprocessed/X_test_seq.npy')
        self.X_test_flat = np.load('fixed_data/preprocessed/X_test_flat.npy')
        self.y_test = np.load('fixed_data/preprocessed/y_test.npy')
        self.test_dates = np.load('fixed_data/preprocessed/test_dates.npy', allow_pickle=True)
        
        # Load models with error handling
        model_files = {
            'rf': 'fixed_data/models/random_forest_model.pkl',
            'gb': 'fixed_data/models/gradient_boosting_model.pkl',
            'linear': 'fixed_data/models/linear_regression_model.pkl',
            'ridge': 'fixed_data/models/ridge_model.pkl'
        }
        
        for name, path in model_files.items():
            try:
                self.models[name] = joblib.load(path)
                print(f"  ✅ {name.upper()} model loaded")
            except Exception as e:
                print(f"  ❌ {name.upper()} model failed: {str(e)}")
        
        print(f"  📊 Loaded {len(self.models)} models")
        print(f"  📈 Test data: {self.X_test_seq.shape[0]} samples")
        
    def analyze_market_regime(self):
        """
        Analyze current market regime (trending, ranging, volatile)
        """
        print(f"{Fore.YELLOW}📊 Analyzing Market Regime...{Style.RESET_ALL}")
        
        # Get actual prices
        actual_prices = self.scalers['target'].inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        
        # Calculate various market indicators
        returns = pd.Series(actual_prices).pct_change().dropna()
        
        # Trend analysis
        price_series = pd.Series(actual_prices)
        sma_5 = price_series.rolling(5).mean()
        sma_10 = price_series.rolling(10).mean()
        
        # Volatility analysis
        volatility = returns.rolling(5).std() * np.sqrt(252)
        avg_volatility = volatility.mean()
        
        # Trend strength
        trend_strength = ((sma_5.iloc[-1] - sma_10.iloc[-1]) / sma_10.iloc[-1]) * 100
        
        # Determine market regime
        if trend_strength > 2:
            self.market_regime = "STRONG_UPTREND"
            regime_color = Fore.GREEN
        elif trend_strength > 0.5:
            self.market_regime = "UPTREND"
            regime_color = Fore.LIGHTGREEN_EX
        elif trend_strength < -2:
            self.market_regime = "STRONG_DOWNTREND"
            regime_color = Fore.RED
        elif trend_strength < -0.5:
            self.market_regime = "DOWNTREND"
            regime_color = Fore.LIGHTRED_EX
        else:
            self.market_regime = "SIDEWAYS"
            regime_color = Fore.YELLOW
            
        # Determine volatility regime
        if avg_volatility > 0.3:
            self.volatility_regime = "HIGH_VOLATILITY"
        elif avg_volatility > 0.15:
            self.volatility_regime = "MEDIUM_VOLATILITY"
        else:
            self.volatility_regime = "LOW_VOLATILITY"
            
        print(f"  🎯 Market Regime: {regime_color}{self.market_regime}{Style.RESET_ALL}")
        print(f"  📈 Trend Strength: {trend_strength:.2f}%")
        print(f"  📊 Volatility Regime: {self.volatility_regime}")
        print(f"  🌊 Avg Volatility: {avg_volatility:.1%}")
        
        return {
            'regime': self.market_regime,
            'trend_strength': trend_strength,
            'volatility': avg_volatility,
            'volatility_regime': self.volatility_regime
        }
    
    def generate_ensemble_predictions(self):
        """
        Generate predictions using ensemble of models with weights
        """
        print(f"{Fore.CYAN}🧠 Generating AI Predictions...{Style.RESET_ALL}")
        
        # Model weights based on performance (you can adjust these)
        model_weights = {
            'rf': 0.3,    # Random Forest - good for non-linear patterns
            'gb': 0.4,    # Gradient Boosting - often best performer
            'linear': 0.15, # Linear - good for trends
            'ridge': 0.15   # Ridge - regularized linear
        }
        
        individual_predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name in model_weights:
                try:
                    # Use reshaped sequence data for ML models
                    X_test_flat_reshaped = self.X_test_seq.reshape(self.X_test_seq.shape[0], -1)
                    pred_scaled = model.predict(X_test_flat_reshaped)
                    
                    # Inverse transform
                    pred_original = self.scalers['target'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                    individual_predictions[model_name] = pred_original
                    
                    print(f"  ✅ {model_name.upper()}: Generated {len(pred_original)} predictions")
                    
                except Exception as e:
                    print(f"  ❌ {model_name.upper()}: {str(e)}")
        
        # Create weighted ensemble prediction
        if individual_predictions:
            ensemble_pred = np.zeros(len(list(individual_predictions.values())[0]))
            total_weight = 0
            
            for model_name, predictions in individual_predictions.items():
                weight = model_weights.get(model_name, 0)
                ensemble_pred += predictions * weight
                total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                ensemble_pred /= total_weight
                individual_predictions['ensemble'] = ensemble_pred
                print(f"  🎯 ENSEMBLE: Weighted prediction created")
        
        self.predictions = individual_predictions
        return individual_predictions
    
    def calculate_confidence_scores(self):
        """
        Calculate confidence scores for predictions
        """
        print(f"{Fore.MAGENTA}🎯 Calculating Confidence Scores...{Style.RESET_ALL}")
        
        if not self.predictions:
            return {}
            
        actual_prices = self.scalers['target'].inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        
        confidence_scores = {}
        
        for model_name, predictions in self.predictions.items():
            # Calculate prediction confidence based on multiple factors
            
            # 1. Model agreement (how close predictions are to ensemble)
            if 'ensemble' in self.predictions and model_name != 'ensemble':
                agreement = 1 - np.mean(np.abs(predictions - self.predictions['ensemble']) / actual_prices)
                agreement = max(0, min(1, agreement))
            else:
                agreement = 0.8  # Default for ensemble
            
            # 2. Prediction stability (how consistent predictions are)
            pred_changes = np.abs(np.diff(predictions))
            stability = 1 - np.std(pred_changes) / np.mean(actual_prices)
            stability = max(0, min(1, stability))
            
            # 3. Market regime compatibility
            regime_bonus = 0
            if self.market_regime in ['STRONG_UPTREND', 'UPTREND']:
                # Boost confidence for models predicting upward movement
                avg_pred_return = np.mean((predictions / actual_prices - 1) * 100)
                if avg_pred_return > 0:
                    regime_bonus = 0.2
            elif self.market_regime in ['STRONG_DOWNTREND', 'DOWNTREND']:
                # Boost confidence for models predicting downward movement
                avg_pred_return = np.mean((predictions / actual_prices - 1) * 100)
                if avg_pred_return < 0:
                    regime_bonus = 0.2
            
            # Combine factors
            total_confidence = (agreement * 0.4 + stability * 0.4 + 0.2) + regime_bonus
            total_confidence = max(0, min(1, total_confidence))
            
            confidence_scores[model_name] = {
                'total': total_confidence,
                'agreement': agreement,
                'stability': stability,
                'regime_bonus': regime_bonus
            }
            
            print(f"  📊 {model_name.upper()}: {total_confidence:.1%} confidence")
        
        self.confidence_scores = confidence_scores
        return confidence_scores
    
    def generate_intelligent_signals(self):
        """
        Generate intelligent trading signals with sophisticated logic
        """
        print(f"{Fore.GREEN}🚨 Generating Intelligent Trading Signals...{Style.RESET_ALL}")
        
        if 'ensemble' not in self.predictions:
            print("❌ No ensemble predictions available")
            return None
            
        predictions = self.predictions['ensemble']
        actual_prices = self.scalers['target'].inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        dates = pd.to_datetime(self.test_dates)
        
        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'Date': dates,
            'Actual_Price': actual_prices,
            'Predicted_Price': predictions,
            'Predicted_Return': (predictions / actual_prices - 1) * 100
        })
        
        # Add technical indicators
        signals_df['SMA_3'] = signals_df['Actual_Price'].rolling(3).mean()
        signals_df['SMA_5'] = signals_df['Actual_Price'].rolling(5).mean()
        signals_df['SMA_10'] = signals_df['Actual_Price'].rolling(10).mean()
        signals_df['RSI'] = self.calculate_rsi(signals_df['Actual_Price'])
        signals_df['MACD'] = self.calculate_macd(signals_df['Actual_Price'])
        signals_df['Volatility'] = signals_df['Actual_Price'].rolling(5).std()
        
        # Initialize signals
        signals_df['Signal'] = 0  # -1: SELL, 0: HOLD, 1: BUY
        signals_df['Signal_Strength'] = 0.0
        signals_df['Confidence'] = 0.0
        signals_df['Action'] = 'HOLD'
        signals_df['Reasoning'] = ''
        
        # Get ensemble confidence
        ensemble_confidence = self.confidence_scores.get('ensemble', {}).get('total', 0.5)
        
        # Generate signals for each time point
        for i in range(len(signals_df)):
            signal, strength, confidence, action, reasoning = self.analyze_trading_opportunity(
                signals_df.iloc[i], ensemble_confidence
            )
            
            signals_df.iloc[i, signals_df.columns.get_loc('Signal')] = signal
            signals_df.iloc[i, signals_df.columns.get_loc('Signal_Strength')] = strength
            signals_df.iloc[i, signals_df.columns.get_loc('Confidence')] = confidence
            signals_df.iloc[i, signals_df.columns.get_loc('Action')] = action
            signals_df.iloc[i, signals_df.columns.get_loc('Reasoning')] = reasoning
        
        self.signals['intelligent'] = signals_df
        return signals_df
    
    def analyze_trading_opportunity(self, row, base_confidence):
        """
        Analyze individual trading opportunity with sophisticated logic
        """
        pred_return = row['Predicted_Return']
        price = row['Actual_Price']
        sma_3 = row.get('SMA_3', price)
        sma_5 = row.get('SMA_5', price)
        rsi = row.get('RSI', 50)
        macd = row.get('MACD', 0)
        volatility = row.get('Volatility', 0)
        
        # Technical indicators
        short_trend = (price - sma_3) / sma_3 * 100 if sma_3 > 0 else 0
        medium_trend = (sma_3 - sma_5) / sma_5 * 100 if sma_5 > 0 else 0
        
        # Score components
        ml_score = 0
        technical_score = 0
        regime_score = 0
        risk_score = 0
        
        # 1. ML Model Score
        if pred_return > 5:
            ml_score = 3  # Strong buy
        elif pred_return > 2:
            ml_score = 2  # Buy
        elif pred_return > -2:
            ml_score = 1  # Weak buy
        elif pred_return > -5:
            ml_score = -1  # Weak sell
        elif pred_return > -10:
            ml_score = -2  # Sell
        else:
            ml_score = -3  # Strong sell
        
        # 2. Technical Analysis Score
        tech_score = 0
        
        # RSI component
        if rsi < 30:
            tech_score += 1  # Oversold - buy signal
        elif rsi > 70:
            tech_score -= 1  # Overbought - sell signal
            
        # MACD component
        if macd > 0:
            tech_score += 1
        else:
            tech_score -= 1
            
        # Trend component
        if short_trend > 1 and medium_trend > 0.5:
            tech_score += 2  # Strong uptrend
        elif short_trend < -1 and medium_trend < -0.5:
            tech_score -= 2  # Strong downtrend
            
        technical_score = tech_score
        
        # 3. Market Regime Score
        if self.market_regime == 'STRONG_UPTREND':
            regime_score = 2
        elif self.market_regime == 'UPTREND':
            regime_score = 1
        elif self.market_regime == 'STRONG_DOWNTREND':
            regime_score = -2
        elif self.market_regime == 'DOWNTREND':
            regime_score = -1
        else:
            regime_score = 0
            
        # 4. Risk Score (based on volatility)
        if self.volatility_regime == 'HIGH_VOLATILITY':
            risk_score = -1  # Reduce position in high vol
        elif self.volatility_regime == 'LOW_VOLATILITY':
            risk_score = 1   # Increase position in low vol
        else:
            risk_score = 0
        
        # Combine scores
        total_score = ml_score + technical_score + regime_score + risk_score
        
        # Determine action
        if total_score >= 4:
            signal = 1
            action = "🚀 STRONG BUY"
            strength = min(100, total_score * 10)
            confidence = min(0.95, base_confidence + 0.2)
            reasoning = f"ML:{ml_score}, Tech:{technical_score}, Regime:{regime_score}, Risk:{risk_score}"
        elif total_score >= 2:
            signal = 1
            action = "📈 BUY"
            strength = min(80, total_score * 10)
            confidence = min(0.85, base_confidence + 0.1)
            reasoning = f"ML:{ml_score}, Tech:{technical_score}, Regime:{regime_score}"
        elif total_score <= -4:
            signal = -1
            action = "🔥 STRONG SELL"
            strength = min(100, abs(total_score) * 10)
            confidence = min(0.95, base_confidence + 0.2)
            reasoning = f"ML:{ml_score}, Tech:{technical_score}, Regime:{regime_score}, Risk:{risk_score}"
        elif total_score <= -2:
            signal = -1
            action = "📉 SELL"
            strength = min(80, abs(total_score) * 10)
            confidence = min(0.85, base_confidence + 0.1)
            reasoning = f"ML:{ml_score}, Tech:{technical_score}, Regime:{regime_score}"
        else:
            signal = 0
            action = "⏸️ HOLD"
            strength = 0
            confidence = base_confidence
            reasoning = f"Mixed signals - ML:{ml_score}, Tech:{technical_score}"
        
        return signal, strength, confidence, action, reasoning
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period:
            return pd.Series([50] * len(prices), index=prices.index)
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return pd.Series([0] * len(prices), index=prices.index)
            
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)
    
    def generate_alerts(self):
        """
        Generate intelligent alerts with different urgency levels
        """
        if 'intelligent' not in self.signals:
            return []
            
        signals_df = self.signals['intelligent']
        alerts = []
        
        print(f"\n{Back.BLACK}{Fore.WHITE}{'='*80}{Style.RESET_ALL}")
        print(f"{Back.BLACK}{Fore.WHITE}🚨 SOPHISTICATED TRADING ADVISOR ALERTS 🚨{Style.RESET_ALL}")
        print(f"{Back.BLACK}{Fore.WHITE}{'='*80}{Style.RESET_ALL}\n")
        
        # Get latest signals
        latest_signals = signals_df.tail(5)  # Last 5 trading days
        
        for idx, row in latest_signals.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            action = row['Action']
            confidence = row['Confidence']
            reasoning = row['Reasoning']
            price = row['Actual_Price']
            pred_return = row['Predicted_Return']
            
            # Color coding based on action
            if 'STRONG BUY' in action:
                color = f"{Back.GREEN}{Fore.WHITE}"
                icon = "🚀🚀🚀"
            elif 'BUY' in action:
                color = f"{Back.LIGHTGREEN_EX}{Fore.BLACK}"
                icon = "📈📈"
            elif 'STRONG SELL' in action:
                color = f"{Back.RED}{Fore.WHITE}"
                icon = "🔥🔥🔥"
            elif 'SELL' in action:
                color = f"{Back.LIGHTRED_EX}{Fore.BLACK}"
                icon = "📉📉"
            else:
                color = f"{Back.YELLOW}{Fore.BLACK}"
                icon = "⏸️"
            
            alert = {
                'date': date,
                'action': action,
                'confidence': confidence,
                'price': price,
                'predicted_return': pred_return,
                'reasoning': reasoning
            }
            alerts.append(alert)
            
            # Print alert
            print(f"{color} {icon} {date} | {action} | ${price:.2f} {Style.RESET_ALL}")
            print(f"   Confidence: {confidence:.1%} | Predicted Return: {pred_return:+.1f}%")
            print(f"   Reasoning: {reasoning}")
            print()
        
        # Summary statistics
        buy_signals = len(signals_df[signals_df['Signal'] == 1])
        sell_signals = len(signals_df[signals_df['Signal'] == -1])
        hold_signals = len(signals_df[signals_df['Signal'] == 0])
        avg_confidence = signals_df['Confidence'].mean()
        
        print(f"{Back.BLUE}{Fore.WHITE} SIGNAL SUMMARY {Style.RESET_ALL}")
        print(f"📈 Buy Signals: {buy_signals}")
        print(f"📉 Sell Signals: {sell_signals}")
        print(f"⏸️ Hold Signals: {hold_signals}")
        print(f"🎯 Average Confidence: {avg_confidence:.1%}")
        print(f"🏛️ Market Regime: {self.market_regime}")
        print()
        
        return alerts
    
    def run_sophisticated_analysis(self):
        """
        Run complete sophisticated trading analysis
        """
        print(f"\n{Back.MAGENTA}{Fore.WHITE}🤖 SOPHISTICATED AI TRADING ADVISOR 🤖{Style.RESET_ALL}\n")
        
        try:
            # Step 1: Load models and data
            self.load_models_and_data()
            
            # Step 2: Analyze market regime
            market_analysis = self.analyze_market_regime()
            
            # Step 3: Generate ensemble predictions
            predictions = self.generate_ensemble_predictions()
            
            # Step 4: Calculate confidence scores
            confidence_scores = self.calculate_confidence_scores()
            
            # Step 5: Generate intelligent signals
            signals = self.generate_intelligent_signals()
            
            # Step 6: Generate alerts
            alerts = self.generate_alerts()
            
            print(f"{Back.GREEN}{Fore.WHITE}✅ ANALYSIS COMPLETE! CHECK ALERTS ABOVE ✅{Style.RESET_ALL}\n")
            
            return {
                'market_analysis': market_analysis,
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'signals': signals,
                'alerts': alerts
            }
            
        except Exception as e:
            print(f"{Back.RED}{Fore.WHITE}❌ ERROR: {str(e)}{Style.RESET_ALL}")
            raise

def main():
    """
    Main function to run sophisticated trading advisor
    """
    advisor = SophisticatedTradingAdvisor()
    results = advisor.run_sophisticated_analysis()
    return advisor, results

if __name__ == "__main__":
    advisor, results = main()
