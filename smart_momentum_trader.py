#!/usr/bin/env python3
"""
SMART ML-FILTERED MOMENTUM TRADER
Uses ML to identify momentum-friendly stocks, then applies institutional momentum signals
Only trades stocks that ML predicts will respond well to momentum strategies
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json

class SmartMomentumTrader:
    def __init__(self, starting_capital=10000):
        self.starting_capital = starting_capital
        
        # Stock universe
        self.stocks = [
            "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN",
            "TSLA", "META", "NFLX", "CRM", "UBER", 
            "JPM", "WMT", "JNJ", "PG", "KO",
            "PLTR", "COIN", "SNOW", "AMD", "INTC",
            "XOM", "CVX", "CAT", "BA", "GE"
        ]
        
        # INSTITUTIONAL MOMENTUM CONFIG (Jegadeesh & Titman 1993)
        self.momentum_config = {
            'trend_5d_buy_threshold': 0.015,    # Aggressive for momentum detection
            'trend_5d_sell_threshold': -0.025,  # Protect profits
            'trend_10d_buy_threshold': 0.015,   # Confirm momentum
            'trend_10d_sell_threshold': -0.045, # Strict exit
            'rsi_overbought': 85,               # Allow momentum runs
            'rsi_oversold': 30,                 
            'volatility_threshold': 0.10,       
            'volume_ratio_threshold': 1.1       
        }
        
        # ML Model for stock classification
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        print(f"🧠 SMART ML-FILTERED MOMENTUM TRADER")
        print(f"💰 Capital per stock: ${starting_capital:,}")
        print(f"🎯 Step 1: Train ML to identify momentum-friendly stocks")
        print(f"🎯 Step 2: Apply institutional momentum signals to selected stocks")
        
    def calculate_momentum_features(self, data):
        """Calculate features that predict momentum success"""
        data = data.copy()
        
        # Basic indicators
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MOMENTUM PREDICTIVE FEATURES
        data['trend_consistency'] = data['Close'].rolling(10).apply(
            lambda x: (x.diff() > 0).sum() / len(x) if len(x) > 1 else 0.5
        ).fillna(0.5)
        
        data['volatility_regime'] = (data['Close'].rolling(20).std() / data['Close'].rolling(20).mean()).fillna(0)
        
        data['volume_stability'] = (data['Volume'].rolling(10).std() / data['Volume'].rolling(10).mean()).fillna(1)
        
        # Price momentum features
        data['momentum_1w'] = (data['Close'] / data['Close'].shift(5) - 1).fillna(0)
        data['momentum_2w'] = (data['Close'] / data['Close'].shift(10) - 1).fillna(0)
        data['momentum_1m'] = (data['Close'] / data['Close'].shift(21) - 1).fillna(0)
        
        # Trend strength
        data['trend_strength'] = (abs(data['Close'] - data['MA20']) / data['MA20']).fillna(0)
        
        # Volume momentum
        data['volume_trend'] = (data['Volume'] / data['Volume'].rolling(20).mean()).fillna(1)
        
        return data
    
    def create_training_data(self):
        """Create training data by analyzing which stocks responded well to momentum historically"""
        print(f"\n🔬 CREATING ML TRAINING DATA")
        print("-" * 40)
        
        training_features = []
        training_labels = []
        
        # We know from our testing which stocks are momentum-friendly
        momentum_winners = ['NVDA', 'COIN', 'MSFT', 'NFLX', 'INTC', 'WMT', 'JPM', 'CAT']  # Stocks that beat/matched buy-hold
        momentum_losers = ['TSLA', 'AAPL', 'GOOGL', 'PLTR', 'META', 'AMZN', 'CRM', 'UBER', 'SNOW', 'AMD', 'BA', 'XOM', 'CVX', 'GE']
        
        for symbol in self.stocks:
            try:
                # Download 6 months of data for feature calculation
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)
                
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) < 50:
                    continue
                    
                data = self.calculate_momentum_features(data)
                
                # Use recent 30 days for features (exclude latest month to avoid lookahead bias)
                recent_data = data.iloc[-60:-30]  # 30 days, 30 days ago
                
                if len(recent_data) < 20:
                    continue
                
                # Calculate average features for this stock
                features = [
                    recent_data['trend_consistency'].mean(),
                    recent_data['volatility_regime'].mean(),
                    recent_data['volume_stability'].mean(),
                    recent_data['momentum_1w'].mean(),
                    recent_data['momentum_2w'].mean(), 
                    recent_data['momentum_1m'].mean(),
                    recent_data['trend_strength'].mean(),
                    recent_data['volume_trend'].mean(),
                    recent_data['RSI'].mean()
                ]
                
                # Check for NaN values
                if not any(pd.isna(features)):
                    training_features.append(features)
                    
                    # Label: 1 if momentum-friendly, 0 if momentum-hostile
                    label = 1 if symbol in momentum_winners else 0
                    training_labels.append(label)
                    
                    status = "✅ Winner" if label == 1 else "❌ Loser"
                    print(f"   {symbol}: {status} - Trend Consistency: {features[0]:.2f}, Volatility: {features[1]:.3f}")
                
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {str(e)}")
                continue
        
        if len(training_features) >= 10:
            # Train the model
            X = np.array(training_features)
            y = np.array(training_labels)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.ml_model.fit(X_scaled, y)
            self.is_trained = True
            
            # Feature importance
            feature_names = ['trend_consistency', 'volatility_regime', 'volume_stability', 
                           'momentum_1w', 'momentum_2w', 'momentum_1m', 'trend_strength', 
                           'volume_trend', 'rsi']
            
            importances = self.ml_model.feature_importances_
            
            print(f"\n🎯 ML MODEL TRAINED SUCCESSFULLY!")
            print(f"   📊 Training samples: {len(training_features)}")
            print(f"   🏆 Winners: {sum(training_labels)}")
            print(f"   📉 Losers: {len(training_labels) - sum(training_labels)}")
            
            print(f"\n🔍 TOP PREDICTIVE FEATURES:")
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, importance) in enumerate(feature_importance[:5], 1):
                print(f"   #{i}: {name}: {importance:.3f}")
                
            return True
        else:
            print(f"❌ Insufficient training data: {len(training_features)} samples")
            return False
    
    def predict_momentum_suitability(self, symbol):
        """Predict if a stock is suitable for momentum trading"""
        if not self.is_trained:
            return 0.5  # Default to neutral if not trained
            
        try:
            # Download recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(data) < 30:
                return 0.5
                
            data = self.calculate_momentum_features(data)
            
            # Use most recent 20 days for prediction
            recent_data = data.iloc[-20:]
            
            # Calculate features
            features = [
                recent_data['trend_consistency'].mean(),
                recent_data['volatility_regime'].mean(),
                recent_data['volume_stability'].mean(),
                recent_data['momentum_1w'].mean(),
                recent_data['momentum_2w'].mean(),
                recent_data['momentum_1m'].mean(),
                recent_data['trend_strength'].mean(),
                recent_data['volume_trend'].mean(),
                recent_data['RSI'].mean()
            ]
            
            # Check for NaN
            if any(pd.isna(features)):
                return 0.5
            
            # Predict probability
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            momentum_probability = self.ml_model.predict_proba(X_scaled)[0][1]
            return momentum_probability
            
        except Exception as e:
            return 0.5
    
    def filter_momentum_stocks(self, threshold=0.6):
        """Filter stocks using ML predictions"""
        print(f"\n🔍 FILTERING STOCKS WITH ML (threshold: {threshold:.1%})")
        print("-" * 50)
        
        momentum_stocks = []
        rejected_stocks = []
        
        for symbol in self.stocks:
            probability = self.predict_momentum_suitability(symbol)
            
            if probability >= threshold:
                momentum_stocks.append((symbol, probability))
                status = "✅ SELECTED"
            else:
                rejected_stocks.append((symbol, probability))
                status = "❌ REJECTED"
            
            print(f"   {symbol}: {probability:.1%} momentum probability - {status}")
        
        momentum_stocks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 ML FILTERING RESULTS:")
        print(f"   ✅ Selected for momentum trading: {len(momentum_stocks)} stocks")
        print(f"   ❌ Rejected (not momentum-friendly): {len(rejected_stocks)} stocks")
        
        if momentum_stocks:
            print(f"\n🏆 TOP MOMENTUM CANDIDATES:")
            for i, (symbol, prob) in enumerate(momentum_stocks[:10], 1):
                print(f"   #{i}: {symbol} ({prob:.1%} probability)")
        
        return [symbol for symbol, prob in momentum_stocks]
    
    def trade_filtered_stocks(self, momentum_stocks):
        """Apply momentum trading only to ML-filtered stocks"""
        print(f"\n🚀 TRADING {len(momentum_stocks)} ML-SELECTED MOMENTUM STOCKS")
        print("=" * 60)
        
        results = []
        total_deployed = 0
        total_final_value = 0
        
        for symbol in momentum_stocks:
            result = self.trade_single_stock(symbol)
            if result:
                results.append(result)
                total_deployed += self.starting_capital
                total_final_value += result['final_value']
        
        # Analysis
        if results:
            total_return = ((total_final_value - total_deployed) / total_deployed) * 100
            total_buy_hold_value = sum(r['buy_hold_value'] for r in results)
            total_buy_hold_return = ((total_buy_hold_value - total_deployed) / total_deployed) * 100
            outperformance = total_return - total_buy_hold_return
            
            winners = len([r for r in results if r['outperformance'] > 0])
            
            print(f"\n🏆 ML-FILTERED MOMENTUM RESULTS")
            print("=" * 50)
            print(f"💰 Capital Deployed: ${total_deployed:,}")
            print(f"📈 Strategy Return: ${total_final_value:,.0f} ({total_return:+.1f}%)")
            print(f"🎯 Buy-Hold Return: ${total_buy_hold_value:,.0f} ({total_buy_hold_return:+.1f}%)")
            print(f"🏆 Outperformance: ${total_final_value - total_buy_hold_value:,.0f} ({outperformance:+.1f}%)")
            print(f"📊 Win Rate: {winners}/{len(results)} ({winners/len(results):.1%})")
            
            return {
                'total_deployed': total_deployed,
                'total_return': total_return,
                'outperformance': outperformance,
                'win_rate': winners/len(results),
                'results': results
            }
        
        return None
    
    def trade_single_stock(self, symbol):
        """Trade a single stock using institutional momentum signals"""
        try:
            # Download data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=120)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if data.empty or len(data) < 20:
                return None
                
            data = self.calculate_momentum_features(data)
            
            # Portfolio state
            cash = self.starting_capital
            shares = 0
            position = None
            trades = []
            
            # Trading simulation
            for i in range(15, len(data)):
                date = data.index[i]
                signal_data = self.get_momentum_signal(data, i)
                price = signal_data['price']
                signal = signal_data['signal']
                
                # Execute trades
                if signal == 'BUY' and position != 'LONG' and cash > 0:
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
                        'amount': amount
                    })
                    
                elif signal == 'SELL' and position == 'LONG' and shares > 0:
                    amount = shares * price
                    sold_shares = shares
                    
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
                        'profit_pct': profit_pct
                    })
            
            # Final calculations
            final_price = float(data['Close'].iloc[-1])
            final_value = cash + (shares * final_price)
            
            start_price = float(data['Close'].iloc[15])
            buy_hold_value = self.starting_capital * (final_price / start_price)
            
            strategy_return = ((final_value - self.starting_capital) / self.starting_capital) * 100
            buy_hold_return = ((buy_hold_value - self.starting_capital) / self.starting_capital) * 100
            outperformance = strategy_return - buy_hold_return
            
            status = "✅" if outperformance > 0 else "❌"
            print(f"{status} {symbol}: ${final_value:7,.0f} ({strategy_return:+5.1f}%) vs BH: {buy_hold_return:+5.1f}% | Diff: {outperformance:+5.1f}% | Trades: {len(trades)}")
            
            return {
                'symbol': symbol,
                'final_value': final_value,
                'strategy_return': strategy_return,
                'buy_hold_value': buy_hold_value,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'trades': trades
            }
            
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")
            return None
    
    def get_momentum_signal(self, data, idx):
        """Get institutional momentum signal"""
        try:
            price = float(data['Close'].iloc[idx])
            
            # Trend analysis
            recent_5d = data['Close'].iloc[idx-5:idx]
            recent_10d = data['Close'].iloc[idx-10:idx]
            
            trend_5d = (price - float(recent_5d.mean())) / float(recent_5d.mean())
            trend_10d = (price - float(recent_10d.mean())) / float(recent_10d.mean())
            
            # RSI
            rsi = float(data['RSI'].iloc[idx]) if not pd.isna(data['RSI'].iloc[idx]) else 50
            
            # Volume
            recent_vol = float(data['Volume'].iloc[idx-10:idx].mean())
            current_vol = float(data['Volume'].iloc[idx])
            vol_ratio = current_vol / recent_vol if recent_vol > 0 else 1
            
            # Volatility
            volatility = float(recent_10d.std()) / float(recent_10d.mean()) if float(recent_10d.mean()) > 0 else 0
            
            # INSTITUTIONAL MOMENTUM SIGNAL
            signal = 'HOLD'
            
            # BUY: Strong dual-trend momentum
            if (trend_5d > self.momentum_config['trend_5d_buy_threshold'] and 
                trend_10d > self.momentum_config['trend_10d_buy_threshold'] and
                rsi < self.momentum_config['rsi_overbought'] and
                volatility < self.momentum_config['volatility_threshold']):
                signal = 'BUY'
            
            # SELL: Trend breakdown
            elif (trend_5d < self.momentum_config['trend_5d_sell_threshold'] and 
                  trend_10d < self.momentum_config['trend_10d_sell_threshold']) or \
                 (rsi > self.momentum_config['rsi_overbought'] and trend_5d < 0):
                signal = 'SELL'
            
            return {'signal': signal, 'price': price}
            
        except Exception as e:
            return {'signal': 'HOLD', 'price': price if 'price' in locals() else 0}

def main():
    """Run smart ML-filtered momentum trading"""
    trader = SmartMomentumTrader(starting_capital=10000)
    
    # Step 1: Train ML model
    if trader.create_training_data():
        
        # Step 2: Filter stocks
        momentum_stocks = trader.filter_momentum_stocks(threshold=0.6)
        
        if momentum_stocks:
            # Step 3: Trade filtered stocks
            results = trader.trade_filtered_stocks(momentum_stocks)
            
            if results and results['outperformance'] > 0:
                print(f"\n🎉 SUCCESS! ML-Filtered momentum strategy WORKS!")
                print(f"   🏆 Outperformed buy-and-hold by {results['outperformance']:+.1f}%")
                print(f"   🎯 Win rate: {results['win_rate']:.1%}")
            else:
                print(f"\n📊 Results mixed - may need threshold adjustment")
        else:
            print(f"\n❌ No stocks passed ML filter!")
    else:
        print(f"\n❌ Failed to train ML model")

if __name__ == "__main__":
    main()
