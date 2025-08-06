#!/usr/bin/env python3
"""
Interactive Trading Dashboard
Real-time web interface for NVDA trading analysis and strategy recommendations
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import json
import os
from threading import Timer
import warnings
warnings.filterwarnings('ignore')

class TradingDashboard:
    def __init__(self, symbol='NVDA'):
        self.symbol = symbol
        self.app = dash.Dash(__name__)
        self.models = {}
        self.scalers = {}
        self.current_data = None
        self.setup_layout()
        self.load_models()
        
    def load_models(self):
        """Load trained models for predictions"""
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
                except:
                    pass
            print(f"✅ Loaded {len(self.models)} models for predictions")
        except Exception as e:
            print(f"⚠️ Could not load models: {e}")
    
    def get_live_data(self):
        """Fetch live market data"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Get recent data for analysis
            hist = ticker.history(period="30d", interval="1d")
            info = ticker.info
            
            if len(hist) == 0:
                return None
                
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100
            
            # Technical indicators
            sma_5 = hist['Close'].rolling(5).mean().iloc[-1] if len(hist) >= 5 else current_price
            sma_10 = hist['Close'].rolling(10).mean().iloc[-1] if len(hist) >= 10 else current_price
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(hist) >= 14 else 50
            
            # Volume analysis
            avg_volume = hist['Volume'].mean()
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            self.current_data = {
                'price': current_price,
                'change': change,
                'change_pct': change_pct,
                'volume': current_volume,
                'volume_ratio': volume_ratio,
                'sma_5': sma_5,
                'sma_10': sma_10,
                'sma_20': sma_20,
                'rsi': rsi,
                'volatility': volatility,
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'day_high': hist['High'].iloc[-1],
                'day_low': hist['Low'].iloc[-1],
                'hist': hist,
                'timestamp': datetime.now()
            }
            
            return self.current_data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def analyze_trading_strategy(self, data):
        """Analyze current trading strategy"""
        if not data:
            return {
                'action': 'HOLD',
                'confidence': 'LOW',
                'reasoning': 'No data available',
                'score': 0,
                'target_price': 0,
                'stop_loss': 0
            }
        
        # Score different factors
        scores = []
        reasons = []
        
        # 1. Trend Analysis
        if data['price'] > data['sma_5'] > data['sma_10'] > data['sma_20']:
            scores.append(3)
            reasons.append("Strong uptrend (all SMAs aligned)")
        elif data['price'] > data['sma_5'] > data['sma_10']:
            scores.append(2)
            reasons.append("Uptrend (price above short-term SMAs)")
        elif data['price'] < data['sma_5'] < data['sma_10'] < data['sma_20']:
            scores.append(-3)
            reasons.append("Strong downtrend (all SMAs declining)")
        elif data['price'] < data['sma_5'] < data['sma_10']:
            scores.append(-2)
            reasons.append("Downtrend (price below short-term SMAs)")
        else:
            scores.append(0)
            reasons.append("Sideways trend")
        
        # 2. Momentum Analysis
        if data['change_pct'] > 3:
            scores.append(2)
            reasons.append("Strong positive momentum (+3%+)")
        elif data['change_pct'] > 1:
            scores.append(1)
            reasons.append("Positive momentum")
        elif data['change_pct'] < -3:
            scores.append(-2)
            reasons.append("Strong negative momentum (-3%+)")
        elif data['change_pct'] < -1:
            scores.append(-1)
            reasons.append("Negative momentum")
        
        # 3. RSI Analysis
        if data['rsi'] < 30:
            scores.append(2)
            reasons.append("Oversold (RSI < 30) - potential buy")
        elif data['rsi'] > 70:
            scores.append(-2)
            reasons.append("Overbought (RSI > 70) - potential sell")
        elif data['rsi'] < 40:
            scores.append(1)
            reasons.append("Approaching oversold")
        elif data['rsi'] > 60:
            scores.append(-1)
            reasons.append("Approaching overbought")
        
        # 4. Volume Analysis
        if data['volume_ratio'] > 2:
            scores.append(1)
            reasons.append("High volume confirms move")
        elif data['volume_ratio'] < 0.5:
            scores.append(-1)
            reasons.append("Low volume - weak signal")
        
        # 5. Volatility Risk
        if data['volatility'] > 40:
            scores.append(-1)
            reasons.append("High volatility increases risk")
        elif data['volatility'] < 20:
            scores.append(1)
            reasons.append("Low volatility environment")
        
        # Calculate total score
        total_score = sum(scores)
        
        # Determine action
        if total_score >= 4:
            action = "🚀 STRONG BUY"
            confidence = "HIGH"
            color = "success"
            target_price = data['price'] * 1.05
            stop_loss = data['price'] * 0.95
        elif total_score >= 2:
            action = "📈 BUY"
            confidence = "MEDIUM"
            color = "success"
            target_price = data['price'] * 1.03
            stop_loss = data['price'] * 0.97
        elif total_score <= -4:
            action = "🔥 STRONG SELL"
            confidence = "HIGH"
            color = "danger"
            target_price = data['price'] * 0.95
            stop_loss = data['price'] * 1.05
        elif total_score <= -2:
            action = "📉 SELL"
            confidence = "MEDIUM"
            color = "danger"
            target_price = data['price'] * 0.97
            stop_loss = data['price'] * 1.03
        else:
            action = "⏸️ HOLD"
            confidence = "LOW"
            color = "warning"
            target_price = data['price']
            stop_loss = data['price'] * 0.98
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': "; ".join(reasons),
            'score': total_score,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'color': color,
            'individual_scores': scores,
            'reasons': reasons
        }
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🤖 AI Trading Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '0px'}),
                html.H3(f"Real-time Analysis for {self.symbol}", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0px'}),
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # Update every 10 seconds
                n_intervals=0
            ),
            
            # Main content
            html.Div([
                # Left column - Current data and strategy
                html.Div([
                    # Current Price Card
                    html.Div(id='price-card', style={'marginBottom': '20px'}),
                    
                    # Strategy Recommendation Card
                    html.Div(id='strategy-card', style={'marginBottom': '20px'}),
                    
                    # Technical Indicators Card
                    html.Div(id='technical-card'),
                    
                ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 20px'}),
                
                # Right column - Charts
                html.Div([
                    # Price chart
                    dcc.Graph(id='price-chart', style={'height': '400px'}),
                    
                    # Technical indicators chart
                    dcc.Graph(id='indicators-chart', style={'height': '300px'}),
                    
                ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 20px'}),
                
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            
            # Footer
            html.Div([
                html.P(f"Last updated: ", id='last-updated', 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'})
            ], style={'marginTop': '40px', 'borderTop': '1px solid #bdc3c7', 'paddingTop': '20px'})
            
        ], style={'fontFamily': 'Arial, sans-serif', 'margin': '0', 'padding': '0'})
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            [Output('price-card', 'children'),
             Output('strategy-card', 'children'),
             Output('technical-card', 'children'),
             Output('price-chart', 'figure'),
             Output('indicators-chart', 'figure'),
             Output('last-updated', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Get live data
            data = self.get_live_data()
            
            if not data:
                return "Error loading data", "Error", "Error", {}, {}, "Error updating"
            
            # Analyze strategy
            strategy = self.analyze_trading_strategy(data)
            
            # Create price card
            price_card = html.Div([
                html.H3("💰 Current Price", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.H1(f"${data['price']:.2f}", 
                        style={'color': '#e74c3c' if data['change'] < 0 else '#27ae60', 
                               'fontSize': '48px', 'marginBottom': '10px'}),
                html.P([
                    html.Span(f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)", 
                             style={'fontSize': '18px', 'fontWeight': 'bold',
                                   'color': '#e74c3c' if data['change'] < 0 else '#27ae60'}),
                    html.Br(),
                    html.Small(f"Range: ${data['day_low']:.2f} - ${data['day_high']:.2f}")
                ]),
                html.P([
                    html.Strong("Volume: "), f"{data['volume']:,.0f} ({data['volume_ratio']:.1f}x avg)",
                    html.Br(),
                    html.Strong("Volatility: "), f"{data['volatility']:.1f}%"
                ], style={'fontSize': '14px', 'color': '#7f8c8d'})
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 
                     'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'border': '1px solid #bdc3c7'})
            
            # Create strategy card
            strategy_color = {
                'success': '#27ae60',
                'danger': '#e74c3c', 
                'warning': '#f39c12'
            }.get(strategy['color'], '#7f8c8d')
            
            strategy_card = html.Div([
                html.H3("🎯 AI Recommendation", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.H2(strategy['action'], 
                           style={'color': strategy_color, 'marginBottom': '10px', 'fontSize': '32px'}),
                    html.P([
                        html.Strong("Confidence: "), strategy['confidence'],
                        html.Br(),
                        html.Strong("Score: "), f"{strategy['score']}/6"
                    ], style={'fontSize': '16px', 'marginBottom': '15px'}),
                    
                    html.Div([
                        html.P([html.Strong("🎯 Target: "), f"${strategy['target_price']:.2f}"]),
                        html.P([html.Strong("🛡️ Stop Loss: "), f"${strategy['stop_loss']:.2f}"]),
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '15px'}),
                    
                    html.P([
                        html.Strong("💭 Reasoning: "),
                        html.Br(),
                        strategy['reasoning']
                    ], style={'fontSize': '14px', 'color': '#7f8c8d'})
                ])
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 
                     'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'border': f'2px solid {strategy_color}'})
            
            # Create technical indicators card
            technical_card = html.Div([
                html.H3("📊 Technical Analysis", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.P([html.Strong("RSI: "), f"{data['rsi']:.1f}", 
                           html.Span(" (Oversold)" if data['rsi'] < 30 else " (Overbought)" if data['rsi'] > 70 else " (Neutral)",
                                   style={'color': '#27ae60' if data['rsi'] < 30 else '#e74c3c' if data['rsi'] > 70 else '#7f8c8d'})]),
                    html.P([html.Strong("SMA 5: "), f"${data['sma_5']:.2f}"]),
                    html.P([html.Strong("SMA 10: "), f"${data['sma_10']:.2f}"]),
                    html.P([html.Strong("SMA 20: "), f"${data['sma_20']:.2f}"]),
                    html.Hr(),
                    html.P([html.Strong("Market Cap: "), f"{data['market_cap']:,}" if isinstance(data['market_cap'], (int, float)) else str(data['market_cap'])]),
                    html.P([html.Strong("P/E Ratio: "), f"{data['pe_ratio']:.2f}" if isinstance(data['pe_ratio'], (int, float)) else str(data['pe_ratio'])]),
                ])
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 
                     'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'border': '1px solid #bdc3c7'})
            
            # Create price chart
            hist = data['hist'].tail(30)  # Last 30 days
            
            price_fig = go.Figure()
            
            # Candlestick chart
            price_fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='NVDA'
            ))
            
            # Add moving averages
            price_fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'].rolling(5).mean(),
                mode='lines',
                name='SMA 5',
                line=dict(color='blue', width=1)
            ))
            
            price_fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'].rolling(10).mean(),
                mode='lines',
                name='SMA 10',
                line=dict(color='orange', width=1)
            ))
            
            price_fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'].rolling(20).mean(),
                mode='lines',
                name='SMA 20',
                line=dict(color='red', width=1)
            ))
            
            price_fig.update_layout(
                title=f'{self.symbol} Price Chart (30 Days)',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white',
                showlegend=True,
                height=400
            )
            
            # Create indicators chart
            indicators_fig = go.Figure()
            
            # RSI
            rsi_values = []
            for i in range(len(hist)):
                if i >= 13:  # Need 14 periods for RSI
                    period_data = hist['Close'].iloc[i-13:i+1]
                    delta = period_data.diff()
                    gain = (delta.where(delta > 0, 0)).mean()
                    loss = (-delta.where(delta < 0, 0)).mean()
                    rs = gain / loss if loss != 0 else 0
                    rsi_val = 100 - (100 / (1 + rs)) if rs != 0 else 50
                    rsi_values.append(rsi_val)
                else:
                    rsi_values.append(50)
            
            indicators_fig.add_trace(go.Scatter(
                x=hist.index,
                y=rsi_values,
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            # RSI levels
            indicators_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            indicators_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            indicators_fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
            
            indicators_fig.update_layout(
                title='RSI Indicator',
                xaxis_title='Date',
                yaxis_title='RSI',
                template='plotly_white',
                height=300,
                yaxis=dict(range=[0, 100])
            )
            
            # Last updated timestamp
            last_updated = f"Last updated: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
            
            return price_card, strategy_card, technical_card, price_fig, indicators_fig, last_updated
    
    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        print(f"🚀 Starting AI Trading Dashboard...")
        print(f"📊 Analyzing {self.symbol} with real-time data")
        print(f"🌐 Dashboard will be available at: http://localhost:{port}")
        print(f"⚡ Auto-refreshes every 10 seconds")
        print(f"💡 Press Ctrl+C to stop")
        
        self.app.run(debug=debug, port=port, host='0.0.0.0')

def main():
    """Run the trading dashboard"""
    dashboard = TradingDashboard('NVDA')
    dashboard.run(debug=False, port=8050)

if __name__ == "__main__":
    main()
