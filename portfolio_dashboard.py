#!/usr/bin/env python3
"""
AI Portfolio Management Dashboard
Web interface for the AI portfolio management system
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from ai_portfolio_manager import AIPortfolioManager, STOCK_UNIVERSE
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__)

# Custom CSS
app.layout = html.Div([
    html.H1("🤖 AI Portfolio Management Dashboard", 
            style={'textAlign': 'center', 'color': '#2E86AB', 'marginBottom': 30}),
    
    # Control panel
    html.Div([
        html.Div([
            html.Label("Initial Capital ($):", style={'fontWeight': 'bold'}),
            dcc.Input(id='capital-input', type='number', value=10000, 
                     style={'width': '100%', 'marginBottom': 10}),
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label("Backtest Start:", style={'fontWeight': 'bold'}),
            dcc.Input(id='start-date', type='text', value='2025-05-01', 
                     style={'width': '100%', 'marginBottom': 10}),
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Label("Backtest End:", style={'fontWeight': 'bold'}),
            dcc.Input(id='end-date', type='text', value='2025-08-01', 
                     style={'width': '100%', 'marginBottom': 10}),
        ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.Button('Run Backtest', id='run-button', n_clicks=0,
                       style={'width': '100%', 'height': 40, 'backgroundColor': '#2E86AB', 
                             'color': 'white', 'border': 'none', 'borderRadius': 5,
                             'fontSize': 16, 'fontWeight': 'bold', 'cursor': 'pointer'}),
        ], style={'width': '23%', 'display': 'inline-block'}),
    ], style={'marginBottom': 30, 'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 10}),
    
    # Results section
    html.Div(id='results-container', children=[
        
        # Performance summary cards
        html.Div(id='summary-cards', style={'marginBottom': 30}),
        
        # Portfolio value chart
        html.Div([
            html.H3("📈 Portfolio Performance", style={'color': '#2E86AB'}),
            dcc.Graph(id='portfolio-chart'),
        ], style={'marginBottom': 30}),
        
        # Current AI signals
        html.Div([
            html.H3("🧠 Current AI Signals", style={'color': '#2E86AB'}),
            html.Div(id='signals-table'),
        ], style={'marginBottom': 30}),
        
        # Trade history
        html.Div([
            html.H3("📋 Trade History", style={'color': '#2E86AB'}),
            html.Div(id='trades-table'),
        ], style={'marginBottom': 30}),
        
        # Portfolio allocation
        html.Div([
            html.H3("🎯 Portfolio Allocation", style={'color': '#2E86AB'}),
            dcc.Graph(id='allocation-chart'),
        ], style={'marginBottom': 30}),
        
    ], style={'display': 'none'}),
    
    # Loading indicator
    html.Div(id='loading-div', children=[
        html.H3("Click 'Run Backtest' to start AI portfolio analysis", 
               style={'textAlign': 'center', 'color': '#666'})
    ]),
    
], style={'padding': 20, 'fontFamily': 'Arial, sans-serif'})

@app.callback(
    [Output('results-container', 'style'),
     Output('loading-div', 'style'),
     Output('summary-cards', 'children'),
     Output('portfolio-chart', 'figure'),
     Output('signals-table', 'children'),
     Output('trades-table', 'children'),
     Output('allocation-chart', 'figure')],
    [Input('run-button', 'n_clicks')],
    [dash.dependencies.State('capital-input', 'value'),
     dash.dependencies.State('start-date', 'value'),
     dash.dependencies.State('end-date', 'value')]
)
def run_backtest_callback(n_clicks, capital, start_date, end_date):
    if n_clicks == 0:
        # Initial state
        empty_fig = go.Figure()
        return ({'display': 'none'}, {'display': 'block'}, [], empty_fig, [], [], empty_fig)
    
    try:
        # Run the backtest
        print(f"Running backtest with ${capital} from {start_date} to {end_date}")
        
        manager = AIPortfolioManager(capital=capital)
        results = manager.backtest_portfolio(start_date, end_date)
        
        if not results:
            return ({'display': 'none'}, {'display': 'block'}, 
                   [html.P("No results generated", style={'color': 'red'})], 
                   go.Figure(), [], [], go.Figure())
        
        # Calculate performance metrics
        initial_value = results[0]['portfolio_value']
        final_value = results[-1]['portfolio_value']
        total_return = (final_value / initial_value - 1) * 100
        
        # Create summary cards
        summary_cards = html.Div([
            html.Div([
                html.H4(f"${initial_value:,.0f}", style={'color': '#2E86AB', 'margin': 0}),
                html.P("Initial Capital", style={'margin': 0, 'color': '#666'})
            ], style={'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'padding': 20, 
                     'borderRadius': 10, 'width': '22%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.H4(f"${final_value:,.0f}", style={'color': '#28A745', 'margin': 0}),
                html.P("Final Value", style={'margin': 0, 'color': '#666'})
            ], style={'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'padding': 20, 
                     'borderRadius': 10, 'width': '22%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.H4(f"{total_return:+.1f}%", 
                       style={'color': '#28A745' if total_return > 0 else '#DC3545', 'margin': 0}),
                html.P("Total Return", style={'margin': 0, 'color': '#666'})
            ], style={'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'padding': 20, 
                     'borderRadius': 10, 'width': '22%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.H4(f"{len(manager.trades)}", style={'color': '#2E86AB', 'margin': 0}),
                html.P("Total Trades", style={'margin': 0, 'color': '#666'})
            ], style={'textAlign': 'center', 'backgroundColor': '#f8f9fa', 'padding': 20, 
                     'borderRadius': 10, 'width': '22%', 'display': 'inline-block'}),
        ])
        
        # Create portfolio performance chart
        dates = [r['date'] for r in results]
        values = [r['portfolio_value'] for r in results]
        returns = [(v/values[0] - 1) * 100 for v in values]
        
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=dates, y=values, mode='lines+markers',
            name='Portfolio Value', line=dict(color='#2E86AB', width=3),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ))
        
        portfolio_fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400,
            hovermode='closest'
        )
        
        # Get current AI signals
        signals_data = get_current_signals()
        
        signals_table = dash_table.DataTable(
            data=signals_data,
            columns=[
                {'name': 'Stock', 'id': 'symbol'},
                {'name': 'AI Strength', 'id': 'strength', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                {'name': 'Signal', 'id': 'signal'},
                {'name': 'Current Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                {'name': 'Recommendation', 'id': 'recommendation'}
            ],
            style_cell={'textAlign': 'center', 'padding': 10},
            style_header={'backgroundColor': '#2E86AB', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{strength} > 70'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{strength} < 40'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ]
        )
        
        # Trade history table
        trades_data = []
        for trade in manager.trades[-10:]:  # Show last 10 trades
            trades_data.append({
                'date': trade['date'].strftime('%Y-%m-%d'),
                'symbol': trade['symbol'],
                'action': trade['action'],
                'shares': trade['shares'],
                'price': trade['price'],
                'value': trade['value']
            })
        
        trades_table = dash_table.DataTable(
            data=trades_data,
            columns=[
                {'name': 'Date', 'id': 'date'},
                {'name': 'Stock', 'id': 'symbol'},
                {'name': 'Action', 'id': 'action'},
                {'name': 'Shares', 'id': 'shares', 'type': 'numeric'},
                {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': '$.2f'}},
                {'name': 'Value', 'id': 'value', 'type': 'numeric', 'format': {'specifier': '$.2f'}}
            ],
            style_cell={'textAlign': 'center', 'padding': 10},
            style_header={'backgroundColor': '#2E86AB', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{action} = BUY'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{action} = SELL'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ]
        )
        
        # Portfolio allocation chart
        if manager.positions:
            symbols = list(manager.positions.keys())
            shares = list(manager.positions.values())
            
            # Get current prices for allocation calculation
            allocation_values = []
            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    current_price = stock.history(period='1d')['Close'].iloc[-1]
                    allocation_values.append(shares[symbols.index(symbol)] * current_price)
                except:
                    allocation_values.append(0)
            
            allocation_fig = go.Figure(data=[go.Pie(
                labels=symbols,
                values=allocation_values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            allocation_fig.update_layout(
                title="Current Portfolio Allocation",
                height=400
            )
        else:
            allocation_fig = go.Figure()
            allocation_fig.add_annotation(
                text="No current positions",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            allocation_fig.update_layout(title="Current Portfolio Allocation", height=400)
        
        return ({'display': 'block'}, {'display': 'none'}, summary_cards, 
               portfolio_fig, signals_table, trades_table, allocation_fig)
        
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error running backtest:", style={'color': 'red'}),
            html.P(str(e), style={'color': 'red'})
        ])
        return ({'display': 'block'}, {'display': 'none'}, error_msg, 
               go.Figure(), [], [], go.Figure())

def get_current_signals():
    """Get current AI signals for all stocks"""
    signals_data = []
    
    for symbol in STOCK_UNIVERSE:
        try:
            # Simple technical analysis for current signals
            stock = yf.Ticker(symbol)
            df = stock.history(period='3mo')
            
            if df.empty:
                continue
            
            # Calculate simple strength score
            current_price = df['Close'].iloc[-1]
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            rsi = calculate_rsi(df['Close']).iloc[-1]
            
            # Simple scoring
            price_score = 50 + (current_price / sma_20 - 1) * 100
            rsi_score = 100 - abs(rsi - 50)  # Closer to 50 is better
            
            strength = (price_score + rsi_score) / 2
            strength = max(0, min(100, strength))
            
            if strength > 70:
                signal = "🟢 Strong Buy"
                recommendation = "BUY"
            elif strength > 60:
                signal = "🔵 Buy"
                recommendation = "BUY"
            elif strength > 40:
                signal = "🟡 Hold"
                recommendation = "HOLD"
            else:
                signal = "🔴 Sell"
                recommendation = "SELL"
            
            signals_data.append({
                'symbol': symbol,
                'strength': strength,
                'signal': signal,
                'price': current_price,
                'recommendation': recommendation
            })
            
        except Exception as e:
            print(f"Error getting signal for {symbol}: {e}")
            continue
    
    return sorted(signals_data, key=lambda x: x['strength'], reverse=True)

def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if __name__ == '__main__':
    print("🚀 Starting AI Portfolio Management Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8051")
    print("💡 Features:")
    print("  - AI-driven portfolio management")
    print("  - 3-month backtest with $10k")
    print("  - Real-time stock signals")
    print("  - Portfolio allocation visualization")
    print("  - Trade history tracking")
    
    app.run(debug=True, host='0.0.0.0', port=8051)
