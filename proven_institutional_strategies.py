#!/usr/bin/env python3
"""
PROVEN INSTITUTIONAL STRATEGIES - SIMPLIFIED ROBUST VERSION
Tests the 3 most proven strategies with institutional validation:

1. MOMENTUM STRATEGY (Jegadeesh & Titman 1993) - 12% annual excess returns
2. FACTOR STRATEGY (Fama-French + Quality) - Nobel Prize foundation
3. LOW VOLATILITY STRATEGY (Haugen & Baker 1991) - Consistent outperformance

All strategies have 30+ years of academic validation and $500B+ institutional usage.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProvenInstitutionalStrategies:
    def __init__(self):
        # LIQUID MEGA-CAP UNIVERSE (Institutional favorites)
        self.stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
            "TSLA", "JPM", "UNH", "JNJ", "PG", "WMT",
            "HD", "CAT", "CRM", "ADBE", "NFLX", "AMD"
        ]
        
    def download_robust_data(self, days_lookback=365):
        """Robust data download with error handling"""
        print("📥 DOWNLOADING INSTITUTIONAL UNIVERSE DATA")
        print("=" * 42)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_lookback + 30)
        
        stock_data = {}
        fundamentals = {}
        
        for symbol in self.stocks:
            try:
                print(f"📊 {symbol}...", end=" ")
                
                # Download price data
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if not data.empty and len(data) >= 50:  # Minimum 50 days
                    stock_data[symbol] = data
                    
                    # Get basic fundamental info
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    fundamentals[symbol] = {
                        'pe_ratio': info.get('trailingPE'),
                        'market_cap': info.get('marketCap', 0),
                        'roe': info.get('returnOnEquity'),
                        'dividend_yield': info.get('dividendYield', 0)
                    }
                    
                    print(f"✅ ({len(data)} days)")
                else:
                    print("❌ No data")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}")
                continue
                
        print(f"\n✅ Downloaded data for {len(stock_data)} stocks")
        return stock_data, fundamentals
    
    def momentum_strategy(self, stock_data):
        """
        JEGADEESH & TITMAN MOMENTUM STRATEGY (1993)
        - Most robust trading anomaly in academic literature
        - 12% annual excess returns across 30+ years
        - Used by AQR ($200B), Two Sigma, Renaissance
        """
        print("\n🚀 MOMENTUM STRATEGY (Jegadeesh & Titman 1993)")
        print("=" * 48)
        print("📚 30+ years of academic validation, 12% excess returns")
        
        momentum_scores = []
        
        for symbol, data in stock_data.items():
            try:
                prices = data['Close']
                
                # Calculate momentum metrics
                if len(prices) >= 90:  # Need 3+ months
                    # 3-month momentum (academic standard)
                    mom_3m = (prices.iloc[-1] - prices.iloc[-63]) / prices.iloc[-63]
                    
                    # 6-month momentum (if available)
                    if len(prices) >= 126:
                        mom_6m = (prices.iloc[-1] - prices.iloc[-126]) / prices.iloc[-126]
                    else:
                        mom_6m = mom_3m
                    
                    # Risk adjustment (volatility scaling)
                    returns = prices.pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)
                    risk_adj_momentum = (mom_3m + mom_6m) / 2 / volatility if volatility > 0 else 0
                    
                    momentum_scores.append({
                        'symbol': symbol,
                        'momentum_3m': mom_3m,
                        'momentum_6m': mom_6m,
                        'risk_adj_momentum': risk_adj_momentum,
                        'volatility': volatility
                    })
                    
            except Exception as e:
                continue
        
        # Sort by risk-adjusted momentum
        momentum_scores.sort(key=lambda x: x['risk_adj_momentum'], reverse=True)
        
        print("🏆 TOP MOMENTUM STOCKS:")
        for i, stock in enumerate(momentum_scores[:8], 1):
            print(f"{i}. {stock['symbol']:<6} "
                  f"3M: {stock['momentum_3m']:+6.1%} | "
                  f"6M: {stock['momentum_6m']:+6.1%} | "
                  f"Risk-Adj: {stock['risk_adj_momentum']:+.3f}")
        
        return momentum_scores
    
    def factor_strategy(self, stock_data, fundamentals):
        """
        FAMA-FRENCH FACTOR STRATEGY + QUALITY
        - Nobel Prize foundation (Eugene Fama 2013)
        - Multi-factor model with quality overlay
        - Used by institutional investors globally
        """
        print("\n🏛️ FACTOR STRATEGY (Fama-French + Quality)")
        print("=" * 42)
        print("🏆 Nobel Prize foundation, institutional standard")
        
        factor_scores = []
        
        for symbol, data in stock_data.items():
            try:
                prices = data['Close']
                fund_data = fundamentals.get(symbol, {})
                
                # Momentum factor (past 6 months)
                if len(prices) >= 126:
                    momentum = (prices.iloc[-1] - prices.iloc[-126]) / prices.iloc[-126]
                else:
                    momentum = 0
                
                # Quality factor
                quality_score = 0
                roe = fund_data.get('roe')
                if roe and roe > 0:
                    quality_score += min(roe / 0.15, 1)  # Cap at 15% ROE
                
                # Value factor
                value_score = 0
                pe_ratio = fund_data.get('pe_ratio')
                if pe_ratio and pe_ratio > 0:
                    # Prefer reasonable P/E (10-30 range)
                    if 10 <= pe_ratio <= 30:
                        value_score = 1
                    elif pe_ratio < 10:
                        value_score = 0.7  # Potential value trap
                    else:
                        value_score = max(0, 1 - (pe_ratio - 30) / 50)
                
                # Low volatility factor
                returns = prices.pct_change().dropna()
                if len(returns) > 30:
                    volatility = returns.std() * np.sqrt(252)
                    vol_score = max(0, 1 - volatility * 3)  # Penalty for high vol
                else:
                    vol_score = 0.5
                
                # Size factor (prefer large cap for stability)
                market_cap = fund_data.get('market_cap', 0)
                size_score = min(1, market_cap / 100e9) if market_cap > 0 else 0.5  # $100B cap
                
                # Composite factor score
                composite_score = (
                    0.30 * momentum +        # Momentum factor
                    0.25 * quality_score +   # Quality factor
                    0.20 * value_score +     # Value factor
                    0.15 * vol_score +       # Low volatility
                    0.10 * size_score        # Size factor
                )
                
                factor_scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'momentum': momentum,
                    'quality': quality_score,
                    'value': value_score,
                    'volatility': volatility if 'volatility' in locals() else 0
                })
                
            except Exception as e:
                continue
        
        # Sort by composite score
        factor_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print("🏆 TOP FACTOR STOCKS:")
        for i, stock in enumerate(factor_scores[:8], 1):
            print(f"{i}. {stock['symbol']:<6} "
                  f"Score: {stock['composite_score']:+.3f} | "
                  f"Mom: {stock['momentum']:+.2f} | "
                  f"Qual: {stock['quality']:.2f}")
        
        return factor_scores
    
    def low_volatility_strategy(self, stock_data):
        """
        LOW VOLATILITY ANOMALY (Haugen & Baker 1991)
        - Low vol stocks outperform high vol stocks
        - Violates CAPM theory but consistently works
        - Used in $300B+ of institutional assets
        """
        print("\n📉 LOW VOLATILITY STRATEGY (Haugen & Baker 1991)")
        print("=" * 48)
        print("📊 Violates CAPM but works: Lower risk = Higher returns")
        
        vol_scores = []
        
        for symbol, data in stock_data.items():
            try:
                prices = data['Close']
                returns = prices.pct_change().dropna()
                
                if len(returns) >= 60:  # Need 60+ days for volatility calc
                    # Calculate volatility metrics
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    
                    # Calculate risk-adjusted return
                    if len(prices) >= 126:
                        total_return = (prices.iloc[-1] - prices.iloc[-126]) / prices.iloc[-126]
                    else:
                        total_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                    
                    risk_adj_return = total_return / volatility if volatility > 0 else 0
                    
                    # Downside deviation (more sophisticated risk measure)
                    negative_returns = returns[returns < 0]
                    downside_dev = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
                    
                    # Sortino ratio (return/downside deviation)
                    sortino_ratio = total_return / downside_dev if downside_dev > 0 else 0
                    
                    # Maximum drawdown
                    cumulative = (1 + returns).cumprod()
                    rolling_max = cumulative.expanding().max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    # Low volatility score (inverse of volatility with quality adjustments)
                    low_vol_score = (1 / (1 + volatility)) * (1 + risk_adj_return) * (1 - abs(max_drawdown))
                    
                    vol_scores.append({
                        'symbol': symbol,
                        'volatility': volatility,
                        'risk_adj_return': risk_adj_return,
                        'sortino_ratio': sortino_ratio,
                        'max_drawdown': max_drawdown,
                        'low_vol_score': low_vol_score,
                        'total_return': total_return
                    })
                    
            except Exception as e:
                continue
        
        # Sort by low volatility score (higher is better)
        vol_scores.sort(key=lambda x: x['low_vol_score'], reverse=True)
        
        print("🏆 TOP LOW VOLATILITY STOCKS:")
        for i, stock in enumerate(vol_scores[:8], 1):
            print(f"{i}. {stock['symbol']:<6} "
                  f"Vol: {stock['volatility']:5.1%} | "
                  f"Return: {stock['total_return']:+6.1%} | "
                  f"Sortino: {stock['sortino_ratio']:+.2f} | "
                  f"Score: {stock['low_vol_score']:.3f}")
        
        return vol_scores
    
    def backtest_strategies(self, momentum_scores, factor_scores, vol_scores, stock_data, days=180):
        """Backtest all three strategies"""
        print(f"\n🧪 BACKTESTING ALL STRATEGIES ({days} days)")
        print("=" * 40)
        
        strategies = {
            'Momentum': momentum_scores[:6],
            'Factor': factor_scores[:6],
            'Low_Vol': vol_scores[:6]
        }
        
        results = {}
        
        for strategy_name, stocks in strategies.items():
            if not stocks:
                continue
                
            print(f"\n📊 {strategy_name} Strategy:")
            
            portfolio_returns = []
            symbols = [s['symbol'] for s in stocks]
            
            # Calculate equal-weighted portfolio performance
            for symbol in symbols:
                if symbol in stock_data:
                    data = stock_data[symbol]
                    prices = data['Close'].tail(days)
                    
                    if len(prices) >= days:
                        stock_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                        portfolio_returns.append(stock_return)
            
            if portfolio_returns:
                avg_return = np.mean(portfolio_returns) * 100
                return_std = np.std(portfolio_returns) * 100
                sharpe_estimate = avg_return / return_std if return_std > 0 else 0
                
                results[strategy_name] = {
                    'return': avg_return,
                    'volatility': return_std,
                    'sharpe': sharpe_estimate,
                    'num_stocks': len(portfolio_returns)
                }
                
                print(f"   📈 Return: {avg_return:+.1f}%")
                print(f"   📉 Volatility: {return_std:.1f}%")
                print(f"   ⚡ Sharpe Est: {sharpe_estimate:.2f}")
                print(f"   📊 Stocks: {len(portfolio_returns)}")
        
        return results
    
    def recommend_optimal_strategy(self, backtest_results):
        """Recommend the best strategy based on risk-adjusted returns"""
        print(f"\n🏆 STRATEGY RECOMMENDATION")
        print("=" * 30)
        
        if not backtest_results:
            print("❌ No backtest results available")
            return None
        
        # Rank strategies by Sharpe ratio (risk-adjusted return)
        ranked = sorted(backtest_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        
        print("📊 STRATEGY RANKINGS (by Sharpe Ratio):")
        print("-" * 35)
        
        for i, (strategy, metrics) in enumerate(ranked, 1):
            print(f"{i}. {strategy:<12} "
                  f"Return: {metrics['return']:+6.1f}% | "
                  f"Sharpe: {metrics['sharpe']:5.2f}")
        
        if ranked:
            winner = ranked[0]
            strategy_name, metrics = winner
            
            print(f"\n🥇 OPTIMAL STRATEGY: {strategy_name.upper()}")
            print(f"🎯 WHY IT WINS:")
            print(f"   • Highest risk-adjusted returns (Sharpe: {metrics['sharpe']:.2f})")
            print(f"   • Strong absolute performance ({metrics['return']:+.1f}%)")
            print(f"   • Proven institutional track record")
            
            # Strategy-specific recommendations
            if strategy_name == 'Momentum':
                print(f"\n🚀 MOMENTUM STRATEGY DEPLOYMENT:")
                print(f"   📚 Academic Basis: Jegadeesh & Titman (1993)")
                print(f"   💰 Institutional Usage: AQR, Two Sigma, Renaissance")
                print(f"   📊 Expected Performance: 10-15% annual excess returns")
                print(f"   🔄 Rebalancing: Monthly (academic standard)")
                
            elif strategy_name == 'Factor':
                print(f"\n🏛️ FACTOR STRATEGY DEPLOYMENT:")
                print(f"   📚 Academic Basis: Fama-French Model (Nobel Prize)")
                print(f"   💰 Institutional Usage: Vanguard, BlackRock, DFA")
                print(f"   📊 Expected Performance: 8-12% annual excess returns")
                print(f"   🔄 Rebalancing: Quarterly (factor rebalancing)")
                
            elif strategy_name == 'Low_Vol':
                print(f"\n📉 LOW VOLATILITY STRATEGY DEPLOYMENT:")
                print(f"   📚 Academic Basis: Low Vol Anomaly (1991)")
                print(f"   💰 Institutional Usage: $300B+ in smart beta ETFs")
                print(f"   📊 Expected Performance: 6-10% with lower risk")
                print(f"   🔄 Rebalancing: Semi-annually (stability focus)")
            
            return winner
        
        return None

def main():
    """Run comprehensive institutional strategy analysis"""
    print("🏛️ PROVEN INSTITUTIONAL STRATEGIES")
    print("=" * 40)
    print("🎯 Testing the 3 most validated academic strategies:")
    print("   1. Momentum (30+ years proven)")
    print("   2. Multi-Factor (Nobel Prize foundation)")
    print("   3. Low Volatility (Institutional favorite)")
    print("💰 Goal: Find highest risk-adjusted returns")
    print("=" * 40)
    
    # Initialize strategy tester
    strategies = ProvenInstitutionalStrategies()
    
    # Download data
    stock_data, fundamentals = strategies.download_robust_data(days_lookback=400)
    
    if len(stock_data) < 5:
        print("❌ Insufficient data - cannot proceed")
        return None
    
    print(f"✅ Proceeding with {len(stock_data)} stocks")
    
    # Run all three strategies
    momentum_scores = strategies.momentum_strategy(stock_data)
    factor_scores = strategies.factor_strategy(stock_data, fundamentals)
    vol_scores = strategies.low_volatility_strategy(stock_data)
    
    # Backtest strategies
    backtest_results = strategies.backtest_strategies(
        momentum_scores, factor_scores, vol_scores, stock_data
    )
    
    # Get final recommendation
    optimal_strategy = strategies.recommend_optimal_strategy(backtest_results)
    
    print(f"\n📋 INSTITUTIONAL VALIDATION SUMMARY:")
    print("=" * 38)
    print(f"✅ Momentum: $500B+ institutional assets (AQR, Two Sigma)")
    print(f"✅ Factor: $1T+ smart beta strategies (Vanguard, BlackRock)")  
    print(f"✅ Low Vol: $300B+ minimum volatility ETFs")
    print(f"📚 All strategies have 20-30+ years academic validation")
    print(f"🏆 Combined institutional usage: $2T+ in assets")
    
    return {
        'strategies': strategies,
        'momentum_scores': momentum_scores,
        'factor_scores': factor_scores,
        'vol_scores': vol_scores,
        'backtest_results': backtest_results,
        'optimal_strategy': optimal_strategy
    }

if __name__ == "__main__":
    results = main()
