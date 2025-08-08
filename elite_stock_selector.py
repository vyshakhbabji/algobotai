#!/usr/bin/env python3
"""
Elite Stock Selection for AI Trading
Select the 25 best stocks for maximum trading alpha
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

class EliteStockSelector:
    def __init__(self):
        # Candidate universe - High-quality, liquid stocks
        self.candidate_stocks = [
            # Mega Cap Tech (High volume, trending)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            
            # High-Beta Growth (Volatile, trending)
            'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'SQ', 'ROKU', 'ZM',
            
            # Momentum Stocks (Trending behavior)
            'SHOP', 'SNOW', 'PLTR', 'COIN', 'RBLX', 'U', 'NET', 'DDOG',
            
            # Biotech (High volatility, catalyst-driven)
            'GILD', 'BIIB', 'REGN', 'VRTX', 'MRNA', 'BNTX',
            
            # Financials (Economic sensitivity)
            'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',
            
            # Consumer Discretionary (Trending behavior)
            'DIS', 'NKE', 'SBUX', 'HD', 'COST', 'TGT',
            
            # Energy (High volatility)
            'XOM', 'CVX', 'COP', 'EOG', 'SLB',
            
            # Semiconductors (High-beta, cyclical)
            'INTC', 'QCOM', 'AVGO', 'MU', 'AMAT', 'LRCX',
            
            # Cloud/SaaS (Growth trending)
            'ORCL', 'NOW', 'WDAY', 'OKTA', 'ZS'
        ]
        
        self.selection_criteria = {
            'min_avg_volume': 5_000_000,       # 5M+ daily volume (relaxed)
            'min_price': 15,                   # $15+ to avoid penny stocks
            'max_price': 1000,                 # <$1000 to include NVDA etc
            'min_volatility': 0.015,           # 1.5%+ daily volatility (relaxed)
            'max_volatility': 0.12,            # <12% to include more growth stocks
            'min_market_cap': 5_000_000_000,   # $5B+ market cap (relaxed)
            'data_availability': 400           # 1.5+ years of data (relaxed)
        }
    
    def analyze_stock_quality(self, symbol):
        """
        Analyze a stock's trading characteristics for AI suitability
        """
        try:
            print(f"Analyzing {symbol}...")
            
            # Get 2+ years of data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=600)  # Extra buffer
            
            # Get historical data
            df = ticker.history(start=start_date, end=end_date)
            info = ticker.info
            
            if len(df) < self.selection_criteria['data_availability']:
                print(f"  ❌ {symbol}: Insufficient data ({len(df)} days)")
                return None
            
            # Calculate key metrics
            analysis = {
                'symbol': symbol,
                'current_price': df['Close'][-1],
                'market_cap': info.get('marketCap', 0),
                'avg_volume': df['Volume'].mean(),
                'median_volume': df['Volume'].median(),
                'avg_daily_volatility': df['Close'].pct_change().std(),
                'max_drawdown': self._calculate_max_drawdown(df['Close']),
                'annual_return': self._calculate_annual_return(df['Close']),
                'sharpe_ratio': self._calculate_sharpe_ratio(df['Close']),
                'trending_score': self._calculate_trending_score(df['Close']),
                'volume_consistency': df['Volume'].std() / df['Volume'].mean(),
                'price_momentum': self._calculate_momentum_score(df['Close']),
                'options_available': info.get('optionsAvailable', False),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'data_points': len(df)
            }
            
            # Calculate composite score
            analysis['ai_trading_score'] = self._calculate_ai_score(analysis)
            
            time.sleep(0.1)  # Rate limiting
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_annual_return(self, prices):
        """Calculate annualized return"""
        total_return = (prices[-1] / prices[0]) - 1
        days = len(prices)
        return (1 + total_return) ** (252 / days) - 1
    
    def _calculate_sharpe_ratio(self, prices):
        """Calculate Sharpe ratio (assuming 2% risk-free rate)"""
        returns = prices.pct_change().dropna()
        excess_returns = returns.mean() * 252 - 0.02  # Annualized excess return
        volatility = returns.std() * np.sqrt(252)     # Annualized volatility
        return excess_returns / volatility if volatility > 0 else 0
    
    def _calculate_trending_score(self, prices):
        """Calculate how well the stock trends (good for AI)"""
        # Use multiple SMAs to detect trending behavior
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()
        sma_200 = prices.rolling(200).mean()
        
        # Calculate trend consistency
        trend_20_50 = ((sma_20 > sma_50).astype(int).diff().abs().sum()) / len(sma_20)
        trend_50_200 = ((sma_50 > sma_200).astype(int).diff().abs().sum()) / len(sma_50)
        
        # Lower score = more trending (fewer direction changes)
        return 1 - (trend_20_50 + trend_50_200) / 2
    
    def _calculate_momentum_score(self, prices):
        """Calculate momentum persistence (good for AI predictions)"""
        returns = prices.pct_change()
        
        # Calculate auto-correlation of returns (momentum persistence)
        autocorr_1 = returns.autocorr(lag=1)
        autocorr_5 = returns.autocorr(lag=5)
        autocorr_20 = returns.autocorr(lag=20)
        
        # Average auto-correlation (higher = more momentum)
        return (autocorr_1 + autocorr_5 + autocorr_20) / 3
    
    def _calculate_ai_score(self, analysis):
        """
        Calculate composite AI trading suitability score
        Higher score = better for AI trading
        """
        score = 0
        
        # Volume score (higher volume = better)
        if analysis['avg_volume'] > 50_000_000:
            score += 20
        elif analysis['avg_volume'] > 20_000_000:
            score += 15
        elif analysis['avg_volume'] > 10_000_000:
            score += 10
        
        # Volatility score (optimal range)
        vol = analysis['avg_daily_volatility']
        if 0.025 <= vol <= 0.05:  # Sweet spot for AI
            score += 20
        elif 0.02 <= vol <= 0.06:
            score += 15
        elif 0.015 <= vol <= 0.07:
            score += 10
        
        # Trending score (higher = better for AI)
        score += analysis['trending_score'] * 15
        
        # Momentum score (some momentum is good)
        momentum = abs(analysis['price_momentum'])
        if 0.05 <= momentum <= 0.15:
            score += 15
        elif momentum <= 0.2:
            score += 10
        
        # Sharpe ratio (risk-adjusted returns)
        if analysis['sharpe_ratio'] > 1.0:
            score += 10
        elif analysis['sharpe_ratio'] > 0.5:
            score += 5
        
        # Options availability (alternative data)
        if analysis['options_available']:
            score += 10
        
        # Volume consistency (lower = better)
        if analysis['volume_consistency'] < 1.0:
            score += 5
        
        # Market cap bonus for liquidity
        if analysis['market_cap'] > 100_000_000_000:  # $100B+
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    def select_elite_25(self):
        """
        Select the top 25 stocks for AI trading
        """
        print("🎯 Analyzing candidate stocks for AI trading suitability...")
        
        all_analyses = []
        
        for symbol in self.candidate_stocks:
            analysis = self.analyze_stock_quality(symbol)
            if analysis:
                # Apply filters
                passes_filters = True
                
                if analysis['avg_volume'] < self.selection_criteria['min_avg_volume']:
                    print(f"  ❌ {symbol}: Low volume ({analysis['avg_volume']/1e6:.1f}M)")
                    passes_filters = False
                
                if not (self.selection_criteria['min_price'] <= analysis['current_price'] <= self.selection_criteria['max_price']):
                    print(f"  ❌ {symbol}: Price out of range (${analysis['current_price']:.2f})")
                    passes_filters = False
                
                if not (self.selection_criteria['min_volatility'] <= analysis['avg_daily_volatility'] <= self.selection_criteria['max_volatility']):
                    print(f"  ❌ {symbol}: Volatility out of range ({analysis['avg_daily_volatility']*100:.2f}%)")
                    passes_filters = False
                
                if analysis['market_cap'] < self.selection_criteria['min_market_cap']:
                    print(f"  ❌ {symbol}: Market cap too small (${analysis['market_cap']/1e9:.1f}B)")
                    passes_filters = False
                
                if passes_filters:
                    print(f"  ✅ {symbol}: Score {analysis['ai_trading_score']:.1f}")
                    all_analyses.append(analysis)
        
        # Sort by AI trading score
        elite_stocks = sorted(all_analyses, key=lambda x: x['ai_trading_score'], reverse=True)[:25]
        
        return elite_stocks
    
    def create_portfolio_file(self, elite_stocks):
        """
        Create portfolio_universe.json with elite stocks
        """
        # Group by sector for diversification
        sectors = {}
        for stock in elite_stocks:
            sector = stock['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(stock['symbol'])
        
        # Calculate market caps
        market_caps = {}
        for stock in elite_stocks:
            symbol = stock['symbol']
            if stock['market_cap'] > 200_000_000_000:
                market_caps[symbol] = 'Mega Cap'
            elif stock['market_cap'] > 50_000_000_000:
                market_caps[symbol] = 'Large Cap'
            elif stock['market_cap'] > 10_000_000_000:
                market_caps[symbol] = 'Mid Cap'
            else:
                market_caps[symbol] = 'Small Cap'
        
        portfolio_data = {
            'stocks': [stock['symbol'] for stock in elite_stocks],
            'sectors': sectors,
            'market_caps': market_caps,
            'selection_criteria': self.selection_criteria,
            'last_updated': datetime.now().isoformat(),
            'selection_date': datetime.now().strftime('%Y-%m-%d'),
            'total_stocks': len(elite_stocks),
            'avg_ai_score': sum(s['ai_trading_score'] for s in elite_stocks) / len(elite_stocks)
        }
        
        # Save to file
        with open('elite_portfolio_universe.json', 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        return portfolio_data

def main():
    """
    Main execution - select elite 25 stocks
    """
    selector = EliteStockSelector()
    
    print("🚀 Elite Stock Selection for AI Trading")
    print("=" * 50)
    
    # Select top 25 stocks
    elite_stocks = selector.select_elite_25()
    
    if len(elite_stocks) < 25:
        print(f"⚠️ Warning: Only found {len(elite_stocks)} stocks meeting criteria")
    
    # Display results
    print(f"\n🎯 TOP {len(elite_stocks)} ELITE STOCKS FOR AI TRADING:")
    print("=" * 60)
    
    for i, stock in enumerate(elite_stocks, 1):
        print(f"{i:2d}. {stock['symbol']:6s} - Score: {stock['ai_trading_score']:5.1f} - "
              f"Vol: {stock['avg_daily_volatility']*100:4.1f}% - "
              f"Avg Volume: {stock['avg_volume']/1e6:4.1f}M - "
              f"Sector: {stock['sector']}")
    
    # Create portfolio file
    if elite_stocks:
        portfolio_data = selector.create_portfolio_file(elite_stocks)
        
        print(f"\n📊 PORTFOLIO SUMMARY:")
        print(f"Total Stocks: {portfolio_data['total_stocks']}")
        print(f"Average AI Score: {portfolio_data['avg_ai_score']:.1f}")
        print(f"Sectors: {len(portfolio_data['sectors'])}")
        
        print(f"\n💾 Portfolio saved to: elite_portfolio_universe.json")
        print("\n🎯 Ready for elite AI trading!")
    else:
        print("\n❌ No stocks met the criteria. Consider relaxing the filters.")

if __name__ == "__main__":
    main()
