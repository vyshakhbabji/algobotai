#!/usr/bin/env python3
"""
Sophisticated AI Trading System Enhancement
Adding multiple data sources beyond technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SophisticatedTradingSystem:
    def __init__(self):
        self.features_collected = {}
        
    def collect_fundamental_data(self, symbol):
        """Collect fundamental analysis data"""
        print(f"📊 Collecting fundamental data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            financials = ticker.financials
            
            fundamental_features = {
                # Valuation metrics
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'peg_ratio': info.get('pegRatio', np.nan),
                'price_to_book': info.get('priceToBook', np.nan),
                'price_to_sales': info.get('priceToSalesTrailing12Months', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'ev_to_revenue': info.get('enterpriseToRevenue', np.nan),
                'ev_to_ebitda': info.get('enterpriseToEbitda', np.nan),
                
                # Profitability metrics
                'profit_margin': info.get('profitMargins', np.nan),
                'operating_margin': info.get('operatingMargins', np.nan),
                'return_on_assets': info.get('returnOnAssets', np.nan),
                'return_on_equity': info.get('returnOnEquity', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                
                # Financial health
                'total_cash': info.get('totalCash', np.nan),
                'total_debt': info.get('totalDebt', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'current_ratio': info.get('currentRatio', np.nan),
                'quick_ratio': info.get('quickRatio', np.nan),
                
                # Market metrics
                'market_cap': info.get('marketCap', np.nan),
                'shares_outstanding': info.get('sharesOutstanding', np.nan),
                'float_shares': info.get('floatShares', np.nan),
                'shares_short': info.get('sharesShort', np.nan),
                'short_ratio': info.get('shortRatio', np.nan),
                'beta': info.get('beta', np.nan),
                
                # Analyst sentiment
                'target_mean_price': info.get('targetMeanPrice', np.nan),
                'recommendation_mean': info.get('recommendationMean', np.nan),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', np.nan),
            }
            
            print(f"   ✅ Collected {len([v for v in fundamental_features.values() if not pd.isna(v)])} fundamental metrics")
            return fundamental_features
            
        except Exception as e:
            print(f"   ❌ Error collecting fundamental data: {e}")
            return {}
    
    def collect_sentiment_data(self, symbol):
        """Collect sentiment and news-based features"""
        print(f"📰 Collecting sentiment data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent news
            news = ticker.news
            if news:
                # Simple sentiment analysis based on headlines
                positive_words = ['surge', 'rally', 'bullish', 'growth', 'profit', 'beat', 'strong', 'up', 'rise', 'gain']
                negative_words = ['fall', 'drop', 'bearish', 'loss', 'miss', 'weak', 'down', 'decline', 'crash', 'sell']
                
                total_sentiment = 0
                news_count = 0
                
                for article in news[:10]:  # Last 10 articles
                    title = article.get('title', '').lower()
                    
                    positive_score = sum(1 for word in positive_words if word in title)
                    negative_score = sum(1 for word in negative_words if word in title)
                    
                    total_sentiment += (positive_score - negative_score)
                    news_count += 1
                
                avg_sentiment = total_sentiment / max(news_count, 1)
                
                sentiment_features = {
                    'news_sentiment_score': avg_sentiment,
                    'news_count_recent': news_count,
                    'sentiment_positive_ratio': sum(1 for article in news[:10] 
                                                   if any(word in article.get('title', '').lower() 
                                                         for word in positive_words)) / max(len(news[:10]), 1)
                }
                
                print(f"   ✅ Analyzed {news_count} news articles, sentiment: {avg_sentiment:.2f}")
                return sentiment_features
            else:
                print(f"   ⚠️ No recent news found")
                return {}
                
        except Exception as e:
            print(f"   ❌ Error collecting sentiment data: {e}")
            return {}
    
    def collect_macro_economic_data(self):
        """Collect macroeconomic indicators"""
        print(f"🌍 Collecting macroeconomic data...")
        
        try:
            # Get major indices for market sentiment
            indices = {
                'SPY': '^GSPC',  # S&P 500
                'NASDAQ': '^IXIC',  # NASDAQ
                'VIX': '^VIX',     # Volatility Index
                'DXY': 'DX-Y.NYB', # Dollar Index
                'TNX': '^TNX'      # 10-Year Treasury
            }
            
            macro_features = {}
            
            for name, symbol in indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")  # Last 5 days
                    
                    if len(hist) > 0:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[0]
                        change = ((current - prev) / prev) * 100
                        
                        macro_features[f'{name.lower()}_5d_change'] = change
                        macro_features[f'{name.lower()}_current_level'] = current
                        
                        # Volatility
                        returns = hist['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252) * 100
                        macro_features[f'{name.lower()}_volatility'] = volatility
                        
                except Exception as e:
                    print(f"   ⚠️ Could not get {name}: {e}")
            
            # Market regime detection
            if 'spy_5d_change' in macro_features and 'vix_current_level' in macro_features:
                spy_change = macro_features['spy_5d_change']
                vix_level = macro_features['vix_current_level']
                
                if spy_change > 2 and vix_level < 20:
                    market_regime = 1  # Bullish
                elif spy_change < -2 and vix_level > 30:
                    market_regime = -1  # Bearish
                else:
                    market_regime = 0  # Neutral
                
                macro_features['market_regime'] = market_regime
                macro_features['risk_on_sentiment'] = spy_change / max(vix_level, 1)
            
            print(f"   ✅ Collected {len(macro_features)} macro indicators")
            return macro_features
            
        except Exception as e:
            print(f"   ❌ Error collecting macro data: {e}")
            return {}
    
    def collect_sector_performance(self, symbol):
        """Collect sector and peer performance data"""
        print(f"🏭 Collecting sector performance for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get sector information
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Sector ETFs mapping
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV', 
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Industrials': 'XLI',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU',
                'Communication Services': 'XLC'
            }
            
            sector_features = {
                'sector_encoded': hash(sector) % 100,  # Simple encoding
                'industry_encoded': hash(industry) % 100
            }
            
            # Get sector ETF performance
            sector_etf = sector_etfs.get(sector)
            if sector_etf:
                try:
                    etf_ticker = yf.Ticker(sector_etf)
                    etf_hist = etf_ticker.history(period="1mo")
                    
                    if len(etf_hist) > 0:
                        etf_return = ((etf_hist['Close'].iloc[-1] / etf_hist['Close'].iloc[0]) - 1) * 100
                        sector_features['sector_1m_performance'] = etf_return
                        
                        # Relative performance vs SPY
                        spy_ticker = yf.Ticker('SPY')
                        spy_hist = spy_ticker.history(period="1mo")
                        spy_return = ((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1) * 100
                        
                        sector_features['sector_relative_performance'] = etf_return - spy_return
                        
                except Exception as e:
                    print(f"   ⚠️ Could not get sector ETF data: {e}")
            
            print(f"   ✅ Sector: {sector}, collected {len(sector_features)} features")
            return sector_features
            
        except Exception as e:
            print(f"   ❌ Error collecting sector data: {e}")
            return {}
    
    def collect_options_flow_data(self, symbol):
        """Collect options flow and institutional activity indicators"""
        print(f"📋 Collecting options flow data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get options data
            options_dates = ticker.options
            if options_dates:
                # Get nearest expiry options
                nearest_expiry = options_dates[0]
                calls = ticker.option_chain(nearest_expiry).calls
                puts = ticker.option_chain(nearest_expiry).puts
                
                if not calls.empty and not puts.empty:
                    # Calculate put/call ratio
                    total_call_volume = calls['volume'].fillna(0).sum()
                    total_put_volume = puts['volume'].fillna(0).sum()
                    
                    put_call_ratio = total_put_volume / max(total_call_volume, 1)
                    
                    # Calculate options sentiment
                    call_oi = calls['openInterest'].fillna(0).sum()
                    put_oi = puts['openInterest'].fillna(0).sum()
                    
                    options_features = {
                        'put_call_volume_ratio': put_call_ratio,
                        'call_open_interest': call_oi,
                        'put_open_interest': put_oi,
                        'total_options_volume': total_call_volume + total_put_volume,
                        'options_sentiment': (call_oi - put_oi) / max(call_oi + put_oi, 1)
                    }
                    
                    print(f"   ✅ P/C Ratio: {put_call_ratio:.2f}, collected {len(options_features)} features")
                    return options_features
            
            print(f"   ⚠️ No options data available")
            return {}
            
        except Exception as e:
            print(f"   ❌ Error collecting options data: {e}")
            return {}
    
    def collect_all_features(self, symbol):
        """Collect all sophisticated features for a symbol"""
        print(f"\n🚀 COLLECTING SOPHISTICATED FEATURES FOR {symbol}")
        print("=" * 60)
        
        all_features = {}
        
        # 1. Fundamental Analysis
        fundamental = self.collect_fundamental_data(symbol)
        all_features.update({f'fund_{k}': v for k, v in fundamental.items()})
        
        # 2. Sentiment Analysis
        sentiment = self.collect_sentiment_data(symbol)
        all_features.update({f'sent_{k}': v for k, v in sentiment.items()})
        
        # 3. Macroeconomic Data
        macro = self.collect_macro_economic_data()
        all_features.update({f'macro_{k}': v for k, v in macro.items()})
        
        # 4. Sector Performance
        sector = self.collect_sector_performance(symbol)
        all_features.update({f'sector_{k}': v for k, v in sector.items()})
        
        # 5. Options Flow
        options = self.collect_options_flow_data(symbol)
        all_features.update({f'options_{k}': v for k, v in options.items()})
        
        # Filter out NaN values
        clean_features = {k: v for k, v in all_features.items() if not pd.isna(v)}
        
        print(f"\n📊 FEATURE COLLECTION SUMMARY:")
        print(f"   📈 Fundamental Features: {len(fundamental)}")
        print(f"   📰 Sentiment Features: {len(sentiment)}")
        print(f"   🌍 Macro Features: {len(macro)}")
        print(f"   🏭 Sector Features: {len(sector)}")
        print(f"   📋 Options Features: {len(options)}")
        print(f"   ✅ Total Clean Features: {len(clean_features)}")
        
        return clean_features

def main():
    """Demonstrate sophisticated feature collection"""
    system = SophisticatedTradingSystem()
    
    # Test on NVDA
    features = system.collect_all_features('NVDA')
    
    print(f"\n🎯 ENHANCEMENT RECOMMENDATIONS:")
    print("=" * 60)
    print(f"1. 📊 CURRENT SYSTEM: Only technical indicators (38 features)")
    print(f"2. 🚀 ENHANCED SYSTEM: Multi-source features ({len(features)} features)")
    print(f"")
    print(f"📈 ADDITIONAL DATA SOURCES TO ADD:")
    print(f"   • Fundamental Analysis (P/E, ROE, Debt ratios)")
    print(f"   • News Sentiment (Positive/negative headline analysis)")
    print(f"   • Macroeconomic Indicators (VIX, USD, Interest rates)")
    print(f"   • Sector Performance (Relative to sector ETFs)")
    print(f"   • Options Flow (Put/call ratios, institutional activity)")
    print(f"   • Insider Trading Data")
    print(f"   • Earnings Calendar Events")
    print(f"   • Social Media Sentiment")
    print(f"   • Cryptocurrency Correlation (for tech stocks)")
    print(f"   • International Market Performance")
    print(f"")
    print(f"🎯 EXPECTED BENEFITS:")
    print(f"   ✅ Better performance across different market conditions")
    print(f"   ✅ More resilient to sector-specific downturns")
    print(f"   ✅ Early detection of fundamental shifts")
    print(f"   ✅ Improved risk management")
    print(f"   ✅ Higher consistency across stock types")

if __name__ == "__main__":
    main()
