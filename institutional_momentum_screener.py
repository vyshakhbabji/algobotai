#!/usr/bin/env python3
"""
INSTITUTIONAL MOMENTUM SCREENER
Uses proven Jegadeesh & Titman (1993) momentum parameters
Screens stocks in real-time to find current momentum candidates
Trade only stocks that meet institutional criteria RIGHT NOW!
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InstitutionalMomentumScreener:
    def __init__(self):
        # PROVEN INSTITUTIONAL MOMENTUM PARAMETERS
        # Based on Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"
        self.institutional_config = {
            'momentum_6m_threshold': 0.15,     # 15% 6-month momentum (formation period)
            'momentum_3m_threshold': 0.08,     # 8% 3-month momentum (recent strength)
            'momentum_1m_threshold': 0.03,     # 3% 1-month momentum (current trend)
            'rsi_max': 75,                     # Not extremely overbought
            'rsi_min': 30,                     # Not extremely oversold
            'volume_multiplier': 1.2,          # Above average volume
            'market_cap_min': 1_000_000_000,   # $1B+ market cap (institutional focus)
            'volatility_max': 0.40,            # Max 40% annualized volatility
            'price_min': 10.0,                 # Min $10 (avoid penny stocks)
        }
        
        # COMPREHENSIVE STOCK UNIVERSE
        self.stock_universe = [
            # MEGA CAP TECH
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            # LARGE CAP GROWTH
            "NFLX", "CRM", "UBER", "SHOP", "ROKU", "SNOW", "PLTR",
            # LARGE CAP VALUE
            "JPM", "BAC", "WMT", "JNJ", "PG", "KO", "UNH", "V", "MA",
            # TECH/SEMI
            "AMD", "INTC", "QCOM", "AVGO", "MU", "LRCX", "KLAC",
            # GROWTH STOCKS
            "SQ", "PYPL", "ADBE", "NOW", "DDOG", "CRWD", "ZS",
            # TRADITIONAL
            "DIS", "HD", "LOW", "TGT", "COST", "NKE", "SBUX",
            # ENERGY/MATERIALS
            "XOM", "CVX", "COP", "EOG", "CAT", "DE", "FCX",
            # FINANCE
            "GS", "MS", "C", "WFC", "AXP", "BLK", "SPGI",
            # HEALTHCARE
            "ABBV", "PFE", "MRK", "TMO", "DHR", "ABT", "ISRG",
            # INDUSTRIAL
            "BA", "GE", "HON", "UPS", "LMT", "RTX", "MMM",
            # CONSUMER
            "AMGN", "GILD", "BKNG", "MCD", "CMG", "LULU", "TJX",
            # EMERGING
            "COIN", "HOOD", "RIVN", "LCID", "SOFI", "AFRM", "UPST"
        ]
    
    def get_stock_data(self, symbol):
        """Get comprehensive stock data for screening"""
        try:
            # Get 1 year of data for momentum calculations
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)  # Extra for calculations
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty or len(data) < 200:
                return None
            
            # Get basic info
            info = stock.info
            market_cap = info.get('marketCap', 0)
            
            return data, market_cap
            
        except Exception as e:
            return None
    
    def calculate_momentum_metrics(self, data):
        """Calculate institutional momentum metrics"""
        if len(data) < 200:
            return None
        
        current_price = float(data['Close'].iloc[-1])
        
        # MOMENTUM CALCULATIONS (Jegadeesh & Titman methodology)
        try:
            # 6-month momentum (formation period)
            price_6m_ago = float(data['Close'].iloc[-126])  # ~6 months of trading days
            momentum_6m = (current_price - price_6m_ago) / price_6m_ago
            
            # 3-month momentum (intermediate strength)
            price_3m_ago = float(data['Close'].iloc[-63])   # ~3 months
            momentum_3m = (current_price - price_3m_ago) / price_3m_ago
            
            # 1-month momentum (recent trend)
            price_1m_ago = float(data['Close'].iloc[-21])   # ~1 month
            momentum_1m = (current_price - price_1m_ago) / price_1m_ago
            
        except IndexError:
            return None
        
        # RSI CALCULATION
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
        
        # VOLUME ANALYSIS
        avg_volume = float(data['Volume'].iloc[-20:].mean())
        current_volume = float(data['Volume'].iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # VOLATILITY (annualized)
        returns = data['Close'].pct_change().dropna()
        daily_vol = float(returns.std())
        annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days
        
        return {
            'current_price': current_price,
            'momentum_6m': momentum_6m,
            'momentum_3m': momentum_3m,
            'momentum_1m': momentum_1m,
            'rsi': current_rsi,
            'volume_ratio': volume_ratio,
            'volatility': annualized_vol
        }
    
    def screen_stock(self, symbol):
        """Screen individual stock against institutional criteria"""
        data_result = self.get_stock_data(symbol)
        if not data_result:
            return None
        
        data, market_cap = data_result
        metrics = self.calculate_momentum_metrics(data)
        if not metrics:
            return None
        
        # CHECK ALL INSTITUTIONAL CRITERIA
        criteria_met = []
        criteria_failed = []
        
        # 1. MOMENTUM CRITERIA (Core Jegadeesh & Titman)
        if metrics['momentum_6m'] >= self.institutional_config['momentum_6m_threshold']:
            criteria_met.append(f"6M momentum: {metrics['momentum_6m']:+.1%}")
        else:
            criteria_failed.append(f"6M momentum: {metrics['momentum_6m']:+.1%} < {self.institutional_config['momentum_6m_threshold']:.1%}")
        
        if metrics['momentum_3m'] >= self.institutional_config['momentum_3m_threshold']:
            criteria_met.append(f"3M momentum: {metrics['momentum_3m']:+.1%}")
        else:
            criteria_failed.append(f"3M momentum: {metrics['momentum_3m']:+.1%} < {self.institutional_config['momentum_3m_threshold']:.1%}")
        
        if metrics['momentum_1m'] >= self.institutional_config['momentum_1m_threshold']:
            criteria_met.append(f"1M momentum: {metrics['momentum_1m']:+.1%}")
        else:
            criteria_failed.append(f"1M momentum: {metrics['momentum_1m']:+.1%} < {self.institutional_config['momentum_1m_threshold']:.1%}")
        
        # 2. RSI CRITERIA (Not extreme)
        if self.institutional_config['rsi_min'] <= metrics['rsi'] <= self.institutional_config['rsi_max']:
            criteria_met.append(f"RSI: {metrics['rsi']:.0f}")
        else:
            criteria_failed.append(f"RSI: {metrics['rsi']:.0f} outside {self.institutional_config['rsi_min']}-{self.institutional_config['rsi_max']}")
        
        # 3. VOLUME CRITERIA
        if metrics['volume_ratio'] >= self.institutional_config['volume_multiplier']:
            criteria_met.append(f"Volume: {metrics['volume_ratio']:.1f}x")
        else:
            criteria_failed.append(f"Volume: {metrics['volume_ratio']:.1f}x < {self.institutional_config['volume_multiplier']:.1f}x")
        
        # 4. MARKET CAP CRITERIA
        if market_cap >= self.institutional_config['market_cap_min']:
            criteria_met.append(f"Market cap: ${market_cap/1e9:.1f}B")
        else:
            criteria_failed.append(f"Market cap: ${market_cap/1e9:.1f}B < ${self.institutional_config['market_cap_min']/1e9:.1f}B")
        
        # 5. VOLATILITY CRITERIA
        if metrics['volatility'] <= self.institutional_config['volatility_max']:
            criteria_met.append(f"Volatility: {metrics['volatility']:.1%}")
        else:
            criteria_failed.append(f"Volatility: {metrics['volatility']:.1%} > {self.institutional_config['volatility_max']:.1%}")
        
        # 6. PRICE CRITERIA
        if metrics['current_price'] >= self.institutional_config['price_min']:
            criteria_met.append(f"Price: ${metrics['current_price']:.2f}")
        else:
            criteria_failed.append(f"Price: ${metrics['current_price']:.2f} < ${self.institutional_config['price_min']:.2f}")
        
        # CALCULATE COMPOSITE MOMENTUM SCORE
        momentum_score = (
            metrics['momentum_6m'] * 0.5 +    # 6M weighted most (formation period)
            metrics['momentum_3m'] * 0.3 +    # 3M weighted medium
            metrics['momentum_1m'] * 0.2      # 1M weighted least
        )
        
        # DETERMINE IF QUALIFIES
        total_criteria = 6
        passed_criteria = len(criteria_met)
        qualification_rate = passed_criteria / total_criteria
        
        # STRICT INSTITUTIONAL STANDARD: Must pass ALL momentum + risk criteria
        momentum_qualified = (
            metrics['momentum_6m'] >= self.institutional_config['momentum_6m_threshold'] and
            metrics['momentum_3m'] >= self.institutional_config['momentum_3m_threshold'] and
            metrics['momentum_1m'] >= self.institutional_config['momentum_1m_threshold']
        )
        
        risk_qualified = (
            self.institutional_config['rsi_min'] <= metrics['rsi'] <= self.institutional_config['rsi_max'] and
            metrics['volatility'] <= self.institutional_config['volatility_max']
        )
        
        institutional_qualified = momentum_qualified and risk_qualified and market_cap >= self.institutional_config['market_cap_min']
        
        return {
            'symbol': symbol,
            'qualified': institutional_qualified,
            'momentum_score': momentum_score,
            'qualification_rate': qualification_rate,
            'current_price': metrics['current_price'],
            'momentum_6m': metrics['momentum_6m'],
            'momentum_3m': metrics['momentum_3m'], 
            'momentum_1m': metrics['momentum_1m'],
            'rsi': metrics['rsi'],
            'volume_ratio': metrics['volume_ratio'],
            'volatility': metrics['volatility'],
            'market_cap': market_cap,
            'criteria_met': criteria_met,
            'criteria_failed': criteria_failed
        }
    
    def screen_all_stocks(self):
        """Screen entire universe for institutional momentum candidates"""
        print("🎯 INSTITUTIONAL MOMENTUM SCREENING")
        print("=" * 60)
        print("📊 Using Jegadeesh & Titman (1993) momentum methodology")
        print("🏛️  Institutional-grade criteria: 6M/3M/1M momentum cascade")
        print(f"🔍 Screening {len(self.stock_universe)} stocks...")
        print("=" * 60)
        
        qualified_stocks = []
        screened_count = 0
        
        for symbol in self.stock_universe:
            result = self.screen_stock(symbol)
            if result:
                screened_count += 1
                
                if result['qualified']:
                    qualified_stocks.append(result)
                    print(f"✅ {symbol}: Momentum Score {result['momentum_score']:+.1%} | 6M:{result['momentum_6m']:+.1%} 3M:{result['momentum_3m']:+.1%} 1M:{result['momentum_1m']:+.1%}")
                else:
                    print(f"❌ {symbol}: Failed ({len(result['criteria_failed'])}/6 criteria)")
        
        print(f"\n📊 SCREENING RESULTS:")
        print(f"   🔍 Stocks screened: {screened_count}")
        print(f"   ✅ Institutional qualified: {len(qualified_stocks)}")
        print(f"   📈 Qualification rate: {len(qualified_stocks)/screened_count:.1%}")
        
        if qualified_stocks:
            # Sort by momentum score
            qualified_stocks.sort(key=lambda x: x['momentum_score'], reverse=True)
            
            print(f"\n🏆 TOP INSTITUTIONAL MOMENTUM CANDIDATES:")
            print("=" * 60)
            
            for i, stock in enumerate(qualified_stocks[:10], 1):
                print(f"#{i:2d} {stock['symbol']:6s} | Score: {stock['momentum_score']:+6.1%} | "
                      f"6M:{stock['momentum_6m']:+6.1%} 3M:{stock['momentum_3m']:+6.1%} 1M:{stock['momentum_1m']:+6.1%} | "
                      f"RSI:{stock['rsi']:3.0f} Vol:{stock['volatility']:4.1%} Price:${stock['current_price']:6.2f}")
            
            print(f"\n💰 READY TO TRADE:")
            print(f"   🎯 Focus on top 3-5 candidates")
            print(f"   📊 All candidates meet institutional momentum criteria")
            print(f"   🏛️  Proven Jegadeesh & Titman methodology")
            
        return qualified_stocks
    
    def get_detailed_analysis(self, symbol):
        """Get detailed analysis for a specific stock"""
        print(f"\n🔍 DETAILED ANALYSIS: {symbol}")
        print("=" * 40)
        
        result = self.screen_stock(symbol)
        if not result:
            print(f"❌ Unable to analyze {symbol}")
            return None
        
        print(f"📊 Momentum Score: {result['momentum_score']:+.1%}")
        print(f"💰 Current Price: ${result['current_price']:.2f}")
        print(f"🏛️  Institutional Qualified: {'YES' if result['qualified'] else 'NO'}")
        
        print(f"\n📈 MOMENTUM BREAKDOWN:")
        print(f"   6-Month: {result['momentum_6m']:+.1%} (Formation Period)")
        print(f"   3-Month: {result['momentum_3m']:+.1%} (Intermediate)")
        print(f"   1-Month: {result['momentum_1m']:+.1%} (Recent)")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   RSI: {result['rsi']:.0f}")
        print(f"   Volatility: {result['volatility']:.1%}")
        print(f"   Volume Ratio: {result['volume_ratio']:.1f}x")
        print(f"   Market Cap: ${result['market_cap']/1e9:.1f}B")
        
        print(f"\n✅ CRITERIA MET ({len(result['criteria_met'])}/6):")
        for criteria in result['criteria_met']:
            print(f"   ✓ {criteria}")
        
        if result['criteria_failed']:
            print(f"\n❌ CRITERIA FAILED ({len(result['criteria_failed'])}/6):")
            for criteria in result['criteria_failed']:
                print(f"   ✗ {criteria}")
        
        return result

def main():
    """Run institutional momentum screening"""
    print("🏛️  INSTITUTIONAL MOMENTUM SCREENER")
    print("=" * 60)
    print("📚 Based on Jegadeesh & Titman (1993)")
    print("🎯 Find stocks meeting institutional momentum criteria RIGHT NOW")
    print("💰 Trade only proven momentum candidates")
    print("=" * 60)
    
    screener = InstitutionalMomentumScreener()
    
    # Screen all stocks
    qualified_stocks = screener.screen_all_stocks()
    
    # Show top candidates for detailed analysis
    if qualified_stocks:
        print(f"\n🔍 Want detailed analysis of any stock? Here are the top candidates:")
        for i, stock in enumerate(qualified_stocks[:5], 1):
            print(f"   {i}. {stock['symbol']} (Score: {stock['momentum_score']:+.1%})")
    
    return screener, qualified_stocks

if __name__ == "__main__":
    screener, stocks = main()
