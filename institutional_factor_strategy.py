#!/usr/bin/env python3
"""
INSTITUTIONAL FACTOR-BASED STRATEGY
Based on proven academic research and institutional methodologies:

1. FAMA-FRENCH 5-FACTOR MODEL (Eugene Fama, Nobel Prize 2013)
2. MOMENTUM FACTOR (Jegadeesh & Titman, 1993)
3. QUALITY FACTOR (Asness, Frazzini & Pedersen, 2014)
4. LOW VOLATILITY ANOMALY (Haugen & Baker, 1991)
5. MEAN REVERSION (Poterba & Summers, 1988)

These factors have been validated across decades of institutional use
and consistently outperform buy-and-hold in academic studies.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InstitutionalFactorStrategy:
    def __init__(self, stocks=None):
        # INSTITUTIONAL STOCK UNIVERSE - Focus on large-cap liquid stocks
        if stocks:
            self.stocks = stocks
        else:
            self.stocks = [
                # MEGA CAP TECH (High momentum, quality)
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                # FINANCIAL LEADERS (Value, dividend yield)
                "JPM", "BAC", "WFC", "GS", "MS", "C",
                # CONSUMER STAPLES (Low volatility, quality)
                "PG", "KO", "WMT", "PEP", "COST", "HD",
                # HEALTHCARE (Defensive, quality)
                "JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK",
                # INDUSTRIALS (Economic sensitivity)
                "CAT", "BA", "GE", "MMM", "HON", "UPS"
            ]
    
    def fetch_fundamental_data(self, symbol):
        """Fetch fundamental data for factor calculations"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get financial data
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            
            # Extract key metrics for factor analysis
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', None)
            pb_ratio = info.get('priceToBook', None)
            roe = info.get('returnOnEquity', None)
            debt_to_equity = info.get('debtToEquity', None)
            dividend_yield = info.get('dividendYield', 0)
            
            return {
                'market_cap': market_cap,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'dividend_yield': dividend_yield
            }
        except:
            return None
    
    def calculate_momentum_factor(self, data, lookback_days=252):
        """Calculate momentum factor (12-1 month momentum)"""
        if len(data) < lookback_days + 21:
            return 0
        
        # 12-month return excluding last month (industry standard)
        end_idx = -21  # Exclude last month
        start_idx = end_idx - lookback_days
        
        start_price = data['Close'].iloc[start_idx]
        end_price = data['Close'].iloc[end_idx]
        
        momentum = (end_price - start_price) / start_price
        return momentum
    
    def calculate_volatility_factor(self, data, lookback_days=252):
        """Calculate volatility factor (lower volatility = higher score)"""
        if len(data) < lookback_days:
            return 0
        
        recent_data = data['Close'].tail(lookback_days)
        returns = recent_data.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Invert so lower volatility = higher score
        vol_score = 1 / (1 + volatility) if volatility > 0 else 0
        return vol_score
    
    def calculate_mean_reversion_factor(self, data, short_window=20, long_window=252):
        """Calculate mean reversion factor"""
        if len(data) < long_window:
            return 0
        
        current_price = data['Close'].iloc[-1]
        short_ma = data['Close'].tail(short_window).mean()
        long_ma = data['Close'].tail(long_window).mean()
        
        # Mean reversion: buy when below long-term average
        reversion_signal = (long_ma - current_price) / long_ma
        return reversion_signal
    
    def calculate_quality_factor(self, fundamental_data):
        """Calculate quality factor from fundamental metrics"""
        if not fundamental_data:
            return 0
        
        quality_score = 0
        
        # ROE component (higher is better)
        roe = fundamental_data.get('roe')
        if roe and roe > 0:
            quality_score += min(roe / 0.20, 1)  # Cap at 20% ROE
        
        # Low debt component (lower debt-to-equity is better)
        debt_ratio = fundamental_data.get('debt_to_equity')
        if debt_ratio is not None:
            debt_score = max(0, 1 - debt_ratio / 100)  # Normalize
            quality_score += debt_score
        
        # Reasonable valuation (avoid extreme P/E ratios)
        pe_ratio = fundamental_data.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            # Prefer P/E between 10-25
            if 10 <= pe_ratio <= 25:
                quality_score += 1
            elif pe_ratio < 10:
                quality_score += 0.5  # Could be value trap
            else:
                quality_score += max(0, 1 - (pe_ratio - 25) / 50)
        
        return quality_score / 3  # Normalize to 0-1
    
    def calculate_value_factor(self, fundamental_data):
        """Calculate value factor using multiple metrics"""
        if not fundamental_data:
            return 0
        
        value_score = 0
        
        # P/E ratio (lower is better)
        pe_ratio = fundamental_data.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            pe_score = max(0, 1 - (pe_ratio - 10) / 40)  # Normalize around P/E 10-50
            value_score += pe_score
        
        # P/B ratio (lower is better)
        pb_ratio = fundamental_data.get('pb_ratio')
        if pb_ratio and pb_ratio > 0:
            pb_score = max(0, 1 - (pb_ratio - 1) / 5)  # Normalize around P/B 1-6
            value_score += pb_score
        
        # Dividend yield (higher is better for value)
        div_yield = fundamental_data.get('dividend_yield', 0)
        if div_yield > 0:
            div_score = min(div_yield / 0.06, 1)  # Cap at 6% yield
            value_score += div_score
        
        return value_score / 3  # Normalize to 0-1
    
    def calculate_composite_factor_score(self, symbol, data, fundamental_data):
        """Calculate composite factor score using institutional weightings"""
        
        # Calculate individual factors
        momentum = self.calculate_momentum_factor(data)
        volatility = self.calculate_volatility_factor(data)
        mean_reversion = self.calculate_mean_reversion_factor(data)
        quality = self.calculate_quality_factor(fundamental_data)
        value = self.calculate_value_factor(fundamental_data)
        
        # INSTITUTIONAL FACTOR WEIGHTINGS (based on academic research)
        # These weights are derived from factor investing literature
        weights = {
            'momentum': 0.25,      # Strong academic evidence
            'quality': 0.25,       # Growing institutional adoption
            'value': 0.20,         # Traditional factor, lower weight in growth markets
            'low_vol': 0.15,       # Risk-adjusted returns
            'mean_reversion': 0.15  # Short-term tactical
        }
        
        # Calculate weighted composite score
        composite_score = (
            momentum * weights['momentum'] +
            quality * weights['quality'] +
            value * weights['value'] +
            volatility * weights['low_vol'] +
            mean_reversion * weights['mean_reversion']
        )
        
        return {
            'symbol': symbol,
            'composite_score': composite_score,
            'momentum': momentum,
            'quality': quality,
            'value': value,
            'low_volatility': volatility,
            'mean_reversion': mean_reversion,
            'individual_factors': {
                'momentum': momentum,
                'quality': quality,
                'value': value,
                'volatility': volatility,
                'mean_reversion': mean_reversion
            }
        }
    
    def rank_stocks_by_factors(self, lookback_days=252):
        """Rank all stocks by composite factor scores"""
        print("🏛️ INSTITUTIONAL FACTOR ANALYSIS")
        print("=" * 50)
        print("📊 Analyzing stocks using proven academic factors...")
        
        stock_scores = []
        
        for symbol in self.stocks:
            try:
                print(f"🔍 Analyzing {symbol}...")
                
                # Download price data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days + 100)
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if data.empty or len(data) < lookback_days:
                    print(f"   ❌ Insufficient data for {symbol}")
                    continue
                
                # Get fundamental data
                fundamental_data = self.fetch_fundamental_data(symbol)
                
                # Calculate factor scores
                factor_analysis = self.calculate_composite_factor_score(symbol, data, fundamental_data)
                stock_scores.append(factor_analysis)
                
                print(f"   ✅ Score: {factor_analysis['composite_score']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by composite score (descending)
        stock_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"\n🏆 FACTOR-BASED RANKINGS:")
        print("-" * 40)
        for i, stock in enumerate(stock_scores[:10], 1):
            print(f"{i:2d}. {stock['symbol']:<6} Score: {stock['composite_score']:.3f}")
            print(f"     Mom: {stock['momentum']:+.2f} | Qual: {stock['quality']:.2f} | Val: {stock['value']:.2f}")
        
        return stock_scores
    
    def create_factor_portfolio(self, stock_scores, portfolio_size=10):
        """Create factor-tilted portfolio using institutional methodology"""
        
        # Select top stocks by factor score
        top_stocks = stock_scores[:portfolio_size]
        
        print(f"\n💼 INSTITUTIONAL FACTOR PORTFOLIO ({portfolio_size} stocks)")
        print("=" * 55)
        
        # SMART BETA WEIGHTING (institutional approach)
        # Weight by factor score with some diversification constraints
        total_score = sum(stock['composite_score'] for stock in top_stocks)
        
        portfolio = []
        for stock in top_stocks:
            # Base weight from factor score
            base_weight = stock['composite_score'] / total_score
            
            # Apply diversification constraints (no stock > 15%)
            weight = min(base_weight, 0.15)
            
            portfolio.append({
                'symbol': stock['symbol'],
                'weight': weight,
                'factor_score': stock['composite_score'],
                'factors': stock['individual_factors']
            })
        
        # Normalize weights to sum to 1
        total_weight = sum(p['weight'] for p in portfolio)
        for p in portfolio:
            p['weight'] = p['weight'] / total_weight
        
        # Display portfolio
        print("📊 PORTFOLIO COMPOSITION:")
        for i, holding in enumerate(portfolio, 1):
            print(f"{i:2d}. {holding['symbol']:<6} {holding['weight']:6.1%} "
                  f"(Score: {holding['factor_score']:.3f})")
        
        return portfolio
    
    def backtest_factor_strategy(self, portfolio, period_days=730):
        """Backtest the factor-based portfolio strategy"""
        print(f"\n🧪 BACKTESTING FACTOR STRATEGY")
        print("=" * 40)
        print(f"📅 Period: {period_days} days")
        print(f"📊 Portfolio: {len(portfolio)} stocks")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)
        
        # Download data for all portfolio stocks
        portfolio_data = {}
        for holding in portfolio:
            symbol = holding['symbol']
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    portfolio_data[symbol] = data
            except:
                continue
        
        if not portfolio_data:
            print("❌ No data available for backtesting")
            return None
        
        # Calculate portfolio returns
        initial_value = 10000
        portfolio_values = []
        dates = []
        
        # Get common date range
        all_dates = set()
        for data in portfolio_data.values():
            all_dates.update(data.index)
        common_dates = sorted(list(all_dates))[-period_days:]
        
        # Calculate daily portfolio values
        for date in common_dates:
            daily_value = 0
            
            for holding in portfolio:
                symbol = holding['symbol']
                weight = holding['weight']
                
                if symbol in portfolio_data and date in portfolio_data[symbol].index:
                    price = portfolio_data[symbol].loc[date, 'Close']
                    
                    # First day establishes position sizes
                    if not portfolio_values:
                        shares = (initial_value * weight) / price
                        holding['shares'] = shares
                    
                    # Calculate current value
                    shares = holding.get('shares', 0)
                    daily_value += shares * price
            
            if daily_value > 0:
                portfolio_values.append(daily_value)
                dates.append(date)
        
        if not portfolio_values:
            print("❌ Unable to calculate portfolio values")
            return None
        
        # Calculate performance metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate buy-and-hold benchmark (equal-weighted index)
        benchmark_values = []
        for i, date in enumerate(dates):
            bench_value = 0
            valid_stocks = 0
            
            for holding in portfolio:
                symbol = holding['symbol']
                if symbol in portfolio_data and date in portfolio_data[symbol].index:
                    # Equal weight benchmark
                    if i == 0:  # First day
                        start_price = portfolio_data[symbol].loc[date, 'Close']
                        holding['bench_shares'] = (initial_value / len(portfolio)) / start_price
                    
                    current_price = portfolio_data[symbol].loc[date, 'Close']
                    bench_shares = holding.get('bench_shares', 0)
                    bench_value += bench_shares * current_price
                    valid_stocks += 1
            
            if valid_stocks > 0:
                benchmark_values.append(bench_value)
        
        benchmark_return = (benchmark_values[-1] - initial_value) / initial_value * 100 if benchmark_values else 0
        outperformance = total_return - benchmark_return
        
        # Performance analytics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized
        sharpe_ratio = (total_return / (period_days/365)) / volatility if volatility > 0 else 0
        
        max_drawdown = 0
        peak = initial_value
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Results
        print(f"\n📈 BACKTEST RESULTS:")
        print(f"   💰 Portfolio Return: {total_return:+.1f}%")
        print(f"   📊 Benchmark Return: {benchmark_return:+.1f}%")
        print(f"   🎯 Outperformance: {outperformance:+.1f}%")
        print(f"   📉 Volatility: {volatility:.1f}%")
        print(f"   ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   📉 Max Drawdown: {max_drawdown:.1%}")
        
        return {
            'portfolio_return': total_return,
            'benchmark_return': benchmark_return,
            'outperformance': outperformance,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'portfolio_values': portfolio_values,
            'dates': dates
        }
    
    def generate_trading_signals(self, portfolio):
        """Generate institutional-style trading signals"""
        print(f"\n📡 GENERATING TRADING SIGNALS")
        print("=" * 35)
        
        signals = []
        
        for holding in portfolio:
            symbol = holding['symbol']
            weight = holding['weight']
            factors = holding['factors']
            
            # INSTITUTIONAL SIGNAL LOGIC
            signal_strength = 0
            reasons = []
            
            # Momentum signal
            if factors['momentum'] > 0.10:  # Strong momentum
                signal_strength += 2
                reasons.append("Strong momentum")
            elif factors['momentum'] > 0.05:
                signal_strength += 1
                reasons.append("Moderate momentum")
            
            # Quality signal
            if factors['quality'] > 0.7:
                signal_strength += 2
                reasons.append("High quality")
            elif factors['quality'] > 0.5:
                signal_strength += 1
                reasons.append("Good quality")
            
            # Value signal
            if factors['value'] > 0.6:
                signal_strength += 1
                reasons.append("Attractive valuation")
            
            # Determine signal
            if signal_strength >= 4:
                signal = "STRONG BUY"
            elif signal_strength >= 3:
                signal = "BUY"
            elif signal_strength >= 2:
                signal = "HOLD"
            else:
                signal = "WEAK"
            
            signals.append({
                'symbol': symbol,
                'signal': signal,
                'strength': signal_strength,
                'weight': weight,
                'reasons': reasons,
                'factor_score': holding['factor_score']
            })
            
            print(f"📊 {symbol:<6} {signal:<12} ({signal_strength}/6) - {', '.join(reasons)}")
        
        return signals

def main():
    """Run institutional factor-based strategy"""
    print("🏛️ INSTITUTIONAL FACTOR-BASED TRADING STRATEGY")
    print("=" * 55)
    print("📚 Based on Nobel Prize-winning research:")
    print("   • Fama-French Factor Models")
    print("   • Momentum (Jegadeesh & Titman)")
    print("   • Quality Investing (AQR Capital)")
    print("   • Low Volatility Anomaly")
    print("   • Mean Reversion Theory")
    print("=" * 55)
    
    # Initialize strategy
    strategy = InstitutionalFactorStrategy()
    
    # Step 1: Analyze all stocks using factor models
    stock_scores = strategy.rank_stocks_by_factors()
    
    if not stock_scores:
        print("❌ No stocks could be analyzed")
        return None
    
    # Step 2: Create factor-tilted portfolio
    portfolio = strategy.create_factor_portfolio(stock_scores, portfolio_size=12)
    
    # Step 3: Backtest the strategy
    backtest_results = strategy.backtest_factor_strategy(portfolio, period_days=730)
    
    # Step 4: Generate current trading signals
    signals = strategy.generate_trading_signals(portfolio)
    
    print(f"\n🎯 INSTITUTIONAL STRATEGY SUMMARY:")
    print("=" * 40)
    if backtest_results:
        print(f"📈 2-Year Performance: {backtest_results['outperformance']:+.1f}% vs benchmark")
        print(f"⚡ Risk-Adjusted Return (Sharpe): {backtest_results['sharpe_ratio']:.2f}")
        print(f"📊 Portfolio Volatility: {backtest_results['volatility']:.1f}%")
    
    strong_buys = [s for s in signals if s['signal'] == 'STRONG BUY']
    print(f"🚀 Strong Buy Signals: {len(strong_buys)} stocks")
    
    return {
        'strategy': strategy,
        'portfolio': portfolio,
        'backtest_results': backtest_results,
        'signals': signals
    }

if __name__ == "__main__":
    results = main()
