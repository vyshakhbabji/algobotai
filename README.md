# 🚀 AI Paper Trading Bot

A sophisticated AI-powered paper trading system with real-time market data and machine learning predictions.

## 🎯 Features

- **$10,000 Starting Capital** - Fresh paper trading account
- **Real-time Market Data** - Live price feeds via yfinance
- **AI Trading Signals** - Machine learning models with 36.23% backtested returns
- **Risk Management** - Maximum 10% position sizing
- **Portfolio Tracking** - Real-time P&L and performance monitoring
- **Auto-Execution** - High confidence signals (>75%) execute automatically

## 📊 AI Model Performance

- **15 Trained Models** - All stocks showing positive R² scores (0.489-0.619)
- **Ensemble Methods** - Random Forest and Gradient Boosting
- **Cross-Validation** - Time series split validation for reliability
- **Confidence Scoring** - 0-100% confidence levels for each prediction

## 🏗️ Architecture

### Core Components
- `live_paper_trading.py` - Main trading dashboard
- `improved_ai_portfolio_manager.py` - AI prediction engine
- `app.py` - Deployment entry point

### AI Models
- **Feature Engineering** - 16 technical indicators
- **Model Training** - Random Forest and Gradient Boosting
- **Prediction Confidence** - Weighted by model R² performance
- **Signal Generation** - Buy/Hold recommendations

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run live_paper_trading.py
```

### Cloud Deployment
1. **Streamlit Cloud** (FREE)
   - Fork this repository
   - Connect to Streamlit Cloud
   - Deploy with one click

2. **Railway** ($5-20/month)
   - Connect GitHub repository
   - Auto-deploys from main branch

3. **Render** ($7-25/month)
   - Connect repository
   - Deploy as web service

## 💰 Cost Analysis

- **Development**: $0 (local testing)
- **Deployment**: $0-25/month (well under $100 budget)
- **Data**: $0 (using free yfinance API)
- **Total**: Extremely cost-effective

## 📈 Trading Strategy

### Signal Generation
1. **Data Collection** - 60 days of historical data
2. **Feature Calculation** - Technical indicators and momentum
3. **AI Prediction** - Ensemble model confidence scoring
4. **Risk Assessment** - Position sizing based on confidence
5. **Execution** - Manual or automatic trade placement

### Risk Management
- Maximum 10% portfolio allocation per position
- Confidence-based position sizing
- Real-time portfolio value monitoring
- Stop-loss through confidence thresholds

## 🎯 30-Day Challenge

**Objective**: Validate AI models in live market conditions
- **Starting Capital**: $10,000
- **Target Performance**: Beat 36.23% backtested returns
- **Duration**: 30 days of live paper trading
- **Success Metrics**: Total return, Sharpe ratio, win rate

## 🔧 Technical Details

### Dependencies
- Streamlit for web interface
- yfinance for market data
- scikit-learn for AI models
- plotly for visualizations
- pandas/numpy for data processing

### Data Sources
- Yahoo Finance (yfinance) - Real-time and historical data
- No API keys required
- Free tier sufficient for all operations

## 📊 Performance Tracking

The system tracks:
- **Portfolio Value** - Real-time total value
- **Unrealized P&L** - Position-level profit/loss
- **Trade History** - Complete transaction log
- **AI Performance** - Signal accuracy and confidence
- **Risk Metrics** - Exposure and diversification

## 🎮 How to Use

1. **Access Dashboard** - Navigate to deployed URL
2. **Monitor Portfolio** - Check current positions and cash
3. **Generate Signals** - Click "Generate AI Trading Signals"
4. **Execute Trades** - Manual or automatic execution
5. **Track Performance** - Monitor real-time results

## 🚨 Disclaimer

This is a **PAPER TRADING** system for educational and testing purposes. No real money is involved. Past performance does not guarantee future results.

---

**Ready to test AI trading in live market conditions!** 🎯📈
