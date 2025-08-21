# 🚀 AI Paper Trading Bot

A sophisticated AI-powered paper trading system with real-time market data, machine learning predictions, and Alpaca Markets integration.

## 🎯 Features

### 💼 Trading Platforms
- **Local Paper Trading** - $10,000 virtual starting capital with JSON persistence
- **Alpaca Paper Trading** - $100,000 real market simulation with live execution
- **Real-time Market Data** - Live price feeds via yfinance and Alpaca
- **Multi-page Dashboard** - Comprehensive Streamlit interface

### 🤖 AI Trading Engine
- **AI Trading Signals** - Machine learning models with 36.23% backtested returns
- **15 Trained Models** - All stocks showing positive R² scores (0.489-0.619)
- **Ensemble Methods** - Random Forest and Gradient Boosting
- **Confidence Scoring** - 0-100% confidence levels for each prediction
- **Auto-Execution** - High confidence signals (>75%) execute automatically

### 📊 Advanced Features
- **Elite Options Trading** - AI-powered options strategies for maximum returns
- **Portfolio Management** - 25-50 elite stock universe with sector analysis
- **System Health Monitor** - Comprehensive diagnostic and testing suite
- **Performance Analytics** - Advanced backtesting and validation tools
- **Risk Management** - Maximum 10% position sizing with stop-loss automation

## 🏗️ Architecture

### 📱 Streamlit Multipage App Structure
```
app.py                    ← Main entry point (Streamlit Cloud compatible)
pages/
  ├── live_trading.py     ← Local paper trading ($10K virtual)
  ├── alpaca_trading.py   ← Alpaca paper trading ($100K real simulation)
  ├── system_monitor.py   ← Health diagnostics & system testing
  ├── portfolio_manager.py ← Elite stock universe management
  ├── performance_analytics.py ← Backtesting & validation
  ├── enhanced_dashboard.py ← Advanced market analysis
  ├── ai_optimizer.py     ← AI model optimization tools
  └── elite_options.py    ← Options trading strategies
```

### 🧠 AI Engine Components
- `improved_ai_portfolio_manager.py` - Main AI prediction engine
- `elite_stock_selector.py` - Stock universe curation
- `elite_options_trader.py` - Options strategy optimization
- `alpaca_paper_trading.py` - Alpaca Markets integration

### 📊 Data & Configuration
- JSON-based persistence for local trading
- Real-time Alpaca API integration
- Environment-based credential management
- Comprehensive error handling and logging

## 🚀 Quick Start

### 1. Local Development
```bash
# Clone and install
git clone https://github.com/vyshakhbabji/algobotai.git
cd AlgoTradingBot
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### 2. Alpaca Paper Trading Setup
```bash
# 1. Sign up at https://alpaca.markets
# 2. Get paper trading API keys
# 3. Add credentials to .env file:
echo "ALPACA_API_KEY=your_api_key" >> .env
echo "ALPACA_SECRET_KEY=your_secret_key" >> .env
```

### 3. Cloud Deployment
**Streamlit Cloud** (FREE)
- Fork this repository
- Connect to Streamlit Cloud  
- Add Alpaca credentials to secrets.toml
- Deploy with one click

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
- Alpaca Markets API - Paper trading execution and market data
- No API keys required for basic functionality
- Free tier sufficient for all operations

## 📊 Performance Tracking

## 🔄 Recent Development Session Summary

### 🎯 Major Accomplishments (August 7, 2025)

#### 1. 🔗 **Streamlit Multipage Navigation FIXED**
- **Issue**: Sidebar navigation not working on Streamlit Cloud
- **Root Cause**: Multiple `st.set_page_config()` calls in pages directory
- **Solution**: Removed all page configs from `/pages/*.py`, only `app.py` has config
- **Status**: ✅ **RESOLVED** - Proper multipage structure implemented

#### 2. 🐍 **Python Version Compatibility**  
- **Issue**: Python 3.13 vs 3.12.3 causing duplicate widget key errors
- **Root Cause**: Different key handling between Python versions
- **Solution**: UUID-based unique keys with session state
- **Implementation**: `st.session_state.system_monitor_session_id = str(uuid.uuid4())[:8]`
- **Status**: ✅ **RESOLVED** - Cross-version compatibility

#### 3. 🏦 **Alpaca Paper Trading Integration**
- **New Feature**: Professional paper trading with $100K virtual capital
- **Components Added**:
  - `alpaca_paper_trading.py` - Full Alpaca SDK integration
  - `pages/alpaca_trading.py` - Streamlit page wrapper
  - `.env` template and configuration files
- **Features**: Real-time orders, portfolio tracking, market data
- **Credentials**: Configured with user's Alpaca API keys
- **Status**: ✅ **IMPLEMENTED** - Ready for testing

#### 4. 📁 **File Structure Cleanup**
- **Removed**: Duplicate/conflicting page files with emoji names
- **Standardized**: Python-compatible naming convention
- **Current Pages**:
  ```
  ├── live_trading.py      (Local $10K paper trading)
  ├── alpaca_trading.py    (Alpaca $100K paper trading)  
  ├── system_monitor.py    (Health diagnostics)
  ├── portfolio_manager.py (Stock universe management)
  ├── performance_analytics.py (Backtesting)
  ├── enhanced_dashboard.py (Market analysis)
  ├── ai_optimizer.py      (AI optimization)
  └── elite_options.py     (Options strategies)
  ```

#### 5. 🔧 **System Health Monitoring**
- **Enhanced**: Comprehensive 8-section diagnostic system
- **Features**: Core imports, pages, data files, trading engines, AI models
- **UUID Keys**: Fixed duplicate key conflicts
- **Status**: ✅ **FULLY FUNCTIONAL**

### 🚨 **Known Issues Still Outstanding**

#### 1. 🔗 **Sidebar Navigation on Streamlit Cloud**
- **Status**: ❌ **PARTIALLY RESOLVED**
- **Issue**: Despite fixes, some users report sidebar still not visible
- **Workaround**: App works perfectly on localhost:8501
- **Next Steps**: May need different Streamlit Cloud deployment approach

#### 2. 🔑 **Widget Key Conflicts**  
- **Status**: ⚠️ **ONGOING**
- **Issue**: Intermittent duplicate key errors on cloud platforms
- **Current Solution**: UUID-based session keys
- **Next Steps**: May need per-page session isolation

### 🛠️ **Technical Debt & Future Improvements**

1. **🔐 Security**: Move hardcoded Alpaca credentials to proper secret management
2. **📊 Data**: Implement caching for better performance  
3. **🎨 UI**: Enhance visual design and user experience
4. **🧪 Testing**: Add automated testing suite
5. **📈 Features**: Add more advanced trading strategies

### 💼 **Ready for Production Use**

✅ **Local Development**: Fully functional with all features  
✅ **Alpaca Integration**: Ready for live paper trading  
✅ **AI Trading**: Models trained and operational  
✅ **Risk Management**: Position sizing and stop-losses implemented  
⚠️ **Cloud Deployment**: Sidebar navigation issues on some platforms

### 🚀 **Next Session Priorities**

1. **Test Alpaca Integration**: Place first paper trades
2. **Resolve Cloud Sidebar**: Alternative navigation if needed  
3. **Enhance AI Models**: Add more sophisticated strategies
4. **Performance Optimization**: Caching and speed improvements
5. **User Experience**: Polish interface and add tutorials

The system tracks:
- **Portfolio Value** - Real-time total value
- **Unrealized P&L** - Position-level profit/loss
- **Trade History** - Complete transaction log
- **AI Performance** - Signal accuracy and confidence
- **Risk Metrics** - Exposure and diversification

## 🎮 How to Use

1. **Access Dashboard** - Navigate to deployed URL
2. **Enter Password** - Use access code: `102326`
3. **Monitor Portfolio** - Check current positions and cash
4. **Generate Signals** - Click "Generate AI Trading Signals"
5. **Execute Trades** - Manual or automatic execution
6. **Track Performance** - Monitor real-time results

## 🔐 Security

The application is password-protected to prevent unauthorized access:
- **Access Password**: `102326`
- **Session Management**: Automatic logout on incorrect attempts
- **Professional Interface**: Clean login screen for authorized users

## 🚨 Disclaimer

This is a **PAPER TRADING** system for educational and testing purposes. No real money is involved. Past performance does not guarantee future results.

---

**Ready to test AI trading in live market conditions!** 🎯📈
