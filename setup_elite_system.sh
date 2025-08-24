#!/bin/bash

# Elite ML Trading System Setup Script
echo "🚀 Setting up Elite ML Trading System..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv elite_trading_env
source elite_trading_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install TA-Lib (requires system dependencies)
echo "📊 Installing TA-Lib..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        brew install ta-lib
    else
        echo "❌ Homebrew not found. Please install TA-Lib manually:"
        echo "   brew install ta-lib"
        echo "   OR download from: https://ta-lib.org/hdr_dw.html"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    sudo apt-get update
    sudo apt-get install -y ta-lib-dev
else
    echo "⚠️  Please install TA-Lib manually for your OS"
    echo "   Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
    echo "   Linux: sudo apt-get install ta-lib-dev"
    echo "   macOS: brew install ta-lib"
fi

# Install Python requirements
echo "🐍 Installing Python packages..."
pip install -r elite_requirements.txt

# Install additional packages that might fail
echo "🔧 Installing additional packages..."
pip install backtrader
pip install TA-Lib || pip install talib-binary

# Install XGBoost and LightGBM with specific versions
pip install xgboost==1.7.5
pip install lightgbm==3.3.5

# Install enhanced packages
pip install yfinance --upgrade
pip install pandas --upgrade
pip install numpy --upgrade
pip install scikit-learn --upgrade

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
required_packages = [
    'numpy', 'pandas', 'sklearn', 'scipy',
    'xgboost', 'lightgbm', 'yfinance', 'backtrader'
]

failed = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        print(f'❌ {package}')
        failed.append(package)

if failed:
    print(f'\\n⚠️  Failed to import: {failed}')
    print('Please install manually with: pip install ' + ' '.join(failed))
    sys.exit(1)
else:
    print('\\n🎉 All packages installed successfully!')
"

# Create directories
echo "📁 Creating result directories..."
mkdir -p elite_results
mkdir -p elite_results/models
mkdir -p elite_results/plots
mkdir -p elite_results/reports

# Set permissions
chmod +x elite_backtest_runner.py

echo "✅ Elite ML Trading System setup complete!"
echo ""
echo "🚀 To run the elite backtest:"
echo "   source elite_trading_env/bin/activate"
echo "   python elite_backtest_runner.py --initial-capital 1000000"
echo ""
echo "📊 Features enabled:"
echo "   • XGBoost + LightGBM + Neural Network Ensemble"
echo "   • 100+ Advanced Technical Features"  
echo "   • Multi-horizon Predictions"
echo "   • Risk-adjusted Kelly Position Sizing"
echo "   • Market Regime Detection"
echo "   • Pattern Recognition"
echo "   • Advanced Risk Management"
echo ""
echo "📁 Results will be saved to: elite_results/"
