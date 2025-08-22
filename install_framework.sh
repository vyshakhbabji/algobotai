#!/bin/bash
# 🚀 ROBUST TRADING FRAMEWORK INSTALLATION SCRIPT

echo "🚀 Installing Robust Trading Framework Dependencies..."
echo "=================================================="

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv trading_env
source trading_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📊 Installing core packages..."
pip install pandas numpy scipy scikit-learn

# Install data sources
echo "📈 Installing market data packages..."
pip install yfinance alpaca-trade-api

# Install visualization
echo "🎨 Installing visualization packages..."
pip install plotly matplotlib seaborn

# Install backtesting frameworks
echo "🔬 Installing backtesting frameworks..."

# Try to install qlib
echo "🤖 Installing Qlib (Microsoft)..."
pip install pyqlib || echo "⚠️ Qlib installation failed - will use manual implementation"

# Try to install vectorbt  
echo "⚡ Installing VectorBT..."
pip install vectorbt || echo "⚠️ VectorBT installation failed - will use manual implementation"

# Install alternative backtesting
echo "🧪 Installing Backtesting.py..."
pip install backtesting || echo "⚠️ Backtesting.py installation failed"

# Install additional ML libraries
echo "🤖 Installing ML libraries..."
pip install lightgbm xgboost joblib

# Install utilities
echo "🛠️ Installing utilities..."
pip install tqdm requests jupyter ipykernel

echo ""
echo "✅ Installation completed!"
echo ""
echo "🚀 To activate the environment:"
echo "   source trading_env/bin/activate"
echo ""
echo "📊 To run the framework:"
echo "   python robust_trading_framework.py"
echo ""
echo "📋 Check installation status:"
python3 -c "
import sys
packages = ['pandas', 'numpy', 'scikit-learn', 'yfinance', 'plotly', 'lightgbm']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'\\n⚠️ Missing packages: {missing}')
    print('Please install manually: pip install ' + ' '.join(missing))
else:
    print('\\n🎉 All core packages installed successfully!')
"
