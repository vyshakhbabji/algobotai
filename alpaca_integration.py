#!/usr/bin/env python3
"""
ALPACA INTEGRATION SETUP
Easy configuration and testing for Alpaca API integration
"""

import os
import json
from typing import Dict, Optional

def create_alpaca_config_template():
    """Create template for Alpaca API configuration"""
    
    config_template = {
        "alpaca": {
            "api_key": "YOUR_ALPACA_API_KEY_HERE",
            "secret_key": "YOUR_ALPACA_SECRET_KEY_HERE",
            "paper_trading": True,
            "base_url": "https://paper-api.alpaca.markets"
        },
        "trading_limits": {
            "max_position_value": 15000,
            "max_daily_trades": 10,
            "stop_loss_percent": 0.08,
            "take_profit_percent": 0.15
        }
    }
    
    config_file = "alpaca_config.json"
    
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump(config_template, f, indent=2)
        
        print("📝 ALPACA CONFIGURATION TEMPLATE CREATED")
        print("=" * 45)
        print(f"✅ Created: {config_file}")
        print("🔧 Next steps:")
        print("   1. Get your Alpaca API keys from alpaca.markets")
        print("   2. Edit alpaca_config.json with your keys")
        print("   3. Set paper_trading: true for testing")
        print("   4. Run test_alpaca_connection() to verify")
        print()
        print("📋 How to get Alpaca API keys:")
        print("   1. Go to alpaca.markets")
        print("   2. Create account (free paper trading)")
        print("   3. Navigate to API Keys section")
        print("   4. Generate new API key pair")
        print("   5. Copy keys to alpaca_config.json")
        
        return config_template
    else:
        print(f"⚠️ {config_file} already exists")
        return None

def load_alpaca_config(config_file: str = "alpaca_config.json") -> Optional[Dict]:
    """Load Alpaca configuration from file"""
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        alpaca_config = config.get('alpaca', {})
        required_fields = ['api_key', 'secret_key']
        
        for field in required_fields:
            if not alpaca_config.get(field) or alpaca_config[field] == f"YOUR_ALPACA_{field.upper()}_HERE":
                print(f"❌ Missing or invalid {field} in {config_file}")
                return None
        
        print(f"✅ Alpaca configuration loaded from {config_file}")
        return config
        
    except FileNotFoundError:
        print(f"❌ Configuration file {config_file} not found")
        print("💡 Run create_alpaca_config_template() first")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {config_file}")
        return None

def test_alpaca_connection(config_file: str = "alpaca_config.json"):
    """Test Alpaca API connection"""
    
    print("🧪 TESTING ALPACA API CONNECTION")
    print("=" * 35)
    
    # Load configuration
    config = load_alpaca_config(config_file)
    if not config:
        return False
    
    alpaca_config = config['alpaca']
    
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        
        # Test trading client
        trading_client = TradingClient(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key'],
            paper=alpaca_config.get('paper_trading', True)
        )
        
        # Test connection by getting account info
        account = trading_client.get_account()
        
        print(f"✅ Trading Client Connected")
        print(f"📊 Account Status: {account.status}")
        print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
        print(f"📈 Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"⚡ Paper Trading: {alpaca_config.get('paper_trading', True)}")
        
        # Test data client
        data_client = StockHistoricalDataClient(
            api_key=alpaca_config['api_key'],
            secret_key=alpaca_config['secret_key']
        )
        
        print(f"✅ Data Client Connected")
        
        print(f"\n🎯 ALPACA INTEGRATION READY!")
        print(f"   Your auto trader can now execute real trades")
        print(f"   (currently in {'paper' if alpaca_config.get('paper_trading') else 'live'} mode)")
        
        return True
        
    except ImportError:
        print("❌ Alpaca SDK not installed")
        print("💡 Install with: pip install alpaca-py")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("🔧 Check your API keys and network connection")
        return False

def setup_alpaca_integration():
    """Complete Alpaca integration setup wizard"""
    
    print("🚀 ALPACA INTEGRATION SETUP WIZARD")
    print("=" * 40)
    
    # Step 1: Create configuration template
    print("\n📝 STEP 1: Creating configuration template...")
    template = create_alpaca_config_template()
    
    if not template:
        print("⚠️ Configuration file already exists")
        
        # Test existing configuration
        print("\n🧪 STEP 2: Testing existing configuration...")
        connection_success = test_alpaca_connection()
        
        if connection_success:
            print("\n✅ SETUP COMPLETE - Alpaca integration ready!")
            return True
        else:
            print("\n❌ Setup incomplete - check your configuration")
            return False
    
    print("\n📋 STEP 2: Manual configuration required")
    print("=" * 35)
    print("Please complete these steps:")
    print("1. Get Alpaca API keys from alpaca.markets")
    print("2. Edit alpaca_config.json with your keys")
    print("3. Run test_alpaca_connection() to verify")
    print("\nThen run the auto trader with Alpaca integration!")
    
    return False

def create_auto_trader_with_alpaca():
    """Create auto trader instance with Alpaca integration"""
    
    from complete_auto_trader import CompleteAutoTrader
    
    # Load Alpaca configuration
    config = load_alpaca_config()
    if not config:
        print("❌ Cannot create auto trader - Alpaca configuration missing")
        return None
    
    # Test connection first
    if not test_alpaca_connection():
        print("❌ Cannot create auto trader - Alpaca connection failed")
        return None
    
    # Create auto trader with Alpaca integration
    auto_trader = CompleteAutoTrader(
        account_size=100000,  # $100K demo
        paper_trading=config['alpaca'].get('paper_trading', True),
        alpaca_config=config['alpaca']
    )
    
    print("✅ AUTO TRADER WITH ALPACA INTEGRATION CREATED!")
    print("🚀 Ready for automated trading with real market execution")
    
    return auto_trader

if __name__ == "__main__":
    # Run setup wizard
    setup_success = setup_alpaca_integration()
    
    if setup_success:
        # Create auto trader instance
        auto_trader = create_auto_trader_with_alpaca()
        
        if auto_trader:
            print(f"\n🎯 READY TO TRADE!")
            print(f"   Use: auto_trader.run_trading_session(['NVDA', 'AAPL', 'TSLA'])")
    
    print(f"\n📚 ALPACA INTEGRATION GUIDE:")
    print(f"   1. Get free account at alpaca.markets")
    print(f"   2. Generate API keys in dashboard")
    print(f"   3. Update alpaca_config.json")
    print(f"   4. Test with test_alpaca_connection()")
    print(f"   5. Run auto trader with live execution!")
