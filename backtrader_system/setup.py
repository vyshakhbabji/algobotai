#!/usr/bin/env python3
"""
Installation and setup script for Backtrader ML Trading System
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install all required packages"""
    print("\n🔧 Installing required packages...")
    
    packages = [
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "yfinance>=0.2.0",
        "PyYAML>=6.0",
        "matplotlib>=3.5.0",
        "backtrader",
        "ta-lib-binary"  # Alternative to TA-Lib that's easier to install
    ]
    
    failed_packages = []
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed to install: {', '.join(failed_packages)}")
        print("\nTrying alternative installation methods...")
        
        # Try alternative TA-Lib installation
        if any("ta-lib" in pkg.lower() for pkg in failed_packages):
            print("\n📦 Trying alternative TA-Lib installation...")
            alternatives = [
                "pip install TA-Lib",
                "conda install -c conda-forge ta-lib",
                "pip install --only-binary=all TA-Lib"
            ]
            
            for alt in alternatives:
                if run_command(alt, f"Alternative TA-Lib: {alt}"):
                    break
        
        return len(failed_packages) == 0
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    base_dir = Path(__file__).parent
    directories = [
        base_dir / "results",
        base_dir / "logs",
        base_dir / "data"
    ]
    
    for directory in directories:
        try:
            directory.mkdir(exist_ok=True)
            print(f"✅ Created {directory}")
        except Exception as e:
            print(f"❌ Failed to create {directory}: {e}")
            return False
    
    return True

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    test_script = Path(__file__).parent / "test_system.py"
    
    if not test_script.exists():
        print("❌ Test script not found")
        return False
    
    return run_command(f"python {test_script}", "Running system tests")

def create_sample_run_script():
    """Create a sample run script"""
    print("\n📝 Creating sample run script...")
    
    script_content = """#!/usr/bin/env python3
\"\"\"
Sample run script for Backtrader ML Trading System
\"\"\"

import subprocess
import sys
from datetime import datetime, timedelta

def run_quick_test():
    \"\"\"Run a quick backtest on a small date range\"\"\"
    print("🚀 Running quick test backtest...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    command = [
        "python", "backtest_runner.py",
        "--symbols", "AAPL", "TSLA",
        "--start-date", start_date.strftime("%Y-%m-%d"),
        "--end-date", end_date.strftime("%Y-%m-%d"),
        "--initial-capital", "10000"
    ]
    
    try:
        subprocess.run(command, check=True)
        print("✅ Quick test completed successfully!")
        print("📊 Check the results/ directory for output files")
    except subprocess.CalledProcessError as e:
        print(f"❌ Quick test failed: {e}")

def run_full_backtest():
    \"\"\"Run a full backtest on all symbols\"\"\"
    print("🚀 Running full backtest...")
    
    command = ["python", "backtest_runner.py"]
    
    try:
        subprocess.run(command, check=True)
        print("✅ Full backtest completed successfully!")
        print("📊 Check the results/ directory for output files")
    except subprocess.CalledProcessError as e:
        print(f"❌ Full backtest failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_backtest()
    else:
        run_quick_test()
"""
    
    script_path = Path(__file__).parent / "run_sample.py"
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        print(f"✅ Created sample run script: {script_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create run script: {e}")
        return False

def main():
    """Main installation process"""
    print("🎯 Backtrader ML Trading System Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install packages
    if not install_requirements():
        print("\n⚠️  Some packages failed to install. You may need to install them manually.")
        print("See README.md for troubleshooting steps.")
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Create sample script
    create_sample_run_script()
    
    # Test installation
    if test_installation():
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Run quick test: python run_sample.py")
        print("2. Run full test: python run_sample.py --full")
        print("3. Customize config/backtest_config.yaml as needed")
        print("4. Check README.md for detailed usage instructions")
    else:
        print("\n⚠️  Installation completed but tests failed.")
        print("Please check error messages above and see README.md for troubleshooting.")

if __name__ == "__main__":
    main()
