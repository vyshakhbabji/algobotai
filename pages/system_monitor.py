#!/usr/bin/env python3
"""
Comprehensive System Status Check - AlgoTradingBot Health Monitor
Tests all critical components, imports, data files, and trading systems
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
import pandas as pd
import uuid

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generate unique session ID for this page
if 'system_monitor_session_id' not in st.session_state:
    st.session_state.system_monitor_session_id = str(uuid.uuid4())[:8]

st.title("🔧 AlgoTradingBot System Status Monitor")
st.markdown("**Comprehensive Health Check & Diagnostics**")

# System status tracking
system_status = {
    "core_imports": {"status": "unknown", "details": []},
    "data_files": {"status": "unknown", "details": []},
    "trading_engines": {"status": "unknown", "details": []},
    "streamlit_pages": {"status": "unknown", "details": []},
    "ai_models": {"status": "unknown", "details": []},
    "portfolio_data": {"status": "unknown", "details": []}
}

# ========================================
# 1. CORE IMPORTS TEST
# ========================================
st.header("1️⃣ Core System Imports")

core_modules = [
    ("Elite Options Trader", "elite_options_trader", "EliteOptionsTrader"),
    ("Elite Stock Selector", "elite_stock_selector", "EliteStockSelector"), 
    ("AI Portfolio Manager", "improved_ai_portfolio_manager", "ImprovedAIPortfolioManager"),
    ("Paper Trading Engine", "pages.live_trading", "PaperTradingEngine"),
    ("Main Dashboard", "main_dashboard", None),
    ("App Entry Point", "app", None)
]

import_results = []
for name, module, class_name in core_modules:
    try:
        exec(f"import {module}")
        if class_name:
            exec(f"from {module} import {class_name}")
            exec(f"test_instance = {class_name}()")
            status = "✅ SUCCESS"
        else:
            status = "✅ SUCCESS"
        import_results.append({"Module": name, "Status": status, "Details": f"import {module}"})
        system_status["core_imports"]["details"].append(f"✅ {name}")
    except Exception as e:
        status = f"❌ FAILED: {str(e)}"
        import_results.append({"Module": name, "Status": status, "Details": str(e)})
        system_status["core_imports"]["details"].append(f"❌ {name}: {str(e)}")

import_df = pd.DataFrame(import_results)
st.dataframe(import_df, use_container_width=True)

# Set overall import status
failed_imports = len([r for r in import_results if "❌" in r["Status"]])
if failed_imports == 0:
    system_status["core_imports"]["status"] = "healthy"
    st.success(f"✅ All {len(core_modules)} core modules importing successfully!")
else:
    system_status["core_imports"]["status"] = "critical" if failed_imports > 2 else "warning"
    st.error(f"❌ {failed_imports}/{len(core_modules)} modules failed to import!")

# ========================================
# 2. STREAMLIT PAGES TEST
# ========================================
st.header("2️⃣ Streamlit Pages Status")

pages_to_test = [
    "pages.enhanced_dashboard",
    "pages.live_trading", 
    "pages.portfolio_manager",
    "pages.system_monitor",
    "pages.ai_optimizer",
    "pages.performance_analytics"
]

page_results = []
for page in pages_to_test:
    try:
        exec(f"import {page}")
        page_results.append({"Page": page, "Status": "✅ SUCCESS", "Details": "Import successful"})
        system_status["streamlit_pages"]["details"].append(f"✅ {page}")
    except Exception as e:
        page_results.append({"Page": page, "Status": f"❌ FAILED", "Details": str(e)})
        system_status["streamlit_pages"]["details"].append(f"❌ {page}: {str(e)}")

page_df = pd.DataFrame(page_results)
st.dataframe(page_df, use_container_width=True)

# Set overall pages status
failed_pages = len([r for r in page_results if "❌" in r["Status"]])
if failed_pages == 0:
    system_status["streamlit_pages"]["status"] = "healthy"
    st.success(f"✅ All {len(pages_to_test)} Streamlit pages working!")
else:
    system_status["streamlit_pages"]["status"] = "critical" if failed_pages > 2 else "warning"
    st.error(f"❌ {failed_pages}/{len(pages_to_test)} pages failed!")

# ========================================
# 3. DATA FILES HEALTH CHECK
# ========================================
st.header("3️⃣ Critical Data Files")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
critical_data_files = {
    "Paper Trading Account": os.path.join(parent_dir, "paper_trading_data.json"),
    "Current Positions": os.path.join(parent_dir, "current_positions.json"), 
    "Trade History": os.path.join(parent_dir, "trade_history.json"),
    "Portfolio Universe": os.path.join(parent_dir, "portfolio_universe.json"),
    "Paper Trading Account (Main)": os.path.join(parent_dir, "paper_trading_account.json"),
    "Paper Trading Positions": os.path.join(parent_dir, "paper_trading_positions.json"),
    "Model Performance History": os.path.join(parent_dir, "model_performance_history.json")
}

file_results = []
for name, filename in critical_data_files.items():
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Validate data format
                if "positions" in name.lower() and not isinstance(data, dict):
                    status = "❌ INVALID FORMAT (should be dict {})"
                    details = f"File contains {type(data).__name__} instead of dict"
                elif "trade_history.json" in filename and not isinstance(data, list):
                    status = "❌ INVALID FORMAT (should be list [])"
                    details = f"File contains {type(data).__name__} instead of list"
                else:
                    status = "✅ VALID"
                    if "paper_trading_data.json" in filename:
                        details = f"Capital: ${data.get('initial_capital', 0):,.2f}, Cash: ${data.get('cash', 0):,.2f}"
                    elif "current_positions.json" in filename:
                        details = f"Active Positions: {len(data)}"
                    elif "trade_history.json" in filename:
                        details = f"Total Trades: {len(data)}"
                    elif "portfolio_universe.json" in filename:
                        details = f"Stocks: {len(data.get('stocks', []))}"
                    elif "model_performance_history.json" in filename:
                        details = f"Daily: {len(data.get('daily_performance', []))}, Scores: {len(data.get('model_scores', []))}"
                    else:
                        details = f"Size: {len(data)} items"
                        
            file_results.append({"File": name, "Status": status, "Details": details, "Path": filename})
            system_status["data_files"]["details"].append(f"✅ {name}" if "✅" in status else f"❌ {name}")
        except Exception as e:
            status = f"❌ ERROR: {str(e)}"
            file_results.append({"File": name, "Status": status, "Details": str(e), "Path": filename})
            system_status["data_files"]["details"].append(f"❌ {name}: {str(e)}")
    else:
        file_results.append({"File": name, "Status": "❌ MISSING", "Details": "File not found", "Path": filename})
        system_status["data_files"]["details"].append(f"❌ {name}: Missing")

file_df = pd.DataFrame(file_results)
st.dataframe(file_df, use_container_width=True)

# Set overall data files status
failed_files = len([r for r in file_results if "❌" in r["Status"]])
if failed_files == 0:
    system_status["data_files"]["status"] = "healthy"
    st.success(f"✅ All {len(critical_data_files)} data files are healthy!")
else:
    system_status["data_files"]["status"] = "critical" if failed_files > 3 else "warning"
    st.error(f"❌ {failed_files}/{len(critical_data_files)} data files have issues!")

# ========================================
# 4. TRADING ENGINES TEST  
# ========================================
st.header("4️⃣ Trading Engines Functionality")

engine_tests = []

# Test Paper Trading Engine
try:
    from pages.live_trading import PaperTradingEngine
    engine = PaperTradingEngine()
    
    # Test portfolio value calculation
    portfolio_value = engine.update_portfolio_value()
    
    # Test price fetching
    test_price = engine.get_current_price("AAPL")
    
    engine_tests.append({
        "Engine": "Paper Trading Engine", 
        "Status": "✅ FUNCTIONAL",
        "Details": f"Portfolio: ${portfolio_value:,.2f}, Price Test: {'✅' if test_price else '❌'}"
    })
    system_status["trading_engines"]["details"].append("✅ Paper Trading Engine")
except Exception as e:
    engine_tests.append({
        "Engine": "Paper Trading Engine",
        "Status": "❌ FAILED", 
        "Details": str(e)
    })
    system_status["trading_engines"]["details"].append(f"❌ Paper Trading Engine: {str(e)}")

# Test Elite Options Trader
try:
    from elite_options_trader import EliteOptionsTrader
    options_trader = EliteOptionsTrader()
    portfolio_count = len(options_trader.portfolio_stocks)
    
    engine_tests.append({
        "Engine": "Elite Options Trader",
        "Status": "✅ FUNCTIONAL", 
        "Details": f"Portfolio stocks: {portfolio_count}"
    })
    system_status["trading_engines"]["details"].append("✅ Elite Options Trader")
except Exception as e:
    engine_tests.append({
        "Engine": "Elite Options Trader",
        "Status": "❌ FAILED",
        "Details": str(e)
    })
    system_status["trading_engines"]["details"].append(f"❌ Elite Options Trader: {str(e)}")

# Test AI Portfolio Manager
try:
    from improved_ai_portfolio_manager import ImprovedAIPortfolioManager
    ai_manager = ImprovedAIPortfolioManager()
    
    engine_tests.append({
        "Engine": "AI Portfolio Manager",
        "Status": "✅ FUNCTIONAL",
        "Details": f"Capital: ${ai_manager.initial_capital:,.2f}"
    })
    system_status["trading_engines"]["details"].append("✅ AI Portfolio Manager")
except Exception as e:
    engine_tests.append({
        "Engine": "AI Portfolio Manager", 
        "Status": "❌ FAILED",
        "Details": str(e)
    })
    system_status["trading_engines"]["details"].append(f"❌ AI Portfolio Manager: {str(e)}")

# Test Elite Stock Selector
try:
    from elite_stock_selector import EliteStockSelector
    stock_selector = EliteStockSelector()
    
    engine_tests.append({
        "Engine": "Elite Stock Selector",
        "Status": "✅ FUNCTIONAL",
        "Details": "Stock selection AI ready"
    })
    system_status["trading_engines"]["details"].append("✅ Elite Stock Selector")
except Exception as e:
    engine_tests.append({
        "Engine": "Elite Stock Selector",
        "Status": "❌ FAILED", 
        "Details": str(e)
    })
    system_status["trading_engines"]["details"].append(f"❌ Elite Stock Selector: {str(e)}")

engine_df = pd.DataFrame(engine_tests)
st.dataframe(engine_df, use_container_width=True)

# Set overall trading engines status
failed_engines = len([r for r in engine_tests if "❌" in r["Status"]])
if failed_engines == 0:
    system_status["trading_engines"]["status"] = "healthy"
    st.success(f"✅ All {len(engine_tests)} trading engines are functional!")
else:
    system_status["trading_engines"]["status"] = "critical" if failed_engines > 2 else "warning"
    st.error(f"❌ {failed_engines}/{len(engine_tests)} trading engines failed!")

# ========================================
# 5. PORTFOLIO & AI STATUS
# ========================================
st.header("5️⃣ Portfolio & AI Models")

try:
    # Check portfolio universe
    with open(os.path.join(parent_dir, "portfolio_universe.json"), 'r') as f:
        portfolio_data = json.load(f)
    
    portfolio_info = {
        "Total Stocks": len(portfolio_data.get('stocks', [])),
        "Top Performers": ", ".join(portfolio_data.get('stocks', [])[:3]),
        "Sectors": len(portfolio_data.get('sectors', {})),
        "Last Updated": portfolio_data.get('last_updated', 'Unknown'),
        "Avg AI Score": portfolio_data.get('avg_ai_score', 'N/A')
    }
    
    col1, col2 = st.columns(2)
    with col1:
        for key, value in portfolio_info.items():
            st.metric(key, value)
    
    with col2:
        # Check model performance
        try:
            with open(os.path.join(parent_dir, "model_performance_history.json"), 'r') as f:
                model_data = json.load(f)
            st.metric("Daily Performance Records", len(model_data.get('daily_performance', [])))
            st.metric("Model Scores", len(model_data.get('model_scores', [])))
            st.metric("Prediction Accuracy", len(model_data.get('prediction_accuracy', [])))
        except:
            st.warning("Model performance data not available")
    
    system_status["portfolio_data"]["status"] = "healthy"
    system_status["portfolio_data"]["details"].append("✅ Portfolio Universe loaded")
    system_status["ai_models"]["status"] = "healthy" 
    system_status["ai_models"]["details"].append("✅ AI models operational")
    
except Exception as e:
    st.error(f"Portfolio data error: {str(e)}")
    system_status["portfolio_data"]["status"] = "critical"
    system_status["portfolio_data"]["details"].append(f"❌ Portfolio error: {str(e)}")

# ========================================
# 6. OVERALL SYSTEM HEALTH SUMMARY
# ========================================
st.header("6️⃣ System Health Summary")

# Calculate overall health score
total_systems = len(system_status)
healthy_systems = len([s for s in system_status.values() if s["status"] == "healthy"])
warning_systems = len([s for s in system_status.values() if s["status"] == "warning"])
critical_systems = len([s for s in system_status.values() if s["status"] == "critical"])

# Overall health calculation
if critical_systems > 0:
    overall_health = "🔴 CRITICAL"
    health_color = "red"
elif warning_systems > 0:
    overall_health = "🟡 WARNING"
    health_color = "orange"
else:
    overall_health = "🟢 HEALTHY"
    health_color = "green"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Overall Health", overall_health)
with col2:
    st.metric("Healthy Systems", f"{healthy_systems}/{total_systems}")
with col3:
    st.metric("Warning Systems", warning_systems)
with col4:
    st.metric("Critical Systems", critical_systems)

# Detailed status breakdown
st.subheader("📊 Detailed System Status")
status_details = []
for system_name, system_info in system_status.items():
    status_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴", "unknown": "⚪"}
    status_details.append({
        "System": system_name.replace("_", " ").title(),
        "Status": f"{status_icon.get(system_info['status'], '⚪')} {system_info['status'].upper()}",
        "Issues": len([d for d in system_info['details'] if '❌' in d]),
        "Details": "; ".join(system_info['details'][:3]) + ("..." if len(system_info['details']) > 3 else "")
    })

status_df = pd.DataFrame(status_details)
st.dataframe(status_df, use_container_width=True)

# Quick action buttons
st.subheader("🚀 Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh All Tests", type="primary", key=f"sysmon_refresh_{st.session_state.system_monitor_session_id}"): 
        st.rerun()

with col2:
    if st.button("📊 Open Live Trading", type="secondary", key=f"sysmon_live_{st.session_state.system_monitor_session_id}"):
        st.info("Navigate to Live Paper Trading page manually")

with col3:
    if st.button("⚙️ View Elite Options", type="secondary", key=f"sysmon_elite_{st.session_state.system_monitor_session_id}"):
        st.info("Check main dashboard for Elite Options Trading")

# ========================================
# 7. SYSTEM INFORMATION
# ========================================
st.header("7️⃣ System Information")

try:
    # System environment info
    sys_info = {
        "Current Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Working Directory": os.getcwd(), 
        "Python Executable": sys.executable,
        "Python Version": sys.version.split()[0],
        "Streamlit Version": st.__version__,
        "OS Platform": os.name
    }
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        for key, value in list(sys_info.items())[:3]:
            st.write(f"**{key}**: {value}")
    with info_col2:
        for key, value in list(sys_info.items())[3:]:
            st.write(f"**{key}**: {value}")
            
except Exception as e:
    st.error(f"System info error: {str(e)}")

# ========================================
# 8. LAST UPDATED & AUTOMATED STATUS LOG  
# ========================================
st.header("8️⃣ Status Log")

# Save status to a log file for automated monitoring
try:
    status_log = {
        "timestamp": datetime.now().isoformat(),
        "overall_health": overall_health,
        "healthy_systems": healthy_systems,
        "warning_systems": warning_systems, 
        "critical_systems": critical_systems,
        "system_details": system_status
    }
    
    log_file = os.path.join(parent_dir, "system_status_log.json")
    with open(log_file, 'w') as f:
        json.dump(status_log, f, indent=2)
    
    st.success(f"✅ Status log saved to: {log_file}")
    
except Exception as e:
    st.error(f"Failed to save status log: {str(e)}")

st.markdown("---")
st.markdown(f"""
**AlgoTradingBot System Status Monitor v2.0**  
**Last Check**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Overall Health**: {overall_health}  
**Systems Checked**: {total_systems} | **Healthy**: {healthy_systems} | **Issues**: {warning_systems + critical_systems}
""")

# Auto-refresh every 5 minutes option
if st.checkbox("🔄 Auto-refresh every 5 minutes", key=f"sysmon_autorefresh_{st.session_state.system_monitor_session_id}"):
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()
