#!/usr/bin/env python3
"""
Clean Repository for Deployment
Keep only essential files for the AI Paper Trading Bot
"""

import os
import shutil

# Essential files to keep
KEEP_FILES = {
    # Core application files
    'live_paper_trading.py',           # Main trading dashboard
    'improved_ai_portfolio_manager.py', # AI engine
    'app.py',                          # Deployment entry point
    'enhanced_paper_trading_dashboard.py', # Enhanced dashboard
    'test_paper_trading.py',           # Test dashboard
    
    # Configuration files
    'requirements.txt',                # Dependencies
    'Procfile',                       # Deployment config
    'README.md',                      # Documentation
    'DEPLOYMENT_GUIDE.md',            # Deployment guide
    'DEPLOYMENT_STATUS.txt',          # Status file
    '.gitignore',                     # Git ignore rules
}

KEEP_DIRS = {
    '.git',                           # Git repository
    '.streamlit',                     # Streamlit config
}

def clean_repository():
    """Remove unnecessary files and directories"""
    print("🧹 CLEANING REPOSITORY FOR DEPLOYMENT")
    print("=" * 50)
    
    current_dir = os.getcwd()
    print(f"Working in: {current_dir}")
    
    # Get all items in directory
    all_items = os.listdir('.')
    
    removed_files = []
    removed_dirs = []
    kept_files = []
    kept_dirs = []
    
    for item in all_items:
        if os.path.isfile(item):
            if item in KEEP_FILES:
                kept_files.append(item)
                print(f"✅ KEEP: {item}")
            else:
                try:
                    os.remove(item)
                    removed_files.append(item)
                    print(f"🗑️  REMOVE: {item}")
                except Exception as e:
                    print(f"❌ ERROR removing {item}: {e}")
        
        elif os.path.isdir(item):
            if item in KEEP_DIRS:
                kept_dirs.append(item)
                print(f"✅ KEEP DIR: {item}/")
            else:
                try:
                    shutil.rmtree(item)
                    removed_dirs.append(item)
                    print(f"🗑️  REMOVE DIR: {item}/")
                except Exception as e:
                    print(f"❌ ERROR removing {item}/: {e}")
    
    print("\n" + "=" * 50)
    print("📊 CLEANUP SUMMARY:")
    print(f"✅ Files kept: {len(kept_files)}")
    print(f"✅ Directories kept: {len(kept_dirs)}")
    print(f"🗑️  Files removed: {len(removed_files)}")
    print(f"🗑️  Directories removed: {len(removed_dirs)}")
    
    print("\n📂 FINAL REPOSITORY STRUCTURE:")
    final_items = os.listdir('.')
    for item in sorted(final_items):
        if os.path.isdir(item):
            print(f"📁 {item}/")
        else:
            print(f"📄 {item}")
    
    print("\n🎉 Repository cleaned and ready for deployment!")

if __name__ == "__main__":
    clean_repository()
