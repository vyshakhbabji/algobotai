#!/usr/bin/env python3
"""
Alpaca Paper Trading Page
Streamlit page for Alpaca paper trading integration
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main Alpaca trading engine
from alpaca_paper_trading import main

# Run the main function
if __name__ == "__main__":
    main()
else:
    main()
