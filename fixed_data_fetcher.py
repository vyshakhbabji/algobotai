#!/usr/bin/env python3
"""
Fixed Data Fetcher for Algorithmic Trading Bot
Fetches NVDA data and prepares it for modeling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_nvda_data(months=6):
    """
    Fetch NVDA data for the specified number of months
    """
    print(f"Fetching {months} months of NVDA data...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    # Fetch data
    ticker = yf.Ticker("NVDA")
    data = ticker.history(start=start_date, end=end_date, interval="1d")
    
    if data.empty:
        raise ValueError("No data fetched. Please check your internet connection.")
    
    # Reset index to get Date as a column
    data.reset_index(inplace=True)
    
    # Clean column names
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    # Drop unnecessary columns
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"Fetched {len(data)} days of data from {data['Date'].min()} to {data['Date'].max()}")
    return data

def calculate_technical_indicators(df):
    """
    Calculate technical indicators using pandas and numpy
    """
    print("Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Ensure we have enough data
    if len(data) < 50:
        raise ValueError("Not enough data to calculate technical indicators")
    
    # Basic price indicators
    data['Returns'] = data['Close'].pct_change()
    data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100
    data['PC_PCT'] = (data['Close'] - data['Open']) / data['Open'] * 100
    
    # Simple Moving Averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12).mean()
    data['EMA_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI calculation
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    data['RSI_14'] = calculate_rsi(data['Close'])
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
    data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
    data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
    
    # Stochastic Oscillator
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    data['Stoch_k'], data['Stoch_d'] = calculate_stochastic(data['High'], data['Low'], data['Close'])
    
    # Average True Range (ATR)
    def calculate_atr(high, low, close, period=14):
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    data['ATR'] = calculate_atr(data['High'], data['Low'], data['Close'])
    
    # On Balance Volume (OBV)
    def calculate_obv(close, volume):
        obv = np.zeros(len(close))
        obv[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]
        return pd.Series(obv, index=close.index)
    
    data['OBV'] = calculate_obv(data['Close'], data['Volume'])
    
    # VWAP (simplified version)
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    
    # Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
    
    # Price action features
    data['Price_above_SMA20'] = (data['Close'] > data['SMA_20']).astype(int)
    data['Price_above_SMA50'] = (data['Close'] > data['SMA_50']).astype(int)
    data['RSI_oversold'] = (data['RSI_14'] < 30).astype(int)
    data['RSI_overbought'] = (data['RSI_14'] > 70).astype(int)
    
    # Momentum indicators
    data['Price_momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['Price_momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    
    # Support and Resistance (simplified)
    data['High_20'] = data['High'].rolling(window=20).max()
    data['Low_20'] = data['Low'].rolling(window=20).min()
    data['Distance_to_high'] = (data['High_20'] - data['Close']) / data['Close']
    data['Distance_to_low'] = (data['Close'] - data['Low_20']) / data['Close']
    
    print(f"Added technical indicators. Dataset now has {len(data.columns)} columns.")
    
    # Forward fill missing values for initial periods
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    return data

def prepare_features_and_target(data):
    """
    Prepare feature set and target variable
    """
    print("Preparing features and target...")
    
    # Define feature columns (excluding non-predictive ones)
    feature_columns = [
        'Open', 'High', 'Low', 'Volume',
        'Returns', 'HL_PCT', 'PC_PCT',
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_position',
        'Stoch_k', 'Stoch_d', 'ATR', 'OBV', 'VWAP',
        'Volume_SMA', 'Volume_ratio',
        'Price_above_SMA20', 'Price_above_SMA50', 
        'RSI_oversold', 'RSI_overbought',
        'Price_momentum_5', 'Price_momentum_10',
        'High_20', 'Low_20', 'Distance_to_high', 'Distance_to_low'
    ]
    
    # Create target: next day's close price
    data['Target'] = data['Close'].shift(-1)
    
    # Create classification target for direction
    data['Target_Direction'] = (data['Target'] > data['Close']).astype(int)
    
    # Create target for percentage change
    data['Target_Return'] = (data['Target'] / data['Close'] - 1) * 100
    
    # Remove rows with missing target (last row)
    data = data.dropna(subset=['Target'])
    
    # Ensure we have all feature columns
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        feature_columns = [col for col in feature_columns if col in data.columns]
    
    print(f"Using {len(feature_columns)} features for modeling")
    print(f"Final dataset shape: {data.shape}")
    
    return data, feature_columns

def save_data(data, feature_columns):
    """
    Save processed data
    """
    print("Saving processed data...")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('fixed_data', exist_ok=True)
    
    # Save full dataset
    data.to_csv('fixed_data/nvda_data_processed.csv', index=False)
    
    # Save feature list
    with open('fixed_data/feature_columns.txt', 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    
    # Save summary statistics
    summary = {
        'total_samples': len(data),
        'date_range': f"{data['Date'].min()} to {data['Date'].max()}",
        'num_features': len(feature_columns),
        'target_mean': data['Target'].mean(),
        'target_std': data['Target'].std(),
        'positive_days': (data['Target_Direction'] == 1).sum(),
        'negative_days': (data['Target_Direction'] == 0).sum()
    }
    
    with open('fixed_data/data_summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print("Data saved successfully!")
    print(f"Positive days: {summary['positive_days']}")
    print(f"Negative days: {summary['negative_days']}")
    print(f"Target mean: ${summary['target_mean']:.2f}")
    print(f"Target std: ${summary['target_std']:.2f}")

def main():
    """
    Main function to fetch and process NVDA data
    """
    try:
        # Fetch NVDA data
        raw_data = fetch_nvda_data(months=6)
        
        # Calculate technical indicators
        processed_data = calculate_technical_indicators(raw_data)
        
        # Prepare features and target
        final_data, feature_columns = prepare_features_and_target(processed_data)
        
        # Save data
        save_data(final_data, feature_columns)
        
        print("\n" + "="*50)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return final_data, feature_columns
        
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        raise

if __name__ == "__main__":
    data, features = main()
