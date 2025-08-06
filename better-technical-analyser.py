import pandas as pd
import numpy as np
from talib import RSI, MACD, BBANDS, ATR, STOCH, OBV, EMA , SMA

def calculate_technical_indicators(df, timeperiod=14):

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # RSI
    df['RSI_14'] = RSI(df['Close'], timeperiod=14)

    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = BBANDS(df['Close'], timeperiod=20)

    # EMAs
    df['EMA_12'] = EMA(df['Close'], timeperiod=12)
    df['EMA_26'] = EMA(df['Close'], timeperiod=26)

    # SMAs
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # OBV
    df['OBV'] = OBV(df['Close'], df['Volume'])

    # ATR
    df['ATR'] = ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Stochastic Oscillator
    df['Stoch_k'], df['Stoch_d'] = STOCH(df['High'], df['Low'], df['Close'])


    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()


    # Calculate Tenkan-sen (Conversion Line) - (9-period high + 9-period low) / 2
    df['tenkan_sen'] = SMA((df['High'] + df['Low']) / 2, timeperiod=9)

    # Calculate Kijun-sen (Base Line) - (26-period high + 26-period low) / 2
    df['kijun_sen'] = SMA((df['High'] + df['Low']) / 2, timeperiod=26)

    # Calculate Senkou Span A - (Tenkan-sen + Kijun-sen) / 2, shifted forward 26 periods
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Calculate Senkou Span B - (52-period high + 52-period low) / 2, shifted forward 26 periods
    df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)


# Calculate Pivot Points
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
    df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
    df['R3'] = df['High'] + 2 * (df['Pivot'] - df['Low'])
    df['S3'] = df['Low'] - 2 * (df['High'] - df['Pivot'])

    # Now, df contains support and resistance levels in columns 'R1', 'S1', 'R2', 'S2', 'R3', and 'S3'.


    # Normalization
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Close'])
    df_normalized = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())

    df_final = df[['Close']].join(df_normalized)
    return df_final

# Usage
df = pd.read_csv('OldFiles/data/AAPL_data.csv')
enhanced_df = calculate_technical_indicators(df)
enhanced_df.to_csv('analysis/stock_data_enhanced.csv')

# df_future = pd.read_csv('data/tesla_future_data.csv')
# enhanced_future_df = calculate_technical_indicators(df_future)
# enhanced_future_df.to_csv('better/tesla_future_data_enhanced.csv')

print("Enhanced datasets with technical indicators created and saved.")
