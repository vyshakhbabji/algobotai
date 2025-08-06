import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

# Function to create sequences
def create_sequences(data, target, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = target[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load the dataset
df = pd.read_csv('better/clean-data/stock_data_enhanced.csv', parse_dates=['Date'], index_col='Date')

# Assuming 'Close' is the target
target_column = 'Close'


# Date,Close,High,Low,n,Open,Volume,vw,RSI_14,MACD,MACD_signal,MACD_hist,BB_upper,BB_middle,BB_lower,EMA_12,EMA_26,SMA_10,SMA_20,SMA_50,SMA_200,OBV,ATR,Stoch_k,Stoch_d,VWAP,Pivot,R1,S1,R2,S2,R3,S3

# feature_columns = ['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d', 'VWAP', 'Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']
feature_columns = ['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d', 'VWAP', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']


# feature_columns = ['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal','MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d']
# feature_columns = ['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal','MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d']


sequence_length = 30
# Impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[feature_columns] = imputer.fit_transform(df[feature_columns])
# Save the imputer for later use
joblib.dump(imputer, 'better/imputer.pkl')

# Normalize features and target
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])
target_scaler = MinMaxScaler()
df[target_column] = target_scaler.fit_transform(df[[target_column]])

# Time-Based Split
# train_size = int(len(df) * 0.8)
# train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size - sequence_length:]  # Include extra rows to construct sequences correctly


# Create sequences
X_train, y_train = create_sequences(train_df[feature_columns].values, train_df[target_column].values, sequence_length)
X_test, y_test = create_sequences(test_df[feature_columns].values, test_df[target_column].values, sequence_length)

# Exclude the first 'sequence_length' sequences from X_test and y_test to avoid leakage from train set
X_test = X_test[sequence_length:]
y_test = y_test[sequence_length:]

# Reshape data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(feature_columns)))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(feature_columns)))

# Save preprocessed data and scalers
np.save('better/X_train.npy', X_train)
np.save('better/X_test.npy', X_test)
np.save('better/y_train.npy', y_train)
np.save('better/y_test.npy', y_test)
joblib.dump(scaler, 'better/feature_scaler.pkl')
joblib.dump(target_scaler, 'better/target_scaler.pkl')
