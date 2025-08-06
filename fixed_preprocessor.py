#!/usr/bin/env python3
"""
Fixed Data Preprocessing Pipeline
Clean, scale, and prepare data for machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, sequence_length=10, test_size=0.3):
        """
        Initialize the data preprocessor
        
        Args:
            sequence_length: Number of time steps to use for sequence data
            test_size: Proportion of data to use for testing
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.feature_scaler = None
        self.target_scaler = None
        self.imputer = None
        self.feature_columns = None
        
    def load_data(self, data_path='fixed_data/nvda_data_processed.csv'):
        """
        Load the processed data
        """
        print("Loading processed data...")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Load feature columns
        feature_path = 'fixed_data/feature_columns.txt'
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
        else:
            # Fallback to auto-detection
            exclude_cols = ['Date', 'Close', 'Target', 'Target_Direction', 'Target_Return']
            self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"Loaded {len(self.data)} samples with {len(self.feature_columns)} features")
        return self.data
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("Handling missing values...")
        
        # Check for missing values
        missing_counts = self.data[self.feature_columns].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found {missing_counts.sum()} missing values")
            
            # Use SimpleImputer for missing values
            self.imputer = SimpleImputer(strategy='median', missing_values=np.nan)
            self.data[self.feature_columns] = self.imputer.fit_transform(self.data[self.feature_columns])
            print("Missing values imputed using median strategy")
        else:
            print("No missing values found")
    
    def remove_outliers(self, method='iqr', threshold=3):
        """
        Remove outliers from the dataset
        """
        print(f"Removing outliers using {method} method...")
        
        if method == 'iqr':
            # Use IQR method for outlier detection
            Q1 = self.data[self.feature_columns].quantile(0.25)
            Q3 = self.data[self.feature_columns].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outlier_mask = ((self.data[self.feature_columns] < lower_bound) | 
                           (self.data[self.feature_columns] > upper_bound)).any(axis=1)
            
        elif method == 'zscore':
            # Use Z-score method
            z_scores = np.abs((self.data[self.feature_columns] - self.data[self.feature_columns].mean()) / 
                             self.data[self.feature_columns].std())
            outlier_mask = (z_scores > threshold).any(axis=1)
        
        outliers_count = outlier_mask.sum()
        if outliers_count > 0:
            print(f"Removing {outliers_count} outlier rows ({outliers_count/len(self.data)*100:.2f}%)")
            self.data = self.data[~outlier_mask].reset_index(drop=True)
        else:
            print("No outliers detected")
    
    def scale_features(self, scaler_type='robust'):
        """
        Scale features and target variables
        """
        print(f"Scaling features using {scaler_type} scaler...")
        
        # Choose scaler
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            raise ValueError("Scaler type must be 'standard' or 'robust'")
        
        # Scale features
        self.data[self.feature_columns] = self.feature_scaler.fit_transform(
            self.data[self.feature_columns]
        )
        
        # Scale target
        self.data[['Target']] = self.target_scaler.fit_transform(
            self.data[['Target']]
        )
        
        print("Features and targets scaled successfully")
    
    def create_sequences(self, data_array, target_array):
        """
        Create sequences for time series modeling
        """
        X, y = [], []
        
        for i in range(len(data_array) - self.sequence_length):
            # Get sequence of features
            X.append(data_array[i:(i + self.sequence_length)])
            # Get corresponding target
            y.append(target_array[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self):
        """
        Split data into train and test sets using time-based splitting
        """
        print("Splitting data into train and test sets...")
        
        # Sort by date to ensure chronological order
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(self.data) * (1 - self.test_size))
        
        # Split the data
        train_data = self.data.iloc[:split_idx].copy()
        test_data = self.data.iloc[split_idx:].copy()
        
        print(f"Train set: {len(train_data)} samples ({train_data['Date'].min()} to {train_data['Date'].max()})")
        print(f"Test set: {len(test_data)} samples ({test_data['Date'].min()} to {test_data['Date'].max()})")
        
        return train_data, test_data
    
    def prepare_model_data(self):
        """
        Prepare final datasets for model training
        """
        print("Preparing model data...")
        
        # Split data
        train_data, test_data = self.split_data()
        
        # Extract features and targets
        X_train_flat = train_data[self.feature_columns].values
        X_test_flat = test_data[self.feature_columns].values
        y_train = train_data['Target'].values
        y_test = test_data['Target'].values
        
        # Create sequences for LSTM/CNN models
        X_train_seq, y_train_seq = self.create_sequences(X_train_flat, y_train)
        X_test_seq, y_test_seq = self.create_sequences(X_test_flat, y_test)
        
        # Also prepare classification targets
        y_train_class = train_data['Target_Direction'].values
        y_test_class = test_data['Target_Direction'].values
        
        # Check if we have enough data for sequences
        if len(X_test_seq) == 0:
            print("Warning: Not enough test data for sequences. Adjusting split...")
            # Use a smaller test size
            self.test_size = 0.3
            train_data, test_data = self.split_data()
            
            X_train_flat = train_data[self.feature_columns].values
            X_test_flat = test_data[self.feature_columns].values
            y_train = train_data['Target'].values
            y_test = test_data['Target'].values
            
            X_train_seq, y_train_seq = self.create_sequences(X_train_flat, y_train)
            X_test_seq, y_test_seq = self.create_sequences(X_test_flat, y_test)
            
            y_train_class = train_data['Target_Direction'].values
            y_test_class = test_data['Target_Direction'].values
        
        # Handle classification sequences
        if len(X_train_seq) > 0:
            y_train_class_seq, _ = self.create_sequences(X_train_flat, y_train_class)
            y_train_class_final = y_train_class_seq[:, -1] if len(y_train_class_seq.shape) > 1 else y_train_class_seq
        else:
            y_train_class_final = np.array([])
            
        if len(X_test_seq) > 0:
            y_test_class_seq, _ = self.create_sequences(X_test_flat, y_test_class)
            y_test_class_final = y_test_class_seq[:, -1] if len(y_test_class_seq.shape) > 1 else y_test_class_seq
        else:
            y_test_class_final = np.array([])
        
        print(f"Sequence data shapes:")
        print(f"X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
        print(f"X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")
        
        # Store shapes for later use
        if len(X_train_seq) > 0:
            self.input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        else:
            self.input_shape = (self.sequence_length, len(self.feature_columns))
        
        return {
            'X_train_flat': X_train_flat,
            'X_test_flat': X_test_flat,
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train': y_train_seq,
            'y_test': y_test_seq,
            'y_train_class': y_train_class_final,
            'y_test_class': y_test_class_final,
            'train_dates': train_data['Date'].iloc[self.sequence_length:].values if len(train_data) > self.sequence_length else train_data['Date'].values,
            'test_dates': test_data['Date'].iloc[self.sequence_length:].values if len(test_data) > self.sequence_length else test_data['Date'].values
        }
    
    def save_preprocessed_data(self, model_data):
        """
        Save preprocessed data and scalers
        """
        print("Saving preprocessed data...")
        
        # Create directory
        os.makedirs('fixed_data/preprocessed', exist_ok=True)
        
        # Save numpy arrays
        for key, value in model_data.items():
            if isinstance(value, np.ndarray):
                np.save(f'fixed_data/preprocessed/{key}.npy', value)
        
        # Save scalers and imputer
        if self.feature_scaler:
            joblib.dump(self.feature_scaler, 'fixed_data/preprocessed/feature_scaler.pkl')
        if self.target_scaler:
            joblib.dump(self.target_scaler, 'fixed_data/preprocessed/target_scaler.pkl')
        if self.imputer:
            joblib.dump(self.imputer, 'fixed_data/preprocessed/imputer.pkl')
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'test_size': self.test_size,
            'feature_columns': self.feature_columns,
            'input_shape': self.input_shape,
            'num_features': len(self.feature_columns),
            'train_samples': len(model_data['X_train_seq']),
            'test_samples': len(model_data['X_test_seq'])
        }
        
        pd.Series(metadata).to_json('fixed_data/preprocessed/metadata.json')
        
        print("Preprocessed data saved successfully!")
    
    def get_train_validation_split(self, model_data, validation_size=0.2):
        """
        Create a validation split from training data
        """
        print("Creating train-validation split...")
        
        # Calculate split point
        val_split_idx = int(len(model_data['X_train_seq']) * (1 - validation_size))
        
        # Split training data
        X_train = model_data['X_train_seq'][:val_split_idx]
        X_val = model_data['X_train_seq'][val_split_idx:]
        y_train = model_data['y_train'][:val_split_idx]
        y_val = model_data['y_train'][val_split_idx:]
        
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val

def main():
    """
    Main preprocessing pipeline
    """
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(sequence_length=10, test_size=0.3)
        
        # Load data
        preprocessor.load_data()
        
        # Handle missing values
        preprocessor.handle_missing_values()
        
        # Remove outliers
        preprocessor.remove_outliers(method='iqr')
        
        # Scale features
        preprocessor.scale_features(scaler_type='robust')
        
        # Prepare model data
        model_data = preprocessor.prepare_model_data()
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(model_data)
        
        print("\n" + "="*50)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return preprocessor, model_data
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    preprocessor, data = main()
