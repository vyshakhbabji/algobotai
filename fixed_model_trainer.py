#!/usr/bin/env python3
"""
Fixed Model Training Pipeline
Train multiple models and create ensemble
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ModelTrainer:
    def __init__(self):
        """
        Initialize the model trainer
        """
        self.models = {}
        self.model_scores = {}
        self.feature_columns = None
        self.input_shape = None
        
    def load_preprocessed_data(self):
        """
        Load preprocessed data
        """
        print("Loading preprocessed data...")
        
        # Load metadata
        with open('fixed_data/preprocessed/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_columns = metadata['feature_columns']
        self.input_shape = tuple(metadata['input_shape'])
        
        # Load numpy arrays
        data = {}
        for file_name in ['X_train_seq', 'X_test_seq', 'X_train_flat', 'X_test_flat', 
                         'y_train', 'y_test', 'y_train_class', 'y_test_class']:
            data[file_name] = np.load(f'fixed_data/preprocessed/{file_name}.npy')
        
        # Load scalers
        self.feature_scaler = joblib.load('fixed_data/preprocessed/feature_scaler.pkl')
        self.target_scaler = joblib.load('fixed_data/preprocessed/target_scaler.pkl')
        
        print(f"Loaded data with input shape: {self.input_shape}")
        print(f"Train samples: {len(data['X_train_seq'])}, Test samples: {len(data['X_test_seq'])}")
        
        return data
    
    def train_lstm_model(self, X_train, X_test, y_train, y_test):
        """
        Train LSTM model
        """
        print("Training LSTM model...")
        
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=self.input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001)),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0
        )
        
        # Save model
        model.save('fixed_data/models/lstm_model.h5')
        
        return model, history
    
    def train_cnn_model(self, X_train, X_test, y_train, y_test):
        """
        Train CNN model
        """
        print("Training CNN model...")
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001)),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0
        )
        
        # Save model
        model.save('fixed_data/models/cnn_model.h5')
        
        return model, history
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """
        Train Random Forest model
        """
        print("Training Random Forest model...")
        
        # Parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Use RandomizedSearchCV for efficiency
        grid_search = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=3, 
            scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Save model
        joblib.dump(best_model, 'fixed_data/models/random_forest_model.pkl')
        
        print(f"Best RF parameters: {grid_search.best_params_}")
        
        return best_model
    
    def train_gradient_boosting_model(self, X_train, X_test, y_train, y_test):
        """
        Train Gradient Boosting model
        """
        print("Training Gradient Boosting model...")
        
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        gb_model = GradientBoostingRegressor(random_state=42)
        
        # Use RandomizedSearchCV
        grid_search = RandomizedSearchCV(
            gb_model, param_grid, n_iter=10, cv=3,
            scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Save model
        joblib.dump(best_model, 'fixed_data/models/gradient_boosting_model.pkl')
        
        print(f"Best GB parameters: {grid_search.best_params_}")
        
        return best_model
    
    def train_linear_models(self, X_train, X_test, y_train, y_test):
        """
        Train linear models (Linear Regression and Ridge)
        """
        print("Training linear models...")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        joblib.dump(lr_model, 'fixed_data/models/linear_regression_model.pkl')
        
        # Ridge Regression with hyperparameter tuning
        ridge_model = Ridge()
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        ridge_grid = GridSearchCV(ridge_model, ridge_params, cv=3, scoring='neg_mean_absolute_error')
        ridge_grid.fit(X_train, y_train)
        
        best_ridge = ridge_grid.best_estimator_
        joblib.dump(best_ridge, 'fixed_data/models/ridge_model.pkl')
        
        print(f"Best Ridge alpha: {ridge_grid.best_params_}")
        
        return lr_model, best_ridge
    
    def evaluate_model(self, model, X_test, y_test, model_name, model_type='sklearn'):
        """
        Evaluate a model and return metrics
        """
        if model_type == 'tensorflow':
            y_pred = model.predict(X_test, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store scores
        self.model_scores[model_name] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        return y_pred
    
    def create_ensemble_model(self, X_train, X_test, y_train, y_test):
        """
        Create ensemble model using trained base models
        """
        print("Creating ensemble model...")
        
        # Get predictions from all models
        ensemble_train_preds = []
        ensemble_test_preds = []
        
        for model_name, model in self.models.items():
            if 'lstm' in model_name or 'cnn' in model_name:
                train_pred = model.predict(X_train, verbose=0).flatten()
                test_pred = model.predict(X_test, verbose=0).flatten()
            else:
                # For sklearn models, we need flat data
                X_train_flat_ens = X_train.reshape(X_train.shape[0], -1)
                X_test_flat_ens = X_test.reshape(X_test.shape[0], -1)
                train_pred = model.predict(X_train_flat_ens)
                test_pred = model.predict(X_test_flat_ens)
            
            ensemble_train_preds.append(train_pred)
            ensemble_test_preds.append(test_pred)
        
        # Stack predictions
        ensemble_train_features = np.column_stack(ensemble_train_preds)
        ensemble_test_features = np.column_stack(ensemble_test_preds)
        
        # Train meta-model (Ridge regression)
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(ensemble_train_features, y_train)
        
        # Save meta-model
        joblib.dump(meta_model, 'fixed_data/models/ensemble_meta_model.pkl')
        
        # Evaluate ensemble
        ensemble_pred = meta_model.predict(ensemble_test_features)
        self.evaluate_model(meta_model, ensemble_test_features, y_test, 'Ensemble', 'sklearn')
        
        return meta_model, ensemble_pred
    
    def plot_training_history(self, histories):
        """
        Plot training history for neural networks
        """
        print("Plotting training history...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (model_name, history) in enumerate(histories.items()):
            row = i // 2
            col = i % 2
            
            # Plot loss
            axes[row, col].plot(history.history['loss'], label='Training Loss')
            axes[row, col].plot(history.history['val_loss'], label='Validation Loss')
            axes[row, col].set_title(f'{model_name} Training History')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig('fixed_data/models/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions(self, y_test, predictions, model_names):
        """
        Plot predictions vs actual values
        """
        print("Plotting predictions...")
        
        # Inverse transform for plotting
        y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, (model_name, pred) in enumerate(zip(model_names, predictions)):
            if i < len(axes):
                pred_orig = self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                
                axes[i].scatter(y_test_orig, pred_orig, alpha=0.6)
                axes[i].plot([y_test_orig.min(), y_test_orig.max()], 
                           [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
                axes[i].set_xlabel('Actual')
                axes[i].set_ylabel('Predicted')
                axes[i].set_title(f'{model_name} Predictions')
                axes[i].grid(True)
        
        # Remove empty subplots
        for i in range(len(model_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('fixed_data/models/predictions_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """
        Save model results and scores
        """
        print("Saving results...")
        
        # Save scores
        scores_df = pd.DataFrame(self.model_scores).T
        scores_df.to_csv('fixed_data/models/model_scores.csv')
        
        # Find best model
        best_model_name = scores_df['MAE'].idxmin()
        best_mae = scores_df.loc[best_model_name, 'MAE']
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"Best MAE: {best_mae:.4f}")
        
        # Save summary
        summary = {
            'best_model': best_model_name,
            'best_mae': float(best_mae),
            'all_scores': self.model_scores
        }
        
        with open('fixed_data/models/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def train_all_models(self):
        """
        Train all models
        """
        # Create models directory
        os.makedirs('fixed_data/models', exist_ok=True)
        
        # Load data
        data = self.load_preprocessed_data()
        
        X_train_seq = data['X_train_seq']
        X_test_seq = data['X_test_seq']
        X_train_flat = data['X_train_flat']
        X_test_flat = data['X_test_flat']
        y_train = data['y_train']
        y_test = data['y_test']
        
        histories = {}
        predictions = []
        model_names = []
        
        # Train deep learning models
        print("\n" + "="*50)
        print("TRAINING DEEP LEARNING MODELS")
        print("="*50)
        
        lstm_model, lstm_history = self.train_lstm_model(X_train_seq, X_test_seq, y_train, y_test)
        self.models['LSTM'] = lstm_model
        histories['LSTM'] = lstm_history
        lstm_pred = self.evaluate_model(lstm_model, X_test_seq, y_test, 'LSTM', 'tensorflow')
        predictions.append(lstm_pred)
        model_names.append('LSTM')
        
        cnn_model, cnn_history = self.train_cnn_model(X_train_seq, X_test_seq, y_train, y_test)
        self.models['CNN'] = cnn_model
        histories['CNN'] = cnn_history
        cnn_pred = self.evaluate_model(cnn_model, X_test_seq, y_test, 'CNN', 'tensorflow')
        predictions.append(cnn_pred)
        model_names.append('CNN')
        
        # Train machine learning models
        print("\n" + "="*50)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*50)
        
        # Train machine learning models using sequence data reshaped to flat
        X_train_flat_adjusted = X_train_seq.reshape(X_train_seq.shape[0], -1)
        X_test_flat_adjusted = X_test_seq.reshape(X_test_seq.shape[0], -1)
        
        rf_model = self.train_random_forest(X_train_flat_adjusted, X_test_flat_adjusted, y_train, y_test)
        self.models['Random Forest'] = rf_model
        rf_pred = self.evaluate_model(rf_model, X_test_flat_adjusted, y_test, 'Random Forest')
        predictions.append(rf_pred)
        model_names.append('Random Forest')
        
        gb_model = self.train_gradient_boosting_model(X_train_flat_adjusted, X_test_flat_adjusted, y_train, y_test)
        self.models['Gradient Boosting'] = gb_model
        gb_pred = self.evaluate_model(gb_model, X_test_flat_adjusted, y_test, 'Gradient Boosting')
        predictions.append(gb_pred)
        model_names.append('Gradient Boosting')
        
        lr_model, ridge_model = self.train_linear_models(X_train_flat_adjusted, X_test_flat_adjusted, y_train, y_test)
        self.models['Linear Regression'] = lr_model
        self.models['Ridge'] = ridge_model
        lr_pred = self.evaluate_model(lr_model, X_test_flat_adjusted, y_test, 'Linear Regression')
        ridge_pred = self.evaluate_model(ridge_model, X_test_flat_adjusted, y_test, 'Ridge')
        predictions.extend([lr_pred, ridge_pred])
        model_names.extend(['Linear Regression', 'Ridge'])
        
        # Create ensemble
        print("\n" + "="*50)
        print("CREATING ENSEMBLE MODEL")
        print("="*50)
        
        try:
            ensemble_model, ensemble_pred = self.create_ensemble_model(
                X_train_seq, X_test_seq, y_train, y_test
            )
            self.models['Ensemble'] = ensemble_model
            predictions.append(ensemble_pred)
            model_names.append('Ensemble')
        except Exception as e:
            print(f"Warning: Could not create ensemble model: {str(e)}")
            print("Continuing without ensemble...")
        
        # Plot results
        self.plot_training_history(histories)
        self.plot_predictions(y_test, predictions, model_names)
        
        # Save results
        self.save_results()
        
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        return self.models, self.model_scores

def main():
    """
    Main training pipeline
    """
    try:
        trainer = ModelTrainer()
        models, scores = trainer.train_all_models()
        return trainer, models, scores
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    trainer, models, scores = main()
