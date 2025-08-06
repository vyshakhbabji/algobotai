
# Updated imports to remove SVR
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from keras_tuner import RandomSearch

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer, Conv1D, MaxPooling1D, LSTM, Dense

from tensorflow.keras.optimizers.legacy import Adam

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(1, 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        print("Input shape to AttentionLayer:", x.shape)
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        print("Output shape after attention application:", output.shape)

        # Instead of summing over the time dimension, take the mean or another operation that maintains the batch dimension
        return tf.keras.backend.mean(output, axis=1, keepdims=True)

# Load the preprocessed data and scalers
X_train = np.load('better/X_train.npy')
X_test = np.load('better/X_test.npy')
y_train = np.load('better/y_train.npy')
y_test = np.load('better/y_test.npy')
feature_scaler = joblib.load('better/feature_scaler.pkl')
target_scaler = joblib.load('better/target_scaler.pkl')
input_shape = (X_train.shape[1], X_train.shape[2])

# with custom_object_scope({'AttentionLayer': AttentionLayer}):
#     hybrid_model_tuning= load_model('better/tunable_hybrid_model.h5')
#     hybrid_model_optimized = load_model('better/hybrid_model_optimized.h5')



# best_rf_model = train_optimized_rf_model(X_train, y_train, feature_names)


def train_cnn_model(X_train, y_train, X_test, y_test):
    def build_cnn_model(hp):
        model = Sequential()
        model.add(Conv1D(
            filters=hp.Int('filters', min_value=16, max_value=64, step=16),
            kernel_size=hp.Choice('kernel_size', values=[2, 3]),
            activation='relu',
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(
            units=hp.Int('dense_units', min_value=30, max_value=100, step=10),
            activation='relu'
        ))
        model.add(Dense(1))
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        return model


    # Hyperparameter tuning with the updated build function
    tuner = RandomSearch(
        build_cnn_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='model_tuning',
        project_name='cnn_tuning_v2'
    )

    # Start the search for the best hyperparameters
    tuner.search(
        X_train,
        y_train,
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )


    # Retrieve the best hyperparameters and build the model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_cnn_model = tuner.hypermodel.build(best_hps)
    # Before fitting the model, check if the 'batch_size' hyperparameter was tuned and retrieve it
    batch_size = best_hps.get('batch_size') if 'batch_size' in best_hps.values else 64
    # Fit the best model
    # callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    history_cnn = best_cnn_model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    best_cnn_model.save('better/cnn_model.h5')
    return best_cnn_model, history_cnn




def train_optimized_xgb_model(X_train, y_train):
    xgb_param_grid = {
        'n_estimators': [100, 150],  # Reduced to the most promising values
        'learning_rate': [0.05, 0.1],  # Focus on values that provided better results
        'max_depth': [3, 5],  # Limit depth to prevent overfitting
        'subsample': [0.8, 0.9],  # Slightly higher to keep most data
        'colsample_bytree': [0.8, 0.9],  # Consider adding this for feature subsampling
    }
    xgb_model = XGBRegressor()
    xgb_grid_search = GridSearchCV(
        estimator=xgb_model, param_grid=xgb_param_grid, cv=3, n_jobs=-1, verbose=1
    )
    xgb_grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())
    best_xgb_model = xgb_grid_search.best_estimator_
    print("Best parameters found for XGBoost: ", xgb_grid_search.best_params_)
    joblib.dump(best_xgb_model, 'better/best_xgb_model.pkl')
    return best_xgb_model



import matplotlib.pyplot as plt


def train_optimized_rf_model(X_train, y_train):
    # Reshape X_train from 3D to 2D for RandomForest
    nsamples, nx, ny = X_train.shape
    X_train_reshaped = X_train.reshape((nsamples, nx * ny))
    # feature_columns = ['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal','MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d']
    # feature_columns = ['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d', 'VWAP', 'Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']
    feature_columns = ['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d', 'VWAP', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']
    sequence_length = 30
    rf_param_grid = {
        'n_estimators': [100, 150],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_model = RandomForestRegressor()
    rf_grid_search = GridSearchCV(
        estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=1
    )
    rf_grid_search.fit(X_train_reshaped, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    print("Best parameters found for RandomForest: ", rf_grid_search.best_params_)

    # Feature importance
    importances = best_rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    updated_feature_names = [f"{feature}_{t}" for feature in feature_columns for t in range(sequence_length)]


    # Print and plot feature ranking
    print("Feature ranking (Top 20):")
    top_features = 300  # Number of top features to display
    for f in range(top_features):
        print(f"{f + 1}. feature {updated_feature_names[indices[f]]} ({importances[indices[f]]})")



    summed_feature_importances = {}

    # Loop through the features and group them by removing the last underscore
    for f in range(top_features):
        feature_name = updated_feature_names[indices[f]]
        feature_importance = importances[indices[f]]

        # Remove the last underscore and get the base feature name
        base_feature_name = feature_name.rsplit('_', 1)[0]

        # Check if the base feature name is already in the dictionary
        if base_feature_name in summed_feature_importances:
            # If it exists, add the current feature importance to it
            summed_feature_importances[base_feature_name] += feature_importance
        else:
            # If it doesn't exist, create a new entry for it
            summed_feature_importances[base_feature_name] = feature_importance

    # Sort the summed feature importances in descending order
    sorted_summed_features = sorted(summed_feature_importances.items(), key=lambda x: x[1], reverse=True)

    # Print and rank the top features
    print("Summed Feature ranking (Top 20):")
    for rank, (feature_name, importance) in enumerate(sorted_summed_features[:30]):
        print(f"{rank + 1}. feature {feature_name} ({importance})")


    # # Plotting only the top 20 feature importances
    # plt.figure(figsize=(12, 6))
    # plt.title("Feature importances (Top 20)")
    # plt.bar(range(top_features), importances[indices][:top_features], color="r", align="center")
    # plt.xticks(range(top_features), [updated_feature_names[i] for i in indices[:top_features]], rotation=45)
    # plt.xlim([-1, top_features])
    # plt.show()


    joblib.dump(best_rf_model, 'better/best_rf_model.pkl')
    return best_rf_model


# Call the function with feature names




def train_lstm_model(X_train, y_train, X_test, y_test):
    def build_lstm_model(hp):
        model = Sequential()
        for i in range(hp.Int('num_lstm_layers', 1, 3)):
            if i == 0:
                model.add(LSTM(units=hp.Int('units_l0', min_value=50, max_value=200, step=50),
                               return_sequences=hp.Int('num_lstm_layers', 1, 3) > 1,
                               input_shape=X_train.shape[1:]))
            else:
                model.add(LSTM(units=hp.Int(f'units_l{i}', min_value=50, max_value=200, step=50),
                               return_sequences=i < hp.Int('num_lstm_layers', 1, 3) - 1))

            dropout_rate = hp.Float(f'dropout_rate_l{i}', min_value=0.1, max_value=0.5, step=0.1)
            model.add(Dropout(rate=dropout_rate))

        model.add(Dense(1))
        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='mean_squared_error')

        return model

    tuner = RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=2,
        directory='model_tuning',
        project_name='lstm_tuning'
    )

    tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    model.save('better/lstm_model_tuned')
    model.save('better/lstm_model_tuned.h5')
    return model, history



def tune_gradient_boosting_model(X_train, y_train):
    # Define the parameter grid
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    # Initialize the GradientBoostingRegressor model
    gb_model = GradientBoostingRegressor()

    # Perform grid search
    gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=3, n_jobs=-1, verbose=2)
    gb_grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())

    # Retrieve the best parameters and model
    best_gb_model = gb_grid_search.best_estimator_
    print("Best parameters found for GradientBoostingRegressor: ", gb_grid_search.best_params_)

    # Save the best model
    joblib.dump(best_gb_model, 'better/best_gb_model.pkl')

    return best_gb_model

def tune_bagging_model(X_train, y_train):
    # Define the parameter grid
    bagging_param_grid = {
        'n_estimators': [10, 50, 100],
        'base_estimator__max_depth': [10, 20, 30]
    }

    # Initialize the BaggingRegressor model with a DecisionTreeRegressor as the base estimator
    dt = DecisionTreeRegressor()
    bagging_model = BaggingRegressor(base_estimator=dt)

    # Perform grid search
    bagging_grid_search = GridSearchCV(estimator=bagging_model, param_grid=bagging_param_grid, cv=3, n_jobs=-1, verbose=2)
    bagging_grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())

    # Retrieve the best parameters and model
    best_bagging_model = bagging_grid_search.best_estimator_
    print("Best parameters found for BaggingRegressor: ", bagging_grid_search.best_params_)

    # Save the best model
    joblib.dump(best_bagging_model, 'better/best_bagging_model.pkl')

    return best_bagging_model


def train_dt_bag_model(X_train, y_train):
    dt_bag = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100)
    dt_bag.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())
    joblib.dump(dt_bag, 'better/dt_bag_model.pkl')
    return dt_bag


def train_dt_boost_model(X_train, y_train):
    dt_boost = GradientBoostingRegressor()
    dt_boost.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())
    joblib.dump(dt_boost, 'better/dt_boost_model.pkl')
    return dt_boost

def train_lr_model(X_train, y_train):
    lr_model = LinearRegression()
    lr_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    joblib.dump(lr_model, 'better/lr_model.pkl')
    return lr_model

# Dictionary to hold scaled predictions
predictions_scaled = {}

# def generate_ensemble_predictions(models, X_test):
#     predictions = {}
#     for model_name, model in models.items():
#         if model_name in ['lstm', 'cnn']:
#             # Reshape the data for LSTM and CNN
#             X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))
#             predictions[model_name] = model.predict(X_test_reshaped).flatten()
#         else:
#             # Flatten the data for other models like RF, XGB, GB, Bagging, and LR
#             X_test_flat = X_test.reshape(X_test.shape[0], -1)
#             predictions[model_name] = model.predict(X_test_flat)
#
#     return predictions
def generate_ensemble_predictions(models, X_test):
    predictions = {}
    for model_name, model in models.items():
        # Ensure model_name is in the predefined list
        if model_name in ['lstm', 'cnn', 'hybrid', 'hybrid_model_optimized']:
            # Ensure the shape matches what was used during training
            X_test_reshaped = X_test.reshape(X_test.shape[0], *input_shape)
            predictions[model_name] = model.predict(X_test_reshaped).flatten()
        elif model_name in ['rf', 'xgb', 'lr', 'dt_boost', 'dt_bag', 'gb', 'bagging']:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            predictions[model_name] = model.predict(X_test_flat)
    return predictions

def weighted_ensemble(predictions, weights):
    total_weight = sum(weights.values())
    ensemble_pred = sum(predictions[model_name] * weight for model_name, weight in weights.items()) / total_weight
    return ensemble_pred



def evaluate_model(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = mean_squared_error(y_true, predictions, squared=False)
    return mae, rmse


def train_stacking_ensemble(models, X_test, y_test, save_meta_path='better/meta_model.pkl'):
    # Generate base model predictions
    base_predictions = generate_ensemble_predictions(models, X_test)

    # Reshape predictions to 2D arrays for stacking
    stacked_features = np.column_stack([pred.reshape(-1, 1) for pred in base_predictions.values()])

    # Split for meta-model training
    stacked_train, stacked_val, y_train_meta, y_val_meta = train_test_split(stacked_features, y_test, test_size=0.2, random_state=42)

    # Train the meta-model
    meta_model = GradientBoostingRegressor()
    meta_model.fit(stacked_train, y_train_meta)

    # Save the meta-model
    joblib.dump(meta_model, save_meta_path)

    # Predict using the meta-model
    meta_predictions = meta_model.predict(stacked_val)
    return meta_predictions, y_val_meta, meta_model



# Assuming data loading and preprocessing is already done
# X_train, X_test, y_train, y_test, input_shape are available
# input_shape = (X_train.shape[1], X_train.shape[2])
# Train individual models
# cnn_model, _ = train_cnn_model(X_train, y_train, X_test, y_test)
# lstm_model, _ = train_lstm_model(X_train, y_train, X_test, y_test)
rf_model = train_optimized_rf_model(X_train, y_train)
# xgb_model = train_optimized_xgb_model(X_train, y_train)
# lr_model = train_lr_model(X_train, y_train)
# dt_boost_model = train_dt_boost_model(X_train, y_train)
# dt_bag_model = train_dt_bag_model(X_train, y_train)
# best_bagging_model = tune_bagging_model(X_train, y_train)
# best_gb_model = tune_gradient_boosting_model(X_train, y_train)



# Store models in a dictionary
models = {
    # 'cnn': cnn_model,
    # 'lstm': lstm_model,
    'rf': rf_model,
    # 'xgb': xgb_model,
    # 'lr': lr_model,
    # 'dt_boost': dt_boost_model,
    # 'dt_bag': dt_bag_model,
    # 'gb': best_gb_model,
    # 'bagging': best_bagging_model,
    # 'hybrid_model_optimized': hybrid_model_optimized,
}

# Generate ensemble predictions
predictions = generate_ensemble_predictions(models, X_test)

# Define weights for ensemble (example weights, should be tuned)
weights = {
    # 'lstm': 0.3,
    # 'cnn': 0.2,
    'rf': 0.15,
    # 'xgb': 0.15,
    # 'lr': 0.1,
    # 'dt_boost': 0.1,
    # 'dt_bag': 0.1,
    # 'gb': 0.1,
    # 'bagging': 0.1,
    # 'hybrid_model_optimized': 0.3,
}



# Calculate weighted ensemble predictions
ensemble_pred = weighted_ensemble(predictions, weights)

# Evaluate the ensemble model
ensemble_mae, ensemble_rmse = evaluate_model(y_test, ensemble_pred)
print(f"Weighted Ensemble MAE: {ensemble_mae}, RMSE: {ensemble_rmse}")



meta_predictions, y_val_meta, meta_model = train_stacking_ensemble(models, X_test, y_test, 'better/meta_model.pkl')
meta_mae, meta_rmse = evaluate_model(y_val_meta, meta_predictions)
print(f"Stacking Ensemble MAE: {meta_mae}, RMSE: {meta_rmse}")

