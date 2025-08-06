import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer, Conv1D, MaxPooling1D, LSTM, Dense


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

# with custom_object_scope({'AttentionLayer': AttentionLayer}):
#     models = {
#         # 'lstm': load_model('better/lstm_model_tuned.h5'),
#         # 'cnn': load_model('better/cnn_model.h5'),
#         # Uncomment other models as needed
#         # 'lr': joblib.load('better/lr_model.pkl'),
#         'meta': joblib.load('better/meta_model.pkl'),
#         'hybrid_model_optimized': load_model('better/hybrid_model_optimized.h5')
#         # Add other models if necessary
#     }


def make_rolling_predictions_with_meta(meta_model, hybrid_optimized, last_known_sequence, num_predictions, feature_scaler, target_scaler):
# def make_rolling_predictions_with_meta(meta_model, lstm_model, cnn_model, hybrid_optimized, last_known_sequence, num_predictions, feature_scaler, target_scaler):
    current_sequence = last_known_sequence.copy()
    meta_predictions = []
    lstm_predictions = []
    cnn_predictions = []
    hybrid_optimized_predictions = []

    for _ in range(num_predictions):
        # Reshape sequence for LSTM and CNN models
        X_test_reshaped = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        # Reshape sequence for LR model
        X_test_flat = current_sequence.reshape(1, -1)

        # Generate and store predictions from individual models
        # lstm_pred = lstm_model.predict(X_test_reshaped).flatten()
        # lstm_predictions.append(target_scaler.inverse_transform(lstm_pred.reshape(-1, 1)))  # Assuming single-step prediction

        hybrid_optimized_pred = hybrid_optimized.predict(X_test_reshaped).flatten()
        hybrid_optimized_predictions.append(target_scaler.inverse_transform(hybrid_optimized_pred.reshape(-1, 1)))  # Assuming single-step prediction

        # cnn_pred = cnn_model.predict(X_test_reshaped).flatten()
        # cnn_predictions.append(target_scaler.inverse_transform(cnn_pred.reshape(-1, 1)))




        # Meta-model prediction
        meta_pred = meta_model.predict(meta_features)
        meta_pred_original = target_scaler.inverse_transform(meta_pred.reshape(-1, 1))[0, 0]
        meta_predictions.append(meta_pred_original)

        # Update the sequence with new prediction
        new_feature_vector = current_sequence[-1, :].copy()
        new_feature_vector[0] = meta_pred_original  # Assuming 'Close' price is the first feature
        current_sequence = np.append(current_sequence[1:, :], [new_feature_vector], axis=0)

    return  meta_predictions , hybrid_optimized_predictions
    # return lstm_predictions, cnn_predictions, meta_predictions , hybrid_optimized_predictions

# Usage example:




def create_sequences(data, sequence_length):
    xs = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        xs.append(x)
    return np.array(xs)


# # Load the saved models and scalers
models = {
    # 'lstm': load_model('better/lstm_model_tuned.h5'),
    # 'cnn': load_model('better/cnn_model.h5'),
    'rf': joblib.load('better/best_rf_model.pkl'),
    # 'xgb': joblib.load('better/best_xgb_model.pkl'),
    # 'svm': joblib.load('better/svm_model.pkl'),
    # 'lr': joblib.load('better/lr_model.pkl'),
    # 'dt_bag': joblib.load('better/dt_bag_model.pkl'),
    # 'dt_boost': joblib.load('better/dt_boost_model.pkl'),
    'meta': joblib.load('better/meta_model.pkl'),
    # 'gb': joblib.load('better/best_gb_model.pkl'),
    # 'bagging': joblib.load('better/best_bagging_model.pkl'),
}


feature_scaler = joblib.load('better/feature_scaler.pkl')
target_scaler = joblib.load('better/target_scaler.pkl')


# Define a function to preprocess features
def preprocess_features(df, feature_scaler):
    # Define the features to scale based on the columns used during training
    features = df[['High', 'Low', 'Open', 'Volume', 'vw', 'RSI_14', 'MACD', 'MACD_signal','MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA_12', 'EMA_26', 'SMA_10', 'SMA_20', 'OBV', 'ATR', 'Stoch_k', 'Stoch_d']]

    # Apply the scaler used during training
    scaled_features = feature_scaler.transform(features)
    # No need to fit the imputer again, just transform the data
    imputed_features = imputer.transform(scaled_features)
    return imputed_features


# Load the test dataset
# test_df = pd.read_csv('better/stock_data_enhanced.csv')
test_df = pd.read_csv('better/clean-data/stock_future_data_enhanced.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Load the imputer
imputer = joblib.load('better/imputer.pkl')

# Prepare your features for the model
test_features = preprocess_features(test_df, feature_scaler)
sequence_length = 30  # Ensure this is the same as training
X_test = create_sequences(test_features, sequence_length)

# Prepare your targets for comparison
actual_prices = test_df['Close'][sequence_length:]
print ('actual_prices is ',actual_prices.shape)

# Function to inverse scale predictions
def inverse_scale_predictions(predictions, scaler):
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


predictions_scaled = {}
# Generate predictions for each model
for model_name, model in models.items():
    print(f"Generating {model_name} predictions...")
    if model_name != 'meta':
        if model_name in ['lstm', 'cnn', 'hybrid', 'hybrid_model_optimized']:
            # Reshape the data for LSTM and CNN
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))
            predictions_scaled[model_name] = model.predict(X_test_reshaped).flatten()
        else:
            # Flatten the data for other models
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            predictions_scaled[model_name] = model.predict(X_test_flat)

# Inverse scale the predictions
predictions = {model_name: inverse_scale_predictions(pred, target_scaler)
               for model_name, pred in predictions_scaled.items()}

# Combine predictions from all models for an ensemble prediction
ensemble_predictions = np.mean(list(predictions.values()), axis=0)
mae_ensemble = mean_absolute_error(actual_prices, ensemble_predictions)
print(f"Ensemble Model MAE: {mae_ensemble}")

# Prepare features for the meta model
meta_features = np.column_stack(list(predictions_scaled.values()))
meta_predictions_scaled = models['meta'].predict(meta_features)
meta_predictions = inverse_scale_predictions(meta_predictions_scaled, target_scaler)

# After getting all the base model predictions, stack them horizontally to create meta features
meta_features = np.column_stack([
    # predictions_scaled['lstm'],
    # predictions_scaled['cnn'],
    predictions_scaled['rf'],
    # predictions_scaled['xgb'],
    # predictions_scaled['svm'],
    # predictions_scaled['lr'],
    # predictions_scaled['dt_bag'],
    # predictions_scaled['dt_boost'],
    # predictions_scaled['gb'],
    # predictions_scaled['bagging'],
    # predictions_scaled['hybrid_model_optimized']
])

# Check that the meta_features have the correct shape
n_features_expected_by_meta = models['meta'].n_features_in_
n_features_provided_to_meta = meta_features.shape[1]

if n_features_provided_to_meta != n_features_expected_by_meta:
    raise ValueError(
        f"Meta model expects {n_features_expected_by_meta} features, but got {n_features_provided_to_meta}.")

# If the shape is correct, make predictions with the meta model
meta_predictions_scaled = models['meta'].predict(meta_features)

# Inverse scale the predictions of the meta model
meta_predictions = target_scaler.inverse_transform(meta_predictions_scaled.reshape(-1, 1)).flatten()

# Evaluate the predictions and store the MAE for each model
model_mae = {}
for model_name, pred in predictions.items():
    mae = mean_absolute_error(actual_prices, pred)
    model_mae[model_name] = mae
    print(f"{model_name} Model MAE: {mae}")

# Include MAE for ensemble and meta model
model_mae['ensemble'] = mae_ensemble
model_mae['meta'] = mean_absolute_error(actual_prices, meta_predictions)
print(f"Meta Model MAE: {model_mae['meta']}")

# Determine the best performing model based on MAE
best_model_name = min(model_mae, key=model_mae.get)
best_model_mae = model_mae[best_model_name]
print(f"The best performing model is: {best_model_name} with MAE: {best_model_mae}")


last_known_sequence = X_test[-1]

# Make rolling predictions
num_predictions = 3  # Number of days to predict
# rolling_predictions = make_rolling_predictions_with_meta(models['meta'], models['lstm'], models['cnn'], models['lr'], last_known_sequence, num_predictions, feature_scaler, target_scaler)
#
# # Output the predictions
# for i, pred in enumerate(rolling_predictions, 1):
#     print(f"Day {i} Prediction: {pred}")



# lstm_preds, cnn_preds, meta_preds , hybrid_optimized_pred = make_rolling_predictions_with_meta(models['meta'], models['lstm'] , models['cnn'], models['hybrid_model_optimized'], last_known_sequence, num_predictions, feature_scaler, target_scaler)
meta_preds , hybrid_optimized_pred = make_rolling_predictions_with_meta(models['meta'],  models['hybrid_model_optimized'], last_known_sequence, num_predictions, feature_scaler, target_scaler)

# Print rolling predictions
# print("LSTM Rolling Predictions:", lstm_preds)
# print("CNN Rolling Predictions:", cnn_preds)
print("Hybrid optimized Predictions:", hybrid_optimized_pred)
print("Meta Model Rolling Predictions:", meta_preds)


print ('hybrid_optimized_pred',hybrid_optimized_pred)




import matplotlib.dates as mdates

plt.figure(figsize=(15, 5))

# Plot actual prices
plt.plot(test_df['Date'][sequence_length:], actual_prices, label='Actual Prices', color='blue', linewidth=2)

# Plot predictions for each model
colors = [ 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta', 'gray', 'lime','green',]
for (model_name, pred), color in zip(predictions.items(), colors):
    print(f"Plotting {model_name} predictions... with color {color}")
    plt.plot(test_df['Date'][sequence_length:], pred, label=f'{model_name} Predicted Prices', linestyle='--', color=color)

# Plot meta model predictions
plt.plot(test_df['Date'][sequence_length:], meta_predictions, label='Meta Predicted Prices', color='yellow', linestyle='-', linewidth=2)

# Plot ensemble model predictions
plt.plot(test_df['Date'][sequence_length:], ensemble_predictions, label='Ensemble Predicted Prices', color='black', linestyle='-', linewidth=2)

# rolling predictions
# Add title and labels
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')

# Add legend
plt.legend()

# Format x-axis to show each date
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid
plt.grid(True)

# Ensure everything fits without overlapping
plt.tight_layout()

# Display the plot
plt.show()