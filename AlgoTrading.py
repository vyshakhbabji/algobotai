import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

from keras.layers import Reshape



from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch






# Load preprocessed data
X_train = np.load('better/X_train.npy')
y_train = np.load('better/y_train.npy')
X_test = np.load('better/X_test.npy')
y_test = np.load('better/y_test.npy')

# Load scalers (if needed for post-prediction processing)
feature_scaler = joblib.load('better/feature_scaler.pkl')
target_scaler = joblib.load('better/target_scaler.pkl')

# Assuming X_train is already loaded from the preprocessing step
total_features = X_train.shape[2]  # The number of features is the size of the last dimension

# Assuming the build_hybrid_model function and input shape are defined
input_shape = X_train.shape[1:]  # Shape of the input data









# File: hybrid_model.py

def train_hybrid_model( X_train, y_train, X_test, y_test, epochs=50, batch_size=64):

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

        def get_config(self):
            return super(AttentionLayer, self).get_config()


    def add_cnn_layers(model, input_shape):
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        # Dynamically compute the new shape based on the current output shape
        model.add(Reshape((-1, model.layers[-1].output_shape[-1])))


    def add_lstm_layers(model):
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50))


    def build_tunable_hybrid_model(hp):
        model = Sequential()

        # Tunable parameters for CNN layers
        model.add(Conv1D(
            filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('conv_kernel_size', values=[2, 3, 4]),
            activation='relu',
            input_shape=input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Reshape((-1, model.layers[-1].output_shape[-1])))

        # Tunable parameters for LSTM layers
        for i in range(hp.Int('num_lstm_layers', 1, 3)):
            model.add(LSTM(
                units=hp.Int('units_lstm_{}'.format(i), min_value=30, max_value=100, step=10),
                return_sequences=i < hp.Int('num_lstm_layers', 1, 3) - 1
            ))

        model.add(AttentionLayer())  # Custom Attention Layer
        model.add(Dense(1))


        # Compile model
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='mean_squared_error'
        )

        model.save('better/tunable_hybrid_model.h5')

        return model


    def build_hybrid_model(input_shape):
        try:
            model = Sequential()
            add_cnn_layers(model, input_shape)
            add_lstm_layers(model)
            model.add(AttentionLayer())  # Review and potentially enhance this layer
            model.add(Dense(1))  # Assuming a regression task; adjust if necessary

            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mean_squared_error')

            model.summary()  # Provides a summary of the model
            model.save('better/hybrid_model.h5')
            return model
        except Exception as e:
            print(f"Error occurred while building the model: {e}")
            return None


    # Use a batch size that divides the number of samples
    batch_size = 64  # Adjust as necessary


    model = build_hybrid_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)] , verbose=1)

    # Evaluate the model on test data
    test_loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')


    # Initialize the tuner
    tuner = RandomSearch(
        build_tunable_hybrid_model,
        objective='val_loss',
        max_trials=10,  # Adjust based on how many trials you want to run
        executions_per_trial=2,
        directory='hybrid_model_tuning',
        project_name='tune_hybrid_model'
    )

    # Start hyperparameter search
    tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=5)])


    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the best hyperparameters
    best_model = build_tunable_hybrid_model(best_hps)
    best_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
                   callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    # Save the best model
    best_model.save('better/hybrid_model_optimized.h5')

    return best_model


# Train the model
best_model = train_hybrid_model( X_train, y_train, X_test, y_test, epochs=50, batch_size=64)

