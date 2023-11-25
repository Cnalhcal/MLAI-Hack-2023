import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate dummy dataset
# Replace this with your actual dataset
n_samples = 1000
n_features = 10

X = np.random.rand(n_samples, n_features)
y = np.random.rand(n_samples, 1)  # Replace with your actual target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the LSTM model BASE MODEL 
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_input_shape = (X_train_scaled.shape[1], 1)  # Adjust input shape based on your data
lstm_model = build_lstm_model(lstm_input_shape)

lstm_model.fit(
    np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1)),
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Make predictions using the LSTM model
lstm_preds = lstm_model.predict(np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1)))

# Define and train the LightGBM model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)

# Make predictions using the LightGBM model
lgb_preds = lgb_model.predict(X_test)

# Create an ensemble model using LightGBM
ensemble_preds = lgb_model.predict(X_test_scaled)

# Evaluate the models
lstm_mse = mean_squared_error(y_test, lstm_preds)
lgb_mse = mean_squared_error(y_test, lgb_preds)
ensemble_mse = mean_squared_error(y_test, ensemble_preds)

print(f'LSTM MSE: {lstm_mse}')
print(f'LightGBM MSE: {lgb_mse}')
print(f'Ensemble MSE: {ensemble_mse}')
