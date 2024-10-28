# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("C://Users//Nitin//Downloads//Crude Oil Prices Daily.xlsx")

# Forward fill missing data
df.fillna(method='ffill', inplace=True)

# Rename columns for compatibility with forecasting models
df.rename(columns={'Date': 'ds', 'Closing Value': 'y'}, inplace=True)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['y']])

# Split the data into training and testing sets
train_size = len(scaled_data) - 365  # Use all but the last 365 days for training
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Time step for LSTM
time_step = 30
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with fewer epochs to speed up the deployment
model.fit(X_train, y_train, batch_size=64, epochs=10)

# Make predictions on test data
predictions = model.predict(X_test)

# Inverse transform predictions and test data to original values
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics
mse = mean_squared_error(y_test_actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, predictions)

# Streamlit Dashboard
st.markdown("<h1 style='text-align: center; color: white;'>Closing Price Prediction</h1>", unsafe_allow_html=True)

# User input for future days to predict
st.sidebar.header('Future Prediction Parameters')
num_days = st.sidebar.number_input("Number of days to predict into the future", min_value=1, max_value=365, value=10)

# Get the last 30 days from the dataset for future predictions
last_30_days = scaled_data[-30:]

# Reshape for prediction
last_30_days_input = last_30_days.reshape(1, -1, 1)

# Make future predictions
predicted_prices = []
for _ in range(num_days):
    predicted_price_scaled = model.predict(last_30_days_input)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    predicted_prices.append(predicted_price[0][0])
    
    # Ensure predicted price has correct dimensions and append it to the input
    predicted_price_scaled = predicted_price_scaled.reshape(1, 1, 1)  # Correct the dimensions
    last_30_days_input = np.append(last_30_days_input[:, 1:, :], predicted_price_scaled, axis=1)

# Display future predictions
future_dates = [df['ds'].max() + timedelta(days=i) for i in range(1, num_days + 1)]
predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})

st.write(f"Predicted prices for the next {num_days} days:")
st.dataframe(predicted_df)

# Prepare the full dataframe with predictions and actual values
df['Predicted'] = np.nan
df.iloc[-len(predictions):, df.columns.get_loc('Predicted')] = predictions.flatten()

# Optionally plot the predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='Actual')
plt.plot(df['ds'], df['Predicted'], label='Predicted', color='orange')
plt.xlabel('Date')
plt.ylabel('Closing Value')
plt.title('Actual vs Predicted - LSTM Model')
plt.legend()
plt.show()
