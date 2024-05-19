# streamlit_app.py

import streamlit as st
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf

# Load data from uploaded file
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Load trained model
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate moving averages and Bollinger Bands
def calculate_indicators(data, short_window=5, long_window=10, bollinger_window=20):
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    data['EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['BB_upper'] = data['Close'].rolling(window=bollinger_window).mean() + (data['Close'].rolling(window=bollinger_window).std() * 2)
    data['BB_lower'] = data['Close'].rolling(window=bollinger_window).mean() - (data['Close'].rolling(window=bollinger_window).std() * 2)
    return data

# Function to make predictions
def predict_prices(model, data, date):
    # Prepare data for the model
    data = data.dropna(subset=['Close'])  # Drop rows with NaN values in 'Close'
    data['Date_ordinal'] = data['Date'].map(datetime.datetime.toordinal)

    # Ensure the selected date exists in the data
    if pd.to_datetime(date) not in data['Date'].values:
        raise ValueError("Selected date is not in the dataset")

    # Find the index of the selected date
    date_index = data[data['Date'] == pd.to_datetime(date)].index[0]

    # Ensure we have enough data for the sequence length
    sequence_length = 15
    if date_index < sequence_length:
        raise ValueError("Not enough data to create a sequence for the model")

    # Create input sequence
    sequence_data = data.iloc[date_index-sequence_length:date_index]
    future_dates = pd.date_range(date + datetime.timedelta(days=1), periods=7).tolist()

    # Normalize the data (assuming your model expects normalized inputs)
    features = ['Date_ordinal', 'Open', 'Close']  # Use relevant features
    sequence_data_normalized = (sequence_data[features] - sequence_data[features].mean()) / sequence_data[features].std()
    sequence_input = sequence_data_normalized.values.reshape(1, sequence_length, len(features))

    # Make predictions iteratively
    predictions = []
    for i in range(7):
        pred = model.predict(sequence_input)
        predicted_value = pred[0, -1] * sequence_data[features].std()['Close'] + sequence_data[features].mean()['Close']  # Rescale prediction
        predictions.append(predicted_value)

        # Update sequence_input with the new prediction
        new_row = np.array([[future_dates[i].toordinal(), predicted_value, predicted_value]])
        new_row_normalized = (new_row - sequence_data[features].mean().values) / sequence_data[features].std().values
        sequence_input = np.append(sequence_input[:, 1:, :], new_row_normalized.reshape(1, 1, -1), axis=1)

    return future_dates, predictions

# Function to generate trading strategy
def generate_strategy(data, future_dates, predictions, rsi_window=14, short_window=5, long_window=10, bollinger_window=20, threshold=0.05):
    max_price = max(predictions)
    min_price = min(predictions)
    
    # Add predicted prices to data
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})
    data = pd.concat([data.set_index('Date'), pred_df.set_index('Date')], axis=1)

    # Calculate indicators
    data = calculate_indicators(data, short_window, long_window, bollinger_window)
    data['RSI'] = calculate_rsi(data, rsi_window)
    
    sell_date = None
    buy_date = None

    for i, price in enumerate(predictions):
        if price == max_price:
            sell_date = i
            break  # Break to ensure we only set sell_date once

    for i, price in enumerate(predictions):
        if price == min_price and sell_date is not None and i > sell_date:
            buy_date = i
            break  # Break to ensure we only set buy_date once

    # Strategy decision based on indicators and price movements
    if data.loc[future_dates[sell_date], 'RSI'] > 70:
        return "Sell only", future_dates[sell_date], None  # RSI indicates overbought, sell

    if data.loc[future_dates[buy_date], 'RSI'] < 30:
        return "Sell and Buy", future_dates[sell_date], future_dates[buy_date]  # RSI indicates oversold, buy

    if predictions[sell_date] > data.loc[future_dates[sell_date], 'BB_upper']:
        return "Sell only", future_dates[sell_date], None  # Price is above upper Bollinger Band, sell

    if predictions[buy_date] < data.loc[future_dates[buy_date], 'BB_lower']:
        return "Sell and Buy", future_dates[sell_date], future_dates[buy_date]  # Price is below lower Bollinger Band, buy

    if data.loc[future_dates[sell_date], 'SMA_short'] > data.loc[future_dates[sell_date], 'SMA_long']:
        return "Sell only", future_dates[sell_date], None  # Short MA crossed above long MA, sell

    if data.loc[future_dates[buy_date], 'SMA_short'] < data.loc[future_dates[buy_date], 'SMA_long']:
        return "Sell and Buy", future_dates[sell_date], future_dates[buy_date]  # Short MA crossed below long MA, buy

    price_change = (max_price - min_price) / min_price
    if price_change < threshold:
        return "Hold", None, None  # No significant price movement

    return "Hold", None, None

# Streamlit app
st.title('BitSmart - Bitcoin Price Prediction and Trading Strategy')

st.sidebar.header('User Input Features')
date = st.sidebar.date_input("Select a Date", datetime.date(2024, 4, 20))

# Load data
data_load_state = st.text('Loading data...')
data = load_data('BTC-USD.csv')
data_load_state.text('Loading data...done!')

# Load model
model_load_state = st.text('Loading model...')
model = load_model('Bitcoin_LSTM_Best_Model.keras')
model_load_state.text('Loading model...done!')

st.subheader('Historical Bitcoin Prices')
st.write(data.tail())

# Make predictions
try:
    future_dates, predictions = predict_prices(model, data, date)

    # Check lengths for debugging
    st.write(f"Length of future_dates: {len(future_dates)}")
    st.write(f"Length of predictions: {len(predictions)}")

    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close Price': predictions
    })

    st.subheader('Predicted Bitcoin Prices for the Next 7 Days')
    st.write(predictions_df)

    # Generate trading strategy
    strategy, sell_date, buy_date = generate_strategy(data, future_dates, predictions)

    st.subheader('Swing Trading Strategy')
    if strategy == "Hold":
        st.write("Hold the portfolio and do not trade.")
    elif strategy == "Sell only":
        st.write(f"Sell all on {sell_date.date()} and hold cash.")
    elif strategy == "Sell and Buy":
        st.write(f"Sell all on {sell_date.date()} and buy back on {buy_date.date()}.")
except Exception as e:
    st.error(f"Error: {e}")

if st.button('Predict'):
    st.write('Predictions and strategy generated!')

# Run Streamlit app
if __name__ == '__main__':
    st.write('BitSmart app is running.')
