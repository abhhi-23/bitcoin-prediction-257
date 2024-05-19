import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load data and model
crypto_data = pd.read_csv('BTC-USD.csv', parse_dates=['Date'])
crypto_data.set_index('Date', inplace=True)
model = load_model('Bitcoin_LSTM_Model.keras')
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to make predictions
def predict_prices(input_date, data, model, scaler, time_steps=7):
    input_date = pd.to_datetime(input_date)
    end_date = input_date + timedelta(days=6)
    if input_date not in data.index or end_date not in data.index:
        raise ValueError("The input date or end range is out of the data's index.")

    # Scale the data
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']])

    # Get the last sequences from the input date
    idx = data.index.get_loc(input_date)
    x_input = scaled_data[idx-time_steps+1:idx+1].reshape((1, time_steps, 4))

    # Predict next 7 days
    predictions = model.predict(x_input)
    predictions = scaler.inverse_transform(predictions).flatten()
    
    return {
        "High Prices": predictions[1::4],
        "Low Prices": predictions[2::4],
        "Close Prices": predictions[3::4]
    }

# Function to determine trading strategy
def determine_trading_strategy(predictions):
    high_prices = predictions['High Prices']
    low_prices = predictions['Low Prices']
    closing_prices = predictions['Close Prices']
    
    # Initialize cash and find initial number of bitcoins
    initial_cash = 100000
    initial_bitcoins = initial_cash / predictions['Open Prices'][0]
    
    # Calculate the best sell day (max high price)
    sell_index = np.argmax(high_prices)
    sell_price = high_prices[sell_index]
    
    # Calculate post-sell cash
    cash_after_sell = initial_bitcoins * sell_price
    
    # Calculate the best buy day (min low price after sell)
    buy_index = sell_index + 1 + np.argmin(low_prices[sell_index+1:])
    buy_price = low_prices[buy_index]
    
    # Calculate bitcoins after buy
    bitcoins_after_buy = cash_after_sell / buy_price
    
    # Calculate the final balance (value in bitcoins at last day's close price)
    final_balance = bitcoins_after_buy * closing_prices[-1]
    
    return {
        "Sell Date": input_date + timedelta(days=sell_index),
        "Buy Date": input_date + timedelta(days=buy_index),
        "Final Balance": final_balance
    }

# Example usage
input_date = '2024-04-20'
predictions = predict_prices(input_date, crypto_data, model, scaler)
trading_strategy = determine_trading_strategy(predictions)

print(f"Predictions from {input_date}: {predictions}")
print(f"Recommended Trading Strategy: {trading_strategy}")
