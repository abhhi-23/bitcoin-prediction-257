import streamlit as st
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from datetime import timedelta

# Initialize the Streamlit UI components
st.title('BitSmart Bitcoin Prediction')

# Instructions and date input
start_date = datetime(2018,2,1)
end_date = datetime(2024,5,11)
st.write("Assume today's date is .... ")
selected_date = st.date_input("Select a date", min_value=start_date, max_value=end_date)

if st.button('Predict'):
    start_date = selected_date.strftime('%Y-%m-%d')
    # Load the trained model
    model = load_model('Bitcoin_LSTM_Model.keras')

    data = pd.read_csv('BTC-USD.csv')
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.drop(data[['Open','Adj Close','Volume']],axis=1)
    data_copy = data.copy()
    input_date = datetime.strptime(selected_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
    data_copy = data_copy[['Date','High', 'Low', 'Close']].loc[data_copy['Date'] > '2018-01-01']
    data = data[['High', 'Low', 'Close']].loc[data['Date'] > '2018-01-01']
    data_copy = data_copy.reset_index(drop=True)
    index = data_copy[data_copy['Date'] == input_date].index.tolist()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)  
    
    # Predict the next 7 days' high, low, and close prices
    pred_days = 7
    time_steps = 15
    last_sequence = data[index[0]-8:index[0]+7]
    temp_input = list(last_sequence)
    temp_input = [item for sublist in temp_input for item in sublist] 

    predicted_prices = []

    for _ in range(pred_days):
        if len(temp_input) >= time_steps * data.shape[1]:
            x_input = np.array(temp_input[-time_steps * data.shape[1]:]).reshape((1, time_steps, data.shape[1]))
            y_hat = model.predict(x_input, verbose=0)
            temp_input.extend(y_hat[0].tolist())
            predicted_prices.append(y_hat[0])
        else:
            break

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Extract the high, low, and close prices for the next 7 days
    predicted_high_prices = predicted_prices[:, 0]
    predicted_low_prices = predicted_prices[:, 1]
    predicted_close_prices = predicted_prices[:, 2]

    # Calculate the highest, lowest, and average closing prices for the next 7 days
    highest_price_next_7_days = max(predicted_high_prices)
    lowest_price_next_7_days = min(predicted_low_prices)
    average_close_price_next_7_days = np.mean(predicted_close_prices)

    
    # Placeholder data to simulate predictions
    predictions = {
        "Highest Price": highest_price_next_7_days,
        "Lowest Price": lowest_price_next_7_days,
        "Average Closing Price": average_close_price_next_7_days
    }
    

    def determine_trading_strategy(start_date, predicted_highs, predicted_lows, predicted_closes, initial_cash):
        start_date = pd.to_datetime(start_date)
        initial_bitcoins = initial_cash / predicted_closes[0]
        best_sell_day = None
        best_buy_day = None
        max_profit = -np.inf

        for sell_day in range(7):
            sell_price = predicted_highs[sell_day]
            cash_after_sell = initial_bitcoins * sell_price

            if sell_day < 6:  # Ensure there is at least one day after selling to consider buying
                best_buy_price = np.min(predicted_lows[sell_day+1:])
                buy_day = np.argmin(predicted_lows[sell_day+1:]) + sell_day + 1
                bitcoins_after_buy = cash_after_sell / best_buy_price
                final_balance = bitcoins_after_buy * predicted_closes[-1]  # Evaluate at the last day's closing price

                if final_balance > max_profit:
                    max_profit = final_balance
                    best_sell_day = sell_day
                    best_buy_day = buy_day
            else:
                # If selling on the last day, no buy operation
                final_balance = cash_after_sell
                if final_balance > max_profit:
                    max_profit = final_balance
                    best_sell_day = sell_day
                    best_buy_day = None

        sell_date = (start_date + pd.Timedelta(days=int(best_sell_day))) if best_sell_day is not None else "NA"
        buy_date = (start_date + pd.Timedelta(days=int(buy_day))) if best_buy_day is not None else "NA"

        return {
            "Sell Date": sell_date.strftime("%Y-%m-%d") if best_sell_day is not None else "NA",
            "Buy Date": buy_date.strftime("%Y-%m-%d") if best_buy_day is not None else "NA",
            "Final Balance": f"${max_profit:.2f}"
        }

    initial_cash = 100000
    predicted_highs = predicted_prices[:, 0]
    predicted_lows = predicted_prices[:, 1]
    predicted_closes = predicted_prices[:, 2]
    

    st.write(f"You have selected today as {selected_date.strftime('%m-%d-%Y')}. BitSmart has made the following predictions.")
    strategy = determine_trading_strategy(start_date,predicted_highs, predicted_lows, predicted_closes, initial_cash)
    recommended_strategy = strategy

    # Display predicted prices
    st.write("### Predicted prices (in USD) for the next seven days are:")
    for key, value in predictions.items():
        st.metric(label=key, value=f"${value}")

    # Display recommended swing trading strategy
    st.write("### Recommended swing trading strategy:")
    for key, value in recommended_strategy.items():
        st.metric(label=key, value=value)
