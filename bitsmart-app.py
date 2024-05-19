import streamlit as st
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Initialize the Streamlit UI components
st.title('BitSmart Bitcoin Prediction')

# Instructions and date input
st.write("Enter date for bitcoin prediction.... ")
selected_date = st.date_input("", min_value=datetime.today())

if st.button('Predict'):

    # Load the trained model
    model = load_model('Bitcoin_LSTM_Model.keras')

    data = pd.read_csv('BTC-USD.csv')
    data = data.dropna()
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.drop(data[['Open','Adj Close','Volume']],axis=1)
    data_copy = data.copy()
    data_copy['Index'] = data_copy.index
    input_date = datetime.strptime(selected_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
    index = data_copy.index[data_copy['Date'] == input_date].tolist()
    data = data[['High', 'Low', 'Close']].loc[data['Date'] > '2018-01-01']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)  


    # Predict the next 7 days' high, low, and close prices
    pred_days = 7
    time_steps = 15
    last_sequence = data[len(data)-time_steps:]
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
    recommended_strategy = {"Sell All": "04-23-2024", "All In": "NA"}

    st.write(f"You have selected today as {selected_date.strftime('%m-%d-%Y')}. BitSmart has made the following predictions.")

    # Display predicted prices
    st.write("### Predicted prices (in USD) for the next seven days are:")
    for key, value in predictions.items():
        st.metric(label=key, value=f"${value}")

    # Display recommended swing trading strategy
    st.write("### Recommended swing trading strategy:")
    for key, value in recommended_strategy.items():
        st.metric(label=key, value=value)
