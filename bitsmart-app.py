
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model
st.title('BitSmart - Bitcoin Price Prediction and Trading Strategy')

# st.sidebar.header('User Input Features')
date = st.date_input("Select a Date", datetime.date(2024, 4, 20))

# data_load_state = st.text('Loading data...')
data = load_data('BTC-USD.csv')
# data_load_state.text('Loading data...done!')

# model_load_state = st.text('Loading model...')
model = load_model('Bitcoin_LSTM_Model.keras')
# model_load_state.text('Loading model...done!')


# st.subheader('Historical Bitcoin Prices')
# st.write(data.tail())
def predict():   
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


    
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_indicators(data, window=20):
        data['SMA'] = data['Close'].rolling(window=window).mean()
        data['EMA'] = data['Close'].ewm(span=window, adjust=False).mean()
        data['BB_upper'] = data['SMA'] + (data['Close'].rolling(window=window).std() * 2)
        data['BB_lower'] = data['SMA'] - (data['Close'].rolling(window=window).std() * 2)
        return data

    def predict_prices(model, data, date):
        data = data.dropna(subset=['Close'])  
        data['Date_ordinal'] = data['Date'].map(datetime.datetime.toordinal)
        if pd.to_datetime(date) not in data['Date'].values:
            raise ValueError("Selected date is not in the dataset")
        date_index = data[data['Date'] == pd.to_datetime(date)].index[0]
        sequence_length = 15
        if date_index < sequence_length:
            raise ValueError("Not enough data to create a sequence for the model")
        sequence_data = data.iloc[date_index-sequence_length:date_index]
        future_dates = pd.date_range(date + datetime.timedelta(days=1), periods=7).tolist()

        features = ['Date_ordinal', 'Open', 'Close']  
        sequence_data_normalized = (sequence_data[features] - sequence_data[features].mean()) / sequence_data[features].std()
        sequence_input = sequence_data_normalized.values.reshape(1, sequence_length, len(features))
        predictions = []
        for i in range(7):
            pred = model.predict(sequence_input)
            predicted_value = pred[0, -1] * sequence_data[features].std()['Close'] + sequence_data[features].mean()['Close']  
            predictions.append(predicted_value)

            new_row = np.array([[future_dates[i].toordinal(), predicted_value, predicted_value]])
            new_row_normalized = (new_row - sequence_data[features].mean().values) / sequence_data[features].std().values
            sequence_input = np.append(sequence_input[:, 1:, :], new_row_normalized.reshape(1, 1, -1), axis=1)

        return future_dates, predictions

    def generate_strategy(data, future_dates, predictions, rsi_window=14, ma_window=20, threshold=0.95):
        max_price = max(predictions)
        min_price = min(predictions)
        
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})
        data = pd.concat([data.set_index('Date'), pred_df.set_index('Date')], axis=1)

        data = calculate_indicators(data, ma_window)
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

        # Strategy decisions
        if sell_date is not None and data.loc[future_dates[sell_date], 'RSI'] > 70:
            return "Sell only", future_dates[sell_date], None  # RSI indicates overbought, sell

        if buy_date is not None and data.loc[future_dates[buy_date], 'RSI'] < 30:
            return "Sell and Buy", future_dates[sell_date], future_dates[buy_date]  # RSI indicates oversold, buy

        if sell_date is not None and predictions[sell_date] > data.loc[future_dates[sell_date], 'BB_upper']:
            return "Sell only", future_dates[sell_date], None  # Price is above upper Bollinger Band, sell

        if buy_date is not None and predictions[buy_date] < data.loc[future_dates[buy_date], 'BB_lower']:
            return "Sell and Buy", future_dates[sell_date], future_dates[buy_date]  # Price is below lower Bollinger Band, buy

        price_change = (max_price - min_price) / min_price
        if price_change < threshold:
            return "Hold", None, None 

        return "Hold", None, None



    try:
        future_dates, predictions = predict_prices(model, data, date)
        st.write(f"Length of future_dates: {len(future_dates)}")
        st.write(f"Length of predictions: {len(predictions)}")

        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close Price': predictions
        })
        
        st.subheader('Predicted Bitcoin Prices for the Next 7 Days')
        st.write(predictions_df)
        strategy, sell_date, buy_date = generate_strategy(data, future_dates, predictions)
                
        highest_price = max(predictions)
        lowest_price = min(predictions)
        average_price = np.mean(predictions)

        st.subheader('Summary of Predicted Prices')
        st.write(f"Highest Predicted Price: ${highest_price:.2f}")
        st.write(f"Lowest Predicted Price: ${lowest_price:.2f}")
        st.write(f"Average Predicted Closing Price: ${average_price:.2f}")
        st.subheader('Swing Trading Strategy')
        if strategy == "Hold":
            st.write("Hold the portfolio and do not trade.")
        elif strategy == "Sell only":
            st.write(f"Sell all on **{sell_date.date()}** and hold cash.")
        elif strategy == "Sell and Buy":
            st.write(f"Sell all on **{sell_date.date()}** and buy back on {buy_date.date()}.")
    except Exception as e:
        st.error(f"Error: {e}")

predict()
st.write('Predictions and strategy generated!')

if __name__ == '__main__':
    st.write('')
