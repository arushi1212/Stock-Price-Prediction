import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def fetch_data(stock_list, start, end):
    data = {}
    for stock in stock_list:
        print(stock)
        data[stock] = yf.download(stock, start, end)
    return data

def process_data(data):
    company_list = list(data.values())
    for company, stock in zip(company_list, data.keys()):
        company["company_name"] = stock
    df = pd.concat(company_list, axis=0)
    return df

def prepare_training_data(stock_data, training_split=0.95):
    dataset = stock_data.filter(['Close']).values
    training_data_len = int(np.ceil(len(dataset) * training_split))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler, training_data_len

def train_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

def prepare_testing_data(stock_data, scaler, training_data_len):
    scaled_data = scaler.transform(stock_data.filter(['Close']).values)
    test_data = scaled_data[training_data_len - 60: , :]
    x_test, y_test = [], stock_data.filter(['Close']).values[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test

def make_predictions(model, x_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def get_rmse(predictions, y_test):
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    return rmse

'''end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
stock_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
result = fetch_data(stock_list, start, end)
df = process_data(result)
print(df.head())'''