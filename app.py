import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
import numpy as np


sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from backend import fetch_data, process_data, prepare_training_data, train_model, prepare_testing_data, make_predictions, get_rmse

st.title("Stock Price Prediction Project")

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

data = fetch_data(tech_list, start, end)
df = process_data(data)
st.header("Stock Price dataset using Yahoo finance library in python")
st.write(df.tail(10))

company_data = {ticker: df[df['company_name'] == ticker] for ticker in tech_list}
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]

st.header("Analysing the dataset")
st.markdown("**.describe()** generates descriptive statistics. Descriptive statistics include those that summarize the central tendency, dispersion, and shape of a dataset's distribution, excluding NaN values.")
st.write(company_data['AAPL'].describe())

st.markdown("**.info()** method prints information about a DataFrame including the index dtype and columns, non-null values, and memory usage.")
st.write(company_data['AAPL'].info())

st.markdown("**Closing Price** - is the last price at which the stock is traded during the regular trading day. A stock's closing price is the standard benchmark used by investors to track its perfomance over time.")
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, (ticker, name) in enumerate(zip(tech_list, company_name), 1):
    plt.subplot(2, 2, i)
    company_data[ticker]['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {name}")
plt.tight_layout()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.markdown("**Volume of Sales** - is the amount of an asset or security that changes hands over some period of time, often over the course of a day. For instance, the stock trading volume would refer to the number of shares of security traded between its daily open and close. Trading volume, and changes to volume over the course of time, are important inputs for technical traders.")
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, (ticker, name) in enumerate(zip(tech_list, company_name), 1):
    plt.subplot(2, 2, i)
    company_data[ticker]['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume for {name}")
plt.tight_layout()
st.pyplot()

st.markdown("**Moving Average** - is a simple technical analysis tool that smooths out price data by creating a constantly updated average price. The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks or any time period the trader chooses.")
ma_day = [10, 20, 50]
for ma in ma_day:
    for ticker in tech_list:
        column_name = f"MA for {ma} days"
        company_data[ticker][column_name] = company_data[ticker]['Adj Close'].rolling(ma).mean()
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
for ax, (ticker, name) in zip(axes.flatten(), zip(tech_list, company_name)):
    company_data[ticker][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=ax)
    ax.set_title(name)
fig.tight_layout()
st.pyplot()

st.markdown("To analyze the risk of the stock, we need to take a closer look at the daily changes of the stock.")
for ticker in tech_list:
    company_data[ticker]['Daily Return'] = company_data[ticker]['Adj Close'].pct_change()
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
for ax, (ticker, name) in zip(axes.flatten(), zip(tech_list, company_name)):
    company_data[ticker]['Daily Return'].plot(ax=ax, legend=True, linestyle='--', marker='o')
    ax.set_title(name)
fig.tight_layout()
st.pyplot()

st.markdown("Now we get an overall look at the average daily return using a histogram.")
plt.figure(figsize=(12, 9))
for i, (ticker, name) in enumerate(zip(tech_list, company_name), 1):
    plt.subplot(2, 2, i)
    company_data[ticker]['Daily Return'].hist(bins=50)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(name)
plt.tight_layout()
st.pyplot()

st.markdown("**Correlation** - is a statistic that measures the degree to which two variables move in relation to each other which has a value that must fall between -1.0 and +1.0. Correlation measures association, but doesn't show if x cause y or vice versa, or if the association is caused by a third factor.")
closing_df = yf.download(tech_list, start=start, end=end)['Adj Close']
tech_rets = closing_df.pct_change()
st.write(tech_rets.head())

st.markdown("Now comparing the daily percentage return of two stocks to check how they are correlated.")
joint_plot = sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')
st.pyplot(joint_plot)
joint_plot_2 = sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')
st.pyplot(joint_plot_2)

st.markdown("We can simply call pairplot on our DataFrame for an automatic visual analysis of all the comparisons")
pair_plot = sns.pairplot(tech_rets, kind='reg')
st.pyplot(pair_plot)

returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)
st.pyplot(returns_fig)

st.markdown("We can also visualize correlation by using a heatmap.")
sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
st.pyplot()

st.markdown("Now analyzing the Risk.")
area = np.pi*20
plt.figure(figsize=(10, 10))
plt.scatter(tech_rets.mean(), tech_rets.std(), s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(tech_rets.columns, tech_rets.mean(), tech_rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3'))
st.pyplot()

st.markdown("Predicting stock prices using a Long Short Term Memory (LSTM) method")

st.markdown("Select the stock you want to predict the closing price for:")
ticker = st.selectbox('Select stock', tech_list)
stock_data = company_data.get(ticker)

x_train, y_train, scaler, training_data_len = prepare_training_data(stock_data)
model = train_model(x_train, y_train)

x_test, y_test = prepare_testing_data(stock_data, scaler, training_data_len)
predictions = make_predictions(model, x_test, scaler)

rmse = get_rmse(predictions, y_test)
st.write(f"RMSE for {ticker}: {rmse}")

train = stock_data[:training_data_len]
valid = stock_data[training_data_len:]
valid['Predictions'] = predictions

# Display predictions and actual values
st.header(f"Predicted vs Actual Values for {ticker}")

# Select a range of dates to display
date_range = st.slider('Select Date Range', min_value=valid.index.min(), max_value=valid.index.max(), value=(valid.index.min(), valid.index.max()))

# Filter data based on selected date range
filtered_data = valid[(valid.index >= date_range[0]) & (valid.index <= date_range[1])]

st.dataframe(filtered_data[['Close', 'Predictions']])


plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot()
