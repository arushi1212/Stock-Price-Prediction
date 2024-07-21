import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#matplotlib inline

import yfinance as yf
from datetime import datetime

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Stock Price Prediction Project")

with dataset:
    st.header("Stock Price dataset using Yahoo finance library in python")
    # The tech stocks we'll use for this analysis
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    #Setting up End and Start times for data grab
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    company_data = {}
    for stock in tech_list:
        company_data[stock] = yf.download(stock, start, end)
    
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]
    for stock, com_name in zip(tech_list, company_name):
        company_data[stock]["company_name"] = com_name
    
    df=pd.concat(company_data.values(), axis=0)
    st.write(df.tail(10))

with features:
    st.header("Analysing the dataset")
    st.markdown("**.describe()** generates descriptive statistics. Descriptive statistics include those that summarize the central tendency, dispersion, and shape of a dataset's distribution, excluding NaN values.")
    # Summary Statistics
    st.write(company_data['AAPL'].describe())

    st.markdown("**.info()** method prints information about a DataFrame including the index dtype and columns, non-null values, and memory usage.")
    # General statistics
    st.write(company_data['AAPL'].info())

    st.markdown("**Closing Price** - is the last price at which the stock is traded during the regular trading day. A stock's closing price is the standard benchmark used by investors to track its perfomance over time.")
    # Visualizing the closing price
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, stock in enumerate(tech_list, 1):
        plt.subplot(2, 2, i)
        company_data[stock]['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {company_name[i - 1]}")

    plt.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.markdown("**Volume of Sales** - is the amount of an asset or security that changes hands over some period of time, often over the course of a day. For instance, the stock trading volume would refer to the number of shares of security traded between its daily open and close. Trading volume, and changes to volume over the course of time, are important inputs for technical traders.")
    # Visualizing the volume of stock traded each day
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, stock in enumerate(tech_list, 1):
        plt.subplot(2, 2, i)
        company_data[stock]['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Sales Volume for {company_name[i - 1]}")

    plt.tight_layout()
    st.pyplot()

    st.markdown("**Moving Average** - is a simple technical analysis tool that smooths out price sata by creating a constantly updated average price. The average is taken over a specific period of time, like 10 days, 20 minutes, 30 weeks or any time period the trader chooses.")
    # Visualizing the Moving Average
    ma_day = [10, 20, 50]
    for ma in ma_day:
        for company in tech_list:
            column_name = f"MA for {ma} days"
            company_data[company][column_name] = company_data[company]['Adj Close'].rolling(ma).mean()
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    company_data['AAPL'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
    axes[0,0].set_title('APPLE')

    company_data['GOOG'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
    axes[0,1].set_title('GOOGLE')

    company_data['MSFT'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
    axes[1,0].set_title('MICROSOFT')

    company_data['AMZN'][['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
    axes[1,1].set_title('AMAZON')

    fig.tight_layout()
    st.pyplot()
    st.markdown("From the above plots, we can see that the best values measuring the moving average are 10 and 20 days because we still capture trends in data without the noise.")

    st.markdown("To analyze the risk of the stock, we need to take a closer look at the daily changes of the stock.")
    # We'll use pct_change to find the percent change for each day
    for company in tech_list:
        company_data[company]['Daily Return'] = company_data[company]['Adj Close'].pct_change()

    # Then we'll plot the daily return percentage
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    company_data['AAPL']['Daily Return'].plot(ax=axes[0,0], legend=True, linestyle='--', marker='o')
    axes[0,0].set_title('APPLE')

    company_data['GOOG']['Daily Return'].plot(ax=axes[0,1], legend=True, linestyle='--', marker='o')
    axes[0,1].set_title('GOOGLE')

    company_data['MSFT']['Daily Return'].plot(ax=axes[1,0], legend=True, linestyle='--', marker='o')
    axes[1,0].set_title('MICROSOFT')

    company_data['AMZN']['Daily Return'].plot(ax=axes[1,1], legend=True, linestyle='--', marker='o')
    axes[1,1].set_title('AMAZON')

    fig.tight_layout()
    st.pyplot()

    st.markdown("Now we get an overall look at the average daily return using a histogram.")
    plt.figure(figsize=(12, 9))

    for i, company in enumerate(tech_list, 1):
        plt.subplot(2, 2, i)
        company_data[company]['Daily Return'].hist(bins=50)
        plt.xlabel('Daily Return')
        plt.ylabel('Counts')
        plt.title(f'{company_name[i - 1]}')

    plt.tight_layout()
    st.pyplot()

    st.markdown("**Correlation** - is a statistic that measures the degree to which two variables move in relation to each other which has a value that must fall between -1.0 and +1.0. Correlation measures association, but doesn't show if x cause y or vice versa, or if the association is caused by a third factor.")
    # Grab all the closing prices for the tech stock list into one DataFrame

    closing_df = yf.download(tech_list, start=start, end=end)['Adj Close']

    # Make a new tech returns DataFrame
    tech_rets = closing_df.pct_change()
    st.write(tech_rets.head())

    st.markdown("Now comparing the daily percentage return of two stocks to check how they are correlated.")
    # Comparing Google to itself should show a perfectly linear relationship
    joint_plot = sns.jointplot(x='GOOG', y='GOOG', data=tech_rets, kind='scatter', color='seagreen')
    st.pyplot(joint_plot)
    # We'll use joinplot to compare the daily returns of Google and Microsoft
    joint_plot_2 = sns.jointplot(x='GOOG', y='MSFT', data=tech_rets, kind='scatter')
    st.pyplot(joint_plot_2)
    st.markdown("We can see that if two stocks are perfectly (and positively) correlated with each other a linear relationship between its daily return values should occur.")

    st.markdown("We can simply call pairplot on our DataFrame for an automatic visual analysis of all the comparisons")
    pair_plot = sns.pairplot(tech_rets, kind='reg')
    st.pyplot(pair_plot)

    # Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
    returns_fig = sns.PairGrid(closing_df)

    '''# Using map_upper we can specify what the upper triangle will look like.
    returns_fig.map_upper(plt.scatter,color='purple')

    # We can also define the lower triangle in the figure, inclufing the plot type (kde) or the color map (BluePurple)
    returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

    # Finally we'll define the diagonal as a series of histogram plots of the daily return
    returns_fig.map_diag(plt.hist,bins=30)
    st.pyplot(returns_fig)

    plt.figure(figsize=(12, 10))'''

    plt.subplot(2, 2, 1)
    sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock return')

    plt.subplot(2, 2, 2)
    sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock closing price')
    st.pyplot()
    st.markdown("From above we see that numerically and visually that Microsoft and Amazon have the strongest correlation of daily stock return and stock closing price. Also that all technology companies are positively related while Apple and Amazon are negatively related.")


with model_training:
    st.header("Predicting the Closing Price")
    st.markdown("Select a stcok")
    stock_selected = st.selectbox('Select box', tech_list)
    # Get the stock quote
    df = yf.download(stock_selected, start='2012-01-01', end=datetime.now())
    # Show the data
    #st.write(df)
    sel_col, disp_col = st.columns(2)
    sel_col.markdown("List of features in my data:")
    sel_col.write(df.columns)

    plt.figure(figsize=(16,6))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()
    st.pyplot()

    # Create a new dataframe with only the 'Close column
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    #training_data_len

    #st.markdown("Scaling the data")
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        #if i<= 61:
            #print(x_train)
            #print(y_train)
            #print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    st.markdown("Calculating the root mean square error of the model:")
    st.write(rmse)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    #plt.show()
    st.pyplot()

    st.markdown("Displaying the actual and predicted prices:")
    st.write(valid)
