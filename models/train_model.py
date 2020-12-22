import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

import warnings
warnings.filterwarnings('ignore')

def train(selected_month=1):
    # loading dataset
    frames = [pd.read_csv('data/transactions_1.csv', index_col=0), pd.read_csv('data/transactions_2.csv', index_col=0)]
    df = pd.concat(frames)
    df['date'] = pd.to_datetime(df['date'])

    # calculating product frequency  per months
    prediction_start_date = f'2019-{selected_month}-01'
    prediction_end_date = f'2019-{selected_month+3}-01'
    transactions_per_month_df = df.groupby(['customer_id', pd.Grouper(freq='M', key='date', closed='left')])['customer_id'].count().unstack(fill_value=0)

    # selecting data
    X = transactions_per_month_df.loc[:, transactions_per_month_df.columns.to_series().between('2017-01-01', prediction_start_date)].values
    y = transactions_per_month_df.loc[:, transactions_per_month_df.columns.to_series().between(prediction_start_date, prediction_end_date)].sum(axis=1).values

    # normalizing data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1))[:, 0]

    # spliting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    # reshaping for lstm
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # create model
    model = Sequential()
    model.add(LSTM(16, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(8, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # training model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=0)

    # saveing model
    model.save('models/serialized/lstm_model')

    # predicting data
    trainPredict = model.predict(X_train)
    model.reset_states()
    testPredict = model.predict(X_test)

    # invert predictions
    trainPredict = y_scaler.inverse_transform(trainPredict)
    trainY = y_scaler.inverse_transform([y_train])
    testPredict = y_scaler.inverse_transform(testPredict)
    testY = y_scaler.inverse_transform([y_test])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # ploting
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model loss with 100 epoch')
    plt.legend()
    plt.show()