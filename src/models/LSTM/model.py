import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf
#import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import ModelCheckpoint, History
import numpy as np

# Credit: https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
 
# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test
 
# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons, j):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, return_sequences = True,
                   batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(drop_out)) # added a droppout layer
    model.add(LSTM(n_neurons,return_sequences = True)) #added lstm layer
    model.add(LSTM(n_neurons,return_sequences =False))    #added lstm layer
    model.add(Dropout(drop_out))    # added a dropout layer
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()
        
    # saving best model checkpoint for hyparameter tuning        
    checkpoint_filepath = '/Users/jessi/Onedrive/Desktop/Sem6/DS340W/Final/src/temp/ckpt%s'%j
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)
            
    return model, checkpoint_filepath, model_checkpoint_callback
 
# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]
 
# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts
 
# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    # show the plot
    pyplot.xlabel("Time (days)")
    pyplot.ylabel("Price")
    pyplot.title("Amazon Closing Prices")
    pyplot.show()


if __name__ == '__main__':
    # Load the data
    amazon = read_csv('/Users/jessi/Onedrive/Desktop/Sem6/DS340W/Final/volume/stock_data/AMZN_2006-01-01_to_2017-11-01.csv')
    series = amazon['Close'][2000:2600]

    n_lag = 5
    n_seq = 5
    n_test = 50
    drop_out = 0

    scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
    # forecasts_list = []
    # actual_list = []
    results = []
    batches = 1
    # n_batch = 1
    # n_epochs = 10
    # n_neurons = 8
    epochs = [1, 5, 10, 30] # [5, 50, 100, 250]
    neurons = [1, 2, 5, 8]

    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    j = 0
    
    # implemented for loop for hyperparameter tuning
    for e in epochs:
        for n in neurons:
            # getting model and chekcpoints 
            model, checkpoint_filepath, model_checkpoint_callback = fit_lstm(train, n_lag, n_seq, batches, e, n, j)
            j += 1
            # refitiing model with best checkpoint
            history = model.fit(X, y, epochs=e, callbacks=[model_checkpoint_callback])
            model.load_weights(checkpoint_filepath)
            weights =  model.get_weights()
            loss = history.history['loss']

            forecasts = make_forecasts(model, batches, train, test, n_lag, n_seq)
            forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
            # forecasts_list.append(forecasts)
            actual = [row[n_lag:] for row in test]
            actual = inverse_transform(series, actual, scaler, n_test+2)
            # actual_list.append(actual)

            # saving the best parameters
            results.append({'Loss': loss,
                            'Weights': weights,
                            # 'Batches': b,
                            'Epoch': e,
                            'Neurons': n,
                            'Forecast': forecasts,
                            'Actual': actual})

    results = pd.DataFrame(results)
    # results = results.sort_values(by='Loss', ascending=False)
    results.to_csv('/Users/jessi/Onedrive/Desktop/Sem6/DS340W/Final/volume/results/hp4.csv')




    # prepare data

    # fit model
    # model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

    # make forecasts
    # forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)

    # inverse transform forecasts and test
    # forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
    # actual = [row[n_lag:] for row in test]
    # actual = inverse_transform(series, actual, scaler, n_test+2)

    # evaluate forecasts
    # evaluate_forecasts(actual, forecasts, n_lag, n_seq)

    # plot forecasts
    # plot_forecasts(series, forecasts, n_test+2)
