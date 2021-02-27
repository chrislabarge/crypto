import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import os
import time

import numpy as np
import pandas as pd
import random

np.random.seed(314)
tf.compat.v1.set_random_seed(314)
random.seed(314)

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

"""
- The ticker argument is the ticker we want to load, for instance, you can
use TSLA for Tesla stock market, AAPL for Apple and so on. It can also be a
pandas Dataframe with the condition it includes the columns in feature_columns
as well as date as index.

- n_steps integer indicates the historical sequence length we want to use, some
  people call it the window size, recall that we are going to use a recurrent
  neural network, we need to feed in to the network a sequence data, choosing
  50 means that we will use 50 days of stock prices to predict the next lookup
  time step.

- scale is a boolean variable that indicates whether to scale prices from 0 to
  1, we will set this to True as scaling high values from 0 to 1 will help the
  neural network to learn much faster and more effectively.

- lookup_step is the future lookup step to predict, the default is set to 1
  (e.g next day). 15 means the next 15 days, and so on.

- split_by_date is a boolean which indicates whether we split our training and
  testing sets by date, setting it to False means we randomly split the data
  into training and testing using sklearn's train_test_split() function. If
  it's True (the default), we split the data in date order.  """

def load_data(ticker, n_steps=50, scale=True,
              shuffle=True, lookup_step=1,
              split_by_date=True, test_size=0.2,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low', ]):

    """ Loads data from Yahoo Finance source, as well as scaling, shuffling,
    normalizing and splitting.  Params: ticker (str/pd.DataFrame): the ticker
    you want to load, examples include AAPL, TESL, etc.  n_steps (int): the
    historical sequence length (i.e window size) used to predict, default is 50
    scale (bool): whether to scale prices from 0 to 1, default is True shuffle
    (bool): whether to shuffle the dataset (both training & testing), default
    is True lookup_step (int): the future lookup step to predict, default is 1
    (e.g next day) split_by_date (bool): whether we split the dataset into
    training/testing by date, setting it to False will split datasets in a
    random way test_size (float): ratio for test data, default is 0.2 (20%
    testing data) feature_columns (list): the list of features to use to feed
    into the model, default is everything grabbed from yahoo_fin """

    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticket, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # this will contain all the elements we want to return from this function
    result = {}

    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}'  does not exist in the dataframe."

    # add data as a column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}

        #scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        #add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by difting bt `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with
    # `lookup_step` sequence for instance, if n_steps=50 and lookup_step=10,
    # last_sequence should be of 60 (that is 50+10) length this last_sequence
    # will be used to predict future stock prices that are not available in
    # the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)

    # add to result
    result['last_sequence'] = last_sequence

    X, y = [], []

    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays

    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        #split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]

    # retrieve the test features from the original datafram
    result["test_df"] = result["df"].loc[dates]

    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]

    # remove dates from the training/tesing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

"""
Again, this function is flexible too, you can change the number of layers,
dropout rate, the RNN cell, loss and the optimizer used to compile the model.

The above function constructs a RNN that has a dense layer as output layer with
1 neuron, this model requires a sequence of features of sequence_length (in
this case, we will pass 50 or 100) consecutive time steps (which are days in
this dataset) and outputs a single value which indicates the price of the next
time step.

It also accepts n_features as an argument, which is the number of features we
will pass on each sequence, in our case, we'll pass adjclose, open, high, low
and volume columns (i.e 5 features).

You can tweak the default parameters as you wish, n_layers is the number of RNN
layers you want to stack, dropout is the dropout rate after each RNN layer,
units are the number of RNN cell units (whether its LSTM, SimpleRNN or GRU),
bidirectional is a boolean that indicates whether to use bidirectional RNNs,
experiment with those!
"""

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            #last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

# Window size of the sequence length

N_STEPS = 50

# Lookup step, 1 is the next day

LOOKUP_STEP = 15

# whether to scale feature columns & output price as well

SCALE = True
scale_str = f"sc-{int(SCALE)}"

# whether to shuffle the dataset

SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# whether to split the training/testing set by date

SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# test ratio size, 0.2 is 20%
# The testing set rate. For instance 0.2 means 20% of the total dataset.
TEST_SIZE=0.2

# feature to use
# The features we gonna use to predict the next price value.
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

#date now

date_now = time.strftime("%Y-%m-%d")

### MODEL PARAMETERS

#Number of RNN layers to use.
N_LAYERS = 2

# LSTM cell
#RNN cell to use, default is LSTM.
CELL = LSTM

# 256 LSTM neurons
# Number of cell units
UNITS = 256

# 40% dropout
# The dropout rate is the probability of not training a given node in a layer,
# where 0.0 means no dropout at all. This type of regularization can help the
# model to not overfit on our training data.
DROPOUT = 0.4

# whether to use bidirectional RNN's
# Whether to use bidirectional recurrent neural networks.
BIDIRECTIONAL = False

### TRAINING PARAMETERS
# mean absolute error loss
# LOSS = "mae"
# huber loss
# Loss function to use for this regression problem, we're using Huber loss, you can use mean absolute error (mae) or mean squared error (mse) as well.
LOSS = "huber_loss"
# Optimization algorithm to use, defaulting to Adam.
OPTIMIZER = "adam"

# The number of data samples to use on each training iteration.
BATCH_SIZE = 64
# The number of times that the learning algorithm will pass through the entire
# training dataset, we used 500 here, but try to increase it further more.
EPOCHS = 500

# Amazon stock market
ticker = "AMZN"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"

if BIDIRECTIONAL:
    model_name += "-b"

if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

#save the dataframe
data["df"].to_csv(ticker_data_filename)

#contruct the model

model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                     dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

#some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

# train the model and sace the weights whenever we see a new option model using
# ModelCheckpoint

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

