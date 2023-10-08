import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from keras.optimizers import Adam
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# import keras
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def get_data(years):
    filenames = [f'data/nasa/data_20{x}.csv' for x in years]
    df = pd.DataFrame()
    for file in filenames:
        print(file)
        df = pd.concat([df, pd.read_csv(file)], ignore_index=True)
    return df.iloc[:, 3:]


def build_model(chunk_size, n_features):

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
              input_shape=(chunk_size, n_features)))
    model.add(Conv1D(filters=64, kernel_size=3,
              activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def split_sequence(sequence, chunk_size):
    X, y = list(), list()
    print(sequence.shape)
    for i in range(len(sequence)):
        if (i % 10000 == 0):
            print(i)
        end_ix = i + chunk_size
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix, 54]
        X.append(seq_x)
        y.append(seq_y)
    return np.asarray(X).astype('float32'), np.asarray(y).astype('float32')


df = get_data(range(17, 18))
X, Y = split_sequence(df.loc[0:25000, :], 5)
Y = to_categorical(Y)
print(Y)
xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, test_size=0.2)
model = build_model(5, 55)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint.h5",
    save_weights_only=True)

model.fit(xtrain, ytrain, epochs=100, verbose=1,
          callbacks=[model_checkpoint_callback])
model.save("cnn.h5")
