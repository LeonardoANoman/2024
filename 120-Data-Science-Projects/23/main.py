import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


import tensorflow as tf
from tensorflow import keras
from keras import layers

data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
print(data)

def splitSequence(seq, n_steps):
    
    X = []
    y = []
    
    for i in range(len(seq)):
        lastIndex = i + n_steps
        
        if lastIndex > len(seq) - 1:
            break
            
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]
        
        X.append(seq_X)
        y.append(seq_y)
        pass
    X = np.array(X)
    y = np.array(y)
    
    return X,y

n_steps = 7
X, y = splitSequence(data, n_steps)
print(X)
print(y)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(X[:2])

model = tf.keras.Sequential()
model.add(layers.LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(layers.Dense(1))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError())

print("Fit model on training data")
history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=(X_test, y_test), verbose=1
)

test_data = np.array([90, 100, 110, 120, 130, 140, 150])
test_data = test_data.reshape((1, n_steps, n_features))


predictNextNumber = model.predict(test_data, verbose=1)
print(predictNextNumber)

#