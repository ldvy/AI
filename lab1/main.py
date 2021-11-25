import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/extras/CUPTI/lib64")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/include")
os.add_dll_directory("D:/Tools/cuda/bin")

import pandas as pd
import numpy as np
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense


# ----- Importing datasets and checking for "null" entries -----
red = pd.read_csv('./winequality-red.csv', sep=';')
pd.isnull(red)
white = pd.read_csv('./winequality-white.csv', sep=';')
pd.isnull(white)

# Adding new column 'type' to the dataset, 1 is red wine and 0 is white wine
red['type'] = 1
white['type'] = 0

# Merging two wine datasets together and checking for "null"s again
wines = red.append(white, ignore_index=True)
pd.isnull(wines)

# Specifying the data we will analyze later (wine type and quality)
data = wines.iloc[:, 0:11]
# Targeting the wine type as a label
label = np.ravel(wines.type)

# Creating training and test datasets
data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.33, random_state=42)


# ----- Normalizing data -----
# >>>>> Without normalizing the dataset the end prediction tends to be 2-4% worse
# Defining the scaler for normalization
scaler = StandardScaler().fit(data_train)

# Normalizing the training dataset
data_train = scaler.transform(data_train)
# Normalizing the test dataset
data_test = scaler.transform(data_test)


# ----- Initializing the DNN model (Two-layer Perceptron) -----
model = Sequential()

# Input layer with rectified linear unit activation function
model.add(Dense(18, activation='relu', input_shape=(11,)))
# Output layer with sigmoid activation function
model.add(Dense(1, activation='linear'))

# Compiling and fitting the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

print("\nTraining")
model.fit(data_train, label_train, epochs=50, batch_size=1, verbose=1)


# ----- Testing the model -----
print("\nTesting")
test_loss, test_accuracy = model.evaluate(data_test, label_test, verbose=1)
print('\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
