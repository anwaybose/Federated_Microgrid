import os

import flwr as fl
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


from datetime import datetime

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load data

sns.set()
start_time = datetime.now()

data = pd.read_csv('smart_grid_stability_augmented.csv')

map1 = {'unstable': 0, 'stable': 1}
data['stabf'] = data['stabf'].replace(map1)

data = data.sample(frac=1)

data.head()

# seggregating training and test data
X = data.iloc[:, :12]
y = data.iloc[:, 13]

X_training = X.iloc[:54000, :]
y_training = y.iloc[:54000]

X_testing = X.iloc[54000:, :]
y_testing = y.iloc[54000:]

ratio_training = y_training.value_counts(normalize=True)
ratio_testing = y_testing.value_counts(normalize=True)
ratio_training, ratio_testing

X_training = X_training.values
y_training = y_training.values

X_testing = X_testing.values
y_testing = y_testing.values

# Feature scaling
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)

# Load model
# ANN initialization
model = tf.keras.models.Sequential()

# Input layer and first hidden layer
model.add(tf.keras.layers.Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

# Second hidden layer
model.add(tf.keras.layers.Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))

# Third hidden layer
model.add(tf.keras.layers.Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

# Single-node output layer
model.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# ANN compilation
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Define Flower client
class keggleClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_training, y_training, epochs=20, batch_size=32)
        return model.get_weights(), len(X_training), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_testing, y_testing)
        return loss, len(X_testing), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=keggleClient(), root_certificates=Path(".cache/certificates/ca.crt").read_bytes())