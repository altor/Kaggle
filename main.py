import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import core

NB_FOLD=4
THRESHOLD=0.00001

train_file = h5py.File('kaggle_lille1_2018_train_v1.save', 'r')
data = np.array(train_file['dataset_1'])
target = np.array(train_file['labels'])


l = np.array([(data.T)[i].var() for i in range(4004)])
bad_dim = np.arange(4004)[l>THRESHOLD]
# bad_dim = []
data = np.delete(data, bad_dim, 1)
print("reduced to : " + str(data.shape[1]))

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
print("scale ok")


model = MLPClassifier(activation='relu',
                      hidden_layer_sizes=(20, 20, 20, 20),
                      solver='sgd',
                      learning_rate_init=0.01,
                      learning_rate='adaptive',tol=0.001,
                      shuffle=True)
model.fit(data, target)

# model.fit([data[0], data[9000]], [target[0], target[9000]])

core.gen_solution(model, scaler, bad_dim, model.score(data, target))
# core.gen_solution(model, None, None, model.score(data, target))
