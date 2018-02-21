import time

import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

NB_FOLD=4
THRESHOLD=0.00001

#from sklearn.linear_model import LogisticRegression
# from keras_model import Neural_network

train_file = h5py.File('kaggle_lille1_2018_train_v1.save', 'r')
data = np.array(train_file['dataset_1'])
target = np.array(train_file['labels'])
print("data loaded")

l = np.array([(data.T)[i].var() for i in range(4004)])
bad_dim = np.arange(4004)[l<THRESHOLD]
# bad_dim = []
data = np.delete(data, bad_dim, 1)
print("reduced to : " + str(data.shape[1]))

scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
print("scale ok")

# t = time.time()
# pca = PCA(n_components=1000)
# pca.fit_transform(data)
# t2 = time.time()
# print("pca done in : " + str(t2 - t))

# X_train, X_test, y_train, y_test = train_test_split(data2, target,
#                                                     test_size=0.33)

kf = KFold(n_splits=NB_FOLD, shuffle=True)


# archi_list = [
#     (20),
#     (20, 20),
#     (20, 20, 20),
#     (20, 20, 20, 20),
#     (20, 20, 20, 20, 20),
#     (20, 20, 20, 20, 20, 20),
#     (20, 20, 20, 20, 20, 20, 20),
#     (20, 20, 20, 20, 20, 20, 20, 20)
# ]

archi_list = [
    (10, 10, 10, 10, 10),
    (20, 20, 20, 20, 20),
    (30, 30, 30, 30, 30),
    (40, 40, 40, 40, 40),
    (50, 50, 50, 50, 50),
    (60, 60, 60, 60, 60),
    (70, 70, 70, 70, 70),
    (80, 80, 80, 80, 80),
    (90, 90, 90, 90, 90),
 ]

for archi in archi_list:
    score_test = 0
    score_train = 0
    t = 0
    loss = 0
    i=0
    for train_index, test_index in kf.split(data):
        i+=1
        print(i)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # model = Neural_network(70)
        # model =  LogisticRegression()
        model = MLPClassifier(activation='relu',
                              hidden_layer_sizes=archi,
                              solver='sgd',
                              learning_rate_init=0.01,
                              learning_rate='adaptive',tol=0.001,
                              shuffle=True, verbose=False)
        t2 = time.time()
        model.fit(X_train, y_train)
        s_test = model.score(X_test, y_test)
        s_train = model.score(X_train, y_train)
        t3 = time.time()
        print((s_test, s_train, t3 - t2, model.loss_))
        score_test += s_test
        score_train += s_train
        t = t3 - t2
        loss += model.loss_
    print((archi, score_test/NB_FOLD, score_train/NB_FOLD, t/NB_FOLD, loss/NB_FOLD))
