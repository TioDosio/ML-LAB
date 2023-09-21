import numpy as np

Xtrain = np.load('X_train_regression1.npy')  # load the data 15x10
Ytrain = np.load('Y_train_regression1.npy')  # load the test
a = np.ones((Xtrain.T.shape[0], 1), dtype=int) # array [1] (15x1)
X = np.append(a,Xtrain.T, axis=1) # [1 X]

SSE = sum()

