import numpy as np

Xtrain = np.load('X_train_regression1.npy')
Ytrain = np.load('y_train_regression1.npy')
Xtest = np.load('X_test_regression1.npy')

ones_column = np.ones((Xtrain.shape[0], 1))

X_design = np.hstack((ones_column, Xtrain))

aux1 = X_design.transpose()@X_design
aux2 = X_design.transpose()@Ytrain
beta = np.linalg.inv(aux1)@aux2

output = beta[0] + Xtest@beta[1:]

# SSE(beta) = ||y-^y||^2
error_SSE = (np.linalg.norm(output-Xtest@beta[1:]))**2

print(error_SSE)
