import numpy as np

Xtrain = np.load('X_train_regression1.npy')
Ytrain = np.load('y_train_regression1.npy')
Xtest = np.load('X_test_regression1.npy')

ones_column = np.ones((Xtrain.shape[0], 1))

X_design = np.hstack((ones_column, Xtrain))

aux1 = X_design.transpose()@X_design
aux2 = X_design.transpose()@Ytrain
beta = np.linalg.inv(aux1)@aux2

print(beta)

output = beta[0] + Xtrain@beta[1:]

# SSE(beta) = ||y-X*beta||^2
error_SSE = (np.linalg.norm(Ytrain-X_design@beta))**2
print("SSE ", error_SSE)
avg_y = np.mean(Ytrain)
SStot_aux = (Ytrain - avg_y) ** 2
SStot = np.sum(SStot_aux)
print("SStot ", SStot)
r_squared = 1 - (error_SSE / SStot)
print("r^2 ", r_squared)
