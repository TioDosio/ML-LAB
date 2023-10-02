import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Load the data from .npy files
global X_train
global Y_train
global X_test
global ridge_parameter
global lasso_parameter


X_train = np.load('X_train_regression2.npy')
Y_train = np.load('y_train_regression2.npy')
X_test = np.load('X_test_regression2.npy')
ridge_parameter=2.7
lasso_parameter=0.1

x = X_train
y = Y_train
aux=0
for i in range(X_train.shape[0]):
    linear_model = LinearRegression().fit(x, y)

    y_prediction = linear_model.predict(x)
        #print(y_prediction.shape, y.shape)
    if np.abs(y_prediction[i] - y[i]) > 1:
        print(f"Remove outlier_index={i},  y[i]={y[i]}")
        x_copy = np.delete(arr=x, obj=i, axis=0)
        y_copy = np.delete(arr=y, obj=i, axis=0)
        aux += 1
    else:
        i += 1

    """ y_prediction = linear_model.predict(x)
    sse_baseline = (y_prediction - y) ** 2
    outlier_index = np.argmax(np.abs(y_prediction - y)) # o indice que corresponde à maior diferença de distancia
    outlier_indices = np.where(np.abs(y_prediction - y) > 10)[0] # (todos os indices cuja distancia é maior que 10)
    #print(np.abs(y_prediction - y), np.max(np.abs(y_prediction - y)))

    plt.plot(sse_baseline, zorder = 1)
    plt.plot(outlier_index, sse_baseline[outlier_index], "o", color = 'r', zorder = 2)
    plt.show() """
        

print(f"Total outliers removed: {aux}")

ridge_1 = RidgeCV(alphas=np.arange(0.001, 1, 0.0001))
ridge_1.fit(x_copy, y_copy)
print(f"best alpha ridge1 = {ridge_1.alpha_}")

ridge_2 = RidgeCV(alphas=np.arange(0.001, 1, 0.0001))
ridge_2.fit(x, y)
print(f"best alpha ridge2 = {ridge_2.alpha_}")

