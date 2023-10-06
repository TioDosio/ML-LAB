import numpy as np
from sklearn.linear_model import LinearRegression
import random
import warnings
import itertools
from sklearn.model_selection import KFold


def main():
    global best_inlier_range
    warnings.filterwarnings("ignore")

    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    inlier_range = np.arange(0.2, 2, 0.1)
    best_sse = float('inf')
    combinations = list(itertools.combinations(range(len(X_train)), 2))

    for i in inlier_range:
        print(i)
        a = 0
        for combo in combinations:
            index1, index2 = combo
            x1 = X_train[index1]
            x2 = X_train[index2]
            y1 = Y_train[index1]
            y2 = Y_train[index2]
            x = np.array([x1, x2])
            y = np.array([y1, y2])
            linear = LinearRegression()
            linear.fit(x, y)
            predictions = linear.predict(X_train)
            distances = np.abs(predictions - Y_train)
            inlier_indices = np.where(distances < i)[0]
            outlier_indices = np.where(distances >= i)[0]

            inlier_x = X_train[inlier_indices]
            inlier_y = Y_train[inlier_indices]
            outlier_x = X_train[outlier_indices]
            outlier_y = Y_train[outlier_indices]

            if len(inlier_x) < 30 or len(outlier_x) < 30:
                continue
            inlier_model = cross_val(inlier_x, inlier_y)
            outlier_model = cross_val(outlier_x, outlier_y)

            sse = calculate_error(X_train, Y_train, inlier_model, outlier_model)

            if sse < best_sse:
                best_inlier_range = i
                best_sse = sse
                best_model_in = inlier_model
                best_model_out = outlier_model
            print("a: ", a)
            a += 1
    print("Best Inlier Threshold:", best_sse)
    print("Best Inlier Range:", best_inlier_range)


def cross_val(X, Y):
    kf = KFold(n_splits=int(0.70 * len(X)))
    model = LinearRegression()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)

    return model


def calculate_error(X, Y, inlier_model, outlier_model):
    sse = 0
    for j in range(len(X)):
        error_model_1 = (inlier_model.predict([X[j]]) - Y[j]) ** 2
        error_model_2 = (outlier_model.predict([X[j]]) - Y[j]) ** 2

        if error_model_1 < error_model_2:
            sse += error_model_1
        else:
            sse += error_model_2

    return sse


if __name__ == "__main__":
    main()
