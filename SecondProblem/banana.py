import numpy as np
from sklearn.linear_model import LinearRegression
import random
import warnings
import itertools
from sklearn.model_selection import KFold


def main():
    warnings.filterwarnings("ignore")

    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    index_list = list(range(len(X_train)))
    random.shuffle(index_list)

    inlier_range = np.arange(1, 1.2, 0.01)
    best_sse = float('inf')
    best_model = None
    combinations = list(itertools.combinations(range(len(X_train)), 2))

    for i in inlier_range:
        print(i)
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

            sse_inliers, inlier_model = cross_val(inlier_x, inlier_y, int(len(inlier_x) * 0.8))
            sse_outliers, outlier_model = cross_val(outlier_x, outlier_y, int(len(outlier_x) * 0.8))

            error = calculate_error(X_train, Y_train, inlier_model, outlier_model)

            if error < best_sse:
                best_inlier_range = i
                best_sse = error
                best_model = linear

    print("Best Inlier Threshold:", best_sse)
    print("Best Inlier Range:", best_inlier_range)

    col1 = best_model.predict(X_test)
    col2 = outlier_model.predict(X_test)
    np.save('col1', col1)
    np.save('col2', col2)
    np.save('output_rita', np.column_stack((col1, col2)))


def cross_val(X, Y, num_splits):
    kf = KFold(n_splits=num_splits, shuffle=True)
    sse = 0
    model = LinearRegression()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        sse += np.sum((pred - Y_test) ** 2)

    return sse, model


def calculate_error(X, Y, inlier_model, outlier_model):
    error = 0

    for j in range(len(X)):
        error_model_1 = (inlier_model.predict([X[j]]) - Y[j]) ** 2
        error_model_2 = (outlier_model.predict([X[j]]) - Y[j]) ** 2

        if error_model_1 < error_model_2:
            error += error_model_1
        else:
            error += error_model_2

    return error


if __name__ == "__main__":
    main()
