import numpy as np
from sklearn.linear_model import LinearRegression
import random
import itertools
from sklearn.model_selection import KFold


def main():
    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    inlier_range = np.arange(0.2, 2, 0.1)
    combinations = list(itertools.combinations(range(len(X_train)), 2))
    sse_baseline = float('inf')
    b = 0
    a = 0
    c = 0
    for i in inlier_range:
        print("---------------------------------------------", i)
        f = 1
        for combo in combinations:
            # selects two points and fits them to a linear model
            index1, index2 = combo
            x1 = X_train[index1]
            x2 = X_train[index2]
            y1 = Y_train[index1]
            y2 = Y_train[index2]
            x = np.array([x1, x2])
            y = np.array([y1, y2])
            inlinear = LinearRegression()
            inlinear.fit(x, y)
            predictions = inlinear.predict(X_train)
            distances = np.abs(predictions - Y_train)
            outlier_indices = np.where(distances >= i)[0]
            inlier_indices = np.where(distances < i)[0]

            outliers_x = X_train[outlier_indices]
            outliers_y = Y_train[outlier_indices]
            outlier_model = LinearRegression()

            inliers_x = X_train[inlier_indices]
            inliers_y = Y_train[inlier_indices]
            inlier_model = LinearRegression()

            if f != c:
                c = f
                print("print 1: ", f)
            if len(inliers_x) < 30 or len(outliers_x) < 30:
                continue
            sse_inliers, inlier_model_test = cross_val(inliers_x, inliers_y, inlier_model)
            sse_outliers, outlier_model_test = cross_val(outliers_x, outliers_y, outlier_model)
            if f != a:
                a = f
                print("Print 2: ", f)

            error = 0
            aux1 = 0
            aux2 = 0
            for j in range(len(X_train)):
                error_model_1 = (inlier_model_test.predict([X_train[j]]) - Y_train[j]) ** 2
                error_model_2 = (outlier_model_test.predict([X_train[j]]) - Y_train[j]) ** 2

                if error_model_1 < error_model_2:
                    error += error_model_1
                    aux1 += 1
                else:
                    error += error_model_2
                    aux2 += 1
                if f != b:
                    b = f
                    print("Print 3: ", f)
            if error < sse_baseline:
                sse_baseline = error
                best_parameter = i
                best_inlier_model = inlier_model_test
            f += 1
            print("SSER: ", sse_baseline)


    print(best_parameter)
    print(sse_baseline)


def cross_val(X, Y, model):
    kf = KFold(n_splits=int(0.70 * len(X)))
    kf.get_n_splits(X)
    sse = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        sse += np.sum((pred - Y_test) ** 2)
    return sse, model


if __name__ == "__main__":
    main()
