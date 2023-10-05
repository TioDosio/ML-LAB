import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression, Lasso, RidgeCV, Ridge

import warnings

global best_ransa_parameter, best_sse


def main():
    global best_ransa_parameter, best_sse
    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    warnings.filterwarnings("ignore")
    ransa_parameter = 1.09
    best_sse = 100000
    for i in range(0, 15):
        print(ransa_parameter)
        sse = ransa(X_train, Y_train, ransa_parameter)
        if sse < best_sse:
            best_sse = sse
            best_ransa_parameter = ransa_parameter
        ransa_parameter += 0.001
    print(f"Beste SSE: {best_sse}")
    print(f"Beste MSE: {best_sse / len(X_train)}")
    print(f"Best RANSAC parameter: {best_ransa_parameter}")
    print(ransa_parameter)


def ransa(X_train, Y_train, ransac_parameter):
    # Create and configure the RANSACRegressor
    ransac = RANSACRegressor(LinearRegression(), residual_threshold=ransac_parameter, random_state=0)

    # Fit the RANSAC model to your data
    ransac.fit(X_train, Y_train)

    # Get the inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    """print(f"Number of inliers: {counter(inlier_mask)}")
    print(f"Number of outliers: {counter(outlier_mask)}")"""
    inliers_X = X_train[inlier_mask]
    inliers_y = Y_train[inlier_mask]
    print("Inliers: ", inliers_X.shape[0])
    outliers_X = X_train[outlier_mask]
    outliers_y = Y_train[outlier_mask]
    print("Outliers: ", outliers_X.shape[0])
    lin = linear_model(inliers_X, inliers_y, outliers_X, outliers_y, X_train, Y_train)
    las = lasso_model(inliers_X, inliers_y, outliers_X, outliers_y, X_train, Y_train)
    rid = ridge_model(inliers_X, inliers_y, outliers_X, outliers_y, X_train, Y_train)
    if lin < las and lin < rid:
        print("Linear")
        return lin
    elif las < lin and las < rid:
        print("Lasso")
        return las
    else:
        print("Ridge")
        return rid


def linear_model(inliers_X, inliers_y, outliers_X, outliers_y, X_train, Y_train):
    modelIN = LinearRegression()
    modelOUT = LinearRegression()

    modelIN.fit(inliers_X, inliers_y)
    modelOUT.fit(outliers_X, outliers_y)
    errors = []
    for i in range(len(X_train)):
        abs_error_IN = (modelIN.predict([X_train[i]]) - Y_train[i]) ** 2
        abs_error_OUT = (modelOUT.predict([X_train[i]]) - Y_train[i]) ** 2

        if abs_error_IN < abs_error_OUT:
            errors.append(abs_error_IN)
        else:
            errors.append(abs_error_OUT)
    print("Linear: ", np.sum(errors))
    return np.sum(errors)


def lasso_model(inliers_X, inliers_y, outliers_X, outliers_y, X_train, Y_train):
    modelIN = Lasso()
    modelOUT = Lasso()

    modelIN.fit(inliers_X, inliers_y)
    modelOUT.fit(outliers_X, outliers_y)
    errors = []
    for i in range(len(X_train)):
        abs_error_IN = (modelIN.predict([X_train[i]]) - Y_train[i]) ** 2
        abs_error_OUT = (modelOUT.predict([X_train[i]]) - Y_train[i]) ** 2

        if abs_error_IN < abs_error_OUT:
            errors.append(abs_error_IN)
        else:
            errors.append(abs_error_OUT)
    print("Lasso: ", np.sum(errors))
    return np.sum(errors)


def ridge_model(inliers_X, inliers_y, outliers_X, outliers_y, X_train, Y_train):
    modelIN = RidgeCV(cv=int(inliers_X.shape[0] * 0.9))
    modelOUT = RidgeCV(cv=int(outliers_X.shape[0] * 0.9))

    modelIN.fit(inliers_X, inliers_y)
    modelOUT.fit(outliers_X, outliers_y)
    errors = []
    for i in range(len(X_train)):
        abs_error_IN = (modelIN.predict([X_train[i]]) - Y_train[i]) ** 2
        abs_error_OUT = (modelOUT.predict([X_train[i]]) - Y_train[i]) ** 2

        if abs_error_IN < abs_error_OUT:
            errors.append(abs_error_IN)
        else:
            errors.append(abs_error_OUT)
    print("Ridge: ", np.sum(errors))
    return np.sum(errors)


def counter(mask):
    Counter = 0
    for i in range(len(mask)):
        if mask[i] == 1:
            Counter += 1
    return Counter


if __name__ == "__main__":
    main()
