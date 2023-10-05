import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

# import warnings

global best_ransa_parameter


def main():
    global best_ransa_parameter, best_sse
    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    """warnings.filterwarnings("ignore")"""
    ransa_parameter = 0.1
    best_sse = 100000
    for i in range(0, 2000):
        print(ransa_parameter)
        sse = ransa(X_train, Y_train, ransa_parameter)
        if sse < best_sse:
            best_sse = sse
            best_ransa_parameter = ransa_parameter
        ransa_parameter += 0.001
    print(f"Beste SSE: {best_sse}")
    print(f"Beste MSE: {best_sse/len(X_train)}")
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

    outliers_X = X_train[outlier_mask]
    outliers_y = Y_train[outlier_mask]

    LinearRegIN = LinearRegression()
    LinearRegOUT = LinearRegression()
    LinearRegIN.fit(inliers_X, inliers_y)
    LinearRegOUT.fit(outliers_X, outliers_y)
    Banana = []
    c0 = 0
    c1 = 0
    for i in range(len(X_train)):
        abs_error_IN = (LinearRegIN.predict([X_train[i]]) - Y_train[i])**2
        abs_error_OUT = (LinearRegOUT.predict([X_train[i]]) - Y_train[i])**2

        if abs_error_IN < abs_error_OUT:
            Banana.append(abs_error_IN)
            c0 += 1
        else:
            Banana.append(abs_error_OUT)
            c1 += 1

    return np.sum(Banana)


def counter(mask):
    Counter = 0
    for i in range(len(mask)):
        if mask[i] == 1:
            Counter += 1
    return Counter


if __name__ == "__main__":
    main()
