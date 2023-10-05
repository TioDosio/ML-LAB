import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression, LassoCV, RidgeCV, Ridge, Lasso
from sklearn.model_selection import cross_val_score, cross_val_predict

# import warnings

# Load the data from .npy files
global X_train
global Y_train
global X_test
global ridge_parameter
global lasso_parameter


def main():
    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    """warnings.filterwarnings("ignore")"""

    # Create and configure the RANSACRegressor
    ransac = RANSACRegressor(LinearRegression(), residual_threshold=1.0, random_state=0)

    # Fit the RANSAC model to your data
    ransac.fit(X_train, Y_train)

    # Get the inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

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
        abs_error_IN = np.abs(LinearRegIN.predict([X_train[i]]) - Y_train[i])
        abs_error_OUT = np.abs(LinearRegOUT.predict([X_train[i]]) - Y_train[i])

        if abs_error_IN < abs_error_OUT:
            Banana.append(abs_error_IN)
            c0 += 1
        else:
            Banana.append(abs_error_OUT)
            c1 += 1

    print(c0, c1)
    print(np.mean(Banana))


if __name__ == "__main__":
    main()
