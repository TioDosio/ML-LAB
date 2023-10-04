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
    # Ignore all warnings
    # warnings.filterwarnings("ignore")

    # Import the module that triggers the warning

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

    In_pred = LinearRegIN.predict(X_train)
    Out_pred = LinearRegOUT.predict(X_train)
    sseIn = np.sum((In_pred - Y_train) ** 2, axis=0)
    sseOut = np.sum((Out_pred - Y_train) ** 2, axis=0)
    print(f"SSE Inliers: {sseIn}")
    print(f"SSE Outliers: {sseOut}")
    Soma = 0
    for i in range(len(Y_train)):
        if (In_pred[i]-Y_train[i]) < (Out_pred[i]-Y_train[i]):
            Soma += In_pred[i]-Y_train[i]
        else:
            Soma += Out_pred[i]-Y_train[i]
    print(f"Soma: {Soma**2}")


if __name__ == "__main__":
    main()
