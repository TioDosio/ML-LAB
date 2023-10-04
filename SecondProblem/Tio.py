import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression, LassoCV, RidgeCV, Ridge, Lasso
from sklearn.model_selection import cross_val_score
import warnings

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
    print(X_test.shape)
    # Ignore all warnings
    warnings.filterwarnings("ignore")

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

    print(int(inliers_X.shape[0] * 0.9))
    # Perform cross-validation
    linear_scoresIN = cross_val_score(LinearRegression(), inliers_X, inliers_y, cv=int(inliers_X.shape[0] * 0.9))
    print(f"LINEAR cross-validation scores: {linear_scoresIN}")
    print(max(linear_scoresIN))
    linear_scoresOUT = cross_val_score(LinearRegression(), outliers_X, outliers_y, cv=int(outliers_X.shape[0] * 0.9))
    print(f"LINEAR cross-validation scores: {linear_scoresOUT}")
    print(max(linear_scoresOUT))
    """lasso_scores = cross_val_score(LassoCV(), inliers_X, inliers_y, cv=int(inliers_X.shape[0] * 0.9))
    print(f"LASSO cross-validation scores: {lasso_scores}")

    ridge_scores = cross_val_score(RidgeCV(), inliers_X, inliers_y, cv=int(inliers_X.shape[0] * 0.9))
    print(f"RIDGE cross-validation scores: {ridge_scores}")"""
    predictions_IN = ransac.predict(X_train)
    sse_IN = np.sum((predictions_IN[inlier_mask] - inliers_y) ** 2)

    predictions_OUT = ransac.predict(X_test)
    sse_OUT = np.sum((predictions_OUT[outliers_X] - outliers_y) ** 2)
    print(f"SSE for inliers predictions on X_test: {sse_IN}")
    print(f"SSE for outliers predictions on X_test: {sse_OUT}")


if __name__ == "__main__":
    main()
