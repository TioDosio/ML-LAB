import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def main():
    # Load the data from .npy files
    X_train = np.load('X_train_regression1.npy')
    Y_train = np.load('y_train_regression1.npy')
    X_test = np.load('X_test_regression1.npy')

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Flatten the target matrix if necessary
    Y_train = Y_train.flatten()

    def lasso_fn(X_trainF, Y_trainF):
        lasso_model = LassoCV(cv=5, max_iter=10000)
        lasso_model.fit(X_trainF, Y_trainF)
        prediction = lasso_model.predict(X_trainF)
        return np.sum((prediction - Y_train) ** 2), prediction

    def ridge_fn(X_trainF, Y_trainF):
        ridge_model = RidgeCV(cv=5)
        ridge_model.fit(X_trainF, Y_trainF)
        prediction = ridge_model.predict(X_trainF)
        return np.sum((prediction - Y_train) ** 2), prediction

    baseline_SSE, baseline_predictions = lasso_fn(X_train_scaled, Y_train)
    baseline_SSE_RIDGE, baseline_predictions_RIDGE = ridge_fn(X_train_scaled, Y_train)

    best_feature_idx = None  # The index of the feature to remove
    best_r2 = -float('inf')
    print("BEFORE SSE=", baseline_SSE_RIDGE)
    for feature_idx in range(X_train_scaled.shape[1]):
        # Create a modified dataset with the current feature removed
        modified_data = np.delete(X_train_scaled, feature_idx, axis=1)
        # modified_sse, modified_predictions = lasso_fn(modified_data, Y_train)
        modified_sse, modified_predictions = ridge_fn(modified_data, Y_train)

        # r2 = r2_score(Y_train, modified_predictions)  # Calculate R-squared for the best model

        if modified_sse < baseline_SSE_RIDGE:  # Compare SSE with Lasso baseline
            print(f"Removing feature {feature_idx} improved SSE: {modified_sse}")
            baseline_SSE = modified_sse
            best_feature_idx = feature_idx
        else:
            print(f"Removing feature {feature_idx} worsened SSE: {modified_sse}")

    if best_feature_idx is not None:
        print(f"Best feature to remove: {best_feature_idx} ")
    else:
        print("No improvement found.")

    Xtrain_better = np.delete(X_train_scaled, best_feature_idx, axis=1)
    baseline_SSE_RIDGE_after, baseline_predictions_RIDGE_after = ridge_fn(Xtrain_better, Y_train)
    print("AFTER SSE = ", baseline_SSE_RIDGE_after)


if __name__ == "__main__":
    main()
