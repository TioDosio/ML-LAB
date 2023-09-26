import numpy as np
from sklearn.linear_model import LassoCV
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

    # Define Lasso model with cross-validation
    lasso_model = LassoCV(cv=5, max_iter=10000)
    lasso_model.fit(X_train_scaled, Y_train)
    baseline_predictions = lasso_model.predict(X_train_scaled)
    baseline_sse = np.sum((baseline_predictions - Y_train) ** 2)

    best_feature_idx = None
    best_r2 = -float('inf')

    for feature_idx in range(X_train_scaled.shape[1]):
        # Create a modified dataset with the current feature removed
        modified_data = np.delete(X_train_scaled, feature_idx, axis=1)

        # Train a Lasso model on the modified dataset with cross-validation
        lasso_model = LassoCV(cv=12)  # You can adjust cv as needed
        lasso_model.fit(modified_data, Y_train)
        modified_predictions = lasso_model.predict(modified_data)
        modified_sse = np.sum((modified_predictions - Y_train) ** 2)

        # Calculate R-squared for the best model
        r2 = r2_score(Y_train, modified_predictions)

        # Compare SSE with Lasso baseline
        if modified_sse < baseline_sse:
            print(f"Removing feature {feature_idx} improved SSE: {modified_sse}")
            if r2 > best_r2:
                best_r2 = r2
                best_feature_idx = feature_idx
        else:
            print(f"Removing feature {feature_idx} worsened SSE: {modified_sse}")

    if best_feature_idx is not None:
        print(f"Best feature to remove: {best_feature_idx} with R-squared: {best_r2}")
    else:
        print("No improvement found.")


if __name__ == "__main__":
    main()
