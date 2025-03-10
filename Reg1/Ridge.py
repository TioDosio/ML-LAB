import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold  # Import KFold for custom cross-validation


def main():
    # Load the data from .npy files
    X_train_scaled = np.load('X_train_regression1.npy')
    Y_train = np.load('y_train_regression1.npy')
    X_test = np.load('X_test_regression1.npy')

    # Check the number of samples in your dataset
    num_samples = X_train_scaled.shape[0]

    if num_samples < 2:
        print("Error: You have too few samples for cross-validation.")
        return

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_scaled)

    def function(X_trainF, Y_trainF):
        ridge_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=KFold(n_splits=12, shuffle=True, random_state=42))
        ridge_model.fit(X_trainF, Y_trainF)
        prediction = ridge_model.predict(X_trainF)
        return np.sum((prediction - Y_train) ** 2), prediction

    baseline_sse, baseline_predictions = function(X_train_scaled, Y_train)
    best_feature_idx = None
    best_r2 = -float('inf')

    for feature_idx in range(X_train_scaled.shape[1]):
        # Create a modified dataset with the current feature removed
        modified_data = np.delete(X_train_scaled, feature_idx, axis=1)

        modified_sse, modified_predictions = function(modified_data, Y_train)

        # Calculate R-squared for the best model
        r2 = r2_score(Y_train, modified_predictions)

        # Compare SSE with Ridge baseline
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
