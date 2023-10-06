import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')
    model = LinearRegression()
    score = cross_validation(X_train, Y_train, model)

def predict2_fn(X_train, Y_train, X_test):
    model2 = LinearRegression()
    model2.fit(X_train, Y_train)
    return model2.predict(X_test)

def cross_validation(X, y, model, k=10):
    """
    Perform k-fold cross-validation.

    Parameters:
    - X: The feature matrix (numpy array or pandas DataFrame).
    - y: The target variable (numpy array or pandas Series).
    - model: The machine learning model you want to evaluate.
    - k: The number of folds for cross-validation (default is 5).

    Returns:
    - A list of k accuracy scores (or any other evaluation metric you prefer) for each fold.
    """

    # Split the data into k folds
    n = len(X)
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)

    for i in range(k):
        # Split the data into train and test sets for this fold
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Fit the model on the training data
        model.fit(X_train, y_train)
        predict1 = model.predict(X_test)
        print(predict1)
        predict2 = predict2_fn(X_train, y_train, X_test)
        print(predict2)

    print(np.array_equal(predict1, predict2))
if __name__ == "__main__":
    main()