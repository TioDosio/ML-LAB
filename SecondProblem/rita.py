import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression, Lasso, RidgeCV, Ridge
import random
import warnings
import itertools
import sys
from sklearn.model_selection import KFold


def main():
    warnings.filterwarnings("ignore")

    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    index_list = list(range(len(X_train)))
    random.shuffle(index_list)
    # inlier_range = 0.73
    inlier_range = np.arange(0.721, 0.724, 0.0001)
    best_inlier_count = 0
    inlier_model = None
    combinations = list(itertools.combinations(range(len(X_train)), 2))
    sse_baseline = float('inf')

    for i in (inlier_range):
        print(i)
        for combo in combinations:
            # selects two points and fits them to a linear model
            index1, index2 = combo
            x1 = X_train[index1]
            x2 = X_train[index2]
            y1 = Y_train[index1]
            y2 = Y_train[index2]
            x = np.array([x1, x2])
            y = np.array([y1, y2])
            linear = LinearRegression()
            linear.fit(x, y)
            predictions = linear.predict(X_train)
            distances = np.abs(predictions - Y_train)
            inlier_count = np.sum(distances < i)
            if inlier_count > best_inlier_count:
                outlier_indices = np.where(distances >= i)[0]
                inlier_indices = np.where(distances < i)[0]
                inlier_model = linear
                best_inlier_count = inlier_count

        outlier_x = X_train[outlier_indices]
        outlier_y = Y_train[outlier_indices]
        inlier_x = X_train[inlier_indices]
        inlier_y = Y_train[inlier_indices]
        outlier_model = LinearRegression()
        outlier_model.fit(outlier_x, outlier_y)
        error = 0
        aux = []
        aux2 = []
        for j in range(len(X_train)):
            error_model_1 = (inlier_model.predict([X_train[j]]) - Y_train[j]) ** 2
            error_model_2 = (outlier_model.predict([X_train[j]]) - Y_train[j]) ** 2

            if error_model_1 < error_model_2:
                error += error_model_1
                aux.append(j)
            else:
                error += error_model_2
                aux2.append(j)

        if error < sse_baseline:
            sse_baseline = error
            best_parameter = i

    print("melhor parametro ", best_parameter)
    print("Pontos model 1", len(inlier_indices))
    print("Pontos model 2", len(outlier_indices))
    print("sse ", sse_baseline)

    inlier_model_2 = LinearRegression()
    outlier_model_2 = LinearRegression()
    inlier_model_2.fit(X_train[aux], Y_train[aux])
    outlier_model_2.fit(X_train[aux2], Y_train[aux2])
    error_modified = 0
    confused=0
    what=0
    sse_baseline = float('inf')
    for j in range(len(X_train)):
        error_model_1_modified = (inlier_model_2.predict([X_train[j]]) - Y_train[j]) ** 2
        error_model_2_modified = (outlier_model_2.predict([X_train[j]]) - Y_train[j]) ** 2

        if error_model_1_modified < error_model_2_modified:
            error_modified += error_model_1_modified
            confused+=1
        else:
            error_modified += error_model_2_modified
            what+=1

    if error_modified < sse_baseline:
        sse_baseline = error_modified
        best_parameter = i

    print("melhor parametro ", best_parameter)
    print("Pontos model 1", confused)
    print("Pontos model 2", what)
    print("sse ", sse_baseline)

    col1 = inlier_model.predict(X_test)
    col2 = outlier_model.predict(X_test)
    np.save('col1', col1)
    np.save('col2', col2)
    np.save('output_rita', np.column_stack((col1, col2)))


if __name__ == "__main__":
    main()