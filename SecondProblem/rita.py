import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression, Lasso, RidgeCV, Ridge
import random
import warnings
import itertools

from sklearn.model_selection import KFold


def main():
    warnings.filterwarnings("ignore")

    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    index_list = list(range(len(X_train)))
    random.shuffle(index_list)
    inlier_range = 0.73  # 0.74

    best_inlier_count = 0
    inlier_model = None
    combinations = list(itertools.combinations(range(len(X_train)), 2))
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
        inlier_count = np.sum(distances < inlier_range)  # não fazer isso, fazer com o erro, testar todas as combinações e compará-las com o sse do modelo 1 e 2
        if inlier_count > best_inlier_count:
            outlier_indices = np.where(distances >= inlier_range)[0]
            inlier_indices = np.where(distances < inlier_range)[0]
            inlier_model = linear
            best_inlier_count = inlier_count


    # fit another linear model to the outlier points
    outlier_x = X_train[outlier_indices]
    outlier_y = Y_train[outlier_indices]
    inlier_x = X_train[inlier_indices]
    inlier_y = Y_train[inlier_indices]
    outlier_model = LinearRegression()
    outlier_model.fit(outlier_x, outlier_y)

    error_model1_out, error_model2_out = cross_validation(outlier_x, outlier_y, inlier_model, outlier_model)
    error_model1_in, error_model2_in = cross_validation(inlier_x, inlier_y, inlier_model, outlier_model)

    print(f"[Model 1 Inlier] SSE outlier = {error_model1_out}, SSE inlier = {error_model1_in}")
    print(f"[Model 2 Outlier] SSE outlier = {error_model2_out}, SSE inlier = {error_model2_in}")
    model1_idx = []
    model2_idx = []
    modelpixa = []
    best_modelpixa = 100000
    for i in range(len(X_train)):
        error_model1 = (inlier_model.predict([X_train[i]]) - Y_train[i]) ** 2
        error_model2 = (outlier_model.predict([X_train[i]]) - Y_train[i]) ** 2
        if (error_model1 < error_model2):
            model1_idx.append(i)
            modelpixa.append(error_model1)
        else:
            modelpixa.append(error_model2)
            model2_idx.append(i)
    if np.sum(modelpixa) < best_modelpixa:
        best_modelpixa = np.sum(modelpixa)
    print(f"sse PIXAAAAAAAAAAA: {np.sum(best_modelpixa)}")
    print(f"mse PIXAAAAAAAAAAA: {np.sum(best_modelpixa) / len(X_train)}")
    # print("after error model1", len(model1_idx))
    # rint("after error model2", len(model2_idx))
    # see which model is better for Xtest
    model1_predict = inlier_model.predict(X_test)
    model2_predict = outlier_model.predict(X_test)


def cross_validation(X_train, Y_train, inlier_model, outlier_model):
    error_model1 = 0
    error_model2 = 0
    kf = KFold(n_splits=int(0.75 * len(X_train)))  # usar 75% dos dados para treinar

    for trainID, testID in kf.split(X_train):
        Xtrain_fold = X_train[trainID]
        Ytrain_fold = Y_train[trainID]
        Xtest_fold = X_train[testID]
        Ytest_fold = Y_train[testID]

        inlier_model.fit(Xtrain_fold, Ytrain_fold)
        error_model1 += np.sum((Ytest_fold - (inlier_model.predict(Xtest_fold)))) ** 2

        outlier_model.fit(Xtrain_fold, Ytrain_fold)
        error_model2 += np.sum((Ytest_fold - (outlier_model.predict(Xtest_fold)))) ** 2

    return error_model1 / int(0.75 * len(X_train)), error_model2 / int(0.75 * len(X_train))


if __name__ == "__main__":
    main()
