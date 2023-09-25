import numpy as np

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt


def main():

    Xtrain = np.load('X_train_regression1.npy')
    Ytrain = np.load('y_train_regression1.npy')
    Xtest = np.load('X_test_regression1.npy')

    ones_column = np.ones((Xtrain.shape[0], 1))

    X_design = np.hstack((ones_column, Xtrain))

    aux1 = X_design.transpose()@X_design
    aux2 = X_design.transpose()@Ytrain
    beta = np.linalg.inv(aux1)@aux2

    outcome = beta[0] + Xtrain@beta[1:]

    # SSE(beta) = ||y-X*beta||^2
    error_SSE = (np.linalg.norm(Ytrain-X_design@beta))**2

    avg_y = np.mean(Ytrain)
    SStot_aux = (Ytrain - avg_y) ** 2
    SStot = np.sum(SStot_aux)

    r_squared = 1 - (error_SSE / SStot)

    lambda_var = 1

    k = 6
    kf = KFold(n_splits=k)
    r2 = []
    r2_ridge = []
    r2_lasso = []
    for trainID, testID in kf.split(Xtrain):
        print(trainID)
        Xtrain_fold = Xtrain[trainID]
        Ytrain_fold = Ytrain[trainID]
        Xtest_fold = Xtrain[testID]
        Ytest_fold = Ytrain[testID]
        #Xtrain_fold = feature_removal(Xtrain_fold)

        ridge_regression_model = Ridge(alpha=lambda_var)
        ridge_regression_model.fit(Xtrain_fold, Ytrain_fold)
        beta_ridge = (ridge_regression_model.coef_).transpose()

        lasso_regression_model = Lasso(alpha=lambda_var)
        lasso_regression_model.fit(Xtrain_fold, Ytrain_fold)
        beta_lasso = lasso_regression_model.coef_

        Ytest_predicted = beta[0] + Xtest_fold@beta[1:]
        Ytest_predicted_ridge = 0 + Xtest_fold@beta_ridge
        Ytest_predicted_lasso = 0 + Xtest_fold@beta_lasso

        r2.append(r2_score(Ytest_fold, Ytest_predicted))
        r2_ridge.append(r2_score(Ytest_fold, Ytest_predicted_ridge))
        r2_lasso.append(r2_score(Ytest_fold, Ytest_predicted_lasso))

    squared_errors = (Ytest_fold-Ytest_predicted)**2
    squared_errors_ridge = (Ytest_fold-Ytest_predicted_ridge)**2
    squared_errors_lasso = (Ytest_fold-Ytest_predicted_lasso)**2
    """ print(f"SSE = {np.sum(squared_errors)}")
    print(f"r^2 médio = {np.mean(r2)} ; cross validation com {k} splits")
    print(f"r^2 de cada fold: {r2}")
    print("=============")
    print(f"USANDO RIDGE COM LAMBDA = {lambda_var}")
    print(f"SSE = {np.sum(squared_errors_ridge)}")
    print(f"r^2 médio = {np.mean(r2_ridge)} ; cross validation com {k} splits")
    print(f"r^2 de cada fold: {r2_ridge}")
    print("=============")
    print(f"USANDO LASSO COM LAMBDA = {lambda_var}")
    print(f"SSE = {np.sum(squared_errors_lasso)}")
    print(f"r^2 médio = {np.mean(r2_lasso)} ; cross validation com {k} splits")
    print(f"r^2 de cada fold: {r2_lasso}") """


def feature_removal(Xtrain_fold):

    for i in range(Xtrain_fold.shape[1]):
        new_fold = np.delete(arr=Xtrain_fold, obj=i, axis=1)
        new_y_output
        print(new_fold)
        break

    return new_fold


if __name__ == "__main__":
    main()
