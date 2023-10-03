import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import cross_validate


def metrics(model, x, y, k, N):
    cv_results = cross_validate(model, x, y, cv=k, scoring='neg_mean_squared_error', return_train_score=True)
    SSE = abs(cv_results['test_score'].mean()) * N
    print(cv_results)
    return SSE


def main():
    x_train = np.load('Xtrain_Regression1.npy')
    y_train = np.load('Ytrain_Regression1.npy')
    x_test = np.load('Xtest_Regression1.npy')

    N = x_train.shape[0]
    k = 17

    # Removing redundant/irrelevant features, i.e, the ones that, if removed, make the
    # sse score go down
    indexes_to_remove = []
    null_coefs = []
    regr = LinearRegression().fit(x_train, y_train)
    SSE_baseline = metrics(regr, x_train, y_train, k, N)
    for dim in range(x_train.shape[1]):
        x_new = np.delete(arr=x_train, obj=dim, axis=1)
        regr_new = LinearRegression().fit(x_new, y_train)
        SSE_new = metrics(regr_new, x_new, y_train, k, N)
        if SSE_new < SSE_baseline:
            indexes_to_remove.append(dim)
            print(f"Eliminated feature {dim}")
            null_coefs.append(0)
        else:
            null_coefs.append(1)
    x_train = np.delete(arr=x_train, obj=indexes_to_remove, axis=1)
    x_test = np.delete(arr=x_test, obj=indexes_to_remove, axis=1)

    # ElasticNetCV
    alphas_e = np.arange(0.0001, 1, 0.0001)
    l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elasticnetcv = ElasticNetCV(l1_ratio=0.4, alphas=alphas_e, cv=k).fit(x_train, y_train.ravel())
    # elasticnetcv.coef_ = np.multiply(elasticnetcv.coef_, null_coefs)
    print("Coefficients: ", elasticnetcv.coef_)
    print(f"Best α = {elasticnetcv.alpha_}")
    SSE = metrics(elasticnetcv, x_train, y_train.ravel(), k, N)
    print(f"SSE = {SSE}")

    y_pred = elasticnetcv.predict(x_test).reshape(-1, 1)
    np.save('Ytest_Regression1', y_pred)


if __name__ == "__main__":
    main()
