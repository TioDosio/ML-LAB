from statistics import mean
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV, LassoLars, \
    LassoLarsCV, LassoLarsIC, ARDRegression
from sklearn.model_selection import cross_validate, ShuffleSplit


def metrics(model, x, y, k, N):
    cv_results = cross_validate(model, x, y, cv=k, scoring=('neg_mean_squared_error', 'r2'), return_train_score=True)
    SSE = abs(cv_results['test_neg_mean_squared_error'].mean()) * N
    r2 = cv_results['train_r2'].mean()
    return SSE, r2


def metrics2(model, x, y, k, N):
    cv = ShuffleSplit(n_splits=k, random_state=0, test_size=k / 400)
    cv_results = cross_validate(model, x, y, cv=cv, scoring=('neg_mean_squared_error', 'r2'), return_train_score=True)
    SSE = abs(cv_results['test_neg_mean_squared_error'].mean()) * N
    r2 = cv_results['train_r2'].mean()
    return SSE, r2


def display(models, SSE, r2):
    for i in range(len(SSE)):
        print("\n************", models[i], "************")
        print("SSE =", SSE[i])
        print("r² =", r2[i])
    print("\n************ Conclusion ************")
    print("Best SSE is", min(SSE), "from", models[np.where(SSE == min(SSE))[0]])
    print("Best r² is", max(r2), "from", models[np.where(r2 == max(r2))[0]])


def main():
    x_train = np.load('../X_train_regression1.npy')
    y_train = np.load('../y_train_regression1.npy')
    x_test = np.load('../X_test_regression1.npy')
    print("Bananinhas das boas")
    N = x_train.shape[0]
    k = 17
    models = np.array(['LinearRegression', 'LassoCV', 'RidgeCV', 'ElasticNetCV', 'LassoLarsCV', 'LassoLarsIC'])

    SSE = []
    r2 = []
    SSE_ = []
    r2_ = []

    # Removing redundant/irrelevant features
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
            print(f"SSE lower than with all features => eliminated feature {dim}")
            null_coefs.append(0)
        else:
            null_coefs.append(1)
    x_train = np.delete(arr=x_train, obj=indexes_to_remove, axis=1)
    x_test = np.delete(arr=x_test, obj=indexes_to_remove, axis=1)

    # LinearRegression
    # regr = LinearRegression().fit(x_train, y_train)
    alphas = np.arange(0.0000001, 0.5, 0.00001)
    alphas = 0.01

    regr = LinearRegression().fit(x_train, y_train.ravel())
    SSE.append(metrics(regr, x_train, y_train.ravel(), k, N)[0])
    r2.append(metrics(regr, x_train, y_train.ravel(), k, N)[1])
    SSE_.append(metrics2(regr, x_train, y_train.ravel(), k, N)[0])
    r2_.append(metrics2(regr, x_train, y_train.ravel(), k, N)[1])

    # LassoCV
    alphas_l = np.arange(0.00001, 0.5, 0.001)
    alphas_l = [0.002524999999999993, 0.0025252000000000586, 0.00221]  # best alphas from grid search (previous line)
    lassocv = LassoCV(alphas=alphas_l, cv=k).fit(x_train, y_train.ravel())
    print('Best alpha LASSOCV: ', lassocv.alpha_)
    SSE.append(metrics(lassocv, x_train, y_train.ravel(), k, N)[0])
    r2.append(metrics(lassocv, x_train, y_train.ravel(), k, N)[1])
    SSE_.append(metrics2(lassocv, x_train, y_train.ravel(), k, N)[0])
    r2_.append(metrics2(lassocv, x_train, y_train.ravel(), k, N)[1])

    # RidgeCV
    alphas_r = np.arange(0.00001, 0.5, 0.01)
    alphas_r = [0.35301, 0.350010000]  # best alphas from grid search (previous line)
    ridgecv = RidgeCV(alphas=alphas_r, cv=k).fit(x_train, y_train.ravel())
    print('Best alpha RidgeCV: ', ridgecv.alpha_)
    SSE.append(metrics(ridgecv, x_train, y_train.ravel(), k, N)[0])
    r2.append(metrics(ridgecv, x_train, y_train.ravel(), k, N)[1])
    SSE_.append(metrics(ridgecv, x_train, y_train.ravel(), k, N)[0])
    r2_.append(metrics(ridgecv, x_train, y_train.ravel(), k, N)[1])

    # ElasticNetCV
    l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    alphas_e = np.arange(0.0001, 0.5, 0.01)
    elasticnetcv = ElasticNetCV(l1_ratio=[0.4, 0.5, 0.6], alphas=alphas_e, cv=k).fit(x_train, y_train.ravel())
    print('Best alpha ElasticNetCV: ', elasticnetcv.alpha_)
    SSE.append(metrics(elasticnetcv, x_train, y_train.ravel(), k, N)[0])
    r2.append(metrics(elasticnetcv, x_train, y_train.ravel(), k, N)[1])

    # LassoLarsCV
    lassolarscv = LassoLarsCV(cv=k, normalize=False).fit(x_train, y_train.ravel())
    # lassolarscv.coef_ = np.multiply(lassolarscv.coef_, null_coefs)
    # print('Best alpha:', lassolarscv.alpha_)
    SSE.append(metrics(lassolarscv, x_train, y_train.ravel(), k, N)[0])
    r2.append(metrics(lassolarscv, x_train, y_train.ravel(), k, N)[1])
    SSE_.append(metrics2(lassolarscv, x_train, y_train.ravel(), k, N)[0])
    r2_.append(metrics2(lassolarscv, x_train, y_train.ravel(), k, N)[1])

    # LassoLarsIC
    lassolarsic = LassoLarsIC(criterion='aic', normalize=False).fit(x_train, y_train.ravel())
    # lassolarsic.coef_ = np.multiply(lassolarsic.coef_, null_coefs)
    SSE.append(metrics(lassolarsic, x_train, y_train.ravel(), k, N)[0])
    r2.append(metrics(lassolarsic, x_train, y_train.ravel(), k, N)[1])
    SSE_.append(metrics2(lassolarsic, x_train, y_train.ravel(), k, N)[0])
    r2_.append(metrics2(lassolarsic, x_train, y_train.ravel(), k, N)[1])

    # Display+
    display(models, SSE, r2)


if __name__ == "__main__":
    main()
