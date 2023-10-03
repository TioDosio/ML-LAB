import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, Ridge, LassoCV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
# Load the data from .npy files
global X_train
global Y_train
global X_test
global ridge_parameter
global lasso_parameter


def main():
    warnings.filterwarnings("ignore")
    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')
    
    k = 2
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=20)
    cluster_labels = kmeans.fit_predict(X_train)

    # Split the data into two variables based on cluster labels
    x_c0 = X_train[cluster_labels == 0]
    x_c1 = X_train[cluster_labels == 1]
    
    y_c0 = Y_train[cluster_labels == 0]
    y_c1 = Y_train[cluster_labels == 1]
    
    # cross validation
    print(f"Cluster 0 shape: {x_c0.shape}, Cluster 1 shape: {x_c1.shape}")
    fold_num_c0 = int(x_c0.shape[0]*0.75)
    fold_num_c1 = int(x_c1.shape[0]*0.75)
    linear_c0, lasso_c0, ridge_c0, best_features_c0 = cross_validation(x_c0, y_c0, fold_num_c0,0)
    linear_c1, lasso_c1, ridge_c1,best_features_c1 = cross_validation(x_c1, y_c1, fold_num_c1,0)
    print(f"[Linear] SSE_C0 = {linear_c0}, SSE_C1 = {linear_c1} ")
    print(f"[Lasso] SSE_C0 = {lasso_c0}, SSE_C1 = {lasso_c1} ")
    print(f"[Ridge] SSE_C0 = {ridge_c0}, SSE_C1 = {ridge_c1} ") 
    print("\n--- Feature Removal ---\n")
    print(f"Best features C0 = {best_features_c0}, C1= {best_features_c1}")

    feature_to_remove_c0 = np.argmax(best_features_c0)
    feature_to_remove_c1 = np.argmax(best_features_c1)

    linear_c0_fr, lasso_c0_fr, ridge_c0_fr, best_features_c0_fr = cross_validation(np.delete(x_c0,obj=feature_to_remove_c0, axis=1), y_c0, fold_num_c0,1)
    linear_c1_fr, lasso_c1_fr, ridge_c1_fr,best_features_c1_fr = cross_validation(np.delete(x_c1,obj=feature_to_remove_c1, axis=1), y_c1, fold_num_c1,1)

    print("Best model (without feature removal) is:")
    mini0 = min(linear_c0, lasso_c0, ridge_c0)
    if mini0 == linear_c0:
        print("C0 Linear: ", linear_c0)
    elif mini0 == lasso_c0:
        print("C0 Lasso: ", lasso_c0)
    else: 
        print("C0 Ridge: ", ridge_c0)
        
    mini1 = min(linear_c1, lasso_c1, ridge_c1)
    if mini1 == linear_c1:
        print("C1 Linear: ", linear_c1)
    elif mini1 == lasso_c1:
        print("C1 Lasso: ", lasso_c1)
    else: 
        print("C1 Ridge: ", ridge_c1)

    print(f"[Ridge] C0 - After removing feature {feature_to_remove_c0}, SSE = {ridge_c0_fr}")
    print(f"[Ridge] C1 - After removing feature {feature_to_remove_c1}, SSE = {ridge_c1_fr}")



def cross_validation(X_train, Y_train, fold_num, flag):
    kf = KFold(n_splits=fold_num)
    sse_linear=0
    sse_lasso=0
    sse_ridge=0
    lasso_param, ridge_param = find_best_param(X_train, Y_train, fold_num)
    print(f"[BEST alpha] lasso = {lasso_param}, ridge = {ridge_param}")
    best_features=[0]*X_train.shape[1]
    features=[]
    for trainID, testID in kf.split(X_train):
        Xtrain_fold = X_train[trainID]
        Ytrain_fold = Y_train[trainID]
        Xtest_fold = X_train[testID]
        Ytest_fold = Y_train[testID]

        linear_model = LinearRegression().fit(Xtrain_fold, Ytrain_fold)
        y_linear_pred = linear_model.predict(Xtest_fold)
        sse_linear += calculate_SSE(Ytest_fold, y_linear_pred)

        lasso_model = Lasso(alpha=lasso_param).fit(Xtrain_fold, Ytrain_fold)
        y_lasso_pred = lasso_model.predict(Xtest_fold)
        sse_lasso += calculate_SSE(Ytest_fold, y_lasso_pred)

        ridge_model = Ridge(alpha=ridge_param).fit(Xtrain_fold,Ytrain_fold)
        y_ridge_pred = ridge_model.predict(Xtest_fold)
        sse_ridge += calculate_SSE(Ytest_fold, y_ridge_pred)
        if flag==0:
            features = feature_removal(Xtrain_fold, Ytrain_fold, Xtest_fold, Ytest_fold, y_ridge_pred, ridge_param, best_features)
    return (sse_linear/fold_num), (sse_lasso/fold_num), (sse_ridge/fold_num), features

def find_best_param(X_train, Y_train, fold_num):
    alphas = np.arange(0.1, 20, 0.1)

    lasso_cv = LassoCV(alphas=alphas, cv=fold_num)
    lasso_cv.fit(X_train, Y_train)

    ridge_cv = RidgeCV(alphas=alphas, cv=fold_num)
    ridge_cv.fit(X_train, Y_train)

    return lasso_cv.alpha_, ridge_cv.alpha_


def calculate_SSE(y, y_predicted):
    return np.sum((y-y_predicted)**2)

def feature_removal(Xtrain_fold, Ytrain_fold, Xtest_fold, Ytest_fold, Y_predict, ridge_parameter, best_features):
    # RIDGE FEATURE REMOVAL
    best_feature_idx = -1
    for feature_idx in range(Xtrain_fold.shape[1]):
        # Create a modified Xtrain dataset with the current feature removed
        modified_data = np.delete(Xtrain_fold, feature_idx, axis=1)
        modified_testdata = np.delete(Xtest_fold, feature_idx, axis=1)

        baseline_SSE = calculate_SSE(Ytest_fold, Y_predict)
        ridge_model_test_fm = Ridge(alpha=ridge_parameter)
        ridge_model_test_fm.fit(modified_data, Ytrain_fold)
        ridge_model_test_predict_modified = ridge_model_test_fm.predict(
            modified_testdata)
        modified_sse = calculate_SSE(
            Ytest_fold, ridge_model_test_predict_modified)
        if modified_sse < baseline_SSE:  # Compare SSE with baseline
            #print(f"Removing feature {feature_idx} improved SSE: {modified_sse}")
            baseline_SSE = modified_sse
            best_feature_idx = feature_idx

    if best_feature_idx != -1:
        #print(f"Best feature to remove: {best_feature_idx} ")
        best_features[best_feature_idx] += 1
    else:
        pass
        #print("No improvement found.")
    return best_features


if __name__ == "__main__":
    main()