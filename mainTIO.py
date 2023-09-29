import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, Ridge, Lasso
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold

# Load the data from .npy files
X_train = np.load('X_train_regression1.npy')
Y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')
fold_num = 15 # number of splits for cross-validation

def main():
    warnings.filterwarnings("ignore")

    baseline_SSE_lasso, baseline_predictions_lasso = lasso_fn(X_train, Y_train)
    baseline_SSE_ridge, baseline_predictions_ridge = ridge_fn(X_train, Y_train)
    print(f"Before RIDGE feature removal, SSE={baseline_SSE_ridge:.2f}")

    Xtrain_better = feature_removal(X_train, baseline_SSE_ridge)
    baseline_SSE_ridge_after, baseline_predictions_RIDGE_after = ridge_fn(Xtrain_better, Y_train)
    print(f"After RIDGE feature removal, SSE = {baseline_SSE_ridge_after:.2f}")
    print(f"SSE decreased {((baseline_SSE_ridge-baseline_SSE_ridge_after)/baseline_SSE_ridge)*100:.2f} %")
    plot_alpha_SSE()     # check alpha/SSE evolution with pyplot

    
def calculate_SSE(y, y_predicted):
    """
    Calculates the Sum of Squared Errors (SSE)
    :y: observed y values
    :y_predicted: predicted y values

    :return: SSE
    """ 
    return np.sum((y-y_predicted)**2)

def lasso_fn(X_train, Y_train):
    """
    Trains and predics a Lasso regression model using Cross Validation method
    :X_train: X train values
    :Y_train: Y train values

    :return: SSE of the model, and the predictions array
    """ 
    lassoCV_model = LassoCV(cv=fold_num)
    lassoCV_model.fit(X_train, Y_train)
    prediction = lassoCV_model.predict(X_train)
    return calculate_SSE(Y_train, prediction), prediction

def ridge_fn(X_train, Y_train):
    """
    Trains and predics a Ridge regression model using Cross Validation method
    :X_train: X train values
    :Y_train: Y train values

    :return: SSE of the model, and the predictions array
    """ 
    ridgeCV_model = RidgeCV(cv=fold_num)
    ridgeCV_model.fit(X_train, Y_train)
    prediction = ridgeCV_model.predict(X_train)
    return calculate_SSE(Y_train, prediction), prediction

def feature_removal(X_train, baseline_SSE):
    """
    Checks which feature worsens the most the SSE and removes it.
    :X_train: X train values
    :baseline_SSE: the SSE baseline value without removing features

    :return: modified X-train array, without the worse feature
    """ 
    best_feature_idx=-1
    for feature_idx in range(X_train.shape[1]):
        # Create a modified Xtrain dataset with the current feature removed
        modified_data = np.delete(X_train, feature_idx, axis=1)
        #modified_sse, modified_predictions = lasso_fn(modified_data, Y_train)
        modified_sse, modified_predictions = ridge_fn(modified_data, Y_train)

        if modified_sse < baseline_SSE:  # Compare SSE with baseline
            #print(f"Removing feature {feature_idx} improved SSE: {modified_sse}")
            baseline_SSE = modified_sse
            best_feature_idx = feature_idx
        else:
            z=0 # precisava de por qualquer coisa para nao dar identation error quando comento o print
            #print(f"Removing feature {feature_idx} worsened SSE: {modified_sse}")

    if best_feature_idx is not None:
        print(f"Best feature to remove: {best_feature_idx} ")
    else:
        print("No improvement found.")

    return np.delete(X_train, best_feature_idx, axis=1)


def linear_model(X_train, Y_train, X_test, Y_test):
    """ridge_model_predict
    Simple linear model for comparison purposes
    :X_train: X train values
    :Y_train: Y train values
    :X_test: X test values
    :Y_test: Y test values

    :return: SSE of linear model
    """ 
    ones_column = np.ones((X_train.shape[0], 1))
    X_design = np.hstack((ones_column, X_train))
    aux1 = X_design.transpose()@X_design
    aux2 = X_design.transpose()@Y_train
    beta = np.linalg.inv(aux1)@aux2
    Ytest_predicted = beta[0] + X_test@beta[1:]
    return calculate_SSE(Y_test, Ytest_predicted)

def plot_alpha_SSE():
    """
    Plots the SSE results using a range of different alpha parameters for Lasso and Ridge regression, without feature removal
    """ 
    kf = KFold(n_splits=fold_num)
    linear_SSE=[]
    alphas = np.arange(0.1,5,0.1) # range of alphas to analyse
    ridge_SSE_plot=[0]*(len(alphas))
    lasso_SSE_plot=[0]*len(alphas)
    ridge_SSE_alphas=[0]*(len(alphas))
    lasso_SSE_alphas=[0]*(len(alphas))


    sse_ridge_test=[]
    sse_lasso_test=[]
    linear_SSE = []

    # cross validation without feature removal
    for trainID, testID in kf.split(X_train):
        aux_alphas = 0
        Xtrain_fold = X_train[trainID]
        Ytrain_fold = Y_train[trainID]
        Xtest_fold = X_train[testID]
        Ytest_fold = Y_train[testID]
        
        ridge_model_test = Ridge(alpha=2.4)
        ridge_model_test.fit(Xtrain_fold, Ytrain_fold)
        ridge_model_test_predict = ridge_model_test.predict(Xtest_fold)
        sse_ridge_test.append(calculate_SSE(Ytest_fold, ridge_model_test_predict))

        lasso_model_test = Lasso(alpha=0.1)
        lasso_model_test.fit(Xtrain_fold, Ytrain_fold)
        lasso_model_test_predict = lasso_model_test.predict(Xtest_fold)
        sse_lasso_test.append(calculate_SSE(Ytest_fold, lasso_model_test_predict))

        linear_SSE.append(linear_model(Xtrain_fold, Ytrain_fold, Xtest_fold, Ytest_fold))
        
        for a in alphas:
            ridge_model = Ridge(alpha=a)
            ridge_model.fit(Xtrain_fold, Ytrain_fold)
            ridge_model_predict = ridge_model.predict(Xtest_fold)
            ridge_SSE_alphas[aux_alphas]+=(calculate_SSE(Ytest_fold, ridge_model_predict))

            lasso_model = Lasso(alpha=a)
            lasso_model.fit(Xtrain_fold, Ytrain_fold)
            lasso_model_predict = lasso_model.predict(Xtest_fold)
            lasso_SSE_alphas[aux_alphas]+=(calculate_SSE(Ytest_fold, lasso_model_predict))
            aux_alphas+=1

    print(f"[Ridge] SSE mean of folds = {np.mean(sse_ridge_test):.3f}")
    print(f"[Lasso] SSE mean of folds = {np.mean(sse_lasso_test):.3f}")

    print(f"Linear Regression Cross-Validation SSE = {np.mean(linear_SSE):.3f}")

    ridge_SSE_plot = np.array(ridge_SSE_alphas)/fold_num
    lasso_SSE_plot = np.array(lasso_SSE_alphas)/fold_num
    plt.plot(alphas, ridge_SSE_plot, color='red', label="Ridge regression")
    plt.plot(alphas, lasso_SSE_plot, color='blue', label="Lasso regression")

    min_y_ridge = min(ridge_SSE_plot)
    min_x_ridge = alphas[np.argmin(ridge_SSE_plot)]


    min_y_lasso = min(lasso_SSE_plot)
    min_x_lasso = alphas[np.argmin(lasso_SSE_plot)]

    plt.scatter(min_x_ridge, min_y_ridge, color='red', label=f'Minimum (Alpha={min_x_ridge:.2f}, SSE={min_y_ridge:.2f})', zorder=5)
    plt.scatter(min_x_lasso, min_y_lasso, color='blue', label=f'Minimum (Alpha={min_x_lasso:.2f}, SSE={min_y_lasso:.2f})', zorder=5)

    plt.xlabel("Alpha")
    plt.ylabel("SSE")
    plt.legend()
    plt.grid()
    plt.title(f"Number of folds = {fold_num}")
    plt.show()

if __name__ == "__main__":
    main()