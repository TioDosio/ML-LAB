import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, Ridge, Lasso
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold

# Load the data from .npy files
X_train = np.load('X_train_regression1.npy')
Y_train = np.load('y_train_regression1.npy')
X_test = np.load('X_test_regression1.npy')
fold_num = 5 # number of splits for cross-validation

def main():
    warnings.filterwarnings("ignore")

    # Standardize the data
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #Y_train = Y_train.flatten()

    baseline_SSE, baseline_predictions = lasso_fn(X_train, Y_train)
    baseline_SSE_RIDGE, baseline_predictions_RIDGE = ridge_fn(X_train, Y_train)

    print("BEFORE SSE=",baseline_SSE_RIDGE)
    
    Xtrain_better = feature_removal(X_train, baseline_SSE_RIDGE)
    baseline_SSE_RIDGE_after, baseline_predictions_RIDGE_after = ridge_fn(Xtrain_better, Y_train)

    # check alpha/SSE evolution with pyplot
    plot_alpha_SSE()
    
    print("AFTER SSE = ", baseline_SSE_RIDGE_after)

def calculate_SSE(y, y_predicted):
    return np.sum((y-y_predicted)**2)

# Lasso regression using cross-validation function
def lasso_fn(X_trainF, Y_trainF):
    lassoCV_model = LassoCV(cv=fold_num)
    lassoCV_model.fit(X_trainF, Y_trainF)
    prediction = lassoCV_model.predict(X_trainF)
    return np.sum((prediction - Y_trainF) ** 2), prediction

# Ridge regression using cross-validation function
def ridge_fn(X_trainF, Y_trainF):
    ridgeCV_model = RidgeCV(cv=fold_num)
    ridgeCV_model.fit(X_trainF, Y_trainF)
    prediction = ridgeCV_model.predict(X_trainF)
    return np.sum((prediction - Y_trainF) ** 2), prediction

def feature_removal(X_train, baseline_SSE):
        best_feature_idx=-1
        for feature_idx in range(X_train.shape[1]):
            # Create a modified dataset with the current feature removed
            modified_data = np.delete(X_train, feature_idx, axis=1)
            #modified_sse, modified_predictions = lasso_fn(modified_data, Y_train)
            modified_sse, modified_predictions = ridge_fn(modified_data, Y_train)

            if modified_sse < baseline_SSE:  # Compare SSE with Lasso baseline
                print(f"Removing feature {feature_idx} improved SSE: {modified_sse}")
                baseline_SSE = modified_sse
                best_feature_idx = feature_idx
            else:
                print(f"Removing feature {feature_idx} worsened SSE: {modified_sse}")

        if best_feature_idx is not None:
            print(f"Best feature to remove: {best_feature_idx} ")
        else:
            print("No improvement found.")

        return np.delete(X_train, best_feature_idx, axis=1)

def plot_alpha_SSE():
        kf = KFold(n_splits=fold_num)
        ridge_SSE_alphas=[]
        lasso_SSE_alphas=[]
        alphas = np.arange(0.1,5,0.1) # range of alphas to analyse
        ridge_SSE_plot=[0]*len(alphas)
        lasso_SSE_plot=[0]*len(alphas)

        # cross validation without feature removal
        for trainID, testID in kf.split(X_train):
            Xtrain_fold = X_train[trainID]
            Ytrain_fold = Y_train[trainID]
            Xtest_fold = X_train[testID]
            Ytest_fold = Y_train[testID]

            for a in alphas:
                ridge_model = Ridge(alpha=a)
                ridge_model.fit(Xtrain_fold, Ytrain_fold)
                ridge_model_predict = ridge_model.predict(Xtest_fold)
                ridge_SSE_alphas.append(calculate_SSE(Ytest_fold, ridge_model_predict))

                lasso_model = Lasso(alpha=a)
                lasso_model.fit(Xtrain_fold, Ytrain_fold)
                lasso_model_predict = lasso_model.predict(Xtest_fold)
                lasso_SSE_alphas.append(calculate_SSE(Ytest_fold, lasso_model_predict))
        
        for i in range(0,fold_num):
            for l in range(0,len(alphas)):
                ridge_SSE_plot[l] += ridge_SSE_alphas[l+len(alphas)*i]
                lasso_SSE_plot[l] += lasso_SSE_alphas[l+len(alphas)*i]


        plt.plot(alphas, np.array(ridge_SSE_plot)/len(alphas), color='red', label="Ridge regression")
        plt.plot(alphas, np.array(lasso_SSE_plot)/len(alphas), color='blue', label="Lasso regression")

        min_y_ridge = min(np.array(ridge_SSE_plot)/len(alphas))
        min_x_ridge = alphas[np.argmin(np.array(ridge_SSE_plot)/len(alphas))]

        min_y_lasso = min(np.array(lasso_SSE_plot)/len(alphas))
        min_x_lasso = alphas[np.argmin(np.array(lasso_SSE_plot)/len(alphas))]

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