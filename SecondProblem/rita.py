import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression, Lasso, RidgeCV, Ridge
import random
import warnings
import itertools

def main():
    warnings.filterwarnings("ignore")

    X_train = np.load('X_train_regression2.npy')
    Y_train = np.load('y_train_regression2.npy')
    X_test = np.load('X_test_regression2.npy')

    index_list = list(range(len(X_train)))
    random.shuffle(index_list)

    inlier_range=0.45
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

        x = np.array([x1,x2])
        y = np.array([y1,y2])

        linear = LinearRegression()
        linear.fit(x,y)

        predictions = linear.predict(X_train)
        distances = np.abs(predictions - Y_train)
        inlier_count = np.sum(distances<inlier_range)
        if inlier_count > best_inlier_count:
            outlier_indices = np.where(distances>=inlier_range)[0]
            inlier_indices = np.where(distances<inlier_range)[0]
            inlier_model = linear
            best_inlier_count = inlier_count


    # fit another linear model to the outlier points
    outlier_x = X_train[outlier_indices]
    outlier_y = Y_train[outlier_indices]
    outlier_model = LinearRegression()
    outlier_model.fit(outlier_x, outlier_y)
    print(f"model 1 (inlier) coefs: {inlier_model.coef_}, numero pontos = {best_inlier_count}")
    print(f"model 2 (outlier) coefs: {outlier_model.coef_}, numero pontos = {len(outlier_indices)}")

    model1_idx=[]
    model2_idx=[]
    for i in range(len(X_train)):
        error_model1 = (inlier_model.predict([X_train[i]])-Y_train[i])**2
        error_model2 = (outlier_model.predict([X_train[i]])-Y_train[i])**2

        if (error_model1<error_model2):
            model1_idx.append(i)
        
        else:
            model2_idx.append(i)
    print("after error model1", len(model1_idx))
    print("after error model2", len(model2_idx))

    # see which model is better for Xtest
    model1_predict = inlier_model.predict(X_test)
    model2_predict = outlier_model.predict(X_test)



if __name__ == "__main__":
    main()