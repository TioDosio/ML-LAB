import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def main():
    X_train = np.load('Xtrain_Classification2.npy')
    Y_train = np.load('ytrain_Classification2.npy')
    X_test = np.load('Xtest_Classification2.npy')

    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255

    x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y_train, test_size=0.33, shuffle=True)

    # Define the parameter grid
    """ param_grid = {
        'C': [10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', 0.1, 1,10],
    } """

    # Create the GridSearchCV object
    svc_model = SVC(C=10, kernel='rbf', gamma='scale')
    #grid_search = GridSearchCV(svc_model, cv=5, scoring='balanced_accuracy', verbose=2)

    svc_model.fit(x_train, y_train)
    # Fit the grid search to your data

    svm_predict = svc_model.predict(x_test)

    balanced_acc = balanced_accuracy_score(y_test, np.round(svm_predict))

    print(f'Balanced Accuracy: {balanced_acc}\n')
    #print(f"Best params: {best_svm.get_params()}")
    #Best params: {'C': 10, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 
    # 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': False,
    #  'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
    cm = confusion_matrix(y_test, np.round(svm_predict))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()
