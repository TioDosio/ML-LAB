import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def main():
    X_train = np.load('Xtrain_Classification1.npy')
    Y_train = np.load('ytrain_Classification1.npy')
    X_test = np.load('Xtest_Classification1.npy')

    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255

    x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y_train, test_size=0.33, shuffle=True)

    best_balanced_acc_overall = 0
    best_strategy_overall = ""
    number_of_runs = 1
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', 0.1, 1],
    }

    """ # Create the GridSearchCV object
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='balanced_accuracy')

    # Fit the grid search to your data
    grid_search.fit(x_train, y_train) """

    # Get the best hyperparameters
    #print("PARAMETROS", grid_search.best_params_)
    #print("ESTIMATOR ", grid_search.best_estimator_)
    for run in range(1, number_of_runs + 1):
        best_balanced_acc = {strategy: 0 for strategy in ["over_sampling", "under_sampling"]}
        best_strategy = {strategy: "" for strategy in ["over_sampling", "under_sampling"]}

        strategies = ["over_sampling"]
        for strategy in strategies:
            # Create an RBF SVM classifier
            clf = SVC(kernel='rbf', C=10, gamma='scale')

            # Train the classifier on the training data
            clf.fit(x_train, y_train)

            # Make predictions on the test data
            predict = clf.predict(x_test)

            f1 = f1_score(y_test, np.round(predict))
            balanced_acc = balanced_accuracy_score(y_test, np.round(predict))
            print(f'Run {run}, Strategy: {strategy}')
            print(f'F1-score: {f1}')
            print(f'Balanced Accuracy: {balanced_acc}\n')

            if balanced_acc > best_balanced_acc[strategy]:
                best_balanced_acc[strategy] = balanced_acc
                best_strategy[strategy] = strategy

            if balanced_acc > best_balanced_acc_overall:
                best_balanced_acc_overall = balanced_acc
                best_strategy_overall = strategy

            cm = confusion_matrix(y_test, np.round(predict))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot()

    for strategy in strategies:
        print(f'Best Balanced Accuracy for {strategy}: {best_balanced_acc[strategy]}')

    print(f'\nBest OVERALL Balanced Accuracy: {best_balanced_acc_overall}, Strategy Used: {best_strategy_overall}')

if __name__ == "__main__":
    main()
