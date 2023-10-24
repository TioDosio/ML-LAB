from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay


X_train = np.load('Xtrain_Classification2.npy')
Y_train = np.load('ytrain_Classification2.npy')
X_test = np.load('Xtest_Classification2.npy')

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y_train, test_size=0.33, shuffle=True, stratify=Y_train)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto'],
    'bootstrap': [True],
    'criterion': ['entropy']
}

# random forest classifier
random_forest = RandomForestClassifier()
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, scoring='balanced_accuracy', verbose=2)

# Train the model
# random_forest.fit(x_train, y_train)
grid_search.fit(x_train, y_train)
best_rf = grid_search.best_estimator_
# Make predictions on the test set
forest_predict = best_rf.predict(x_test)

# Calculate accuracy
forest_acc = balanced_accuracy_score(y_test, np.round(forest_predict))
print(f"Random Forest Accuracy: {forest_acc}")
print(f"best params = {best_rf.get_params}")
