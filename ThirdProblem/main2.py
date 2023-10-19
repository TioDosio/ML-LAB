import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
from imblearn.over_sampling import SMOTE

lr = 0.0001 #learning rate
X_train = np.load('Xtrain_Classification1.npy')[0:15]
Y_train = np.load('ytrain_Classification1.npy')[0:15]
X_test = np.load('Xtest_Classification1.npy')

X_train_class1 = X_train[Y_train == 1]
Y_train_class1 = Y_train[Y_train == 1]

j=0
test1 = np.reshape(X_train_class1, (-1,28,28,3))
while j < 5 and j < len(test1):
    plt.imshow(test1[j])
    plt.show()
    j+=1

smote = SMOTE(sampling_strategy='auto',k_neighbors=5, random_state=42)
x_train_reshaped, y_train_reshaped = smote.fit_resample(X_train_class1, Y_train_class1)
test = np.reshape(x_train_reshaped, (-1,28,28,3))
i=0
while i < 5 and i < len(test):
    plt.imshow(test[i])
    plt.show()
    i+=1

