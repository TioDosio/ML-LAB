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

lr = 0.0001 #learning rate
X_train = np.load('Xtrain_Classification1.npy')
Y_train = np.load('ytrain_Classification1.npy')
X_test = np.load('Xtest_Classification1.npy')

X_train_class1 = X_train[Y_train == 1]
Y_train_class1 = Y_train[Y_train == 1]

X_train_class0 = X_train[Y_train == 0]
Y_train_class0 = Y_train[Y_train == 0]

x_reshaped = np.reshape(X_train_class1, (-1,28,28,3))
rotated_x = np.rot90(x_reshaped, k=1, axes=(1, 2)) # rodar 90 graus
rotated_x2 = np.rot90(x_reshaped, k=2, axes=(1, 2)) # rodar 90 graus


i=0
while i < 3:
    plt.imshow(rotated_x[i])
    plt.show()
    plt.imshow(x_reshaped[i])
    plt.show()
    plt.imshow(rotated_x2[i])
    plt.show()
    i+=1