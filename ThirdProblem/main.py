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

lr = 0.0001 #learning rate
X_train = np.load('Xtrain_Classification1.npy')
Y_train = np.load('ytrain_Classification1.npy')
X_test = np.load('Xtest_Classification1.npy')

X_train_scaled = X_train/255

x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, Y_train, test_size=0.33, shuffle= True)
model = Sequential()
model.add(Dense(8, input_dim=28 * 28 * 3, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
adam = Adam(learning_rate=lr)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'] )
history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val))

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()