import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
lr = 0.0001  # Learning rate
X_train = np.load('Xtrain_Classification1.npy')
Y_train = np.load('ytrain_Classification1.npy')
X_test = np.load('Xtest_Classification1.npy')

X_train_scaled = X_train / 255
x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, Y_train, test_size=0.33, shuffle=True)

model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(learning_rate=lr)

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)  # Added random_state for reproducibility
x_train_reshaped, y_train_reshaped = smote.fit_resample(x_train, y_train)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # early stopping

history = model.fit(x_train_reshaped, y_train_reshaped, batch_size=64, epochs=200, validation_data=(x_val, y_val),callbacks=[callback])

predict = model.predict(x_train)
predict_val = model.predict(x_val)

print(f'F1-Score = {f1_score(y_train, np.round(predict))}')
print(f"Balanced Acc = {balanced_accuracy_score(y_train, np.round(predict))}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
