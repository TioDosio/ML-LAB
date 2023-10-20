import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler

X_train = np.load('Xtrain_Classification1.npy')
Y_train = np.load('ytrain_Classification1.npy')

X_train_scaled = X_train / 255

x_train = X_train_scaled[:100]
y_train = Y_train[:100]
print(x_train.shape)
print(y_train.shape)
x_val = x_train.reshape(-1, 28, 28, 3)
plt.imshow(x_val[0])
plt.show()
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
x_train_reshaped, y_train_reshaped = smote.fit_resample(x_train, y_train)
x_final = x_train_reshaped.reshape(-1, 28, 28, 3)
for i in range(170):
    plt.imshow(x_final[i])
    plt.show()
