import numpy as np
from matplotlib import pyplot as plt

X_train = np.load('Xtrain_Classification1.npy')
Y_train = np.load('ytrain_Classification1.npy')
X_test = np.load('Xtest_Classification1.npy')

x = np.reshape(X_train, (-1,28,28,3))
i=0
print(x.shape)
while i < 10:
    plt.imshow(x[i])
    plt.show()
    i+=1


""" multi-layer perceptor example
from keras.models import Squencial # -> to add layers sequencialy

train_img = (X).astype('float32')/255
255 -> maximum pixel intensity mas podemos normalizar como quisermos

train_labels?leras.utils-to_categoriacal()

model_MLP=Sequential()
model_MLP.add(Dense(16, activation = Relu, input_dim = 28*28*3))

adam = keras.optimizers.Adam(learning_rate=lr)
model_MLP.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])

results_MLP = np.argmax(model_MLP.predict(test_images),1)
"""