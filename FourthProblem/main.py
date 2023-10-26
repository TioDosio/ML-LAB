import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical

def main():
    lr = 0.0001  # Learning rate

    X_train = np.load('Xtrain_Classification2.npy')
    Y_train = np.load('ytrain_Classification2.npy')
    X_test = np.load('Xtest_Classification2.npy')

    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255
    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255

    x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y_train, test_size=0.33, shuffle=True, stratify=Y_train)
    for aux in range(0,6):
        print(f"tamanho class {aux} = {len(y_train[y_train==aux])}")
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, shuffle=True, stratify=y_train)

    x_val, y_val = data_augmentation(x_val, y_val)
    x_train, y_train = data_augmentation(x_train, y_train)
    for aux in range(0,6):
        print(f"tamanho class {aux} = {len(y_train[y_train==aux])}")
    
    flipped_class_data = further_data_augmentation(x_train[y_train==2], y_train[y_train==2])
    # juntar train normal com os rotated
    #x_train, y_train = np.concatenate((x_train, x_train_rotated), axis=0), np.concatenate((y_train, y_train_rotated), axis=0)
    #x_val, y_val = np.concatenate((x_train, x_val_rotated), axis=0), np.concatenate((y_train, y_val_rotated), axis=0)

    y_train_onehot = to_categorical(y_train, num_classes=6)
    y_val_onehot = to_categorical(y_val, num_classes=6)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))   # mais camadas de droupout com valores mais baixos
    model.add(Dense(64, activation='relu'))

    model.add(Dense(6, activation='softmax'))
    adam = Adam(learning_rate=lr)


    fig, axes = plt.subplots(2, figsize=(15, 8))

    random_over_sampler = RandomOverSampler()
    x_train_reshaped, y_train_reshaped = random_over_sampler.fit_resample(x_train, y_train)
    y_train_reshaped = to_categorical(y_train_reshaped, num_classes=6)
    x_train_reshaped = x_train_reshaped.reshape(-1, 28, 28, 3)
    x_val = x_val.reshape(-1, 28, 28, 3)
    x_test = x_test.reshape(-1,28,28,3)
    X_test_reshaped = X_test_scaled.reshape(-1,28,28,3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    print(f"len xtrain = {len(x_train_reshaped)}")
    print(f"len xtest = {len(x_test)}")
    history = model.fit(x_train_reshaped, y_train_reshaped,verbose=2, batch_size=2000, epochs=100, validation_data=(x_val, y_val_onehot), callbacks=[callback])
    predict = model.predict(x_test)
    balanced_accuracy = balanced_accuracy_score(y_test, np.argmax(predict, axis=1))
    #print(f"Balanced Accuracy = {balanced_accuracy}")

    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Balanced Acc:{balanced_accuracy:.4f}')
    #axes[0].text(0.1, 0.15, f'F1 Score: {f1:.4f}', fontsize=10,transform=axes[1,aux].transAxes)
    axes[0].legend()

    cm = confusion_matrix(y_test, np.argmax(predict, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5])
    disp.plot(ax=axes[1], values_format='d')
    plt.show()

def data_augmentation(x_train, y_train):
    x = x_train.copy()
    y = y_train.copy()

    for j in range(6):
        X_train_classj = x_train[y_train == j]
        x_reshaped_classj = np.reshape(X_train_classj, (-1, 28, 28, 3))
        x_aug = []

        for i in range(4):
            x_aug.append(np.rot90(x_reshaped_classj, k=i, axes=(1, 2))) # Rotate 90 degrees


        x_aug = np.concatenate(x_aug, axis=0)
        y_aug = np.tile(y_train[y_train == j], 4)

        x_aug_classj = x_aug.reshape(x_aug.shape[0], -1)
        x = np.concatenate((x, x_aug_classj), axis=0)
        y = np.concatenate((y, y_aug), axis=0)

    return x, y

def further_data_augmentation(x_train, y_train):
    print(x_train.shape)
    flipped_images=[]
    x_train_reshaped = x_train.reshape(-1, 28, 28, 3)
    for image in x_train_reshaped:
        flipped_images.append(np.flipud(image))
        flipped_images.append(np.fliplr(image))

    flipped = np.concatenate(flipped_images, axis=0)
    print(flipped.shape)

if __name__ == "__main__":
    main()