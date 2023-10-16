import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf

def main():
    lr = 0.0001  # Learning rate
    X_train = np.load('Xtrain_Classification1.npy')
    Y_train = np.load('ytrain_Classification1.npy')
    X_test = np.load('Xtest_Classification1.npy')

    X_train_scaled = X_train / 255
    x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, Y_train, test_size=0.33)
    
    strategies = ["smote", "over_sampling", "under_sampling"]
    for strategy in strategies:
        model = Sequential()
        model.add(Dense(8, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        adam = Adam(learning_rate=lr)

        if strategy == "smote":
            # other oversampling method
            smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)  # Added random_state for reproducibility
            x_train_reshaped, y_train_reshaped = smote.fit_resample(x_train, y_train)

        elif strategy == "over_sampling":
            # resample minority class
            random_over_sampler = RandomOverSampler(sampling_strategy=1)
            x_train_reshaped, y_train_reshaped = random_over_sampler.fit_resample(x_train, y_train)

        elif strategy == "under_sampling":
            # reduce majority class
            random_under_sampler = RandomUnderSampler(sampling_strategy=1)
            x_train_reshaped, y_train_reshaped = random_under_sampler.fit_resample(x_train, y_train)

        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # early stopping
        history = model.fit(x_train_reshaped, y_train_reshaped, batch_size=64, epochs=200, validation_data=(x_val, y_val),callbacks=[callback])

        predict = model.predict(x_train)
        predict_val = model.predict(x_val)

        f1 = f1_score(y_val, np.round(predict_val))
        balanced_acc = balanced_accuracy_score(y_val, np.round(predict_val))
        print(f'Strategy: {strategy}')
        print(f'F1-score: {f1}')
        print(f'Balanced Accuracy: {balanced_acc}')


        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
        
        # Plot the first subplot (Training Loss)
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss / Cost')
        ax1.legend()

        # Add text to the first subplot
        ax1.text(0.5, 0.8, f'F1-Score: {f1:.4f}', transform=ax1.transAxes, fontsize=12)
        ax1.text(0.5, 0.7, f'Balanced Acc: {balanced_acc:.4f}', transform=ax1.transAxes, fontsize=12)

        ax2.scatter(x_train_reshaped[:, 0][y_train_reshaped == 0], x_train_reshaped[:, 1][y_train_reshaped == 0], c='red', label='y=0')
        ax2.scatter(x_train_reshaped[:, 0][y_train_reshaped == 1], x_train_reshaped[:, 1][y_train_reshaped == 1], c='blue', label='y=1')
        ax2.set_title(f'{strategy} Reshaped Train Split')
        ax2.text(0.2, 0.8, f'class 1 has: {sum(y_train_reshaped)} / {len(y_train_reshaped)}', transform=ax2.transAxes, fontsize=12,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        ax2.legend()

        ax3.scatter(x_train[:, 0][y_train == 0], x_train[:, 1][y_train == 0], c='red', label='y=0')
        ax3.scatter(x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], c='blue', label='y=1')
        ax3.set_title('Normal Train Split')
        ax3.text(0.2, 0.8, f'class 1 has: {sum(y_train)} / {len(y_train)}', transform=ax3.transAxes, fontsize=12,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        ax3.legend()

        plt.savefig(f"{strategy}_plots.png")
        plt.close()  # Close the plot window


if __name__ == "__main__":
    main()