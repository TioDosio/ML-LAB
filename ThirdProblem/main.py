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

    best_balanced_acc_overall = 0
    best_strategy_overall = ""

    for run in range(1, 3):
        best_balanced_acc = {strategy: 0 for strategy in ["smote", "over_sampling", "under_sampling"]}
        best_strategy = {strategy: "" for strategy in ["smote", "over_sampling", "under_sampling"]}

        strategies = ["smote", "over_sampling", "under_sampling"]
        for strategy in strategies:
            model = Sequential()
            model.add(Dense(8, activation='relu'))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            adam = Adam(learning_rate=lr)

            if strategy == "smote":
                smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
                x_train_reshaped, y_train_reshaped = smote.fit_resample(x_train, y_train)

            elif strategy == "over_sampling":
                random_over_sampler = RandomOverSampler()
                x_train_reshaped, y_train_reshaped = random_over_sampler.fit_resample(x_train, y_train)

            elif strategy == "under_sampling":
                random_under_sampler = RandomUnderSampler()
                x_train_reshaped, y_train_reshaped = random_under_sampler.fit_resample(x_train, y_train)

            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history = model.fit(x_train_reshaped, y_train_reshaped, batch_size=64, epochs=200, validation_data=(x_val, y_val), callbacks=[callback])

            predict = model.predict(x_train)
            predict_val = model.predict(x_val)

            f1 = f1_score(y_val, np.round(predict_val))
            balanced_acc = balanced_accuracy_score(y_val, np.round(predict_val))
            print(f'Run {run}, Strategy: {strategy}')
            print(f'F1-score: {f1}')
            print(f'Balanced Accuracy: {balanced_acc}\n')

            if balanced_acc > best_balanced_acc[strategy]:
                best_balanced_acc[strategy] = balanced_acc
                best_strategy[strategy] = strategy

            if balanced_acc > best_balanced_acc_overall:
                best_balanced_acc_overall = balanced_acc
                best_strategy_overall = strategy

        fig, axes = plt.subplots(3, 3, figsize=(30, 15))

        for i, strategy in enumerate(strategies):
            # Plot the first subplot (Training Loss)
            axes[0, i].plot(history.history['loss'], label='Training Loss')
            axes[0, i].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, i].set_xlabel('Epochs')
            axes[0, i].set_ylabel('Loss')
            axes[0, i].set_title(f'Loss / Cost - {strategy}')
            axes[0, i].legend()

            # Add text to the first subplot
            axes[0, i].text(0.5, 0.8, f'Balanced Acc: {best_balanced_acc[strategy]:.4f}', transform=axes[0, i].transAxes, fontsize=12)

        for i in range(3):
            axes[1, i].scatter(x_train_reshaped[:, 0][y_train_reshaped == 0], x_train_reshaped[:, 1][y_train_reshaped == 0], c='red', label='y=0')
            axes[1, i].scatter(x_train_reshaped[:, 0][y_train_reshaped == 1], x_train_reshaped[:, 1][y_train_reshaped == 1], c='blue', label='y=1')
            axes[1, i].set_title(f'{strategies[i]} Reshaped Train Split')
            axes[1, i].text(0.2, 0.8, f'class 1 has: {sum(y_train_reshaped)} / {len(y_train_reshaped)}', transform=axes[1, i].transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            axes[1, i].legend()

        for i in range(3):
            axes[2, i].scatter(x_train[:, 0][y_train == 0], x_train[:, 1][y_train == 0], c='red', label='y=0')
            axes[2, i].scatter(x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], c='blue', label='y=1')
            axes[2, i].set_title(f'Normal Train Split')
            axes[2, i].text(0.2, 0.8, f'class 1 has: {sum(y_train)} / {len(y_train)}', transform=axes[2, i].transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            axes[2, i].legend()

        # Save the figure for the current run
        plt.savefig(f"Run{run}_plots.png")
        plt.close()  # Close the plot window

    for strategy in strategies:
        print(f'Best Balanced Accuracy for {strategy}: {best_balanced_acc[strategy]}')

    print(f'\nBest OVERALL Balanced Accuracy: {best_balanced_acc_overall}, Strategy Used: {best_strategy_overall}')

if __name__ == "__main__":
    main()
