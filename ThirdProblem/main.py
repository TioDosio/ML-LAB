import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
    x_train, x_test, y_train, y_test = train_test_split(X_train_scaled, Y_train, test_size=0.33, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, shuffle=True)

    best_balanced_acc_overall = 0
    best_strategy_overall = ""
    smote_avg = 0
    oversampling_avg=0
    number_of_runs = 2

    for run in range(1, number_of_runs+1):
        best_balanced_acc = {strategy: 0 for strategy in ["smote", "over_sampling", "under_sampling"]}
        best_strategy = {strategy: "" for strategy in ["smote", "over_sampling", "under_sampling"]}

        strategies = ["smote", "over_sampling", "under_sampling"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        aux=0
        for strategy in strategies:
            model = Sequential()
            model.add(Dense(8, activation='relu', input_dim = 28*28*3))
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

            #print("before reshaping:")
            #print(f"class 1 tem {sum(y_train)} e class 0 tem {len(y_train)-sum(y_train)}")
            #print(f"after reshaping {strategy}:")
            #print(f"class 1 tem {sum(y_train_reshaped)} e class 0 tem {len(y_train_reshaped)-sum(y_train_reshaped)}")

            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history = model.fit(x_train_reshaped, y_train_reshaped,verbose=0, batch_size=64, epochs=200, validation_data=(x_val, y_val), callbacks=[callback])

            predict = model.predict(x_test)

            f1 = f1_score(y_test, np.round(predict))
            balanced_acc = balanced_accuracy_score(y_test, np.round(predict))
            print(f'Run {run}, Strategy: {strategy}')
            print(f'F1-score: {f1}')
            print(f'Balanced Accuracy: {balanced_acc}\n')

            if balanced_acc > best_balanced_acc[strategy]:
                best_balanced_acc[strategy] = balanced_acc
                best_strategy[strategy] = strategy

            if balanced_acc > best_balanced_acc_overall:
                best_balanced_acc_overall = balanced_acc
                best_strategy_overall = strategy

            if strategy == "smote":
                smote_avg += balanced_acc

            elif strategy == "over_sampling":
                oversampling_avg += balanced_acc

            axes[0,aux].plot(history.history['loss'], label='Training Loss')
            axes[0,aux].plot(history.history['val_loss'], label='Validation Loss')
            axes[0,aux].set_xlabel('Epochs')
            axes[0,aux].set_ylabel('Loss')
            axes[0,aux].set_title(f'Loss / Cost - {strategy}')
            axes[0,aux].text(0.1, 0.2, f'Balanced Acc: {balanced_acc:.4f}', fontsize=10,transform=axes[1,aux].transAxes)
            axes[0,aux].text(0.1, 0.15, f'F1 Score: {f1:.4f}', fontsize=10,transform=axes[1,aux].transAxes)
            axes[0,aux].legend()

            cm = confusion_matrix(y_test, np.round(predict))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(ax=axes[1, aux], values_format='d')

            aux+=1

        # Save the figure for the current run
        plt.savefig(f"Run{run}_plots.png")
        plt.close()  # Close the plot window

    for strategy in strategies:
        print(f'Best Balanced Accuracy for {strategy}: {best_balanced_acc[strategy]}')

    print(f'\nBest OVERALL Balanced Accuracy: {best_balanced_acc_overall}, Strategy Used: {best_strategy_overall}')
    print(f"SMOTE AVERAGE {number_of_runs} RUNS BALANCED ACCURACY: ", smote_avg/number_of_runs)
    print(f"OVERSAMPLING AVERAGE {number_of_runs} RUNS BALANCED ACCURACY: ", oversampling_avg/number_of_runs)


if __name__ == "__main__":
    main()
