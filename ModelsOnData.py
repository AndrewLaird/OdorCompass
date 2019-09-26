import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# filter warnings
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from keras import Input, Model
from keras import metrics, callbacks
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, LSTM, \
    TimeDistributed, BatchNormalization
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import binarize
from sklearn.svm import SVC
import matplotlib.pyplot as plt

LOAD_DIRECTORY = "./raman_data/"
SAVE_DIRECTORY = "./finished_models/"
EPOCHS_TO_TRAIN = 1

# Early stopping is used if many epochs are going by without much change in the result
patience = 16
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=8, min_lr=0.001)

# num_cores is how many processer cores to use for non-gpu parallel processing
NUM_CORES = 8
# group is the square root of the number of locations
group = 3
# chips is the total number of chips used to train the model
chips = 1
# analyte is the number of unique analytes used
analyte = 2
# cutoff is the wavenumber to cutoff each spectra at, I got rid of some to improve smoothing
cutoff = 1011
# features is the total number of wavenumber in a spectra
features = 1011


def top_2_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def cnn_model(sniffs):
    input = Input(shape=(sniffs, group, group, analyte))
    x = TimeDistributed(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(input)
    x = BatchNormalization()(x)
    x = TimeDistributed(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Conv2D(64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))(x)
    x = TimeDistributed(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(32, dropout=0.5, recurrent_dropout=0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(8, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy', top_2_accuracy])
    # model.summary()
    return model


def ann_model(sniffs):
    input = Input(shape=(sniffs * group * group * analyte,))
    x = Dense(128, activation='relu')(input)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(8, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy', top_2_accuracy])
    # model.summary()
    return model


def load_data(sniffs):
    X_train = np.load(LOAD_DIRECTORY + f"X_train_{sniffs}.npy")
    X_test = np.load(LOAD_DIRECTORY + f"X_test_{sniffs}.npy")
    y_train = np.load(LOAD_DIRECTORY + f"y_train_{sniffs}.npy")
    y_test = np.load(LOAD_DIRECTORY + f"y_test_{sniffs}.npy")
    print("data loaded")

    return X_train, X_test, y_train, y_test


def run_cnn_model(kfold, X_train, y_train, X_test, y_test):
    print("cnn model")
    cnn_results = []
    fold = 0
    for train_idx, val_idx in kfold.split(X_train, y_train):
        print(" Kfold:", fold)
        model = cnn_model(sniffs)
        training_x = X_train[train_idx]
        training_y = y_train[train_idx]
        val_x = X_train[val_idx]
        val_y = y_train[val_idx]
        history = model.fit(training_x, training_y,
                            validation_data=(val_x, val_y),
                            epochs=EPOCHS_TO_TRAIN, verbose=1,
                            callbacks=[early_stop, reduce_lr])

        model.save(SAVE_DIRECTORY + f"cnn_iso_{sniffs}_{fold}.h5")
        # np.save(SAVE_DIRECTORY + f"cnn_val_loss_iso_{sniffs}_{i}.npy", np.array(history.history['val_loss']))
        # np.save(SAVE_DIRECTORY + f"cnn_loss_iso_{sniffs}_{i}.npy", np.array(history.history['loss']))

        predictions = model.predict(X_test)
        result = binarize(predictions, threshold=0.35)

        print(f"CNN Test Total acc {fold}:{accuracy_score(y_test, result)}")
        cnn_results.append(accuracy_score(y_test, result))

        y_pred_label_b = np.array([np.argmax(l) for l in result[:, :4]])
        y_label_b = np.array([np.argmax(l) for l in y_test[:, :4]])
        # print(f"CNN BZT acc {i}:{accuracy_score(y_label_b, y_pred_label_b)}")

        y_pred_label_m = np.array([np.argmax(l) for l in result[:, -4:]])
        y_label_m = np.array([np.argmax(l) for l in y_test[:, -4:]])
        # print(f"CNN MBZT acc {i}:{accuracy_score(y_label_m, y_pred_label_m)}")

        # np.save(SAVE_DIRECTORY + f"cnn_y_label_b_iso_{sniffs}_cv{i}.npy", y_label_b)
        # np.save(SAVE_DIRECTORY + f"cnn_y_label_m_iso_{sniffs}_cv{i}.npy", y_label_m)
        # np.save(SAVE_DIRECTORY + f"cnn_y_pred_label_b_iso_{sniffs}_cv{i}.npy", y_pred_label_b)
        # np.save(SAVE_DIRECTORY + f"cnn_y_pred_label_m_iso_{sniffs}_cv{i}.npy", y_pred_label_m)

        if fold == 0:
            cnn_score = accuracy_score(y_label_b, y_pred_label_b)
        if fold > 0:
            cnn_score = np.append(cnn_score, accuracy_score(y_label_b, y_pred_label_b))
        fold += 1

    # np.save(SAVE_DIRECTORY + f"cnn_score_iso_{sniffs}.npy", cnn_score)
    return cnn_results

def run_knn_model(kfold, X_train_1d, X_test_1d, one_hot_y, one_hot_test_y):
    print("KNN model")
    svc_results = []
    fold = 0
    for train_idx, val_idx in kfold.split(X_train_1d, one_hot_y):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_1d[train_idx], one_hot_y[train_idx])
        score = knn.score(X_train_1d[val_idx], one_hot_y[val_idx])
        svc_results.append(score)
        print(f'KNN_{fold}:{score}')
        Y_pred = knn.predict(X_test_1d)
        pred = np.array([np.argmax(l) for l in Y_pred])
        if fold == 0:
            knn_score = accuracy_score(one_hot_test_y, pred)
        if fold > 0:
            knn_score = np.append(knn_score, accuracy_score(one_hot_test_y, pred))


def run_svc_model(kfold, X_train_1d, X_test_1d, one_hot_y, one_hot_test_y):
    print("SVC model")
    svc_results = []
    fold = 0
    for train_idx, val_idx in kfold.split(X_train_1d, one_hot_y):
        clf = SVC()
        clf.fit(X_train_1d[train_idx], one_hot_y[train_idx])
        score = clf.score(X_train_1d[val_idx], one_hot_y[val_idx])
        print(f'SVC_{fold}:{score}')
        Y_pred = clf.predict(X_test_1d)
        pred = np.array([np.argmax(l) for l in Y_pred])
        if fold == 0:
            svc_score = accuracy_score(one_hot_test_y, pred)
        if fold > 0:
            svc_score = np.append(svc_score, accuracy_score(one_hot_test_y, pred))

        fold += 1
    # np.save(SAVE_DIRECTORY + f"SVC_score_iso_{sniffs}.npy", svc_score)
    # np.save(SAVE_DIRECTORY + f"KNN_score_iso_{sniffs}.npy", knn_score)
    return svc_results


def run_ann_model(kfold, X_train_1d, X_test_1d, y_train, y_test):
    print("ANN model")
    ann_results = []

    fold = 0
    for train_idx, val_idx in kfold.split(X_train_1d, y_train):
        model = ann_model(sniffs)
        history = model.fit(X_train_1d[train_idx], y_train[train_idx],
                            validation_data=(X_train_1d[val_idx], y_train[val_idx]),
                            epochs=EPOCHS_TO_TRAIN, verbose=1, callbacks=[early_stop, reduce_lr])

        model.save(SAVE_DIRECTORY + f"ann_iso_{sniffs}_{fold}.h5")
        # np.save(SAVE_DIRECTORY + f"ann_val_loss_iso_{sniffs}_{fold}.npy", np.array(history.history['val_loss']))
        # np.save(SAVE_DIRECTORY + f"ann_loss_iso_{sniffs}_{fold}.npy", np.array(history.history['loss']))

        predictions = model.predict(X_test_1d)
        result = binarize(predictions, threshold=0.35)

        print(f"ANN Test Total acc {fold}:{accuracy_score(y_test, result)}")
        ann_results.append(accuracy_score(y_test, result))

        y_pred_label_b = np.array([np.argmax(l) for l in result[:, :4]])
        y_label_b = np.array([np.argmax(l) for l in y_test[:, :4]])
        # print(f"ANN BZT acc {fold}:{accuracy_score(y_label_b, y_pred_label_b)}")

        y_pred_label_m = np.array([np.argmax(l) for l in result[:, -4:]])
        y_label_m = np.array([np.argmax(l) for l in y_test[:, -4:]])
        # print(f"ANN MBZT acc {fold}:{accuracy_score(y_label_m, y_pred_label_m)}")

        '''
        np.save(SAVE_DIRECTORY + f"ann_y_label_b_iso_{sniffs}_cv{fold}.npy", y_label_b)
        np.save(SAVE_DIRECTORY + f"ann_y_label_m_iso_{sniffs}_cv{fold}.npy", y_label_m)
        np.save(SAVE_DIRECTORY + f"ann_y_pred_label_b_iso_{sniffs}_cv{fold}.npy", y_pred_label_b)
        np.save(SAVE_DIRECTORY + f"ann_y_pred_label_m_iso_{sniffs}_cv{fold}.npy", y_pred_label_m)
        '''

        if fold == 0:
            ann_score = accuracy_score(y_label_b, y_pred_label_b)
        if fold > 0:
            ann_score = np.append(ann_score, accuracy_score(y_label_b, y_pred_label_b))
        fold += 1
    return ann_results


def one_hot_data(X_train,X_test,y_train,y_test):
    # Flattening the input data
    X_train_1d = np.reshape(X_train, (-1, sniffs * group * group * analyte))
    X_test_1d = np.reshape(X_test, (-1, sniffs * group * group * analyte))

    print("Input Data shape in 1D", X_train_1d.shape)

    counter = 0
    one_hot_y = []
    remap = {}
    # Taking the Labels which are in the form of two integers
    # and making them one one_hot vector
    for i in range(len(y_train)):
        if (tuple(y_train[i]) not in remap.keys()):
            remap[tuple(y_train[i])] = counter
            counter += 1
        one_hot_y.append(remap[tuple(y_train[i])])

    one_hot_test_y = []
    for i in y_test:
        one_hot_test_y.append(remap[tuple(i)])

    one_hot_test_y = np.array(one_hot_test_y)
    one_hot_y = np.array(one_hot_y)

    return X_train_1d,X_test_1d,one_hot_y,one_hot_test_y

def process_sniff(sniffs):
    results = {
        'cnn': [],
        'ann': [],
        'svc': [],
    }

    print(f"Sniffs: {sniffs}")

    X_train, X_test, y_train, y_test = load_data(sniffs)

    print("Input Data shape: ", X_train.shape)

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    # Running CNN model
    results['cnn'] = run_cnn_model(kfold, X_train, y_train, X_test, y_test)


    X_train_1d,X_test_1d,one_hot_y,one_hot_test_y = one_hot_data(X_train,X_test,y_train,y_test)

    results['svc'] = run_svc_model(kfold, X_train_1d, X_test_1d, one_hot_y, one_hot_test_y)

    # KNN model works but is slow due to the size of the dataset
    # and achieves low results
    # knn_results = run_knn_model(kfold, X_train_1d, X_test_1d, one_hot_y, one_hot_test_y)

    results['ann'] = run_ann_model(kfold, X_train_1d, X_test_1d, y_train, y_test)
    print("Sniff Processed")
    return results

def plot_results(results):
    ann = results['ann']
    cnn = results['cnn']
    svc = results['svc']

    plt.plot(ann)
    plt.plot(cnn)
    plt.plot(svc)

    plt.ylabel('accuracy')
    plt.xlabel('fold')
    plt.show()

if __name__ == "__main__":
    # sniffs is the number of maps to use for each prediction
    # if you only have the subset of the data use 20 sniffs
    sniffs = 20
    experiment_results = process_sniff(sniffs)
    plot_results(experiment_results)
