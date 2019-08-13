import numpy as np
from keras import Input, Model
from keras import metrics, callbacks
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, Dropout, AveragePooling3D, Conv2D, GlobalAveragePooling2D, MaxPool2D, LSTM, TimeDistributed, BatchNormalization, GlobalAveragePooling3D,GlobalAveragePooling1D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import binarize
from sklearn.svm import SVC

patience = 16
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=8, min_lr=0.001)
save_directory = "/home/ragan/Downloads/direct_results_test/"

# num_cores is how many processer cores to use for non-gpu parallel processing
num_cores = 8
# group is the square root of the number of locations
group = 3
# sniffs is the number of maps to use for each prediction
sniffs = 3
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

def cnn_model_update(sniffs):
    input = Input(shape=(sniffs, group, group, analyte))
    x = input

    x = TimeDistributed(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    x = TimeDistributed(Conv2D(1024, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))(x)
    # update #1 adding global pooling instead of flatteining

    x = GlobalAveragePooling3D()(x)

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


save_directory = "./direct_results/"

def process_sniff(sniffs):
    print(f"Sniffs: {sniffs}")

    load_directory = "./direct_results/"
    X_train = np.load(load_directory + f"X_train_{sniffs}.npy")
    X_test = np.load(load_directory + f"X_test_{sniffs}.npy")
    y_train = np.load(load_directory + f"y_train_{sniffs}.npy")
    y_test = np.load(load_directory + f"y_test_{sniffs}.npy")

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    i = 0
    for train_idx, val_idx in kfold.split(X_train, y_train):
        print(" Kfold:", i)
        model = cnn_model_update(sniffs)
        history = model.fit(X_train[train_idx], y_train[train_idx],
                            validation_data=(X_train[val_idx], y_train[val_idx]),
                            epochs=100, verbose=0, callbacks=[early_stop, reduce_lr])

        model.save(save_directory + f"cnn_iso_{sniffs}_{i}.h5")
        # np.save(save_directory + f"cnn_val_loss_iso_{sniffs}_{i}.npy", np.array(history.history['val_loss']))
        # np.save(save_directory + f"cnn_loss_iso_{sniffs}_{i}.npy", np.array(history.history['loss']))

        predictions = model.predict(X_test)
        result = binarize(predictions, threshold=0.35)

        print(f"CNN Test Total acc {i}:{accuracy_score(y_test, result)}")

        y_pred_label_b = np.array([np.argmax(l) for l in result[:, :4]])
        y_label_b = np.array([np.argmax(l) for l in y_test[:, :4]])
        # print(f"CNN BZT acc {i}:{accuracy_score(y_label_b, y_pred_label_b)}")

        y_pred_label_m = np.array([np.argmax(l) for l in result[:, -4:]])
        y_label_m = np.array([np.argmax(l) for l in y_test[:, -4:]])
        # print(f"CNN MBZT acc {i}:{accuracy_score(y_label_m, y_pred_label_m)}")

        # np.save(save_directory + f"cnn_y_label_b_iso_{sniffs}_cv{i}.npy", y_label_b)
        # np.save(save_directory + f"cnn_y_label_m_iso_{sniffs}_cv{i}.npy", y_label_m)
        # np.save(save_directory + f"cnn_y_pred_label_b_iso_{sniffs}_cv{i}.npy", y_pred_label_b)
        # np.save(save_directory + f"cnn_y_pred_label_m_iso_{sniffs}_cv{i}.npy", y_pred_label_m)

        if i == 0:
            cnn_score = accuracy_score(y_label_b, y_pred_label_b)
        if i > 0:
            cnn_score = np.append(cnn_score, accuracy_score(y_label_b, y_pred_label_b))
        i += 1

    # np.save(save_directory + f"cnn_score_iso_{sniffs}.npy", cnn_score)

    i = 0
    X_train_1d = np.reshape(X_train, (-1, sniffs * group * group * analyte))
    X_test_1d = np.reshape(X_test, (-1, sniffs * group * group * analyte))

    counter = 0
    new_y = []
    remap = {}

    for i in range(len(y_train)):
        if (tuple(y_train[i]) not in remap.keys()):
            remap[tuple(y_train[i])] = counter
            counter += 1
        new_y.append(remap[tuple(y_train[i])])

    new_test_y = []
    for i in y_test:
        new_test_y.append(remap[tuple(i)])

    new_test_y = np.array(new_test_y)
    new_y = np.array(new_y)

    # np.save(save_directory + "linear_y_test_iso_{sniffs}.npy", new_test_y)

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    i = 0
    for train_idx, val_idx in kfold.split(X_train_1d, new_y):
        clf = SVC()
        clf.fit(X_train_1d[train_idx], new_y[train_idx])
        score = clf.score(X_train_1d[val_idx], new_y[val_idx])
        print(f'SVC_{i}:{score}')
        Y_pred = clf.predict(X_test_1d)
        pred = np.array([np.argmax(l) for l in Y_pred])
        if i == 0:
            svc_score = accuracy_score(new_test_y, pred)
        if i > 0:
            svc_score = np.append(svc_score, accuracy_score(new_test_y, pred))
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_1d[train_idx], new_y[train_idx])
        score = knn.score(X_train_1d[val_idx], new_y[val_idx])
        print(f'KNN_{i}:{score}')
        Y_pred = knn.predict(X_test_1d)
        pred = np.array([np.argmax(l) for l in Y_pred])
        if i == 0:
            knn_score = accuracy_score(new_test_y, pred)
        if i > 0:
            knn_score = np.append(knn_score, accuracy_score(new_test_y, pred))

        i += 1
    # np.save(save_directory + f"SVC_score_iso_{sniffs}.npy", svc_score)
    # np.save(save_directory + f"KNN_score_iso_{sniffs}.npy", knn_score)

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    i = 0
    for train_idx, val_idx in kfold.split(X_train_1d, y_train):
        model = ann_model(sniffs)
        history = model.fit(X_train_1d[train_idx], y_train[train_idx],
                            validation_data=(X_train_1d[val_idx], y_train[val_idx]),
                            epochs=1000, verbose=0, callbacks=[early_stop, reduce_lr])

        model.save(save_directory + f"ann_iso_{sniffs}_{i}.h5")
        # np.save(save_directory + f"ann_val_loss_iso_{sniffs}_{i}.npy", np.array(history.history['val_loss']))
        # np.save(save_directory + f"ann_loss_iso_{sniffs}_{i}.npy", np.array(history.history['loss']))

        predictions = model.predict(X_test_1d)
        result = binarize(predictions, threshold=0.35)

        print(f"ANN Test Total acc {i}:{accuracy_score(y_test, result)}")

        y_pred_label_b = np.array([np.argmax(l) for l in result[:, :4]])
        y_label_b = np.array([np.argmax(l) for l in y_test[:, :4]])
        # print(f"ANN BZT acc {i}:{accuracy_score(y_label_b, y_pred_label_b)}")

        y_pred_label_m = np.array([np.argmax(l) for l in result[:, -4:]])
        y_label_m = np.array([np.argmax(l) for l in y_test[:, -4:]])
        # print(f"ANN MBZT acc {i}:{accuracy_score(y_label_m, y_pred_label_m)}")

        '''
        np.save(save_directory + f"ann_y_label_b_iso_{sniffs}_cv{i}.npy", y_label_b)
        np.save(save_directory + f"ann_y_label_m_iso_{sniffs}_cv{i}.npy", y_label_m)
        np.save(save_directory + f"ann_y_pred_label_b_iso_{sniffs}_cv{i}.npy", y_pred_label_b)
        np.save(save_directory + f"ann_y_pred_label_m_iso_{sniffs}_cv{i}.npy", y_pred_label_m)
        '''

        if i == 0:
            ann_score = accuracy_score(y_label_b, y_pred_label_b)
        if i > 0:
            ann_score = np.append(ann_score, accuracy_score(y_label_b, y_pred_label_b))
        i += 1
    # np.save(save_directory + f"ann_score_iso_{sniffs}.npy", ann_score)


if __name__ == "__main__":
    process_sniff(20)
