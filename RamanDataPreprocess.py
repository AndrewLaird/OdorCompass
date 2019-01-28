import glob
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, minmax_scale
from sklearn.utils import shuffle

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


def organize_data(file_path, lcd):
    file_list = np.array(sorted(glob.glob(file_path + '/*.txt')))
    finder = re.compile("Map (\d[ab]?)")
    numbers = [finder.search(item).group(1) for item in file_list]
    data = [0 for _ in range(9)]
    subtract = 0
    for index in range(len(file_list)):
        i = index - subtract
        if (len(numbers[index]) == 2 and numbers[index][1] == 'b'):
            data[i - 1] = data[i - 1].append(pd.read_table(file_list[index], usecols=[3]))
            subtract += 1
        else:
            data[i] = pd.read_table(file_list[index], usecols=[3])
    data = np.array([np.array(l[:lcd]) for l in data])
    data = np.reshape(data, (len(data), -1, features))
    data = data[:, :, :cutoff]
    return data

def give_labels(data):
    temp = []
    counter = 0
    for i in range(chips):
        for j in range(group * group):
            if i == 0 and j == 0:
                temp = np.column_stack(
                    (np.repeat(i, len(data[counter])), np.repeat(j, len(data[counter])), data[counter]))
            else:
                temp = np.concatenate((temp, np.column_stack(
                    (np.repeat(i, len(data[counter])), np.repeat(j, len(data[counter])), data[counter]))), axis=0)
            counter += 1
    data = np.reshape(temp, (-1, cutoff + 2))
    return data


def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def preprocess(data):
    map = data[:, :2]
    data = data[:, 2:]
    data = savgol_filter(data, 11, 3, axis=0)
    back = Parallel(n_jobs=num_cores)(delayed(baseline_als)(j, 100000, 0.001) for j in data)
    data = np.subtract(data, back)
    data = np.reshape(data, (-1, cutoff))
    if data.min() < 0:
        data = data - data.min()
    data = np.column_stack((map, data))
    return data




def y_gen_a(number):
    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 0, 0, 1, 0])))
    y = m

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 0, 0, 1, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 0, 0, 0, 1])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 1, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 1, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 0, 0, 0, 1])))
    y = np.append(y, m)

    y = y.reshape(-1, 8)
    return y


def y_gen_b(number):
    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 0, 0, 0, 1])))
    y = m

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 0, 0, 1, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 0, 0, 1, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 0, 0, 0, 1])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 1, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 1, 0, 0, 0])))
    y = np.append(y, m)

    m = []

    y = y.reshape(-1, 8)
    return y


def y_gen_c(number):
    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 0, 0, 0, 0])))
    y = m

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 0, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 0, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 0, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 0, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 0, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 0, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 0, 0, 0, 0])))
    y = np.append(y, m)

    m = []

    y = y.reshape(-1, 8)
    return y


def y_gen_d(number):
    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 1, 0, 0, 0])))
    y = m

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 1, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 0, 0, 0, 1])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 0, 0, 1, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 0, 0, 1, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 0, 0, 0, 1])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 0, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []

    y = y.reshape(-1, 8)
    return y


def y_gen_e(number):
    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 0, 0, 1, 0])))
    y = m

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 1, 0, 0, 0, 1, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 0, 0, 0, 1])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 1, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([1, 0, 0, 0, 1, 0, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 1, 0, 0, 0, 1, 0, 0])))
    y = np.append(y, m)

    m = []
    for g in range(number):
        m = np.concatenate((m, np.array([0, 0, 0, 1, 0, 0, 0, 1])))
    y = np.append(y, m)

    m = []

    y = y.reshape(-1, 8)
    return y


def deviation(data):
    e = []
    for i in range(sniffs):
        for j in range(group):
            for k in range(group):
                for l in range(analyte):
                    e = np.append(e, np.std(data[:, i, j, k, l]))
    e = np.reshape(e, (sniffs, group, group, analyte))
    return e


def noise(e):
    rnd = []
    for i in range(sniffs):
        for j in range(group):
            for k in range(group):
                for l in range(analyte):
                    rnd = np.append(rnd, 0.1 * np.random.normal(0, e[i, j, k, l]))
    rnd = np.reshape(rnd, (sniffs, group, group, analyte))
    return rnd


def data_aug_train(data):
    number = len(data)
    e = deviation(data)
    maps_full = data
    first_flips = np.array([np.flip(l, 2) + np.flip(noise(e), 2) for l in data])
    maps_full = np.concatenate((maps_full, first_flips), axis=0)
    for h in range(1, 4):
        temp_maps = np.array([np.rot90(l, h, axes=(1, 2)) + np.rot90(noise(e), h, axes=(1, 2)) for l in data])
        temp_flips = np.array([np.flip(np.rot90(t, h, axes=(1, 2)), 2) +
                               np.flip(np.rot90(noise(e), h, axes=(1, 2)), 2) for t in data])
        maps_full = np.concatenate((maps_full, temp_maps, temp_flips), axis=0)
    return maps_full, number


def data_aug_test(data):
    number = len(data)
    maps_full = data
    first_flips = np.array([np.flip(l, 2) for l in data])
    maps_full = np.concatenate((maps_full, first_flips), axis=0)
    for h in range(1, 4):
        temp_maps = np.array([np.rot90(l, h, axes=(1, 2)) for l in data])
        temp_flips = np.array([np.flip(t, 2) for t in temp_maps])
        maps_full = np.concatenate((maps_full, temp_maps, temp_flips), axis=0)
    return maps_full, number


def bundle(data):
    temp = []
    number = 100000000000
    for i in range(chips):
        for j in range(group * group):
            counter = np.count_nonzero(np.all((data[:, 0] == i, data[:, 1] == j), axis=0))
            if counter < number:
                number = counter
    for i in range(chips):
        for j in range(group * group):
            if i == 0 and j == 0:
                temp = np.array([data[z] for z in range(len(data)) if
                                 np.array_equal(data[z, 0], 0) and np.array_equal(data[z, 1], 0)])[:number]
                temp = np.expand_dims(temp, axis=0)
            else:
                temp_temp = np.array([data[z] for z in range(len(data)) if
                                      np.array_equal(data[z, 0], i) and np.array_equal(data[z, 1], j)])[:number]
                temp_temp = np.expand_dims(temp_temp, axis=0)
                temp = np.concatenate((temp, temp_temp), axis=0)
    data = temp[:, :, 2:]
    data = np.reshape(np.swapaxes(data, 0, 1), (-1, chips, group * group, analyte))
    return data


def scale_data(data):
    # Scale our matrices to improve convergence
    data = np.reshape(data, (-1, chips, group * group * analyte))
    for k in range(len(data)):
        for i in range(chips):
            temp = scale(data[k, i])
            if i == 0 and k == 0:
                scaled = temp
            else:
                scaled = np.concatenate((scaled, temp))
    return np.reshape(scaled, (-1, chips, group * group, analyte))

def process(X_train, X_test, file):
    X_train = np.column_stack((X_train[:, 0], X_train[:, 1], nmf.transform(X_train[:, 2:])[:, :analyte]))
    X_train = bundle(X_train)
    X_train = scale_data(X_train)
    for j in range(sniffs):
        if j == 0:
            temp = shuffle(X_train)
            temp = np.reshape(temp[:(len(temp) // sniffs) * sniffs], (-1, sniffs, group, group, analyte))
        if j > 0:
            temp_temp = shuffle(X_train)
            temp_temp = np.reshape(temp_temp[:(len(temp_temp) // sniffs) * sniffs], (-1, sniffs, group, group, analyte))
            temp = np.concatenate((temp, temp_temp), axis=0)
    X_train = temp
    X_train, number = data_aug_train(X_train)
    if file == 0:
        y_train = y_gen_a(number)
    if file == 1:
        y_train = y_gen_b(number)
    if file == 2:
        y_train = y_gen_c(number)
    if file == 3:
        y_train = y_gen_d(number)
    if file == 4:
        y_train = y_gen_e(number)
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test = np.column_stack((X_test[:, 0], X_test[:, 1], nmf.transform(X_test[:, 2:])[:, :analyte]))
    X_test = bundle(X_test)
    X_test = scale_data(X_test)
    X_test = shuffle(X_test)
    X_test = np.reshape(X_test[:(len(X_test) // sniffs) * sniffs], (-1, sniffs, group, group, analyte))
    X_test, number = data_aug_test(X_test)
    if file == 0:
        y_test = y_gen_a(number)
    if file == 1:
        y_test = y_gen_b(number)
    if file == 2:
        y_test = y_gen_c(number)
    if file == 3:
        y_test = y_gen_d(number)
    if file == 4:
        y_test = y_gen_e(number)
    return X_train, X_test, y_train, y_test


base_location = "/home/andrew/research/directional/direction/"
specific_locations = ["Multiplex/9-21-18", "Multiplex/9-17-18", "8-8-18", "9-14-18", "10-11-18"]
direction_of_smell = [(0, 2), (2, -1), (2, -1), (-1, 0), (2, 2)]

spectra_locations = [base_location + sub_location for sub_location in specific_locations]


X_train, X_test, y_train, y_test = None,None,None,None
X_train_preprocess = []
X_test_preprocess = []

for i in range(len(spectra_locations)):
    print("processing file",spectra_locations[i])
    data = organize_data(spectra_locations[i], 4124880)
    data = give_labels(data)
    X_train_a, X_test_a = train_test_split(data, test_size=0.2)
    print("preprocess section")
    X_train_aa = preprocess(X_train_a)
    X_test_aa = preprocess(X_test_a)

    X_train_preprocess.append(X_train_aa)
    X_train_preprocess.append(X_test_aa)

nmf = NMF(n_components=analyte+1)
X_train_numpy = np.concatenate(X_train_preprocess)
X_train_numpy[:, 2:] = minmax_scale(X_train_numpy[:, 2:], axis=1)
nmf.fit(X_train_numpy[:, 2:])


for i in range(len(spectra_locations)):
    print("process and determine y_labels")
    X_train_a, X_test_a, y_train_a, y_test_a = process(X_train_preprocess[i], X_test_preprocess[i], i)

    if(i == 0):
        X_train = X_train_a
        X_test =  X_test_a
        y_train = y_train_a
        y_test = y_test_a
    else:
        X_train = np.concatenate((X_train,X_train_a),axis=0)
        X_test = np.concatenate((X_test,X_test_a),axis=0)

        y_train = np.concatenate((y_train,y_train_a),axis=0)

        y_test = np.concatenate((y_test,y_test_a),axis=0)


X_test, y_test = shuffle(X_test, y_test)
X_train, y_train = shuffle(X_train, y_train)

save_directory = "/home/andrew/research/Smell-O-ScopePaper/direct_results_test/"

for sniffs in range(1,16):
    np.save(save_directory + f"X_train_{sniffs}.npy", X_train)
    np.save(save_directory + f"X_test_{sniffs}.npy", X_test)
    np.save(save_directory + f"y_train_{sniffs}.npy", y_train)
    np.save(save_directory + f"y_test_{sniffs}.npy", y_test)

