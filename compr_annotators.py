import pickle as pkl

import numpy as np
from sklearn.model_selection import train_test_split


def smooth_annotator_output(signal, tolerance=20):
    for i in range(signal.shape[0]):
        interval_size = 0
        prev_qrs_end = 0
        is_prev_complex_qrs = False

        for j in np.arange(1, signal.shape[1] - 1):

            if signal[i, j] == 2:

                if signal[i, j - 1] == 0:
                    if is_prev_complex_qrs:
                        is_prev_complex_qrs = False
                        interval_size = j - prev_qrs_end
                        if interval_size < tolerance:
                            signal[i, prev_qrs_end:j] = np.ones(interval_size)

                if signal[i, j + 1] == 0:
                    is_prev_complex_qrs = True
                    prev_qrs_end = j

            elif signal[i, j] != 0:
                is_prev_complex_qrs = False
    return signal


def get_qrs_intervals(signal):
    intervals = []

    for i in range(signal.shape[0]):
        tmp = []
        start = end = 0
        for j in range(signal.shape[1] - 2):

            if signal[i, j] == 2:

                if signal[i, j - 1] != 2:
                    start = j
                elif signal[i, j + 1] != 2:
                    end = j
                    tmp.append([start, end])

        intervals.append(tmp)

    return intervals


def get_r_peaks(ecg, intervals):
    R_peaks = []

    for i in range(ecg.shape[0]):
        tmp = []
        for j in intervals[i]:
            maximum = np.argmax((ecg[i, j[0]:j[1] + 1])) + j[0]
            tmp.append(maximum)
        R_peaks.append(tmp)

    return R_peaks


def load_annotation():
    with open('C:\\Users\\donte_000\\PycharmProjects\\ClassificationECG\\raw_output.pkl', 'rb') as f:
        segmentation = pkl.load(f)

    segm = np.zeros((segmentation.shape[0], segmentation.shape[1], 4))

    for i in range(4):
        segm[:, :, i] = np.where(segmentation == i, np.ones(segmentation.shape), np.zeros(segmentation.shape))

    return segm


def load_processed_dataset(diags):
    from keras.utils import np_utils
    xy = load_dataset()
    X = xy["x"]
    annotation = load_annotation()
    X = np.concatenate((X, annotation), axis=2)
    Y = xy["y"]

    Y_new = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        for j in diags:
            if Y[i, j] == 1:
                Y_new[i] = 1
    Y = np_utils.to_categorical(Y_new, 2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


import pickle as pkl

import numpy as np
from scipy.signal import medfilt

from dataset import load_dataset


def find_peaks(x):
    peaks = []
    maximum = max(x[:, 0])

    # minimum = min(x[:, 0])
    tresh = maximum * 1 / 4
    # tresh = min(np.quantile(x[:, 0], 0.95) + maximum*1/10, maximum*1/2)
    for i in range(x.shape[0] - 1):
        if x[i, 0] > tresh:
            if x[i, 0] >= x[i + 1, 0] and x[i, 0] > x[i - 1, 0]:
                peaks.append(i)

    new_peaks = []
    for i in range(len(peaks) - 1):
        if peaks[i] < peaks[i + 1] + 10:
            # if x[peaks[i],0] > x[peaks[i]+25,0]+maximum*1/2:
            new_peaks.append(peaks[i])

    return new_peaks


def find_peaks_div(x):
    len = x.shape[0]
    y0 = np.zeros(len)
    y1 = np.zeros(len)
    y2 = np.zeros(len)
    y3 = np.zeros(len)
    for i in range(len - 2):
        y0[i + 2] = abs(x[i + 2] - x[i])
    for i in range(len - 4):
        y1[i + 4] = abs(x[i + 4] - 2 * x[i + 2] + x[i])
    for i in range(len - 4):
        y2[i + 4] = 1.3 * y0[i + 4] + 1.1 * y1[i + 4]
    for i in range(len - 4 - 7):
        for k in range(7):
            y3[i] += y2[i + 4 - k]
        y3[i] /= 8

    maxes = []

    max_curr = np.argmax(y3)
    max_curr_A = max(y3)
    maxes.append(max(0, max_curr - 40) + np.argmax(x[max(0, max_curr - 40):min(max_curr + 40, len)]))

    y3[max(0, max_curr - 50):min(max_curr + 50, len)] *= 0
    max_prev_A = max_curr_A

    max_curr = np.argmax(y3)
    max_curr_A = max(y3)

    while max_prev_A - max_curr_A < max_prev_A / 4:
        maxes.append(max(0, max_curr - 40) + np.argmax(x[max(0, max_curr - 40):min(max_curr + 40, len)]))
        y3[max(0, max_curr - 50):min(max_curr + 50, len)] *= 0
        max_prev_A = max_curr_A
        max_curr = np.argmax(y3)
        max_curr_A = max(y3)

    return maxes


def find_local_max_min(x):
    sup = []
    inf = []
    eps_max = 20
    eps_min = 10
    eps_up = 10
    eps_down = -5
    for i in range(x.shape[0]):
        tmp_max = x[i]
        tmp_min = x[i]
        for j in range(max(0, i - eps_max), min(x.shape[0], i + eps_max)):
            if tmp_max < x[j]:
                tmp_max = x[j]
                break
        if tmp_max == x[i] and tmp_max > eps_up:
            sup.append(i)

        for j in range(max(0, i - eps_min), min(x.shape[0], i + eps_min)):
            if tmp_min > x[j]:
                tmp_min = x[j]
                break
        if tmp_min == x[i] and tmp_min < eps_down:
            inf.append(i)
    return sup, inf


def cut_ecg_minmax(x, sup, inf):
    x_new = np.zeros(x.shape[0])
    x_new[sup] = x[sup]
    x_new[inf] = x[inf]
    return x_new


def make_thresh(x, sup, inf):
    x_filtred = medfilt(x, 15)
    x_new = np.zeros(x.shape[0])

    for i in sup:
        x_new[i] = x[i]
        j = i
        while x_filtred[j] >= x_filtred[min(j + 1, x.shape[0] - 1)] and x_filtred[
            min(j + 1, x.shape[0] - 1)] > 10 and j < x.shape[0] - 1:
            x_new[j + 1] = x_new[i]
            j += 1
        j = i
        while x_filtred[j] >= x_filtred[max(j - 1, 0)] and x_filtred[max(j - 1, 0)] > 10 and j > 0:
            x_new[j - 1] = x_new[i]
            j -= 1

    for i in inf:
        x_new[i] = x[i]
        j = i
        while x_filtred[j] <= x_filtred[min(j + 1, x.shape[0] - 1)] and x_filtred[
            min(j + 1, x.shape[0] - 1)] < -5 and j < x.shape[0] - 1 and x_new[j + 1] == 0:
            x_new[j + 1] = x_new[i]
            j += 1
        j = i
        while x_filtred[j] <= x_filtred[max(j - 1, 0)] and x_filtred[max(j - 1, 0)] < -5 and j > 0 and x_new[
            j - 1] == 0:
            x_new[j - 1] = x_new[i]
            j -= 1

    return x_new


def cut_ecg_cycles(x, peaks):
    cutted_ecg = []
    cycle = []
    for peak_num in range(len(peaks) - 1):
        cycle.append(peaks[peak_num + 1] - peaks[peak_num])
        cutted_ecg.append(x[peaks[peak_num]:peaks[peak_num + 1]])
    return cutted_ecg, cycle


def scale_ecg_zeros(cutted_ecg):
    length = 250
    new_cutted = []
    for i in range(len(cutted_ecg)):
        # for ii in cutted_ecg[i].shape[1]:
        tmp = np.zeros((length, cutted_ecg[i].shape[1]))
        cur_len = cutted_ecg[i].shape[0]
        if cur_len <= 10:
            continue
        for m in range(min(cur_len // 2, length // 2)):
            tmp[m] = cutted_ecg[i][m]
        for j in range(min(cur_len - cur_len // 2, length - length // 2)):
            tmp[-j] = cutted_ecg[i][-j]

        new_cutted.append(tmp)

    return np.array(new_cutted)


def scale_ecg_reshape(cutted_ecg):
    length = 250
    new_cutted = []
    for i in range(len(cutted_ecg)):
        tmp = np.zeros((length, cutted_ecg[i].shape[1]))
        scale = length / cutted_ecg[i].shape[0]
        for j in range(length):
            tmp[j] = cutted_ecg[i][int(j // scale)]
        new_cutted.append(tmp)

    return np.array(new_cutted)


def make_mean_var(cutted_ecg):
    length = 0
    for i in range(len(cutted_ecg)):
        if cutted_ecg[i].shape[0] > length:
            length = cutted_ecg[i].shape[0]

    mean = np.zeros((length, cutted_ecg[0].shape[1]))
    var = np.zeros((length, cutted_ecg[0].shape[1]))
    for i in range(len(cutted_ecg)):
        mean += cutted_ecg[i] / len(cutted_ecg)

    for i in range(len(cutted_ecg)):
        var += ((cutted_ecg[i] - mean) * (cutted_ecg[i] - mean)) / len(cutted_ecg)
    return mean, var


if __name__ == "__main__":
    infile = open('C:\\Users\\donte_000\\PycharmProjects\\Basic_Methods\\data\\data_old_and_new_without_noise.pkl', 'rb')
    (old, new) = pkl.load(infile)
    infile.close()

    X = old["x"]
    dif_R_peaks = []
    for i in range(X.shape[0]):
        x_i = X[i, :, :]

        peaks = find_peaks_div(x_i[:, 0])
        peaks.sort()
        dif_R_peaks.append(peaks)

    outfile = open('df_peaks_old_wo_noise.pkl', 'wb')
    pkl.dump(dif_R_peaks, outfile)

    ''' 
    infile = open('dif_R_peaks3.pkl', 'rb')
    dif_R_peaks = pkl.load(infile)
    infile.close()
    q1 = []
    q2 = []
    for i in range(X.shape[0]):
        qq = [elem for elem in dif_R_peaks[i] if elem not in neural_R_peaks[i] ]
        ww = [elem for elem in neural_R_peaks[i] if elem not in dif_R_peaks[i] ]
        q1.append(qq)
        q2.append(ww)
    counter = 0
    q3 =[]
    import matplotlib.pyplot as plt
    for i in range(X.shape[0]):
        if len(q1[i])!=0 or len(q2[i])!=0:
            ee = [a_i - b_i for a_i, b_i in zip(q1[i], q2[i])]
            q3.append(ee)
            plt.plot(X[i, :, 0], color = 'k')
            plt.scatter(neural_R_peaks[i], X[i,neural_R_peaks[i],0], color = 'red')
            plt.scatter(dif_R_peaks[i], X[i,dif_R_peaks[i],0], marker='.', color = 'blue')
            plt.show()
            print(ee)


            counter += 1
    print(counter)


    '''
