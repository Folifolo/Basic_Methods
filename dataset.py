import json
import os
import pickle as pkl

from sklearn.model_selection import train_test_split
import BaselineWanderRemoval as bwr
import numpy as np
import sys

# Порядок отведений
LEADS_NAMES = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
DATA_FOLDER = 'C:\\data\\basic\\'
FREQUENCY_OF_DATASET = 250
DATA_SIZE = 6002

RAW_DATASET_PATH = DATA_FOLDER + "data_1078.json"
PKL_FILENAME = DATA_FOLDER + "dataset_fixed_baseline.pkl"
PKL_DICTIONARY = DATA_FOLDER + "dictionary_of_diagnoses.pkl"
OLD_6002_PARH = DATA_FOLDER + "6002_norm_old.pkl"

NEW_6002_PATH = DATA_FOLDER + "6002_norm.pkl"
NEW_DATASET_PATH = DATA_FOLDER + "data_2033.pkl"

MOST_FREQ_DIAGS_NUMS = [179,
                        198,
                        111,
                        127,
                        140,
                        8,
                        138,
                        185,
                        206,
                        186,
                        195,
                        207,
                        66,
                        157,
                        85]
MOST_FREQ_DIAGS_NUMS_NEW = [161,
                            158,
                            2,
                            156,
                            15,
                            44,
                            45,
                            157,
                            159,
                            60,
                            47,
                            46,
                            119,
                            123,
                            0]
MOST_FREQ_DIAGS_NAMES = ['non_specific_repolarisation_abnormalities_apical',
                         'non_specific_repolarisation_abnormalities_septal',
                         'sinus_bradycardia',
                         'non_specific_repolarisation_abnormalities_anterior_wall',
                         'atrial_fibrillation',
                         'electric_axis_vertical',
                         'electric_axis_horizontal',
                         'non_specific_repolarisation_abnormalities_lateral_wall',
                         'non_specific_repolarisation_abnormalities_inferior_wall',
                         'incomplete_right_bundle_branch_block',
                         'electric_axis_left_deviation',
                         'electric_axis_normal',
                         'right_atrial_hypertrophy',
                         'left_ventricular_hypertrophy',
                         'regular_normosystole']

diags_norm_rythm = [0]
diags_fibrilation = [15]
diags_flutter = [16, 17, 18]
diags_hypertrophy = [119, 120]
diags_extrasystole = [69, 70, 71, 72, 73, 74, 75, 86]

NEW_DIAGNOSIS = [diags_norm_rythm, diags_fibrilation, diags_flutter, diags_hypertrophy, diags_extrasystole]

def get_diag_dict():
    def deep(data, diag_list):
        for diag in data:
            if diag['type'] == 'diagnosis':
                diag_list.append(diag['name'])
            else:
                deep(diag['value'], diag_list)

    infile = open('c:\\data\\diagnosis.json', 'rb')
    data = json.load(infile)

    diag_list = []
    deep(data, diag_list)

    diag_num = list(range(len(diag_list)))
    diag_dict = dict(zip(diag_list, diag_num))

    return diag_dict


def load_raw_dataset(raw_dataset):
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X = []
    Y = []
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(LEADS_NAMES)):
            lead_name = LEADS_NAMES[i]
            x.append(leads[lead_name]['Signal'])

        signal_len = 2500
        delineation_tables = leads[LEADS_NAMES[0]]['DelineationDoc']
        p_delin = delineation_tables['p']
        qrs_delin = delineation_tables['qrs']
        t_delin = delineation_tables['t']

        p = get_mask(p_delin, signal_len)
        qrs = get_mask(qrs_delin, signal_len)
        t = get_mask(t_delin, signal_len)
        background = get_background(p, qrs, t)

        y.append(p)
        y.append(qrs)
        y.append(t)
        y.append(background)

        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    # X.shape = (200, 12, 5000)
    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)
    # X.shape = (200, 5000, 12)

    return {"x": X, "y": Y}


def get_mask(table, length):
    mask = [0] * length  # забиваем сначала маску нулями
    for triplet in table:
        start = triplet[0]
        end = triplet[2]
        for i in range(start, end, 1):
            mask[i] = 1
    return mask


def get_background(p, qrs, t):
    background = np.zeros_like(p)
    for i in range(len(p)):
        if p[i] == 0 and qrs[i] == 0 and t[i] == 0:
            background[i] = 1
    return background


def fix_baseline_and_save_to_pkl(xy):
    print("start fixing baseline in the whole dataset. It may take some time, wait...")
    X = xy["x"]
    for i in range(X.shape[0]):
        print(str(i))

        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy['x'] = X
    outfile = open(PKL_FILENAME, 'wb')
    pkl.dump(xy, outfile)
    outfile.close()
    print("dataset saved, number of pacients = " + str(len(xy['x'])))


def save_dictionary_of_diagnoses_to_pkl(data):
    diagnoses = data['60909568']['StructuredDiagnosisDoc'].keys()
    numbers = list(range(212))
    dictionary = dict(zip(diagnoses, numbers))
    outfile = open(PKL_DICTIONARY, 'wb')
    pkl.dump(dictionary, outfile)
    outfile.close()


def get_number_of_diagnosis(diagnosis):
    infile = open(PKL_DICTIONARY, 'rb')
    dictionary = pkl.load(infile)
    try:
        number = dictionary[diagnosis]
    except KeyError:
        print("This diagnosis is not correct.")
        number = None
    return number


def get_statistic_of_diagnosis(diagnosis, Y):
    num_of_patient = Y.shape[0]
    number_of_diagnosis = get_number_of_diagnosis(diagnosis)

    print("\nNumber of diagnosis: " + str(number_of_diagnosis))
    print("\nNumber of sick patients: " + str(sum(Y[:, number_of_diagnosis])) + " / " + str(num_of_patient))
    print("\nSick patient frequency in the data: " + str(sum(Y[:, number_of_diagnosis]) / num_of_patient))

    if sum(Y[:, number_of_diagnosis]) == 0:
        return 1
    else:
        return 0


def load_dataset(raw_dataset=RAW_DATASET_PATH, fixed_baseline=True):
    """
    при первом вызове с параметром fixed_baseline=True может работать очень долго, т.к. выполняет предобработку -
    затем резуотат предобрабоки сохраняется, чтоб не делать эту трудоемкую операцию много раз
    :param raw_dataset:
    :param fixed_baseline: флаг, нужно ли с выровненным дрейфом изолинии
    :return:
    """
    if fixed_baseline is True:
        print("you selected FIXED BASELINE WANDERING")
        if os.path.exists(PKL_FILENAME):  # если файл с предобработанным датасетом уже есть, не выполняем предобработку
            infile = open(PKL_FILENAME, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
        else:
            xy = load_raw_dataset(raw_dataset)  # если файл с обработанным датасетом еще не создан, создаем
            fix_baseline_and_save_to_pkl(xy)
            infile = open(PKL_FILENAME, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
    else:
        print("you selected NOT fixied BASELINE WANDERING")
        return load_raw_dataset(raw_dataset)


def load_new_dataset_6002():
    infile = open(NEW_6002_PATH, 'rb')
    X = pkl.load(infile)
    infile.close()

    infile = open(NEW_DATASET_PATH, 'rb')
    Y = pkl.load(infile)["y"]
    infile.close()

    return {"x": X, "y": Y}


def load_old_dataset_6002():
    outfile = open(OLD_6002_PARH, 'rb')
    X = pkl.load(outfile)
    outfile.close()

    xy = load_dataset()
    Y = xy["y"]

    return {"x": X, "y": Y}


def diagnosis_to_binary(Y, diagnosis):
    from utils import to_categorical

    Y_new = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        for j in diagnosis:
            if Y[i, j] == 1:
                Y_new[i] = 1

    Y = to_categorical(Y_new, 2)
    return Y


def load_split(diagnosis, rnd_state=42, is_new=True):
    if is_new:
        xy = load_new_dataset_6002()
    else:
        xy = load_old_dataset_6002()

    X = xy['x']
    X = np.expand_dims(X, 2)

    Y = xy['y']
    Y = diagnosis_to_binary(Y, diagnosis)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=rnd_state)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=rnd_state)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def num_to_diag(num):
    return MOST_FREQ_DIAGS_NAMES[MOST_FREQ_DIAGS_NUMS.index(num)]


if __name__ == "__main__":
    xy = load_dataset('new')
    Y = xy['y']
    counter = 0
    for i in range(Y.shape[0]):
        if Y[i,119] == 1:
            counter += 1;

    print(counter)
    qwe = get_diag_dict()
    for diag in ["left_atrial_hypertrophy"]:
        print(str(qwe[diag])+',')
    #print(str(get_number_of_diagnosis("atrial_fibrillation")))
