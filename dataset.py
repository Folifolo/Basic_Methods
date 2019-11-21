import os
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import pyedflib
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import BaselineWanderRemoval as bwr

# Порядок отведений
leads_names = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
pkl_filename = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\dataset_fixed_baseline.pkl"
FREQUENCY_OF_DATASET = 250
raw_dataset_path="C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\ecg_data_200.json"
pkl_filename = "C:\\Users\\donte_000\\Documents\\python_projects\\ecg_1078\\\dataset_fixed_baseline.pkl"
pkl_dictionary= "C:\\Users\\donte_000\\Documents\\python_projects\\ecg_1078\\dictionary_of_diagnoses.pkl"

def load_raw_dataset(raw_dataset):
    with open(raw_dataset, 'r') as f:
        data = json.load(f)
    X=[]
    Y=[]
    for case_id in data.keys():
        leads = data[case_id]['Leads']
        x = []
        y = []
        for i in range(len(leads_names)):
            lead_name = leads_names[i]
            x.append(leads[lead_name]['Signal'])

        signal_len = 2500
        delineation_tables = leads[leads_names[0]]['DelineationDoc']
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
    #X.shape = (200, 12, 5000)
    X = np.swapaxes(X, 1, 2)
    Y = np.swapaxes(Y, 1, 2)
    # X.shape = (200, 5000, 12)

    return {"x":X, "y":Y}

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
        if p[i]==0 and qrs[i]==0 and t[i]==0:
            background[i]=1
    return background

def fix_baseline_and_save_to_pkl(xy):
    print("start fixing baseline in the whole dataset. It may take some time, wait...")
    X= xy["x"]
    for i in range(X.shape[0]):
        print(str(i))

        for j in range(X.shape[2]):
            X[i, :, j] = bwr.fix_baseline_wander(X[i, :, j], FREQUENCY_OF_DATASET)
    xy['x']=X
    outfile = open(pkl_filename, 'wb')
    pkl.dump(xy, outfile)
    outfile.close()
    print("dataset saved, number of pacients = " + str(len(xy['x'])))

def seve_dictionary_of_diagnoses_to_pkl(data):
    diagnoses = data['60909568']['StructuredDiagnosisDoc'].keys()
    numbers = list(range(212))
    dictionary = dict(zip(diagnoses, numbers))
    outfile = open(pkl_dictionary, 'wb')
    pkl.dump(dictionary, outfile)
    outfile.close()

def get_number_of_diagnosis(diagnosis):
    infile = open(pkl_dictionary, 'rb')
    dictionary = pkl.load(infile)
    try: number = dictionary[diagnosis]
    except KeyError:
        print("This diagnosis is not correct.")
        number = None
    return number

def get_statistic_of_diagnosis(diagnosis, Y):
    num_of_patient = Y.shape[0]
    number_of_diagnosis = get_number_of_diagnosis(diagnosis)


    print("\nNumber of diagnosis: " + str(number_of_diagnosis))
    print("\nNumber of sick patients: " + str(sum(Y[:,number_of_diagnosis])) + " / " + str(num_of_patient))
    print("\nSick patient frequency in the data: " + str(sum(Y[:,number_of_diagnosis]) / num_of_patient))

    if sum(Y[:,number_of_diagnosis]) == 0:
        return 1
    else: return 0


def load_dataset(raw_dataset=raw_dataset_path, fixed_baseline=True):
    """
    при первом вызове с параметром fixed_baseline=True может работать очень долго, т.к. выполняет предобработку -
    затем резуотат предобрабоки сохраняется, чтоб не делать эту трудоемкую операцию много раз
    :param raw_dataset:
    :param fixed_baseline: флаг, нужно ли с выровненным дрейфом изолинии
    :return:
    """
    if fixed_baseline is True:
        print("you selected FIXED BASELINE WANDERING")
        if os.path.exists(pkl_filename): # если файл с предобработанным датасетом уже есть, не выполняем предобработку
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
        else:
            xy = load_raw_dataset(raw_dataset) # если файл с обработанным датасетом еще не создан, создаем
            fix_baseline_and_save_to_pkl(xy)
            infile = open(pkl_filename, 'rb')
            dataset_with_fixed_baseline = pkl.load(infile)
            infile.close()
            return dataset_with_fixed_baseline
    else:
        print("you selected NOT fixied BASELINE WANDERING")
        return load_raw_dataset(raw_dataset)
diag = [179,
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
def num_to_diag(num):
    diag_text =  ['non_specific_repolarisation_abnormalities_apical',
                  'non_specific_repolarisation_abnormalities_septal', 'sinus_bradycardia',
                  'non_specific_repolarisation_abnormalities_anterior_wall',
                  'atrial_fibrillation', 'electric_axis_vertical', 'electric_axis_horizontal',
                  'non_specific_repolarisation_abnormalities_lateral_wall',
                  'non_specific_repolarisation_abnormalities_inferior_wall',
                  'incomplete_right_bundle_branch_block', 'electric_axis_left_deviation',
                  'electric_axis_normal', 'right_atrial_hypertrophy',
                  'left_ventricular_hypertrophy', 'regular_normosystole']
    diag_num = diag
    return  diag_text[diag_num.index(num)]


if __name__ == "__main__":
    xy = load_dataset()
    Y = xy['y']
    #, 'electric_axis_indeterminate', 'electric_axis_severe_right_deviation'
    #for diag in ['non_specific_repolarisation_abnormalities_apical',
    #             'non_specific_repolarisation_abnormalities_septal', 'sinus_bradycardia',
    ##             'non_specific_repolarisation_abnormalities_anterior_wall',
    #             'atrial_fibrillation', 'electric_axis_vertical', 'electric_axis_horizontal',
    #             'non_specific_repolarisation_abnormalities_lateral_wall',
    #             'non_specific_repolarisation_abnormalities_inferior_wall',
    #             'incomplete_right_bundle_branch_block', 'electric_axis_left_deviation',
    #             'electric_axis_normal', 'right_atrial_hypertrophy',
    #             'left_ventricular_hypertrophy', 'regular_normosystole']:
    #for diag in ["pacemaker_presence_undefined","pacing_issues_undefined","UNIpolar_atrial_pacing","BIpolar_atrial_pacing",
    #             "UNIpolar_ventricular_pacing","BIpolar_ventricular_pacing","biventricular_pacing","p_synchrony",
    #             "ventricular_lead_pacing_issues","atrial_lead_pacing_issues", "atrial_oversensing",
    #             "atrial_undersensing","ventricular_oversensing","ventricular_undersensing","far_field_oversensing",
    #             "cross_talk","cardiac_pacing_other"]:
    print(str(get_number_of_diagnosis("left_atrial_hypertrophy")))

