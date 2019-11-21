from dataset import num_to_diag, diag
import os
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from  scipy.signal import medfilt
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import *
from preprocessing import *
from lda_tree_tests import find_optimal_param



num_components = 100
pkl1 = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\xy_6002.pkl"
pkl2 = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\xy_102.pkl"

infile = open(pkl1, 'rb')
xy = pkl.load(infile)
infile.close()

X = xy["x"]
Y = xy["y"]
print(Y.shape)

pca = PCA(n_components=X.shape[0])
b = X


def predict(lda, x, y, m):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if len(x) != 0:
        probs= lda.predict_proba(x)[:, 1]

        for j in range(len(x)):
            if probs[j] > m:
                if y[j] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if y[j] == 1:
                    fn +=1
                else:
                    tn +=1

    return tp, fp, fn, tn

for d in range(15):
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    train_tp = 0
    train_fp = 0
    train_fn = 0
    train_tn = 0

    for train_index, test_index in cross_val(b.shape[0], 1):
        lda = LinearDiscriminantAnalysis()
        lda.fit(b[train_index, :num_components],Y[train_index,d])
        m = find_optimal_param(lda, b[train_index, :num_components], Y[train_index,d])
        #tree.grow()
        tp, fp, fn, tn = predict(lda, b[test_index, :num_components], Y[test_index,d], m)
        test_tp += tp
        test_fp += fp
        test_fn += fn
        test_tn += tn

        tp, fp, fn, tn = predict(lda, b[train_index, :num_components], Y[train_index,d], m)
        train_tp += tp
        train_fp += fp
        train_fn += fn
        train_tn += tn

    train_se = train_tp/(train_tp+train_fn)
    train_sp = train_tn/(train_tn+train_fp)
    print("Train. Se = %s, Train. Sp = %s" %(round(train_sp,4),  round(train_se,4)))
    test_se = test_tp/(test_tp+test_fn)
    test_sp=test_tn/(test_tn+test_fp)
    print("Val. Se = %s, Val. Sp = %s" %(round(test_sp, 4),  round(test_se, 4)))

print("=============")


infile = open(pkl2, 'rb')
xy = pkl.load(infile)
infile.close()

X = xy["x"]
Y = xy["y"]

b = X


def predict(lda, x, y, m):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if len(x) != 0:
        probs= lda.predict_proba(x)[:, 1]

        for j in range(len(x)):
            if probs[j] > m:
                if y[j] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if y[j] == 1:
                    fn +=1
                else:
                    tn +=1

    return tp, fp, fn, tn

for d in range(15):
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    train_tp = 0
    train_fp = 0
    train_fn = 0
    train_tn = 0

    for train_index, test_index in cross_val(b.shape[0], 1):
        lda = LinearDiscriminantAnalysis()
        lda.fit(b[train_index, :num_components],Y[train_index,d])
        m = find_optimal_param(lda, b[train_index, :num_components], Y[train_index,d])
        #tree.grow()
        tp, fp, fn, tn = predict(lda, b[test_index, :num_components], Y[test_index,d], m)
        test_tp += tp
        test_fp += fp
        test_fn += fn
        test_tn += tn

        tp, fp, fn, tn = predict(lda, b[train_index, :num_components], Y[train_index,d], m)
        train_tp += tp
        train_fp += fp
        train_fn += fn
        train_tn += tn

    train_se = train_tp/(train_tp+train_fn)
    train_sp = train_tn/(train_tn+train_fp)
    print("Train. Se = %s, Train. Sp = %s" %(round(train_sp,4),  round(train_se,4)))
    test_se = test_tp/(test_tp+test_fn)
    test_sp=test_tn/(test_tn+test_fp)
    print("Val. Se = %s, Val. Sp = %s" %(round(test_sp, 4),  round(test_se, 4)))
